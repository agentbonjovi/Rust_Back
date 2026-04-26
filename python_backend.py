#!/usr/bin/env python3
"""Python-бэкенд ML Optimizer.

Точный порт src/main.rs на чистом Python (без numpy / pandas / sklearn),
чтобы сравнение времени выполнения с Rust отражало реальную производительность
языка на CPU-bound нагрузке, а не время работы библиотек, написанных на C.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


# ==================== МОДЕЛЬ ДАННЫХ ====================


class DataType(Enum):
    INTEGER = "Integer"
    FLOAT = "Float"
    CATEGORICAL = "Categorical"


class Distribution(Enum):
    UNIFORM = "Uniform"
    NORMAL = "Normal"
    SKEWED = "Skewed"


class Operator(Enum):
    EQ = "="
    NE = "<>"
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    BETWEEN = "BETWEEN"
    LIKE = "LIKE"


class ScanMethod(Enum):
    SEQ_SCAN = "Seq Scan"
    INDEX_SCAN = "Index Scan"
    BITMAP_SCAN = "Bitmap Scan"

    def estimate_cost(self, table_size: int, selectivity: float, has_index: bool) -> float:
        if self is ScanMethod.SEQ_SCAN:
            return table_size * 1.0
        if self is ScanMethod.INDEX_SCAN:
            if has_index:
                return table_size * 0.1 + table_size * selectivity * 0.5
            return float("inf")
        if self is ScanMethod.BITMAP_SCAN:
            if has_index:
                return table_size * 0.15 + table_size * selectivity * 0.3
            return float("inf")
        return float("inf")


@dataclass
class ColumnStats:
    name: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    unique_values: int
    null_frac: float


@dataclass
class HistogramBucket:
    lower: float
    upper: float
    count: int
    frequency: float


@dataclass
class ColumnMetadata:
    name: str
    data_type: DataType
    distribution: Distribution
    distribution_params: dict
    stats: ColumnStats
    histogram: list


@dataclass
class Predicate:
    column: str
    operator: Operator
    value: float
    value2: Optional[float] = None


@dataclass
class TrainingExample:
    features: list
    actual_selectivity: float
    predicate_type: Operator
    column_name: str


@dataclass
class PredicateSelectivity:
    predicate: Predicate
    actual_selectivity: float
    ml_estimate: float
    histogram_estimate: float
    approx_estimate: float
    error_ml: float
    error_histogram: float
    error_approx: float
    recommended_scan: ScanMethod


# ==================== KNN ====================


class SimpleKNN:
    def __init__(self, k: int) -> None:
        self.training_data: list[TrainingExample] = []
        self.k = k

    def add_example(self, example: TrainingExample) -> None:
        self.training_data.append(example)

    @staticmethod
    def euclidean_distance(a: list[float], b: list[float]) -> float:
        total = 0.0
        for x, y in zip(a, b):
            d = x - y
            total += d * d
        return math.sqrt(total)

    def find_k_nearest(self, features: list[float]) -> list[TrainingExample]:
        distances = [
            (self.euclidean_distance(features, ex.features), ex)
            for ex in self.training_data
        ]
        distances.sort(key=lambda d: d[0])
        return [ex for _, ex in distances[: min(self.k, len(distances))]]

    def predict(self, features: list[float]) -> float:
        if not self.training_data:
            return 0.5
        nearest = self.find_k_nearest(features)
        return sum(ex.actual_selectivity for ex in nearest) / len(nearest)

    def predict_weighted(self, features: list[float]) -> float:
        if not self.training_data:
            return 0.5
        distances = [
            (self.euclidean_distance(features, ex.features), ex)
            for ex in self.training_data
        ]
        distances.sort(key=lambda d: d[0])
        nearest = distances[: min(self.k, len(distances))]
        total_inv = sum(1.0 / (d + 0.0001) for d, _ in nearest)
        weighted = sum(ex.actual_selectivity * (1.0 / (d + 0.0001)) for d, ex in nearest)
        return weighted / total_inv


class KNNSelectivityModel:
    def __init__(self, k: int) -> None:
        self.knn = SimpleKNN(k)
        self.feature_names = [
            "operator_eq",
            "operator_lt",
            "operator_gt",
            "operator_between",
            "value_normalized",
            "value2_normalized",
            "column_mean",
            "column_std",
            "column_unique_ratio",
            "column_min",
            "column_max",
            "value_vs_mean",
        ]

    def extract_features(self, predicate: Predicate, metadata: ColumnMetadata) -> list[float]:
        features: list[float] = []
        features.append(1.0 if predicate.operator is Operator.EQ else 0.0)
        features.append(1.0 if predicate.operator in (Operator.LT, Operator.LE) else 0.0)
        features.append(1.0 if predicate.operator in (Operator.GT, Operator.GE) else 0.0)
        features.append(1.0 if predicate.operator is Operator.BETWEEN else 0.0)

        s = metadata.stats
        if s.max > s.min:
            value_norm = (predicate.value - s.min) / (s.max - s.min)
        else:
            value_norm = 0.5
        features.append(value_norm)

        if predicate.value2 is not None:
            if s.max > s.min:
                value2_norm = (predicate.value2 - s.min) / (s.max - s.min)
            else:
                value2_norm = 0.5
        else:
            value2_norm = value_norm
        features.append(value2_norm)

        features.append(s.mean / 100000.0)
        features.append(s.std / 100000.0)
        features.append(s.unique_values / max(s.count, 1))
        features.append(s.min / 100000.0)
        features.append(s.max / 100000.0)
        features.append((predicate.value - s.mean) / max(s.std, 0.001))
        return features

    def add_training_example(self, example: TrainingExample) -> None:
        self.knn.add_example(example)

    def predict(self, predicate: Predicate, metadata: ColumnMetadata) -> float:
        if not self.knn.training_data:
            return self.heuristic_predict(predicate, metadata)
        features = self.extract_features(predicate, metadata)
        return self.knn.predict_weighted(features)

    def heuristic_predict(self, predicate: Predicate, metadata: ColumnMetadata) -> float:
        s = metadata.stats
        std = max(s.std, 0.001)
        op = predicate.operator
        if op is Operator.EQ:
            return 1.0 / max(s.unique_values, 1)
        if op is Operator.LT:
            z = (predicate.value - s.mean) / std
            return 0.5 * (1.0 + math.tanh(z / 1.414))
        if op is Operator.GT:
            z = (predicate.value - s.mean) / std
            return 0.5 * (1.0 - math.tanh(z / 1.414))
        if op is Operator.BETWEEN:
            if predicate.value2 is not None:
                z1 = (predicate.value - s.mean) / std
                z2 = (predicate.value2 - s.mean) / std
                return 0.5 * (math.tanh(z2 / 1.414) - math.tanh(z1 / 1.414))
            return 0.2
        return 0.5

    def evaluate(self, test_predicates: list[tuple]) -> float:
        if not test_predicates:
            return 0.0
        total = 0.0
        for predicate, actual, metadata in test_predicates:
            predicted = self.predict(predicate, metadata)
            total += abs(predicted - actual)
        return total / len(test_predicates)


# ==================== ГЕНЕРАЦИЯ ДАННЫХ ====================


class DataGenerator:
    def generate_synthetic_csv(self, path: Path, num_rows: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["id", "age", "salary", "department_id", "score", "years_experience"]
            )
            for i in range(num_rows):
                age = int(round(random.gauss(35, 10)))
                u = random.random()
                salary = int(round(50_000 * ((1.0 - u) ** -1.0)))
                dept_id = random.randint(1, 20)
                score = random.uniform(0, 100)
                if age < 22:
                    experience = random.randint(0, 2)
                else:
                    experience = random.randint(0, max(min(age - 22, 40), 0))
                writer.writerow([i, age, salary, dept_id, f"{score:.2f}", experience])
        print(f"✅ Сгенерирован синтетический CSV: {num_rows} строк")


# ==================== АНАЛИЗ ====================


class DataAnalyzer:
    def __init__(self, data: dict[str, list[float]]) -> None:
        self.data = data
        self.metadata: dict[str, ColumnMetadata] = {}
        self.knn_model = KNNSelectivityModel(5)

    @classmethod
    def from_csv(cls, path: Path) -> "DataAnalyzer":
        data: dict[str, list[float]] = {}
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            headers = next(reader)
            for h in headers:
                data[h] = []
            for row in reader:
                for i, h in enumerate(headers):
                    if i < len(row):
                        try:
                            data[h].append(float(row[i]))
                        except ValueError:
                            pass
        return cls(data)

    def compute_statistics(self) -> None:
        for name, values in self.data.items():
            if not values:
                continue
            stats = self._column_stats(name, values)
            histogram = self._build_histogram(values, 20)
            data_type = self._infer_data_type(values)
            distribution, dparams = self._infer_distribution(values)
            self.metadata[name] = ColumnMetadata(
                name=name,
                data_type=data_type,
                distribution=distribution,
                distribution_params=dparams,
                stats=stats,
                histogram=histogram,
            )

    @staticmethod
    def _column_stats(name: str, values: list[float]) -> ColumnStats:
        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(max(var, 0.0))
        unique = len(set(values))
        return ColumnStats(
            name=name,
            count=n,
            mean=mean,
            std=std,
            min=min(values),
            max=max(values),
            unique_values=unique,
            null_frac=0.0,
        )

    def numeric_columns_sorted(self) -> list[str]:
        return sorted(self.metadata.keys())

    @staticmethod
    def _build_histogram(values: list[float], num_buckets: int) -> list[HistogramBucket]:
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if hi == lo:
            return [HistogramBucket(lo, lo, len(values), 1.0)]
        width = (hi - lo) / num_buckets
        counts = [0] * num_buckets
        for v in values:
            idx = int((v - lo) / width)
            if idx >= num_buckets:
                idx = num_buckets - 1
            counts[idx] += 1
        total = float(len(values))
        return [
            HistogramBucket(
                lower=lo + i * width,
                upper=lo + (i + 1) * width,
                count=c,
                frequency=c / total,
            )
            for i, c in enumerate(counts)
        ]

    @staticmethod
    def _infer_data_type(values: list[float]) -> DataType:
        all_int = all(v == int(v) for v in values)
        unique = len(set(values))
        if unique < len(values) // 10 and all_int:
            return DataType.CATEGORICAL
        if all_int:
            return DataType.INTEGER
        return DataType.FLOAT

    @staticmethod
    def _infer_distribution(values: list[float]) -> tuple[Distribution, dict]:
        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(max(var, 0.0))
        if var == 0 or std == 0:
            return Distribution.UNIFORM, {"min": min(values), "max": max(values)}
        skewness = sum((v - mean) ** 3 for v in values) / (n * var * std)
        if abs(skewness) > 1.0:
            return Distribution.SKEWED, {"alpha": 2.0 + abs(skewness)}
        if abs(std - mean * 0.3) < mean * 0.1:
            return Distribution.NORMAL, {"mean": mean, "std": std}
        return Distribution.UNIFORM, {"min": min(values), "max": max(values)}

    def train_knn_model(self) -> None:
        print("\n\U0001f9e0 Обучение KNN модели на исторических данных...")
        examples: list[TrainingExample] = []
        for col_name, values in self.data.items():
            if not values or col_name not in self.metadata:
                continue
            metadata = self.metadata[col_name]
            examples.extend(
                self._generate_training_predicates(col_name, values, metadata)
            )
        for ex in examples:
            self.knn_model.add_training_example(ex)
        print(
            f"✅ KNN модель обучена ({len(self.knn_model.knn.training_data)} примеров)"
        )

    def _generate_training_predicates(
        self, col_name: str, values: list[float], metadata: ColumnMetadata
    ) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        if len(values) < 10:
            return examples
        operators = [Operator.EQ, Operator.LT, Operator.GT, Operator.BETWEEN]
        sample_size = min(20, len(values))
        step = max(len(values) // sample_size, 1)
        for i in range(0, len(values), step):
            val = values[i]
            for op in operators:
                if op is Operator.BETWEEN:
                    val2 = values[i + step] if i + step < len(values) else values[-1]
                    predicate = Predicate(
                        column=col_name,
                        operator=Operator.BETWEEN,
                        value=min(val, val2),
                        value2=max(val, val2),
                    )
                else:
                    predicate = Predicate(column=col_name, operator=op, value=val)
                actual = self.compute_actual_selectivity(predicate)
                features = self.knn_model.extract_features(predicate, metadata)
                examples.append(
                    TrainingExample(
                        features=features,
                        actual_selectivity=actual,
                        predicate_type=op,
                        column_name=col_name,
                    )
                )
        return examples

    def estimate_selectivity_histogram(self, predicate: Predicate) -> float:
        column = self.metadata.get(predicate.column)
        if column is None:
            return 0.5
        op = predicate.operator
        if op is Operator.EQ:
            for b in column.histogram:
                if b.lower <= predicate.value <= b.upper:
                    if b.upper == b.lower:
                        return b.frequency
                    return b.frequency / (b.upper - b.lower)
            return 0.0
        if op is Operator.LT:
            sel = 0.0
            for b in column.histogram:
                if b.upper <= predicate.value:
                    sel += b.frequency
                elif b.lower < predicate.value and b.upper > b.lower:
                    fraction = (predicate.value - b.lower) / (b.upper - b.lower)
                    sel += b.frequency * fraction
            return sel
        if op is Operator.GT:
            return 1.0 - self.estimate_selectivity_histogram(
                Predicate(column=predicate.column, operator=Operator.LT, value=predicate.value)
            )
        if op is Operator.BETWEEN and predicate.value2 is not None:
            lt = self.estimate_selectivity_histogram(
                Predicate(column=predicate.column, operator=Operator.LT, value=predicate.value2)
            )
            lt_low = self.estimate_selectivity_histogram(
                Predicate(column=predicate.column, operator=Operator.LT, value=predicate.value)
            )
            return lt - lt_low
        return 0.5

    def estimate_selectivity_ml(self, predicate: Predicate) -> float:
        column = self.metadata.get(predicate.column)
        if column is None:
            return 0.5
        return self.knn_model.predict(predicate, column)

    @staticmethod
    def estimate_selectivity_approx(predicate: Predicate) -> float:
        op = predicate.operator
        if op is Operator.EQ:
            return 0.01
        if op is Operator.NE:
            return 0.99
        if op in (Operator.LT, Operator.LE):
            return 0.33
        if op in (Operator.GT, Operator.GE):
            return 0.33
        if op is Operator.BETWEEN:
            return 0.25
        if op is Operator.LIKE:
            return 0.05
        return 0.5

    @staticmethod
    def recommend_scan_method(_predicate: Predicate, ml_selectivity: float) -> ScanMethod:
        has_index = True
        if not has_index:
            return ScanMethod.SEQ_SCAN
        if ml_selectivity < 0.05:
            return ScanMethod.INDEX_SCAN
        if ml_selectivity < 0.3:
            table_size = 10000
            cost_index = ScanMethod.INDEX_SCAN.estimate_cost(table_size, ml_selectivity, True)
            cost_bitmap = ScanMethod.BITMAP_SCAN.estimate_cost(table_size, ml_selectivity, True)
            cost_seq = ScanMethod.SEQ_SCAN.estimate_cost(table_size, ml_selectivity, True)
            if cost_bitmap < cost_index and cost_bitmap < cost_seq:
                return ScanMethod.BITMAP_SCAN
            if cost_index < cost_seq:
                return ScanMethod.INDEX_SCAN
            return ScanMethod.SEQ_SCAN
        return ScanMethod.SEQ_SCAN

    def generate_predicates(self) -> list[Predicate]:
        predicates: list[Predicate] = []
        for col in self.numeric_columns_sorted():
            metadata = self.metadata[col]
            s = metadata.stats
            std = max(s.std, 0.001)
            lt_value = max(s.mean - std, s.min)
            gt_value = min(s.mean + std * 0.5, s.max)
            between_low = min(s.min + std, s.max)
            between_high = max(s.max - std, s.min)
            if between_low > between_high:
                between_low, between_high = s.min, s.max
            predicates.append(Predicate(col, Operator.EQ, s.mean))
            predicates.append(Predicate(col, Operator.LT, lt_value))
            predicates.append(Predicate(col, Operator.GT, gt_value))
            predicates.append(Predicate(col, Operator.BETWEEN, between_low, between_high))
        return predicates

    def compute_actual_selectivity(self, predicate: Predicate) -> float:
        values = self.data.get(predicate.column)
        if values is None:
            return 0.5
        if not values:
            return 0.0
        op = predicate.operator
        v = predicate.value
        eps = sys.float_info.epsilon
        if op is Operator.EQ:
            matches = sum(1 for x in values if abs(x - v) < eps)
        elif op is Operator.NE:
            matches = sum(1 for x in values if abs(x - v) >= eps)
        elif op is Operator.LT:
            matches = sum(1 for x in values if x < v)
        elif op is Operator.LE:
            matches = sum(1 for x in values if x <= v)
        elif op is Operator.GT:
            matches = sum(1 for x in values if x > v)
        elif op is Operator.GE:
            matches = sum(1 for x in values if x >= v)
        elif op is Operator.BETWEEN and predicate.value2 is not None:
            v2 = predicate.value2
            matches = sum(1 for x in values if v <= x <= v2)
        elif op is Operator.LIKE:
            matches = len(values) // 10
        else:
            matches = 0
        return matches / len(values)

    def evaluate_predictors_with_knn(self) -> list[PredicateSelectivity]:
        self.compute_statistics()
        self.train_knn_model()
        predicates = self.generate_predicates()
        if not predicates:
            raise RuntimeError(
                "Не удалось сгенерировать предикаты: в CSV нет подходящих числовых столбцов"
            )

        results: list[PredicateSelectivity] = []
        print("\n\U0001f52c Анализ селективности предикатов (с KNN)\n")
        print(
            f"{'Предикат':<15} {'Реальная':<10} {'KNN':<10} {'Гистогр.':<10} "
            f"{'Приближ.':<10} {'Реком.скан':<12} {'KNN ошибка':<8}"
        )

        test_size = max(len(predicates) // 3, 1)
        test_data = []
        for predicate in predicates[:test_size]:
            metadata = self.metadata.get(predicate.column)
            if metadata is None:
                continue
            actual = self.compute_actual_selectivity(predicate)
            test_data.append((predicate, actual, metadata))
        knn_error = self.knn_model.evaluate(test_data)
        print(f"\U0001f4ca Средняя ошибка KNN на тестовых данных: {knn_error:.4f}")

        for predicate in predicates:
            actual = self.compute_actual_selectivity(predicate)
            ml_est = self.estimate_selectivity_ml(predicate)
            hist_est = self.estimate_selectivity_histogram(predicate)
            approx_est = self.estimate_selectivity_approx(predicate)
            error_ml = abs(actual - ml_est)
            recommended = self.recommend_scan_method(predicate, ml_est)
            results.append(
                PredicateSelectivity(
                    predicate=predicate,
                    actual_selectivity=actual,
                    ml_estimate=ml_est,
                    histogram_estimate=hist_est,
                    approx_estimate=approx_est,
                    error_ml=error_ml,
                    error_histogram=abs(actual - hist_est),
                    error_approx=abs(actual - approx_est),
                    recommended_scan=recommended,
                )
            )

            if predicate.operator is Operator.EQ:
                pred_str = f"{predicate.column} = {predicate.value:.1f}"
            elif predicate.operator is Operator.LT:
                pred_str = f"{predicate.column} < {predicate.value:.1f}"
            elif predicate.operator is Operator.GT:
                pred_str = f"{predicate.column} > {predicate.value:.1f}"
            elif predicate.operator is Operator.BETWEEN:
                v2 = predicate.value2 if predicate.value2 is not None else 0.0
                pred_str = f"{predicate.column} BETWEEN {predicate.value:.1f} AND {v2:.1f}"
            else:
                pred_str = f"{predicate.column} op {predicate.value}"
            print(
                f"{pred_str[:15]:<15} {actual:<10.3f} {ml_est:<10.3f} {hist_est:<10.3f} "
                f"{approx_est:<10.3f} {recommended.value:<12} {error_ml:<8.4f}"
            )

        avg_ml = sum(r.error_ml for r in results) / len(results)
        avg_hist = sum(r.error_histogram for r in results) / len(results)
        avg_approx = sum(r.error_approx for r in results) / len(results)
        print("\n\U0001f4ca Итоговая статистика (с KNN):")
        print(f"   Средняя ошибка KNN: {avg_ml:.4f}")
        print(f"   Средняя ошибка гистограммы: {avg_hist:.4f}")
        print(f"   Средняя ошибка приближенной оценки: {avg_approx:.4f}")
        if avg_approx > 0:
            improvement = (avg_approx - avg_ml) / avg_approx * 100
            print(f"   Улучшение точности KNN vs приближенная: {improvement:.2f}%")

        return results


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================


def write_report(
    report_path: Path,
    analyzer: DataAnalyzer,
    results: list[PredicateSelectivity],
    dataset_path: Path,
    elapsed_ms: float,
) -> None:
    numeric_columns = analyzer.numeric_columns_sorted()
    with report_path.open("w", encoding="utf-8") as report:
        report.write("ML OPTIMIZER FOR POSTGRESQL - KNN ANALYSIS REPORT\n")
        report.write("================================================\n\n")
        report.write("Backend: Python\n")
        report.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.write(f"Файл данных: {dataset_path}\n\n")
        report.write("KNN параметры: k=5\n\n")
        report.write(
            f"Количество обучающих примеров: {len(analyzer.knn_model.knn.training_data)}\n\n"
        )
        report.write(f"Числовые столбцы: {', '.join(numeric_columns)}\n\n")
        report.write(f"Время выполнения: {elapsed_ms:.3f} мс\n\n")

        report.write("СТАТИСТИКА ПО СТОЛБЦАМ:\n\n")
        for col in numeric_columns:
            m = analyzer.metadata[col]
            report.write(f"{col}:\n")
            report.write(f"   Data type: {m.data_type.value}\n")
            report.write(f"   Distribution: {m.distribution.value}\n")
            report.write(f"   Count: {m.stats.count}\n")
            report.write(f"   Mean: {m.stats.mean:.6f}\n")
            report.write(f"   Std: {m.stats.std:.6f}\n")
            report.write(f"   Min: {m.stats.min:.6f}\n")
            report.write(f"   Max: {m.stats.max:.6f}\n")
            report.write(f"   Unique values: {m.stats.unique_values}\n")
            report.write(f"   Null fraction: {m.stats.null_frac:.6f}\n\n")

        report.write("СВОДКА ПО ПРЕДИКАТАМ:\n\n")
        for i, r in enumerate(results, 1):
            p = r.predicate
            v2_part = f", value2: Some({p.value2})" if p.value2 is not None else ", value2: None"
            report.write(
                f"{i}. Predicate {{ column: \"{p.column}\", operator: {p.operator.name.title()}, "
                f"value: {p.value}{v2_part} }}\n"
            )
            report.write(
                f"   Actual: {r.actual_selectivity:.6f}, KNN: {r.ml_estimate:.6f}, "
                f"Hist: {r.histogram_estimate:.6f}, Approx: {r.approx_estimate:.6f}\n"
            )
            report.write(
                f"   Error KNN: {r.error_ml:.6f}, Error Hist: {r.error_histogram:.6f}, "
                f"Error Approx: {r.error_approx:.6f}\n"
            )
            report.write(f"   Recommended scan: {r.recommended_scan.value}\n\n")

        avg_ml = sum(r.error_ml for r in results) / len(results)
        avg_hist = sum(r.error_histogram for r in results) / len(results)
        avg_approx = sum(r.error_approx for r in results) / len(results)
        report.write("\nИТОГОВАЯ СТАТИСТИКА:\n")
        report.write(f"Средняя ошибка KNN: {avg_ml:.6f}\n")
        report.write(f"Средняя ошибка гистограммы: {avg_hist:.6f}\n")
        report.write(f"Средняя ошибка приближенной оценки: {avg_approx:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Optimizer Python backend")
    parser.add_argument("dataset", nargs="?", default="synthetic_data.csv")
    args = parser.parse_args()

    print("\U0001f680 ML Optimizer for PostgreSQL - KNN-based Selectivity Estimator (Python backend)\n")

    path = Path(args.dataset)
    if not path.exists():
        print("\U0001f4c1 Файл не найден. Генерирую синтетические данные...")
        DataGenerator().generate_synthetic_csv(path, 10_000)

    print(f"\U0001f4c2 Анализирую файл: {path}")

    start = time.perf_counter()
    analyzer = DataAnalyzer.from_csv(path)
    results = analyzer.evaluate_predictors_with_knn()
    numeric_columns = analyzer.numeric_columns_sorted()
    if not numeric_columns:
        raise RuntimeError("В CSV не найдено числовых столбцов для анализа")
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print("\n\U0001f50d Детальный анализ выборочных предикатов:")
    for r in results[:5]:
        print(f"\n   Предикат: {r.predicate}")
        print(f"     Реальная селективность: {r.actual_selectivity:.4f}")
        print(f"     KNN оценка: {r.ml_estimate:.4f} (ошибка: {r.error_ml:.4f})")
        print(f"     Гистограмма: {r.histogram_estimate:.4f} (ошибка: {r.error_histogram:.4f})")
        print(f"     Приближенная: {r.approx_estimate:.4f} (ошибка: {r.error_approx:.4f})")
        print(f"     \U0001f3af Рекомендуемый метод сканирования: {r.recommended_scan.value}")
        if r.recommended_scan is ScanMethod.SEQ_SCAN:
            print(
                f"        Причина: Низкая селективность ({r.actual_selectivity*100:.1f}% строк) "
                "- дешевле прочитать всю таблицу"
            )
        elif r.recommended_scan is ScanMethod.INDEX_SCAN:
            print(
                f"        Причина: Высокая селективность ({r.actual_selectivity*100:.1f}% строк) "
                "- индекс эффективно отфильтрует"
            )
        else:
            print(
                f"        Причина: Средняя селективность ({r.actual_selectivity*100:.1f}% строк) "
                "- bitmap scan балансирует I/O"
            )

    report_path = Path("ml_optimizer_knn_report.txt")
    write_report(report_path, analyzer, results, path, elapsed_ms)

    print(f"\n⏱  Время выполнения (Python): {elapsed_ms:.3f} мс")
    print(f"\n\U0001f4c4 Отчет сохранен в: {report_path}")
    print("\n✅ Анализ с KNN завершен!")


if __name__ == "__main__":
    main()
