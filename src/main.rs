use burn::backend::NdArray;
use burn::tensor::{ElementConversion, Tensor};
use csv::ReaderBuilder;
use rand::Rng;
use rand::rngs::ThreadRng;
use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

type BackendType = NdArray;

// модель данных

#[derive(Debug, Clone)]
struct ColumnMetadata {
    name: String,
    data_type: DataType,
    distribution: Distribution,
    stats: ColumnStats,
    histogram: Vec<HistogramBucket>,
}

#[derive(Debug, Clone, PartialEq)]
enum DataType {
    Integer,
    Float,
    Categorical,
}

impl DataType {
    fn as_str(&self) -> &'static str {
        match self {
            DataType::Integer => "Integer",
            DataType::Float => "Float",
            DataType::Categorical => "Categorical",
        }
    }
}

#[derive(Debug, Clone)]
enum Distribution {
    Uniform { min: f64, max: f64 },
    Normal { mean: f64, std: f64 },
    Skewed { alpha: f64 }, // Zipf-like distribution
}

impl Distribution {
    fn as_str(&self) -> &'static str {
        match self {
            Distribution::Uniform { .. } => "Uniform",
            Distribution::Normal { .. } => "Normal",
            Distribution::Skewed { .. } => "Skewed",
        }
    }
}

#[derive(Debug, Clone)]
struct HistogramBucket {
    lower: f64,
    upper: f64,
    count: usize,
    frequency: f64,
}

#[derive(Debug, Clone)]
struct Predicate {
    column: String,
    operator: Operator,
    value: f64,
    value2: Option<f64>, // для BETWEEN
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Operator {
    Eq,           // =
    Ne,           // <>
    Lt,           // <
    Le,           // <=
    Gt,           // >
    Ge,           // >=
    Between,      // BETWEEN
    Like,         // LIKE (упрощенно)
}

impl Operator {
    fn as_str(&self) -> &'static str {
        match self {
            Operator::Eq => "=",
            Operator::Ne => "<>",
            Operator::Lt => "<",
            Operator::Le => "<=",
            Operator::Gt => ">",
            Operator::Ge => ">=",
            Operator::Between => "BETWEEN",
            Operator::Like => "LIKE",
        }
    }
}

#[derive(Debug, Clone)]
struct PredicateSelectivity {
    predicate: Predicate,
    actual_selectivity: f64,      // реальная селективность
    ml_estimate: f64,             // оценка ML-моделью (KNN)
    histogram_estimate: f64,      // оценка по гистограмме
    approx_estimate: f64,         // приближенная оценка (по умолчанию)
    error_ml: f64,                // ошибка ML
    error_histogram: f64,         // ошибка гистограммы
    error_approx: f64,            // ошибка приближенной оценки
    recommended_scan: ScanMethod, // рекомендованный метод сканирования
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ScanMethod {
    SeqScan,      // последовательное сканирование
    IndexScan,    // индексное сканирование
    BitmapScan,   // битовое сканирование (комбинация индексов)
}

impl ScanMethod {
    fn as_str(&self) -> &'static str {
        match self {
            ScanMethod::SeqScan => "Seq Scan",
            ScanMethod::IndexScan => "Index Scan",
            ScanMethod::BitmapScan => "Bitmap Scan",
        }
    }
    
    // Оценка стоимости для разных методов
    fn estimate_cost(&self, table_size: usize, selectivity: f64, has_index: bool) -> f64 {
        match self {
            ScanMethod::SeqScan => {
                table_size as f64 * 1.0
            }
            ScanMethod::IndexScan => {
                if has_index {
                    (table_size as f64 * 0.1) + (table_size as f64 * selectivity * 0.5)
                } else {
                    f64::INFINITY // нет индекса - невозможно
                }
            }
            ScanMethod::BitmapScan => {
                if has_index {
                    (table_size as f64 * 0.15) + (table_size as f64 * selectivity * 0.3)
                } else {
                    f64::INFINITY
                }
            }
        }
    }
}

// Статистика

#[derive(Debug, Clone)]
struct ColumnStats {
    name: String,
    count: usize,
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    unique_values: usize,
    null_frac: f64,
}

// Простой knn

#[derive(Clone)]
struct TrainingExample {
    features: Vec<f64>,        // признаки предиката
    actual_selectivity: f64,    // реальная селективность
    predicate_type: Operator,   // тип оператора
    column_name: String,        // колонка
}

#[derive(Clone)]
struct TargetStats {
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
}

// Оценка селективности
#[derive(Clone)]
struct SimpleKNN {
    training_data: Vec<TrainingExample>,
    k: usize,
}

impl SimpleKNN {
    fn new(k: usize) -> Self {
        Self {
            training_data: Vec::new(),
            k,
        }
    }
    
    fn add_example(&mut self, example: TrainingExample) {
        self.training_data.push(example);
    }
    
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
    
    fn find_k_nearest(&self, features: &[f64]) -> Vec<&TrainingExample> {
        let mut distances: Vec<(f64, &TrainingExample)> = self.training_data
            .iter()
            .map(|example| (self.euclidean_distance(features, &example.features), example))
            .collect();
        
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        distances.iter()
            .take(self.k.min(self.training_data.len()))
            .map(|(_, example)| *example)
            .collect()
    }
    
    fn predict(&self, features: &[f64]) -> f64 {
        if self.training_data.is_empty() {
            return 0.5; // значение по умолчанию
        }
        
        let nearest = self.find_k_nearest(features);
        
        let sum: f64 = nearest.iter()
            .map(|ex| ex.actual_selectivity)
            .sum();
        
        sum / nearest.len() as f64
    }
    
    // Взвешенное предсказание (по расстоянию)
    fn predict_weighted(&self, features: &[f64]) -> f64 {
        if self.training_data.is_empty() {
            return 0.5;
        }
        
        let mut distances: Vec<(f64, &TrainingExample)> = self.training_data
            .iter()
            .map(|example| (self.euclidean_distance(features, &example.features), example))
            .collect();
        
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        let nearest = &distances[..self.k.min(distances.len())];
        
        // Избегаем деления на ноль
        let total_inv_dist: f64 = nearest.iter()
            .map(|(dist, _)| 1.0 / (dist + 0.0001))
            .sum();
        
        let weighted_sum: f64 = nearest.iter()
            .map(|(dist, ex)| ex.actual_selectivity * (1.0 / (dist + 0.0001)))
            .sum();
        
        weighted_sum / total_inv_dist
    }
}

#[derive(Clone)]
struct KNNSelectivityModel {
    knn: SimpleKNN,
    feature_names: Vec<String>,
    target_stats: TargetStats,
}


impl KNNSelectivityModel {
    fn new(k: usize) -> Self {
        Self {
            knn: SimpleKNN::new(k),
            feature_names: vec![
                "operator_eq".to_string(),
                "operator_lt".to_string(),
                "operator_gt".to_string(),
                "operator_between".to_string(),
                "value_normalized".to_string(),
                "value2_normalized".to_string(),
                "column_mean".to_string(),
                "column_std".to_string(),
                "column_unique_ratio".to_string(),
                "column_min".to_string(),
                "column_max".to_string(),
                "value_vs_mean".to_string(),
            ],
            target_stats: TargetStats {
                mean: 0.0,
                std: 1.0,
                min: 0.0,
                max: 1.0,
            },
        }
    }
    
    // Преобразует предикат в вектор признаков
    fn extract_features(&self, predicate: &Predicate, metadata: &ColumnMetadata) -> Vec<f64> {
        let mut features = Vec::new();
        
        features.push(match predicate.operator {
            Operator::Eq => 1.0,
            _ => 0.0,
        });
        features.push(match predicate.operator {
            Operator::Lt | Operator::Le => 1.0,
            _ => 0.0,
        });
        features.push(match predicate.operator {
            Operator::Gt | Operator::Ge => 1.0,
            _ => 0.0,
        });
        features.push(match predicate.operator {
            Operator::Between => 1.0,
            _ => 0.0,
        });
        
        // Нормализованное значение
        let value_norm = if metadata.stats.max > metadata.stats.min {
            (predicate.value - metadata.stats.min) / (metadata.stats.max - metadata.stats.min)
        } else {
            0.5
        };
        features.push(value_norm);
        
        // Нормализованное второе значение (для BETWEEN)
        let value2_norm = if let Some(v2) = predicate.value2 {
            if metadata.stats.max > metadata.stats.min {
                (v2 - metadata.stats.min) / (metadata.stats.max - metadata.stats.min)
            } else {
                0.5
            }
        } else {
            value_norm // для не-BETWEEN просто дублируем
        };
        features.push(value2_norm);
        
        // Статистики колонки (масштабируем)
        features.push(metadata.stats.mean / 100000.0);
        features.push(metadata.stats.std / 100000.0);
        features.push(metadata.stats.unique_values as f64 / metadata.stats.count as f64);
        features.push(metadata.stats.min / 100000.0);
        features.push(metadata.stats.max / 100000.0);
        
        // Относительное положение значения
        features.push((predicate.value - metadata.stats.mean) / metadata.stats.std.max(0.001));
        
        features
    }
    
    // Обучающий пример
    fn add_training_example(&mut self, example: TrainingExample) {
        self.knn.add_example(example);
    }
    
    // Селективность для нового предиката
    fn predict(&self, predicate: &Predicate, metadata: &ColumnMetadata) -> f64 {
        if self.knn.training_data.is_empty() {
            // Если модель не обучена, используем эвристику
            return self.heuristic_predict(predicate, metadata);
        }
        
        let features = self.extract_features(predicate, metadata);
        self.knn.predict_weighted(&features)
    }
    
    // Эвристическое предсказание (запасной вариант)
    fn heuristic_predict(&self, predicate: &Predicate, metadata: &ColumnMetadata) -> f64 {
        match predicate.operator {
            Operator::Eq => 1.0 / metadata.stats.unique_values.max(1) as f64,
            Operator::Lt => {
                let z = (predicate.value - metadata.stats.mean) / metadata.stats.std.max(0.001);
                0.5 * (1.0 + (z / 1.414).tanh())
            }
            Operator::Gt => {
                let z = (predicate.value - metadata.stats.mean) / metadata.stats.std.max(0.001);
                0.5 * (1.0 - (z / 1.414).tanh())
            }
            Operator::Between => {
                if let Some(value2) = predicate.value2 {
                    let z1 = (predicate.value - metadata.stats.mean) / metadata.stats.std.max(0.001);
                    let z2 = (value2 - metadata.stats.mean) / metadata.stats.std.max(0.001);
                    0.5 * ((z2 / 1.414).tanh() - (z1 / 1.414).tanh())
                } else {
                    0.2
                }
            }
            _ => 0.5,
        }
    }
    
    // Возвращает важность каждого признака как процент в общей сумме.
    //
    // Чтобы не зависеть от единиц измерения (зарплата в миллионах vs operator_eq
    // в {0,1}), сначала нормализуем каждый признак к [0,1] по его min..max
    // в обучающей выборке, а потом считаем std нормализованных значений.
    // Это и есть честный "вес": чем сильнее признак варьируется относительно
    // своего диапазона, тем сильнее он раздвигает соседей в евклидовом расстоянии.
    fn feature_importance(&self) -> Vec<(String, f64, f64)> {
        let n_features = self.feature_names.len();
        if self.knn.training_data.is_empty() || n_features == 0 {
            return Vec::new();
        }
        let n = self.knn.training_data.len() as f64;
        let mut stds = Vec::with_capacity(n_features);
        for fi in 0..n_features {
            // диапазон значений в обучающей выборке
            let mut fmin = f64::INFINITY;
            let mut fmax = f64::NEG_INFINITY;
            for ex in &self.knn.training_data {
                let v = ex.features.get(fi).copied().unwrap_or(0.0);
                if v < fmin { fmin = v; }
                if v > fmax { fmax = v; }
            }
            let range = fmax - fmin;
            if range <= 1e-12 {
                stds.push(0.0);
                continue;
            }
            // std нормализованных к [0,1] значений
            let mut sum = 0.0;
            let mut sumsq = 0.0;
            for ex in &self.knn.training_data {
                let v = ex.features.get(fi).copied().unwrap_or(0.0);
                let norm = (v - fmin) / range;
                sum += norm;
                sumsq += norm * norm;
            }
            let mean = sum / n;
            let var = (sumsq / n) - mean * mean;
            stds.push(var.max(0.0).sqrt());
        }
        let total: f64 = stds.iter().sum();
        let mut result: Vec<(String, f64, f64)> = self.feature_names.iter()
            .zip(stds.iter())
            .map(|(name, &std)| {
                let pct = if total > 0.0 { std / total * 100.0 } else { 0.0 };
                (name.clone(), std, pct)
            })
            .collect();
        result.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    // Оценивает качество модели
    fn evaluate(&self, test_predicates: &[(Predicate, f64, ColumnMetadata)]) -> f64 {
        if test_predicates.is_empty() {
            return 0.0;
        }
        
        let mut total_error = 0.0;
        for (predicate, actual, metadata) in test_predicates {
            let predicted = self.predict(predicate, metadata);
            total_error += (predicted - actual).abs();
        }
        
        total_error / test_predicates.len() as f64
    }
}

// Генерация случайных данных

struct DataGenerator {
    rng: ThreadRng,
}

impl DataGenerator {
    fn new() -> Self {
        Self { rng: rand::thread_rng() }
    }
    
    // Генерирует синтетический CSV с разными распределениями
    fn generate_synthetic_csv(&mut self, path: &Path, num_rows: usize) -> Result<(), Box<dyn std::error::Error>> {
        let mut wtr = csv::Writer::from_path(path)?;
        
        // Заголовки
        wtr.write_record(&[
            "id", 
            "age",              // нормальное распределение
            "salary",           // скошенное распределение
            "department_id",    // категориальное (равномерное)
            "score",            // равномерное
            "years_experience", // смесь распределений
        ])?;
        
        for i in 0..num_rows {
            let age = self.generate_normal(35.0, 10.0).round() as i32;
            let salary = self.generate_skewed(50000.0, 2.0).round() as i32;
            let dept_id = self.rng.gen_range(1..=20);
            let score = self.rng.gen_range(0.0..100.0);
            let experience = if age < 22 {
                self.rng.gen_range(0..=2)
            } else {
                self.rng.gen_range(0..=(age - 22).min(40))
            };
            
            wtr.write_record(&[
                i.to_string(),
                age.to_string(),
                salary.to_string(),
                dept_id.to_string(),
                format!("{:.2}", score),
                experience.to_string(),
            ])?;
        }
        
        wtr.flush()?;
        println!("✅ Сгенерирован синтетический CSV: {} строк", num_rows);
        Ok(())
    }
    
    fn generate_normal(&mut self, mean: f64, std: f64) -> f64 {
        // Box-Muller transform для генерации нормального распределения
        let u1: f64 = self.rng.gen();
        let u2: f64 = self.rng.gen();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z
    }
    
    fn generate_skewed(&mut self, scale: f64, alpha: f64) -> f64 {
        // Zipf-like skewed distribution
        let u: f64 = self.rng.gen();
        scale * (1.0 - u).powf(-1.0 / (alpha - 1.0))
    }
}

// Анализ данных

struct DataAnalyzer {
    data: HashMap<String, Vec<f64>>,
    metadata: HashMap<String, ColumnMetadata>,
    knn_model: KNNSelectivityModel,
}

impl DataAnalyzer {
    fn from_csv(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
        let headers = rdr.headers()?.clone();
        
        let mut data: HashMap<String, Vec<f64>> = HashMap::new();
        for header in headers.iter() {
            data.insert(header.to_string(), Vec::new());
        }
        
        for result in rdr.records() {
            let record = result?;
            for (i, header) in headers.iter().enumerate() {
                if let Some(value) = record.get(i) {
                    if let Ok(num) = value.parse::<f64>() {
                        data.get_mut(header).unwrap().push(num);
                    }
                }
            }
        }
        
        Ok(Self { 
            data,
            metadata: HashMap::new(),
            knn_model: KNNSelectivityModel::new(5), // k=5
        })
    }
    
    fn compute_statistics(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for (name, values) in &self.data {
            if values.is_empty() {
                continue;
            }
            
            // Базовая статистика через BURN
            let stats = stats_from_burn_tensor(values, name.clone()).unwrap();
            
            let histogram = self.build_histogram(values, 20);
            
            // Определяем тип данных и распределение
            let data_type = self.infer_data_type(values);
            let distribution = self.infer_distribution(values);
            
            self.metadata.insert(name.clone(), ColumnMetadata {
                name: name.clone(),
                data_type,
                distribution,
                stats,
                histogram,
            });
        }
        
        Ok(())
    }

    fn numeric_columns_sorted(&self) -> Vec<String> {
        let mut columns: Vec<String> = self.metadata.keys().cloned().collect();
        columns.sort();
        columns
    }
    
    // Гистограмма
    fn build_histogram(&self, values: &[f64], num_buckets: usize) -> Vec<HistogramBucket> {
        if values.is_empty() {
            return vec![];
        }
        
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let bucket_width = (max - min) / num_buckets as f64;
        
        let mut buckets = vec![0; num_buckets];
        for &v in values {
            let bucket_idx = ((v - min) / bucket_width).floor() as usize;
            if bucket_idx < num_buckets {
                buckets[bucket_idx] += 1;
            } else {
                buckets[num_buckets - 1] += 1;
            }
        }
        
        let total = values.len() as f64;
        buckets.into_iter().enumerate().map(|(i, count)| {
            HistogramBucket {
                lower: min + i as f64 * bucket_width,
                upper: min + (i + 1) as f64 * bucket_width,
                count,
                frequency: count as f64 / total,
            }
        }).collect()
    }
    
    // Определение типа данных
    fn infer_data_type(&self, values: &[f64]) -> DataType {
        // Проверяем, все ли значения целые
        let all_integers = values.iter().all(|&v| v.fract() == 0.0);
        
        // Проверяем, мало ли уникальных значений (категориальные)
        let mut unique = values.to_vec();
        unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique.dedup();
        
        if unique.len() < values.len() / 10 && all_integers {
            DataType::Categorical
        } else if all_integers {
            DataType::Integer
        } else {
            DataType::Float
        }
    }
    
    // Определение распределения
    fn infer_distribution(&self, values: &[f64]) -> Distribution {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();
        
        // Проверяем на скошенность
        let skewness = values.iter()
            .map(|&x| (x - mean).powi(3))
            .sum::<f64>() / (values.len() as f64 * variance * std);
        
        if skewness.abs() > 1.0 {
            Distribution::Skewed { alpha: 2.0 + skewness.abs() }
        } else if (std - mean * 0.3).abs() < mean * 0.1 {
            // Приблизительно нормальное
            Distribution::Normal { mean, std }
        } else {
            Distribution::Uniform { 
                min: values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                max: values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            }
        }
    }
    
    // Обучает KNN на исторических данных (без вывода — отчёт печатает main)
    fn train_knn_model(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut examples = Vec::new();
        for (col_name, values) in &self.data {
            if values.is_empty() || !self.metadata.contains_key(col_name) {
                continue;
            }
            let metadata = self.metadata.get(col_name).unwrap();
            let predicates = self.generate_training_predicates(col_name, values, metadata);
            examples.extend(predicates);
        }
        for example in examples {
            self.knn_model.add_training_example(example);
        }
        Ok(())
    }

    // Сводка обучающей выборки: разбивка по операторам и по столбцам.
    fn training_breakdown(&self) -> (Vec<(Operator, usize)>, Vec<(String, usize)>) {
        let mut by_op: HashMap<Operator, usize> = HashMap::new();
        let mut by_col: HashMap<String, usize> = HashMap::new();
        for ex in &self.knn_model.knn.training_data {
            *by_op.entry(ex.predicate_type).or_insert(0) += 1;
            *by_col.entry(ex.column_name.clone()).or_insert(0) += 1;
        }
        let order = [Operator::Eq, Operator::Lt, Operator::Le, Operator::Gt,
                     Operator::Ge, Operator::Between, Operator::Ne, Operator::Like];
        let mut ops: Vec<(Operator, usize)> = order.iter()
            .filter_map(|op| by_op.get(op).map(|&c| (*op, c)))
            .collect();
        for (op, c) in by_op.iter() {
            if !ops.iter().any(|(o, _)| o == op) {
                ops.push((*op, *c));
            }
        }
        let mut cols: Vec<(String, usize)> = by_col.into_iter().collect();
        cols.sort_by(|a, b| a.0.cmp(&b.0));
        (ops, cols)
    }
    
    // Генерация обучающих примероа из реальных данных
    fn generate_training_predicates(&self, col_name: &str, values: &[f64], metadata: &ColumnMetadata) -> Vec<TrainingExample> {
        let mut examples = Vec::new();
        
        if values.len() < 10 {
            return examples;
        }
        
        // Для разных типов операторов
        let operators = vec![
            Operator::Eq,
            Operator::Lt,
            Operator::Gt,
            Operator::Between,
        ];
        
        // Берем случайные значения из данных
        let sample_size = 20.min(values.len());
        let step = values.len() / sample_size;
        
        for i in (0..values.len()).step_by(step) {
            let val = values[i];
            
            for op in &operators {
                let predicate = match op {
                    Operator::Eq => Predicate {
                        column: col_name.to_string(),
                        operator: Operator::Eq,
                        value: val,
                        value2: None,
                    },
                    Operator::Lt => Predicate {
                        column: col_name.to_string(),
                        operator: Operator::Lt,
                        value: val,
                        value2: None,
                    },
                    Operator::Gt => Predicate {
                        column: col_name.to_string(),
                        operator: Operator::Gt,
                        value: val,
                        value2: None,
                    },
                    Operator::Between => {
                        let val2 = if i + step < values.len() { values[i + step] } else { values[values.len() - 1] };
                        Predicate {
                            column: col_name.to_string(),
                            operator: Operator::Between,
                            value: val.min(val2),
                            value2: Some(val.max(val2)),
                        }
                    },
                    _ => continue,
                };
                
                // Вычисляем реальную селективность
                let actual = self.compute_actual_selectivity(&predicate);
                
                // Извлекаем признаки
                let features = self.knn_model.extract_features(&predicate, metadata);
                
                examples.push(TrainingExample {
                    features,
                    actual_selectivity: actual,
                    predicate_type: *op,
                    column_name: col_name.to_string(),
                });
            }
        }
        
        examples
    }
    
    // Оценивает селективность предиката по гистограмме
    fn estimate_selectivity_histogram(&self, predicate: &Predicate) -> f64 {
        let column = match self.metadata.get(&predicate.column) {
            Some(col) => col,
            None => return 0.5, // default если колонка не найдена
        };
        
        match predicate.operator {
            Operator::Eq => {
                // Ищем бакет, содержащий значение
                for bucket in &column.histogram {
                    if predicate.value >= bucket.lower && predicate.value <= bucket.upper {
                        // Внутри бакета предполагаем равномерное распределение
                        return bucket.frequency / (bucket.upper - bucket.lower);
                    }
                }
                0.0
            }
            Operator::Lt => {
                let mut selectivity = 0.0;
                for bucket in &column.histogram {
                    if bucket.upper <= predicate.value {
                        selectivity += bucket.frequency;
                    } else if bucket.lower < predicate.value {
                        // Частичное покрытие бакета
                        let fraction = (predicate.value - bucket.lower) / (bucket.upper - bucket.lower);
                        selectivity += bucket.frequency * fraction;
                    }
                }
                selectivity
            }
            Operator::Gt => {
                1.0 - self.estimate_selectivity_histogram(&Predicate {
                    column: predicate.column.clone(),
                    operator: Operator::Lt,
                    value: predicate.value,
                    value2: None,
                })
            }
            Operator::Between => {
                if let Some(value2) = predicate.value2 {
                    let lt = self.estimate_selectivity_histogram(&Predicate {
                        column: predicate.column.clone(),
                        operator: Operator::Lt,
                        value: value2,
                        value2: None,
                    });
                    let lt_low = self.estimate_selectivity_histogram(&Predicate {
                        column: predicate.column.clone(),
                        operator: Operator::Lt,
                        value: predicate.value,
                        value2: None,
                    });
                    lt - lt_low
                } else {
                    0.0
                }
            }
            _ => 0.5, // упрощенно для других операторов
        }
    }
    
    // ML-оценка селективности с использованием KNN
    fn estimate_selectivity_ml(&self, predicate: &Predicate) -> f64 {
        let column = match self.metadata.get(&predicate.column) {
            Some(col) => col,
            None => return 0.5,
        };
        
        self.knn_model.predict(predicate, column)
    }
    
    // Приближенная оценка (дефолтная в PostgreSQL)
    fn estimate_selectivity_approx(&self, predicate: &Predicate) -> f64 {
        // PostgreSQL использует эвристики:
        // eq: 0.01, lt: 0.33, gt: 0.33, between: 0.25 и т.д.
        match predicate.operator {
            Operator::Eq => 0.01,
            Operator::Ne => 0.99,
            Operator::Lt | Operator::Le => 0.33,
            Operator::Gt | Operator::Ge => 0.33,
            Operator::Between => 0.25,
            Operator::Like => 0.05,
        }
    }
    
    // Рекомендует метод сканирования на основе селективности
    fn recommend_scan_method(&self, _predicate: &Predicate, ml_selectivity: f64) -> ScanMethod {
        let has_index = true; // предположим, что индексы есть
        
        if !has_index {
            return ScanMethod::SeqScan;
        }
        
        // Правила выбора метода сканирования
        if ml_selectivity < 0.05 {
            // Очень селективно - Index Scan
            ScanMethod::IndexScan
        } else if ml_selectivity < 0.3 {
            // Средняя селективность - сравниваем стоимость
            let table_size = 10000; // предположим
            
            let cost_index = ScanMethod::IndexScan.estimate_cost(table_size, ml_selectivity, true);
            let cost_bitmap = ScanMethod::BitmapScan.estimate_cost(table_size, ml_selectivity, true);
            let cost_seq = ScanMethod::SeqScan.estimate_cost(table_size, ml_selectivity, true);
            
            if cost_bitmap < cost_index && cost_bitmap < cost_seq {
                ScanMethod::BitmapScan
            } else if cost_index < cost_seq {
                ScanMethod::IndexScan
            } else {
                ScanMethod::SeqScan
            }
        } else {
            // Низкая селективность - Seq Scan
            ScanMethod::SeqScan
        }
    }
    
    // Генерирует набор тестовых предикатов
    fn generate_predicates(&self) -> Vec<Predicate> {
        let mut predicates = Vec::new();
        
        for col in self.numeric_columns_sorted() {
            if let Some(metadata) = self.metadata.get(&col) {
                let std = metadata.stats.std.max(0.001);
                let lt_value = (metadata.stats.mean - std).max(metadata.stats.min);
                let gt_value = (metadata.stats.mean + std * 0.5).min(metadata.stats.max);
                let between_low = (metadata.stats.min + std).min(metadata.stats.max);
                let between_high = (metadata.stats.max - std).max(metadata.stats.min);
                let (between_start, between_end) = if between_low <= between_high {
                    (between_low, between_high)
                } else {
                    (metadata.stats.min, metadata.stats.max)
                };

                // Разные типы предикатов
                predicates.push(Predicate {
                    column: col.clone(),
                    operator: Operator::Eq,
                    value: metadata.stats.mean,
                    value2: None,
                });
                
                predicates.push(Predicate {
                    column: col.clone(),
                    operator: Operator::Lt,
                    value: lt_value,
                    value2: None,
                });
                
                predicates.push(Predicate {
                    column: col.clone(),
                    operator: Operator::Gt,
                    value: gt_value,
                    value2: None,
                });
                
                predicates.push(Predicate {
                    column: col.clone(),
                    operator: Operator::Between,
                    value: between_start,
                    value2: Some(between_end),
                });
            }
        }
        
        predicates
    }
    
    // Вычисляет реальную селективность (по данным)
    fn compute_actual_selectivity(&self, predicate: &Predicate) -> f64 {
        let values = match self.data.get(&predicate.column) {
            Some(v) => v,
            None => return 0.5,
        };
        
        if values.is_empty() {
            return 0.0;
        }
        
        let matches = match predicate.operator {
            Operator::Eq => values.iter().filter(|&&v| (v - predicate.value).abs() < f64::EPSILON).count(),
            Operator::Ne => values.iter().filter(|&&v| (v - predicate.value).abs() >= f64::EPSILON).count(),
            Operator::Lt => values.iter().filter(|&&v| v < predicate.value).count(),
            Operator::Le => values.iter().filter(|&&v| v <= predicate.value).count(),
            Operator::Gt => values.iter().filter(|&&v| v > predicate.value).count(),
            Operator::Ge => values.iter().filter(|&&v| v >= predicate.value).count(),
            Operator::Between => {
                if let Some(value2) = predicate.value2 {
                    values.iter().filter(|&&v| v >= predicate.value && v <= value2).count()
                } else {
                    0
                }
            }
            Operator::Like => values.len() / 10, // упрощенно
        };
        
        matches as f64 / values.len() as f64
    }
    
    // Запускает полный анализ с KNN. Возвращает результаты;
    // печать структурированного отчёта берёт на себя main().
    fn evaluate_predictors_with_knn(&mut self) -> Result<Vec<PredicateSelectivity>, Box<dyn std::error::Error>> {
        self.compute_statistics()?;
        self.train_knn_model()?;

        let predicates = self.generate_predicates();
        if predicates.is_empty() {
            return Err("Не удалось сгенерировать предикаты: в CSV нет подходящих числовых столбцов".into());
        }

        let mut results = Vec::new();
        for predicate in predicates {
            let actual = self.compute_actual_selectivity(&predicate);
            let ml_est = self.estimate_selectivity_ml(&predicate);
            let hist_est = self.estimate_selectivity_histogram(&predicate);
            let approx_est = self.estimate_selectivity_approx(&predicate);
            let recommended = self.recommend_scan_method(&predicate, ml_est);
            results.push(PredicateSelectivity {
                predicate: predicate.clone(),
                actual_selectivity: actual,
                ml_estimate: ml_est,
                histogram_estimate: hist_est,
                approx_estimate: approx_est,
                error_ml: (actual - ml_est).abs(),
                error_histogram: (actual - hist_est).abs(),
                error_approx: (actual - approx_est).abs(),
                recommended_scan: recommended,
            });
        }
        Ok(results)
    }
}

// ==================== ФОРМАТИРОВАНИЕ ВЫВОДА ====================

fn format_predicate(p: &Predicate) -> String {
    match p.operator {
        Operator::Between => format!(
            "{} BETWEEN {:.2} AND {:.2}",
            p.column, p.value, p.value2.unwrap_or(0.0)
        ),
        _ => format!("{} {} {:.2}", p.column, p.operator.as_str(), p.value),
    }
}

fn bar(width: usize, frac: f64) -> String {
    let filled = (frac.clamp(0.0, 1.0) * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    "█".repeat(filled) + &"·".repeat(empty)
}

fn write_analysis<W: Write>(
    out: &mut W,
    backend: &str,
    dataset: &Path,
    row_count: usize,
    numeric_columns: &[String],
    analyzer: &DataAnalyzer,
    results: &[PredicateSelectivity],
    elapsed_ms: f64,
) -> std::io::Result<()> {
    writeln!(out, "================================================================")?;
    writeln!(out, "  ML Optimizer — KNN Selectivity Estimator  (backend: {})", backend)?;
    writeln!(out, "================================================================")?;
    writeln!(out)?;

    // [1] Загрузка
    writeln!(out, "[1/4] Загрузка данных")?;
    writeln!(out, "      Файл:    {}", dataset.display())?;
    writeln!(out, "      Строк:   {}", row_count)?;
    writeln!(out, "      Числовые столбцы ({}): {}", numeric_columns.len(), numeric_columns.join(", "))?;
    writeln!(out)?;

    // [2] Обучение
    let (by_op, by_col) = analyzer.training_breakdown();
    let total_train = analyzer.knn_model.knn.training_data.len();
    writeln!(out, "[2/4] Обучение KNN-модели")?;
    writeln!(out, "      k = {}  (взвешенное по евклидову расстоянию)", analyzer.knn_model.knn.k)?;
    writeln!(out, "      Всего обучающих примеров: {}", total_train)?;
    writeln!(out, "      Признаков на пример:      {}", analyzer.knn_model.feature_names.len())?;
    writeln!(out, "      Разбивка по операторам предикатов:")?;
    for (op, count) in &by_op {
        writeln!(out, "        {:<8} : {:>4}", op.as_str(), count)?;
    }
    writeln!(out, "      Разбивка по столбцам:")?;
    for (col, count) in &by_col {
        writeln!(out, "        {:<20} : {:>4}", col, count)?;
    }
    writeln!(out)?;

    // [3] Важность признаков
    let importance = analyzer.knn_model.feature_importance();
    writeln!(out, "[3/4] Приоритеты признаков (что модель считает важным)")?;
    writeln!(out, "      KNN использует евклидово расстояние, поэтому 'вес' признака =")?;
    writeln!(out, "      его std в обучающей выборке: чем сильнее признак варьируется,")?;
    writeln!(out, "      тем сильнее он влияет на расстояние и предсказание.")?;
    writeln!(out)?;
    writeln!(out, "      #  Признак                  Std       Важность")?;
    writeln!(out, "      ---------------------------------------------------------------")?;
    let max_pct = importance.first().map(|(_, _, p)| *p).unwrap_or(0.0).max(1e-9);
    for (i, (name, std, pct)) in importance.iter().enumerate() {
        writeln!(out, "      {:>2}  {:<22}  {:>7.4}   {:>5.1}%  {}",
                 i + 1, name, std, pct, bar(20, pct / max_pct))?;
    }
    writeln!(out)?;
    writeln!(out, "      Топ-3 приоритетных признака:")?;
    for (i, (name, _, pct)) in importance.iter().take(3).enumerate() {
        writeln!(out, "        {}. {:<22} ({:.1}%)", i + 1, name, pct)?;
    }
    writeln!(out)?;

    // [4] Предсказания
    writeln!(out, "[4/4] Предсказания на сгенерированных предикатах")?;
    writeln!(out, "      {:<32} {:>7} {:>7} {:>8} {:>8}  {}",
             "Предикат", "Реал.", "KNN", "Гистогр.", "Приближ.", "Скан")?;
    writeln!(out, "      {}", "-".repeat(78))?;
    for r in results {
        let pred = format_predicate(&r.predicate);
        let pred_short: String = pred.chars().take(32).collect();
        writeln!(out, "      {:<32} {:>7.3} {:>7.3} {:>8.3} {:>8.3}  {}",
                 pred_short,
                 r.actual_selectivity,
                 r.ml_estimate,
                 r.histogram_estimate,
                 r.approx_estimate,
                 r.recommended_scan.as_str())?;
    }
    writeln!(out)?;

    // Итог
    let n = results.len() as f64;
    let avg_ml = results.iter().map(|r| r.error_ml).sum::<f64>() / n;
    let avg_hist = results.iter().map(|r| r.error_histogram).sum::<f64>() / n;
    let avg_approx = results.iter().map(|r| r.error_approx).sum::<f64>() / n;
    let improvement = if avg_approx > 0.0 { (avg_approx - avg_ml) / avg_approx * 100.0 } else { 0.0 };
    writeln!(out, "Итог")?;
    writeln!(out, "----")?;
    writeln!(out, "  Средняя ошибка KNN:           {:.4}", avg_ml)?;
    writeln!(out, "  Средняя ошибка гистограммы:   {:.4}", avg_hist)?;
    writeln!(out, "  Средняя ошибка приближённой:  {:.4}", avg_approx)?;
    writeln!(out, "  KNN точнее приближённой на:   {:.1}%", improvement)?;
    writeln!(out, "  Backend:                      {}", backend)?;
    writeln!(out, "  Время выполнения:             {:.3} мс", elapsed_ms)?;
    Ok(())
}

// ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "synthetic_data.csv".to_string());
    let path = Path::new(&path);

    if !path.exists() {
        println!("Файл не найден, генерирую синтетические данные: {}", path.display());
        let mut generator = DataGenerator::new();
        generator.generate_synthetic_csv(path, 10000)?;
    }

    let start = Instant::now();
    let mut analyzer = DataAnalyzer::from_csv(path)?;
    let results = analyzer.evaluate_predictors_with_knn()?;
    let numeric_columns = analyzer.numeric_columns_sorted();
    if numeric_columns.is_empty() {
        return Err("В CSV не найдено числовых столбцов для анализа".into());
    }
    let row_count = analyzer.data.values().map(|v| v.len()).max().unwrap_or(0);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Структурированный вывод и в stdout, и в файл — чтобы GUI мог распарсить отчёт.
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    write_analysis(&mut handle, "Rust", path, row_count, &numeric_columns,
                   &analyzer, &results, elapsed_ms)?;

    let report_path = Path::new("ml_optimizer_knn_report.txt");
    let mut report = File::create(report_path)?;
    writeln!(report, "Дата: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"))?;
    writeln!(report)?;
    write_analysis(&mut report, "Rust", path, row_count, &numeric_columns,
                   &analyzer, &results, elapsed_ms)?;

    writeln!(handle)?;
    writeln!(handle, "Отчёт сохранён в: {}", report_path.display())?;
    
    Ok(())
}

// ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

fn parse_f64(s: &str) -> Option<f64> {
    s.trim().parse().ok()
}

fn is_numeric_column(values: &[String]) -> bool {
    values.iter().all(|s| parse_f64(s).is_some())
}

fn collect_numeric_column(
    records: &[Vec<String>],
    col_index: usize,
    _name: &str,
) -> Option<Vec<f64>> {
    let values: Vec<String> = records
        .iter()
        .filter_map(|row| row.get(col_index).cloned())
        .collect();
    if values.is_empty() || !is_numeric_column(&values) {
        return None;
    }
    let parsed: Vec<f64> = values
        .iter()
        .filter_map(|s| parse_f64(s))
        .collect();
    if parsed.len() != values.len() {
        return None;
    }
    Some(parsed)
}

fn stats_from_burn_tensor(values: &[f64], name: String) -> Option<ColumnStats> {
    if values.is_empty() {
        return None;
    }
    let device = Default::default();
    let tensor = Tensor::<BackendType, 1>::from_floats(
        values.iter().copied().map(|x| x as f32).collect::<Vec<_>>().as_slice(),
        &device,
    );
    let count = values.len();
    let mean_t = tensor.clone().mean();
    let mean = mean_t.into_scalar().elem::<f32>() as f64;
    let (var_t, _) = tensor.clone().var_mean(0);
    let var = var_t.into_scalar().elem::<f32>() as f64;
    let std = var.max(0.0).sqrt();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    
    // Подсчет уникальных значений (упрощенно)
    let mut unique = values.to_vec();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique.dedup();
    
    Some(ColumnStats {
        name,
        count,
        mean,
        std,
        min,
        max,
        unique_values: unique.len(),
        null_frac: 0.0, // в CSV нет NULL
    })
}
