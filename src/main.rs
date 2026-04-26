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

#[derive(Debug, Clone, Copy, PartialEq)]
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
    
    // Обучает KNN на исторических данных
    fn train_knn_model(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n🧠 Обучение KNN модели на исторических данных...");
        
        // Генерируем обучающие примеры из данных
        let mut examples = Vec::new();
        
        for (col_name, values) in &self.data {
            if values.is_empty() || !self.metadata.contains_key(col_name) {
                continue;
            }
            
            let metadata = self.metadata.get(col_name).unwrap();
            
            // Генерируем разные типы предикатов для обучения
            let predicates = self.generate_training_predicates(col_name, values, metadata);
            examples.extend(predicates);
        }
        
        // Добавляем примеры в модель
        for example in examples {
            self.knn_model.add_training_example(example);
        }
        
        println!("✅ KNN модель обучена ({} примеров)", self.knn_model.knn.training_data.len());
        Ok(())
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
    
    // Запускает полный анализ с KNN
    fn evaluate_predictors_with_knn(&mut self) -> Result<Vec<PredicateSelectivity>, Box<dyn std::error::Error>> {
        // Сначала вычисляем статистики
        self.compute_statistics()?;
        
        // Обучаем KNN модель
        self.train_knn_model()?;
        
        // Тестируем на новых предикатах
        let predicates = self.generate_predicates();
        if predicates.is_empty() {
            return Err("Не удалось сгенерировать предикаты: в CSV нет подходящих числовых столбцов".into());
        }
        let mut results = Vec::new();
        
        println!("\n🔬 Анализ селективности предикатов (с KNN)\n");
        println!("{:<15} {:<10} {:<10} {:<10} {:<10} {:<12} {:<8}", 
                 "Предикат", "Реальная", "KNN", "Гистогр.", "Приближ.", "Реком.скан", "KNN ошибка");
        
        // Разделим на обучающие и тестовые для оценки
        let test_size = (predicates.len() / 3).max(1);
        let test_predicates: Vec<_> = predicates.iter().take(test_size).collect();
        
        let mut test_data = Vec::new();
        for predicate in test_predicates {
            if let Some(metadata) = self.metadata.get(&predicate.column) {
                let actual = self.compute_actual_selectivity(predicate);
                test_data.push((predicate.clone(), actual, metadata.clone()));
            }
        }
        
        // Оцениваем качество
        let knn_error = self.knn_model.evaluate(&test_data);
        println!("📊 Средняя ошибка KNN на тестовых данных: {:.4}", knn_error);
        
        // Предсказания для всех предикатов
        for predicate in predicates {
            let actual = self.compute_actual_selectivity(&predicate);
            let ml_est = self.estimate_selectivity_ml(&predicate);
            let hist_est = self.estimate_selectivity_histogram(&predicate);
            let approx_est = self.estimate_selectivity_approx(&predicate);
            
            let error_ml = (actual - ml_est).abs();
            
            let recommended = self.recommend_scan_method(&predicate, ml_est);
            
            results.push(PredicateSelectivity {
                predicate: predicate.clone(),
                actual_selectivity: actual,
                ml_estimate: ml_est,
                histogram_estimate: hist_est,
                approx_estimate: approx_est,
                error_ml,
                error_histogram: (actual - hist_est).abs(),
                error_approx: (actual - approx_est).abs(),
                recommended_scan: recommended,
            });
            
            // Формируем строку предиката для вывода
            let pred_str = match predicate.operator {
                Operator::Eq => format!("{} = {:.1}", predicate.column, predicate.value),
                Operator::Lt => format!("{} < {:.1}", predicate.column, predicate.value),
                Operator::Gt => format!("{} > {:.1}", predicate.column, predicate.value),
                Operator::Between => format!("{} BETWEEN {:.1} AND {:.1}", 
                    predicate.column, predicate.value, predicate.value2.unwrap_or(0.0)),
                _ => format!("{} op {}", predicate.column, predicate.value),
            };
            
            println!("{:<15} {:<10.3} {:<10.3} {:<10.3} {:<10.3} {:<12} {:<8.4}",
                     &pred_str[..15.min(pred_str.len())],
                     actual,
                     ml_est,
                     hist_est,
                     approx_est,
                     recommended.as_str(),
                     error_ml);
        }
        
        // Общая статистика
        let avg_error_ml = results.iter().map(|r| r.error_ml).sum::<f64>() / results.len() as f64;
        let avg_error_hist = results.iter().map(|r| r.error_histogram).sum::<f64>() / results.len() as f64;
        let avg_error_approx = results.iter().map(|r| r.error_approx).sum::<f64>() / results.len() as f64;
        
        println!("\n📊 Итоговая статистика (с KNN):");
        println!("   Средняя ошибка KNN: {:.4}", avg_error_ml);
        println!("   Средняя ошибка гистограммы: {:.4}", avg_error_hist);
        println!("   Средняя ошибка приближенной оценки: {:.4}", avg_error_approx);
        
        if avg_error_approx > 0.0 {
            println!("   Улучшение точности KNN vs приближенная: {:.2}%", 
                     (avg_error_approx - avg_error_ml) / avg_error_approx * 100.0);
        }
        
        Ok(results)
    }
}

// ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ML Optimizer for PostgreSQL - KNN-based Selectivity Estimator\n");
    
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "synthetic_data.csv".to_string());
    let path = Path::new(&path);
    
    // Генерируем синтетические данные, если файл не существует
    if !path.exists() {
        println!("📁 Файл не найден. Генерирую синтетические данные...");
        let mut generator = DataGenerator::new();
        generator.generate_synthetic_csv(path, 10000)?;
    }
    
    println!("📂 Анализирую файл: {}", path.display());

    // Анализируем данные с KNN
    let start = Instant::now();
    let mut analyzer = DataAnalyzer::from_csv(path)?;
    let results = analyzer.evaluate_predictors_with_knn()?;
    let numeric_columns = analyzer.numeric_columns_sorted();
    if numeric_columns.is_empty() {
        return Err("В CSV не найдено числовых столбцов для анализа".into());
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    
    // Детальный разбор для нескольких примеров
    println!("\n🔍 Детальный анализ выборочных предикатов:");
    
    for result in results.iter().take(5) {
        println!("\n   Предикат: {:?}", result.predicate);
        println!("     Реальная селективность: {:.4}", result.actual_selectivity);
        println!("     KNN оценка: {:.4} (ошибка: {:.4})", result.ml_estimate, result.error_ml);
        println!("     Гистограмма: {:.4} (ошибка: {:.4})", result.histogram_estimate, result.error_histogram);
        println!("     Приближенная: {:.4} (ошибка: {:.4})", result.approx_estimate, result.error_approx);
        println!("     🎯 Рекомендуемый метод сканирования: {}", result.recommended_scan.as_str());
        
        // Объясняем почему выбран этот метод
        match result.recommended_scan {
            ScanMethod::SeqScan => {
                println!("        Причина: Низкая селективность ({:.1}% строк) - дешевле прочитать всю таблицу", 
                         result.actual_selectivity * 100.0);
            }
            ScanMethod::IndexScan => {
                println!("        Причина: Высокая селективность ({:.1}% строк) - индекс эффективно отфильтрует", 
                         result.actual_selectivity * 100.0);
            }
            ScanMethod::BitmapScan => {
                println!("        Причина: Средняя селективность ({:.1}% строк) - bitmap scan балансирует I/O", 
                         result.actual_selectivity * 100.0);
            }
        }
    }
    
    // Генерируем отчет
    let report_path = Path::new("ml_optimizer_knn_report.txt");
    let mut report = File::create(report_path)?;
    
    writeln!(report, "ML OPTIMIZER FOR POSTGRESQL - KNN ANALYSIS REPORT")?;
    writeln!(report, "================================================\n")?;
    writeln!(report, "Backend: Rust")?;
    writeln!(report, "Дата: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"))?;
    writeln!(report, "Файл данных: {}\n", path.display())?;
    writeln!(report, "KNN параметры: k=5\n")?;
    writeln!(report, "Количество обучающих примеров: {}\n", analyzer.knn_model.knn.training_data.len())?;
    writeln!(report, "Числовые столбцы: {}\n", numeric_columns.join(", "))?;
    writeln!(report, "Время выполнения: {:.3} мс\n", elapsed_ms)?;
    
    writeln!(report, "СТАТИСТИКА ПО СТОЛБЦАМ:\n")?;
    for column_name in &numeric_columns {
        if let Some(metadata) = analyzer.metadata.get(column_name) {
            writeln!(report, "{}:", column_name)?;
            writeln!(report, "   Data type: {}", metadata.data_type.as_str())?;
            writeln!(report, "   Distribution: {}", metadata.distribution.as_str())?;
            writeln!(report, "   Count: {}", metadata.stats.count)?;
            writeln!(report, "   Mean: {:.6}", metadata.stats.mean)?;
            writeln!(report, "   Std: {:.6}", metadata.stats.std)?;
            writeln!(report, "   Min: {:.6}", metadata.stats.min)?;
            writeln!(report, "   Max: {:.6}", metadata.stats.max)?;
            writeln!(report, "   Unique values: {}", metadata.stats.unique_values)?;
            writeln!(report, "   Null fraction: {:.6}\n", metadata.stats.null_frac)?;
        }
    }
    
    writeln!(report, "СВОДКА ПО ПРЕДИКАТАМ:\n")?;
    for (i, result) in results.iter().enumerate() {
        writeln!(report, "{}. {:?}", i+1, result.predicate)?;
        writeln!(report, "   Actual: {:.6}, KNN: {:.6}, Hist: {:.6}, Approx: {:.6}", 
                 result.actual_selectivity, 
                 result.ml_estimate, 
                 result.histogram_estimate,
                 result.approx_estimate)?;
        writeln!(report, "   Error KNN: {:.6}, Error Hist: {:.6}, Error Approx: {:.6}", 
                 result.error_ml, result.error_histogram, result.error_approx)?;
        writeln!(report, "   Recommended scan: {}\n", result.recommended_scan.as_str())?;
    }
    
    let avg_error_ml = results.iter().map(|r| r.error_ml).sum::<f64>() / results.len() as f64;
    let avg_error_hist = results.iter().map(|r| r.error_histogram).sum::<f64>() / results.len() as f64;
    let avg_error_approx = results.iter().map(|r| r.error_approx).sum::<f64>() / results.len() as f64;
    
    writeln!(report, "\nИТОГОВАЯ СТАТИСТИКА:")?;
    writeln!(report, "Средняя ошибка KNN: {:.6}", avg_error_ml)?;
    writeln!(report, "Средняя ошибка гистограммы: {:.6}", avg_error_hist)?;
    writeln!(report, "Средняя ошибка приближенной оценки: {:.6}", avg_error_approx)?;
    
    println!("\n⏱  Время выполнения (Rust): {:.3} мс", elapsed_ms);
    println!("\n📄 Отчет сохранен в: {}", report_path.display());
    println!("\n✅ Анализ с KNN завершен!");
    
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
