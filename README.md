# Statistical Learning Lab - Machine Failure Detection

## 📋 Project Overview

This repository contains a comprehensive statistical learning laboratory focused on **predictive maintenance and machine failure detection** using advanced machine learning techniques. The project implements anomaly detection algorithms to identify potential machine failures in industrial equipment using the **AI4I 2020 Predictive Maintenance Dataset**.

The primary goal is to build and evaluate machine learning models that can accurately predict equipment failures before they occur, enabling proactive maintenance strategies and reducing operational downtime.

---

## 🎯 Objectives

- **Exploratory Data Analysis (EDA)**: Conduct thorough statistical and visual analysis of industrial machine operational parameters
- **Feature Engineering**: Prepare and transform raw sensor data for machine learning models
- **Anomaly Detection**: Implement deep learning-based anomaly detection using autoencoders
- **Predictive Modeling**: Build classification models to detect machine failures
- **Model Evaluation**: Assess performance using multiple metrics (Precision, Recall, F1-Score, ROC-AUC)
- **Statistical Interpretation**: Provide actionable insights from model results

---

## 📊 Dataset

### **AI4I 2020 Predictive Maintenance Dataset**

**File**: `ai4i2020.csv`

**Dataset Size**: ~500 KB | **Records**: 10,000 observations

**Features**:
| Feature | Description | Type |
|---------|-------------|------|
| UDI | Unique Device Identifier | Numeric |
| Product ID | Product identifier (L, M, H types) | Categorical |
| Type | Product quality type (L/M/H: Low/Medium/High) | Categorical |
| Air temperature [K] | Ambient air temperature in Kelvin | Numeric |
| Process temperature [K] | Machine process temperature in Kelvin | Numeric |
| Rotational speed [rpm] | Motor rotational speed in RPM | Numeric |
| Torque [Nm] | Applied torque in Newton-meters | Numeric |
| Tool wear [min] | Cumulative tool wear time in minutes | Numeric |
| Machine failure | Target: Binary failure indicator (0/1) | Binary |
| TWF, HDF, PWF, OSF, RNF | Specific failure mode indicators | Binary |

**Class Distribution**: Highly imbalanced dataset with majority non-failure cases (realistic for maintenance scenarios)

---

## 📁 Repository Structure

```
Statistical-Learning-Lab-Test/
├── README.md                          # Project documentation (this file)
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── ai4i2020.csv                       # Raw dataset
├── autoencoder.ipynb                  # Deep learning anomaly detection model
├── stat-lab-test.ipynb                # Comprehensive EDA and initial analysis
└── stat_lab_test_extended.ipynb       # Extended analysis and advanced modeling
```

---

## 🔍 Notebooks Overview

### **1. stat-lab-test.ipynb** (Main Analysis Notebook)
**Size**: 2.7 MB | **Cells**: Comprehensive analysis workflow

**Contents**:
- **Data Loading & Exploration**
  - Load AI4I2020 dataset
  - Display basic statistics (shape, data types, missing values)
  - Initial data quality assessment

- **Exploratory Data Analysis (EDA)**
  - Descriptive statistics (mean, std, min, max, quantiles)
  - Data distribution visualization (histograms, KDE plots)
  - Correlation analysis and heatmaps
  - Feature relationships with target variable
  - Handling of class imbalance

- **Feature Analysis**
  - Temporal patterns in sensor readings
  - Temperature relationships (Air vs. Process)
  - Speed and torque characteristics
  - Tool wear progression analysis
  - Product type comparisons

- **Statistical Tests**
  - Distribution normality tests
  - Feature significance testing
  - Chi-square tests for categorical features

- **Data Preprocessing**
  - Standardization/Normalization of numeric features
  - One-hot encoding of categorical variables
  - Train-test split (80/20) with stratification

- **Visualization Suite**
  - Box plots for outlier detection
  - Scatter plots for feature relationships
  - Violin plots for distribution comparison
  - ROC curves for model evaluation

---

### **2. autoencoder.ipynb** (Deep Learning Model)
**Size**: 13.7 KB | **Focus**: Anomaly Detection via Autoencoders

**Model Architecture**:

**Autoencoder Structure**:
```
Input Layer (9 features)
    ↓
Encoder:
    Linear(9 → 64) + ReLU
    ↓
    Linear(64 → 32) + ReLU
    ↓
    Linear(32 → 16)  [Bottleneck]
    ↓
Decoder:
    Linear(16 → 32) + ReLU
    ↓
    Linear(32 → 64) + ReLU
    ↓
    Linear(64 → 9)   [Output]
```

**Training Details**:
- **Framework**: PyTorch
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 50
- **Batch Size**: 64
- **Data Strategy**: Trained ONLY on normal (non-failure) data

**Methodology**:
1. Train autoencoder on healthy machine data (Machine failure = 0)
2. Calculate reconstruction error (MSE per sample)
3. Set anomaly threshold at 95th percentile of training errors
4. Classify test samples with error > threshold as failures

**Results**:
```
Classification Report:
                precision    recall  f1-score   support
           0       0.97      0.95      0.96      1932
           1       0.17      0.31      0.22        68

    accuracy       0.92                          2000
ROC-AUC Score: 0.7565
```

**Performance Interpretation**:
- **High True Negative Rate**: 95% of normal operations correctly identified
- **Moderate Detection Rate**: 31% of failures successfully detected (high false-negative rate)
- **Trade-off**: Lower recall reflects the challenge of imbalanced classification and threshold tuning

---

### **3. stat_lab_test_extended.ipynb** (Advanced Analysis)
**Size**: 2.7 MB | **Focus**: Extended statistical and machine learning techniques

**Additional Content**:
- Advanced feature engineering techniques
- Multiple classification algorithms comparison
- Hyperparameter tuning and cross-validation
- Extended statistical tests and hypothesis testing
- Advanced visualization and interpretation
- Model comparison and ensemble methods
- Detailed error analysis and misclassification patterns

---

## 🛠️ Technologies & Libraries

### **Core Libraries**
- **Python 3.12.3** - Programming language
- **PyTorch** - Deep learning framework
- **scikit-learn** - Machine learning algorithms and metrics
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### **Analysis & Visualization**
- **Matplotlib** - Static plotting
- **Seaborn** - Statistical visualization
- **Jupyter Notebook** - Interactive computing

### **Preprocessing**
- StandardScaler - Feature normalization
- OneHotEncoder - Categorical encoding
- ColumnTransformer - Pipeline feature processing
- train_test_split - Data splitting with stratification

---

## 🚀 Getting Started

### **Prerequisites**
```bash
Python 3.8+
pip or conda
Jupyter Notebook
```

### **Installation**

1. **Clone the repository**:
```bash
git clone https://github.com/johnfelix2911/Statistical-Learning-Lab-Test.git
cd Statistical-Learning-Lab-Test
```

2. **Create virtual environment** (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn jupyter
```

### **Running the Notebooks**

1. **Start Jupyter**:
```bash
jupyter notebook
```

2. **Open notebooks in order**:
   - Start with `stat-lab-test.ipynb` for EDA
   - Then explore `autoencoder.ipynb` for anomaly detection
   - Finally, review `stat_lab_test_extended.ipynb` for advanced analysis

### **Data Preparation**

The `ai4i2020.csv` file must be in the repository root directory. Update file paths in notebooks if necessary:

```python
df = pd.read_csv('ai4i2020.csv')
```

---

## 📈 Key Findings

### **Data Characteristics**
- Machine failures are **rare events** (~0.8% of data), reflecting realistic maintenance scenarios
- **Temperature readings** show tight control with minimal variance
- **Rotational speed** and **Torque** exhibit wider operational ranges
- **Product Type** significantly influences failure patterns

### **Model Performance**
- **Autoencoder achieves 92% accuracy** on the test set
- **Strong specificity** (97% precision on normal cases)
- **Moderate sensitivity** (31% recall on failures) - reflects difficulty of rare-event detection
- **ROC-AUC of 0.76** indicates reasonable discriminative power

### **Anomaly Detection Insights**
- Reconstruction error effectively captures deviation from normal operation
- **Threshold optimization** is critical - higher thresholds reduce false alarms but miss failures
- Model performance limited by class imbalance - consideration for techniques like:
  - Weighted loss functions
  - SMOTE oversampling
  - Cost-sensitive learning

---

## 💡 Interpretation & Recommendations

### **Machine Failure Patterns**
1. **Temperature deviations** from optimal range indicate potential issues
2. **Rotational speed anomalies** precede several failure modes
3. **Tool wear accumulation** shows degradation patterns
4. **Torque fluctuations** may signal mechanical problems

### **For Production Use**
1. **Combine multiple models** - use both autoencoder and traditional classifiers
2. **Implement ensemble methods** - boost detection reliability
3. **Real-time monitoring** - stream processing for continuous anomaly detection
4. **Actionable alerts** - set thresholds based on maintenance response capacity
5. **Regular retraining** - update models as equipment degrades or improves

### **Data Collection Recommendations**
- Capture more failure instances for better model training
- Include contextual metadata (maintenance history, part age)
- Implement continuous monitoring for temporal patterns
- Document failure root causes for supervised learning

---

## 📊 Advanced Topics Covered

### **Statistical Methods**
- Hypothesis testing and significance testing
- Distribution analysis and normality checks
- Correlation and multicollinearity assessment
- Dimensionality reduction techniques

### **Machine Learning Techniques**
- Supervised classification (logistic regression, decision trees, ensemble methods)
- Unsupervised anomaly detection
- Deep learning with autoencoders
- Cross-validation and hyperparameter tuning

### **Evaluation Metrics**
- **Accuracy**: Overall correctness
- **Precision**: False positive rate
- **Recall/Sensitivity**: False negative rate (critical for safety)
- **F1-Score**: Harmonic mean balancing precision and recall
- **ROC-AUC**: Model discrimination ability across thresholds
- **Confusion Matrix**: Detailed classification breakdown

---

## 🔗 References & Resources

**Datasets**:
- [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)

**Related Topics**:
- Anomaly Detection: [Scikit-learn Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- Autoencoders: [PyTorch Tutorials](https://pytorch.org/tutorials/)
- Imbalanced Learning: [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- Time Series: [Statistical Forecasting](https://www.statsmodels.org/)
