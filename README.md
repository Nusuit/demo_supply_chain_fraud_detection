# Fraud Supply Chain Detection - Web Demo

An interactive web application for detecting fraud in supply chain transactions using Deep Learning and Social Network Analysis.

## 🎯 Features

- **Dashboard**: Display model performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- **Fraud Prediction**: Real-time fraud detection for individual transactions or batch processing
- **Network Analysis**: Visualize customer-product networks and detect fraud rings
- **Documentation**: Comprehensive information about dataset, methodology, and model architecture

## 🏗️ Architecture

### Model Pipeline

The system employs a sophisticated multi-stage architecture optimized for imbalanced fraud detection:

**1. Deep Neural Network Ensemble**

- **Configuration**: 3 independent DNNs with different random initializations (seeds: 42, 123, 456)
- **Architecture**: Each network follows a progressive layer reduction pattern:
  - Input Layer: 45 features (post-PCA)
  - Hidden Layer 1: 256 neurons (ReLU activation + Dropout 0.3)
  - Hidden Layer 2: 128 neurons (ReLU activation + Dropout 0.3)
  - Hidden Layer 3: 64 neurons (ReLU activation + Dropout 0.3)
  - Output Layer: 1 neuron (Sigmoid activation)
- **Total Parameters**: ~50,000 trainable parameters per model

**2. Feature Engineering**

- **Total Features**: 61 engineered features
  - **Transaction Features (57)**: Temporal patterns, customer behavior, order patterns, product categories, geographical indicators, delivery metrics
  - **Network Features (4)**: Degree centrality, betweenness centrality, clustering coefficient, community detection scores
- **Dimensionality Reduction**: PCA compression from 61 → 45 components
  - **Variance Retained**: 95.12% of original information
  - **Benefit**: Reduces overfitting while preserving critical patterns

**3. Preprocessing Pipeline**

- **StandardScaler**: Normalize features to zero mean and unit variance
- **SMOTE**: Synthetic Minority Over-sampling for handling class imbalance (98.43% legitimate vs 1.57% fraud)
- **PCA**: Principal Component Analysis for feature optimization

**4. Cost-Sensitive Learning**

- **Loss Function**: Custom Focal Loss with False Negative penalty
- **FN_COST**: 15.0 (heavily penalizes missed fraud cases)
- **Decision Threshold**: 0.20 (optimized for high recall - detecting 75% of fraud cases)

**5. Ensemble Prediction**

- **Method**: Average probability from 3 models
- **Benefit**: Reduces variance and improves generalization

## 📦 Installation

### Prerequisites

- Python 3.9+ (tested on Python 3.13)
- pip package manager
- 8GB RAM minimum (for model loading)

### Setup Steps

1. Clone the repository:

```bash
git clone https://github.com/Nusuit/demo_supply_chain_fraud_detection.git
cd demo_supply_chain_fraud_detection
```

2. Create and activate virtual environment (recommended):

```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Linux/Mac
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Running the Application

Launch the Streamlit web interface:

```bash
streamlit run Home.py
```

The application will be available at: **http://localhost:8501**

## 🔧 Usage Guide

### 1. Dashboard (Home Page)

- **Performance Overview**: Comprehensive model evaluation metrics
- **Confusion Matrix**: Visual breakdown of predictions vs actual labels
- **ROC Curve**: Model discrimination capability across thresholds
- **Key Metrics**:
  - **Accuracy**: 90.53% - Overall correctness rate
  - **Precision**: 50.00% - Ratio of true fraud among flagged cases
  - **Recall**: 74.83% - Detection rate of actual fraud cases (primary optimization target)
  - **F1-Score**: 51.17% - Harmonic mean of precision and recall
  - **ROC-AUC**: 82.16% - Area under ROC curve (excellent discrimination)

### 2. Fraud Prediction

**Single Transaction Mode:**

- Input transaction details through interactive form
- Model computes fraud probability in real-time
- Displays individual predictions from all 3 models plus ensemble score
- Visual risk indicator (Low/Medium/High risk)

**Batch Processing Mode:**

- Upload CSV file with multiple transactions
- Parallel processing for efficiency
- Download results with fraud scores and classifications
- Supports thousands of transactions

### 3. Network Analysis

- **Network Statistics**: Node count, edge count, community structures
- **Centrality Analysis**: Identify influential nodes in the transaction network
- **Fraud Ring Detection**: Visualize connected fraud patterns using Cytoscape layout
- **Community Detection**: Detect clusters of suspicious activity

### 4. Documentation

- **Dataset Information**: 180K transactions, 20K customers, statistical overview
- **Model Architecture**: Detailed pipeline and training methodology
- **Evaluation Results**: Comprehensive performance analysis

## 📝 Dataset

**DataCo Supply Chain Dataset** (Kaggle)

### Test Set Statistics

- **Total Transactions**: 18,196 samples (after 90-10 train-test split)
- **Legitimate Transactions**: 17,910 (98.43%)
- **Fraud Transactions**: 286 (1.57%)
- **Class Imbalance Ratio**: 62.6:1 (highly imbalanced)

### Full Dataset Overview

- **Total Transactions**: 180,519 orders
- **Unique Customers**: 20,652 individuals
- **Unique Products**: 118 SKUs
- **Time Period**: Multi-year transaction history
- **Fraud Rate**: ~2.25% (realistic imbalance scenario)

### Feature Categories

1. **Temporal Features**: Order timestamps, delivery schedules, processing times
2. **Customer Behavior**: Purchase frequency, order value patterns, payment methods
3. **Product Information**: Category, department, price range
4. **Geographical Data**: Customer location, shipping destination, distance
5. **Order Characteristics**: Quantity, discount rates, shipping modes
6. **Network Features**: Social network centrality measures derived from customer-product bipartite graph

## 🧠 Model Details

### Ensemble Configuration (AGGRESSIVE Strategy)

**Design Philosophy**: Prioritize fraud detection (high recall) over precision to minimize financial losses from undetected fraud.

**Model Architecture:**

```
Input (45 features)
    ↓
Dense Layer (256 neurons) + ReLU + Dropout(0.3)
    ↓
Dense Layer (128 neurons) + ReLU + Dropout(0.3)
    ↓
Dense Layer (64 neurons) + ReLU + Dropout(0.3)
    ↓
Output (1 neuron) + Sigmoid
```

**Training Configuration:**

- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Custom Cost-Sensitive Focal Loss
  - **False Negative Cost**: 15.0 (missing fraud is 15x worse than false alarm)
  - **Focal Gamma**: 2.0 (focus on hard-to-classify examples)
- **Batch Size**: 128
- **Epochs**: 50 with early stopping
- **Class Balancing**: SMOTE over-sampling on training set

**Ensemble Strategy:**

- **Number of Models**: 3 independent DNNs
- **Random Seeds**: [42, 123, 456] for diversity
- **Aggregation**: Simple averaging of predicted probabilities
- **Decision Threshold**: 0.20 (calibrated for recall optimization)

### Performance Metrics

#### Confusion Matrix (Test Set)

|                       | Predicted Legitimate | Predicted Fraud |
| --------------------- | -------------------- | --------------- |
| **Actual Legitimate** | 17,062 (TN)          | 848 (FP)        |
| **Actual Fraud**      | 72 (FN)              | 214 (TP)        |

**Key Insights:**

- **True Negatives (17,062)**: Correctly identified legitimate transactions - 95.26% of legitimate cases
- **False Positives (848)**: Legitimate flagged as fraud - 4.74% false alarm rate (acceptable trade-off)
- **False Negatives (72)**: Missed fraud cases - 25.17% miss rate (room for improvement)
- **True Positives (214)**: Correctly caught fraud - 74.83% detection rate ✓

#### Classification Metrics

| Metric        | Value  | Interpretation                                               |
| ------------- | ------ | ------------------------------------------------------------ |
| **Accuracy**  | 90.53% | Overall correctness across both classes                      |
| **Precision** | 50.00% | Half of fraud alerts are true fraud (1 in 2 alerts is valid) |
| **Recall**    | 74.83% | Detects 3 out of 4 fraud cases (primary goal achieved)       |
| **F1-Score**  | 51.17% | Balanced measure considering both precision and recall       |
| **ROC-AUC**   | 82.16% | Excellent discrimination ability between classes             |

#### Performance Analysis

**Strengths:**

- ✅ **High Recall (74.83%)**: Successfully detects majority of fraud cases, minimizing financial losses
- ✅ **Strong ROC-AUC (82.16%)**: Model has excellent separability between fraud and legitimate transactions
- ✅ **Good Accuracy (90.53%)**: Maintains high overall performance despite class imbalance
- ✅ **Ensemble Robustness**: Combining 3 models reduces overfitting and improves generalization

**Trade-offs:**

- ⚖️ **Moderate Precision (50%)**: Half of fraud alerts are false positives
  - **Business Impact**: Requires manual review of flagged transactions
  - **Cost Analysis**: Manual review cost < undetected fraud losses
- ⚖️ **Threshold Tuning**: 0.20 threshold chosen to maximize recall while maintaining acceptable precision
  - Lower threshold → Higher recall but lower precision
  - Higher threshold → Higher precision but lower recall

**Optimization Strategy:**
The "AGGRESSIVE" configuration intentionally sacrifices precision for recall because:

1. **Cost Asymmetry**: Missing a fraud case (False Negative) is 15x more expensive than investigating a false alarm (False Positive)
2. **Risk Management**: In financial fraud, catching more cases is more critical than reducing false alerts
3. **Human-in-the-Loop**: False positives can be filtered through manual review, but false negatives result in direct losses

### Why This Configuration?

- **Recall > Precision**: For fraud detection, missing fraud is worse than false alarms
- **Deep Architecture**: Captures complex non-linear patterns in transaction behavior
- **Dropout Regularization**: Prevents overfitting on the imbalanced dataset
- **Ensemble Diversity**: Multiple seeds ensure robust predictions across different model initializations
- **Cost-Sensitive Loss**: Explicitly penalizes false negatives during training

## 🛠️ Tech Stack

- **Framework**: Streamlit 1.51.0 (web interface)
- **Deep Learning**: TensorFlow 2.20.0 / Keras
- **Data Processing**: Pandas 2.2.3, NumPy 2.2.1
- **Visualization**: Matplotlib 3.10.7, Plotly 6.4.0
- **Network Analysis**: NetworkX 3.5
- **Machine Learning**: Scikit-learn 1.6.1

## 📊 Project Structure

```
fraud_demo_streamlit/
├── Home.py                    # Main dashboard and entry point
├── pages/
│   ├── 1_Du_Doan.py          # Fraud prediction interface
│   ├── 2_Phan_Tich_Mang.py   # Network analysis visualization
│   └── 3_Gioi_Thieu.py        # Documentation and methodology
├── models/
│   ├── combined_model_seed42.keras
│   ├── combined_model_seed123.keras
│   └── combined_model_seed456.keras
├── data/
│   ├── sample_transactions.csv    # Demo transaction data
│   ├── test_results.json          # Model performance metrics
│   └── sna_results.json           # Network analysis results
├── utils/
│   ├── preprocessing.py       # Feature engineering pipeline
│   ├── predictor.py          # Model ensemble inference
│   ├── visualizations.py     # Charts and graphs
│   └── constants.py          # Configuration constants
└── requirements.txt          # Python dependencies
```

## 🤝 Contributing

This is a demonstration project for academic and research purposes. The system showcases:

- Deep learning for fraud detection
- Handling severely imbalanced datasets
- Ensemble learning strategies
- Cost-sensitive classification
- Interactive ML web applications

Contributions, suggestions, and improvements are welcome!

## 📄 License

MIT License - Free to use for educational and research purposes.

## 👥 Authors

**Supply Chain Management Project**

- University of Information Technology (UIT)
- Year 4, Semester 1

**Team Members:**

- **Phan Huy Kiên** - Model Architecture & Training
- **Trần Thị Kim Ngân** - Feature Engineering & Data Analysis
- **Lê Trọng Hoàng Dũng** (Dungle-24) - Network Analysis & Visualization
- **Nguyễn Quốc Huy** - Web Application & Deployment

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on the [GitHub repository](https://github.com/Nusuit/demo_supply_chain_fraud_detection).

---

**⚠️ Disclaimer**: This system is designed for educational and research purposes. For production deployment in real fraud detection systems, additional validation, security measures, and compliance with data protection regulations are required.
