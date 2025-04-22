# Dynamic Time Warping Applications

A repository showcasing multiple applications of Dynamic Time Warping (DTW) for time series analysis, including financial market analysis and time series clustering.

## Overview

This project demonstrates practical applications of Dynamic Time Warping in two main domains:

1. **Financial Time Series Analysis**: Comparing QQQ (NASDAQ ETF) and Bitcoin Futures to discover temporal relationships and synchronization patterns
2. **Time Series Clustering**: Comparing DTW with Euclidean distance for clustering time series data from the "Trace" dataset

DTW provides a robust method for measuring similarity between sequences that may vary in speed or time, making it especially valuable for time series data where traditional distance metrics fall short.

## What is Dynamic Time Warping?

Dynamic Time Warping (DTW) is an algorithm for measuring similarity between two temporal sequences that may vary in speed. Unlike Euclidean distance, which compares points at the same time index, DTW finds the optimal alignment between sequences by warping the time axis.

Key advantages of DTW:
- Handles sequences of different lengths
- Accounts for shifts, stretches, and compressions in time
- More robust to time distortions and different sampling rates
- Better captures similarity in pattern rather than exact point-to-point matching

## Project Components

### 1. Financial Market Analysis (dtw_btc_qqq.ipynb)

This notebook explores the relationship between QQQ (tracking NASDAQ) and Bitcoin Futures using DTW:

- **Multiple preprocessing approaches**:
  - Business days alignment
  - Moving Average (MA) smoothing
  - Exponential Moving Average (EMA) smoothing
  - Mixed time frequencies analysis
  - Returns vs. price analysis

- **Analysis techniques**:
  - FastDTW implementation for efficient computation
  - Warping path visualization
  - Lag frequency analysis
  - DTW-aligned time series visualization
  - Correlation analysis (Pearson, Spearman, Kendall's Tau)
  - Tests for non-linear relationships

- **Key findings**:
  - Lag patterns between financial assets
  - Comparison of DTW distances across different time periods
  - Analysis of linear vs. non-linear relationships

### 2. Time Series Clustering Comparison (dtw_vs_euclidean.ipynb)

This notebook compares the effectiveness of DTW vs. Euclidean distance for clustering:

- **Clustering implementation**:
  - TimeSeriesKMeans with both distance metrics
  - Standardized preprocessing
  - Optimal cluster number determination

- **Evaluation methods**:
  - Silhouette analysis
  - Elbow method
  - PCA-based visualization
  - Cluster centroid visualization

- **Comparative results**:
  - DTW vs. Euclidean silhouette scores
  - Cluster quality assessment
  - Pattern capture capabilities

## Datasets

This project utilizes two main datasets:

1. **Financial Market Data**:
   - QQQ (NASDAQ tracking ETF) from Yahoo Finance
   - Bitcoin Futures (BTC-F) from Yahoo Finance
   - Date ranges configurable (examples from 2020-2024)

2. **Trace Dataset**:
   - Synthetic time series data from tslearn.datasets.CachedDatasets
   - Commonly used for classification and clustering benchmarking
   - Contains 200 time series with 275 time points each

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
sklearn
tslearn
yfinance
fastdtw
dcor
```

## Installation

```bash
# Clone this repository
git clone https://github.com/username/dynamic-time-warping.git

# Navigate to the project directory
cd dynamic-time-warping

# Install required packages
pip install numpy pandas matplotlib seaborn scipy scikit-learn tslearn yfinance fastdtw dcor
```

## Usage

The project consists of two main Jupyter notebooks:

```bash
# For financial market analysis
jupyter notebook dtw_btc_qqq.ipynb

# For time series clustering comparison
jupyter notebook dtw_vs_euclidean.ipynb
```

### Key Code Examples

#### 1. Financial Market Analysis with FastDTW

```python
# Apply FastDTW to normalized financial time series
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Normalize data
scaler = MinMaxScaler()
x_s = scaler.fit_transform(qqq_data)  # Normalize QQQ
y_s = scaler.fit_transform(btc_data)  # Normalize BTC_F

# Compute DTW distance and path
distance, path = fastdtw(x_s, y_s, dist=euclidean)
print("DTW distance:", distance)

# Analyze lags
lags = [j - i for (i, j) in path]
```

#### 2. TimeSeriesKMeans with DTW vs Euclidean

```python
# DTW Clustering
km_dtw = TimeSeriesKMeans(
    n_clusters=n_clusters,
    n_init=2,
    metric="dtw",
    verbose=False,
    max_iter_barycenter=10,
    random_state=0
)
y_pred_dtw = km_dtw.fit_predict(X_train)
sil_score_dtw = silhouette_score(X_train, y_pred_dtw, metric="dtw")

# Euclidean Clustering
km_euclidean = TimeSeriesKMeans(
    n_clusters=n_clusters,
    n_init=2,
    metric="euclidean",
    verbose=False,
    max_iter_barycenter=10,
    random_state=0
)
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1])
y_pred_euclidean = km_euclidean.fit_predict(X_train_reshaped)
sil_score_euclidean = silhouette_score(X_train_reshaped, y_pred_euclidean, metric="euclidean")
```

## Applications of Dynamic Time Warping

This project demonstrates two key applications of DTW:

### 1. Financial Market Analysis

- **Market Synchronization**: Detect when markets move in similar patterns regardless of timing
- **Lead-Lag Relationships**: Identify which asset tends to lead or lag the other
- **Non-Linear Correlations**: Capture relationships that traditional correlation measures miss
- **Market Regime Change**: Analyze how relationships evolve over different time periods

### 2. Time Series Clustering

- **Pattern Identification**: Group similar time series patterns even when they're shifted in time
- **Superior Clustering**: Achieve better cluster separation compared to Euclidean distance
- **Feature Extraction**: Use cluster membership as features for downstream tasks
- **Anomaly Detection**: Identify unusual patterns that don't fit well in any cluster

## Additional Applications of DTW

Beyond the applications demonstrated in this repository, DTW has broad utility across many domains:

1. **Speech Recognition**: Aligning speech patterns regardless of speaking speed
2. **Bioinformatics**: Comparing gene expression profiles over time
3. **Gesture Recognition**: Identifying similar movements regardless of execution speed
4. **Medical Data Analysis**: Comparing physiological signals like ECG or EEG
5. **Industrial Process Monitoring**: Detecting anomalies in machine operations
6. **Activity Recognition**: Identifying human activities from sensor data
7. **Signature Verification**: Matching handwritten signatures despite timing variations

## Repository Structure

```
dynamic-time-warping/
├── dtw_btc_qqq.ipynb         # Financial market analysis with DTW
├── dtw_vs_euclidean.ipynb    # Comparison of DTW vs Euclidean for clustering
├── README.md
└── requirements.txt
```

## Future Work

Potential extensions to this project include:
- Implementing other variants of DTW (FastDTW, Derivative DTW, MultiscaleDTW)
- Expanding financial analysis to more asset pairs and market regimes
- Developing trading strategies based on DTW-identified patterns
- Applying DTW-based clustering to real-world multivariate time series
- Exploring soft clustering approaches with DTW

## Conclusion

This repository demonstrates the versatility and power of Dynamic Time Warping across different applications. The results highlight how DTW can capture temporal relationships that traditional distance metrics miss, making it an essential tool for time series analysis.

In the financial market analysis, we observed that DTW can detect subtle relationships between different asset classes even when traditional correlation measures show weak linear relationships. In the clustering comparison, DTW consistently outperformed Euclidean distance in terms of cluster quality and silhouette scores.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Berndt, D. J., & Clifford, J. (1994). Using dynamic time warping to find patterns in time series. KDD Workshop.
- Salvador, S., & Chan, P. (2007). Toward accurate dynamic time warping in linear time and space. Intelligent Data Analysis.
- Tavenard, R. et al. (2020). Tslearn, A Machine Learning Toolkit for Time Series Data.
- Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition.
