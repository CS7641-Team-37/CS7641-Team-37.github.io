---
layout: null
---

<div align="center">

<h1>PolyStock: Dimensionality Comparison of E-Markets</h1>
<h2>Project Proposal</h2>
<p><strong>Team 37</strong></p>

</div>

## 1 INTRODUCTION

The stock market and Polymarket involve money changing hands, and almost anyone can participate. Both rely on making predictions, leading to greater profits. However, the stock market has long-term share investment, existing since 1602 (1792 in the US), whereas Polymarket has short-term event betting, created in 2020.

### 1.1 Literature Review

It is important to make accurate and efficient predictions in both markets, vitalizing machine learning algorithms. For the stock market, one study analyzed “the effectiveness of algorithms like decision trees, random forests, support vector machines (SVM) with different kernels, and K-Means Clustering… [1].” Radial Base Function SVM performed the best (88% prediction accuracy) but ran longer than the other algorithms. For Polymarket, one model used Reinforcement Learning with Verifiable Rewards and Large Language Models, competing with the best models and leading to about 10% ROI in a Polymarket simulation [2]. Also, studies have shown that prediction markets were more accurate than analysts regarding company earnings predictions, with 68% accuracy a week out and 77% accuracy a day out, but analysts were only 62% accurate [3]. This suggests there's a strong connection between making predictions with both markets.

### 1.2 Dataset Description

<ul>
	<li>Stock market dataset</li>
	<ul>
		<li>Historical daily prices for all current NASDAQ tickers</li>
		<li>Features: date, open, high, low, close, adj close, and volume for each ticker</li>
	</ul>
	<li>Polymarket prediction markets</li>
	<ul>
		<li>Deduplicated prediction market events and individual markets from Polymarket</li>
		<li>Events features: event metadata, trading metrics, status flags, timestamps, category tags, competitiveness scores, market counts per event, etc.</li>
		<li>Markets features: market details, trading activity, order book data, event references, reward structures, etc.</li>
	</ul>
	<li>Full market data from polymarket</li>
	<ul>
		<li>Raw market data from Polymarket Gamma API</li>
		<li>Features: book, holder, price every 4 hours, trade for each market, etc.</li>
	</ul>
</ul>

### 1.3 Dataset Links

https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset
https://www.kaggle.com/datasets/ismetsemedov/polymarket-prediction-markets/data?select=polymarket_markets.csv
https://www.kaggle.com/datasets/sandeepkumarfromin/full-market-data-from-polymarket

## 2 PROBLEM DEFINITION
Stock prediction algorithms are a staple in the machine learning world. Recently, however, polymarkets where users can place bets on things ranging from elections to sports outcomes have exploded, reaching ~$10b in monthly trade volume within less than two years of launch (see Figure 1). Our high level goal is to build a system which can optimally make investment decisions across both the stock and poly markets.

<figure>
	<iframe width="100%" height="420" frameborder="0" src="https://www.theblock.co/data/decentralized-finance/prediction-markets-and-betting/polymarket-and-kalshi-volume-monthly/embed" title="Polymarket, Polymarket US and Kalshi Volume (Monthly)"></iframe>
	<figcaption><em>Figure 1: Polymarket, Polymarket US, and Kalshi monthly volume trends.</em></figcaption>
</figure>

### 2.1 Problem

Because of the relative infancy of polymarkets, there is a lot of volatility in prices, risk, and options. This is contrary to more mature equity markets. There also may be much more diverse features which play a role in these younger markets, like social sentiment and narrative polarization. Similar trends may exist in the stock market, but it is hard to gauge transferability.

### 2.2 Motivation

This motivates a project which 1. identifies if the dimensionality of stock/poly markets is meaningfully different and 2. frames both domains as comparable forecasting tasks. This could be useful to identify which signals are transferable across markets and which are market-specific. This can guide future systems and help quantify when to prioritize cross-market and specialized learning.

## 3 METHODS

### 3.1 Data Preprocessing

#### Missing Value Handling
Missing values in time series data (due to holidays or collection gaps) could be handled using `sklearn.impute.SimpleImputer` with median strategy as baseline, and `sklearn.impute.KNNImputer` (k=5) for preserving local structure. Time-series-specific interpolation (linear/spline) via `pandas` will also be evaluated.

#### Feature Scaling
To ensure fair comparison across markets with different scales, `sklearn.preprocessing.StandardScaler` applys Z-score normalization (mean=0, std=1). For datasets with outliers, `sklearn.preprocessing.RobustScaler` (based on median and IQR) provides stronger robustness.

#### Feature Engineering and Time Series Processing
Technical indicators could be computed using `pandas` and `numpy`: volatility (standard deviation of returns), volume trends (moving average slopes), and log returns. For time series, lagged features (1/3/7 periods), rolling statistics, and cyclical encodings (day-of-week, month) will be generated. `sklearn.preprocessing.PolynomialFeatures` will create interaction features to capture non-linear relationships.

#### Target Variable Construction
For stock prediction, continuous targets (log returns: log(Price_t+1/Price_t)) and discrete targets (three-class: up/down/flat based on ±1% threshold) could be constructed using `pandas` and `numpy`. For Polymarket, the target will be the difference between actual settlement results and current implied probability.

### 3.2 Unsupervised Learning Methods

#### Principal Component Analysis (PCA)
`sklearn.decomposition.PCA` could reduce dimensionality while maximizing variance retention. The number of components explaining 95% variance will quantify intrinsic dimensionality differences between stock and Polymarket markets. PCA-reduced features will also serve as inputs to supervised models.

#### K-Means Clustering and Gaussian Mixture Models (GMM)
`sklearn.cluster.KMeans` will identify market states by minimizing within-cluster sum of squares, with optimal K determined by silhouette score (`sklearn.metrics.silhouette_score`) or elbow method. `sklearn.mixture.GaussianMixture` will handle irregular cluster shapes using probabilistic clustering, with model selection via Bayesian Information Criterion (BIC).

#### Nonlinear Dimensionality Reduction and Visualization
A `tensorflow.keras` autoencoder could extract nonlinear representations of complex market structures. For 2D visualization, UMAP (`umap-learn`) and t-SNE (`sklearn.manifold.TSNE`) will project the data to highlight distribution differences between markets.

### 3.3 Supervised Learning Methods

#### Random Forest and XGBoost
`sklearn.ensemble.RandomForestClassifier/Regressor` could serve as the ensemble method, with hyperparameters (n_estimators, max_depth, min_samples_split) optimized via `sklearn.model_selection.GridSearchCV`. XGBoost (`xgboost` library) provides gradient boosting with parallel computation and built-in regularization to prevent overfitting.

#### Support Vector Machines (SVM)
`sklearn.svm.SVC` and `sklearn.svm.SVR` implement classification and regression with RBF kernels for capturing nonlinear relationships. Parameters C and gamma will be tuned via grid search. PCA preprocessing will be applied for computational efficiency on large datasets.

#### Linear and Regularized Regression
`sklearn.linear_model.LinearRegression` could provide the baseline. Ridge (`Ridge`, L2 regularization), Lasso (`Lasso`, L1 feature selection), and Elastic Net (`ElasticNet`, combined L1/L2) will prevent overfitting. Regularization strength alpha will be selected via cross-validation.

#### Long Short-Term Memory (LSTM)
LSTM networks implemented via `tensorflow.keras.layers.LSTM` or `torch.nn.LSTM` models long-term dependencies in time series data. Unlike traditional ML with fixed lag features, LSTM learns adaptive temporal patterns through gating mechanisms. Architecture: LSTM layers (50-100 units) followed by dense output layers. Dropout (0.2-0.3) prevents overfitting. Early stopping halts training when validation loss plateaus.

## 4 RESULTS AND DISCUSSION

### 4.1 Project Goals & Expected Results (Quantitative)

Our primary goal is to quantitatively determine if prediction markets (Polymarket) exhibit lower intrinsic dimensionality than equity markets (NASDAQ, NYSE), leveraging these findings to build optimal forecasting models. We expect Polymarket to show lower intrinsic dimensionality due to its binary, outcome-specific nature.

To evaluate our methodology, we will use the following scikit-learn metrics:

#### Explained Variance Ratio ('sklearn.metrics.explained_variance_score')
Quantifies intrinsic dimensionality via PCA. If traditional markets exist on highly complex, nonlinear manifolds, we expect nonlinear algorithms (Isomap) or Sparse PCA to better preserve geometry and interpretability. We hypothesize Polymarket will require significantly fewer components to reach 95% explained variance.
#### Mean Squared Error ('sklearn.metrics.mean_squared_error')
The primary predictive risk metric. Given heavy market noise, we expect Random Forests to provide a robust, low-MSE baseline that resists overfitting.
#### R^2 Score ('sklearn.metrics.r2_score')
Measures the proportion of future price variance successfully explained by our independent variables.
#### Silhouette Score ('sklearn.metrics.silhouette_score')
Evaluates clustering performance to ensure discovered market regimes are cohesive and well-separated.


### 4.2 Expected Results (quantitative)

#### Ethical Considerations
Financial forecasting carries inherent risks. Prediction markets are volatile and heavily driven by social sentiment rather than objective financial reality. We will transparently document model limitations to prevent algorithmic bias or blind trust in high-stakes decisions.
#### Sustainability Considerations
Training deep learning models (LSTMs) consumes vast computational power. Proving these markets exhibit lower intrinsic dimensionality justifies using computationally efficient methods, reducing our carbon footprint and energy overhead.

## 5 REFERENCES

[1]		A. Chakravorty and N. Elsayed, “A Comparative Study of Machine Learning Algorithms for Stock Price Prediction Using Insider Trading Data,” *arXiv.org*, 2025. [Online]. Available: https://arxiv.org/html/2502.08728v1. [Accessed February 23, 2026].

[2]		B. Turtel, D. Franklin, K. Skotheim, L. Hewitt, and P. Schoenegger, “Outcome-based Reinforcement Learning to Predict the Future,” *arXiv.org*, 2025. [Online]. Available: https://arxiv.org/pdf/2505.17989. [Accessed February 23, 2026].

[3]		G. McCubbing, “Can Polymarket-style prediction markets beat analysts this earnings season? Polymarket thinks so,” *Australian Financial Review*, February 4, 2026. [Online]. Available: https://www.afr.com/markets/equity-markets/can-prediction-markets-beat-analysts-this-earnings-season-20260202-p5nyul [Accessed February 23, 2026].

[4]     I. Sorokin and J. F. Puget, "NVARC solution to ARC-AGI-2 2025," *Google Drive*, 2025. [Online]. Available: https://drive.google.com/file/d/1vkEluaaJTzaZiJL69TkZovJUkPSDH5Xc/view. [Accessed February 26, 2026].

[5]     Scikit-learn developers, "3.4. Metrics and scoring: quantifying the quality of predictions," *scikit-learn 1.8.0 documentation*. [Online]. Available: https://scikit-learn.org/stable/modules/model_evaluation.html. [Accessed February 26, 2026]. 

[6]     L. Breiman, "Random forests," *Machine Learning*, vol. 45, pp. 5–32, 2001. [Online]. Available: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf. [Accessed February 26, 2026].

[7]     H. Zou, T. Hastie, and R. Tibshirani, "Sparse principal component analysis," *Journal of 
Computational and Graphical Statistics*, vol. 15, no. 2, pp. 265–286, 2006. [Online]. Available: https://hastie.su.domains/Papers/spc_jcgs.pdf. [Accessed February 26, 2026].

[8]     J. B. Tenenbaum, V. de Silva, and J. C. Langford, "A global geometric framework for nonlinear dimensionality reduction," *Science*, vol. 290, pp. 2319–2323, 2000. [Online]. Available: https://www.robots.ox.ac.uk/~az/lectures/ml/tenenbaum-isomap-Science2000.pdf. [Accessed February 26, 2026].

## 6 CONTRIBUTION TABLE

| Name | Proposal Contributions |
| :--- | :--- |
| Name | Contributions |
| Name | Contributions |
| Name | Contributions |
| Name | Contributions |
| Name | Contributions |

## 7 GANTT CHART
