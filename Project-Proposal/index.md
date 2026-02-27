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

For the stock market, one study analyzed “...algorithms like decision trees, random forests, [SVM] with different kernels, and K-Means… [1].” Radial Base Function SVM performed the best (88% prediction accuracy) but ran the longest. For Polymarket, one model used Reinforcement Learning with Verifiable Rewards and LLMs, challenging the best models and leading to about 10% ROI in a Polymarket simulation [2]. Also, studies showed prediction markets were more accurate than analysts regarding company earnings predictions, with 68% accuracy a week out and 77% accuracy a day out, but analysts were only 62% accurate [3]. This suggests a strong connection between predictions with both markets.

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

### Preprocessing Methods

- **Data Cleaning:** Interpolation will handle missing values stemming from market holidays or collection gaps. SimpleImputer and KNNImputer will preserve local temporal structures during imputation.
- **Data Normalization:** We normalize features with StandardScaler (Z-score) for fair comparison across markets, and RobustScaler for outlier-heavy datasets.
- **Feature Engineering:** We will compute key technical indicators (volatility, volume trends, log returns) and generate temporal features, including rolling statistics, cyclical encodings (e.g., day-of-week), and lagged variables to capture short-term momentum.
- **Target Variable Construction:** We frame both domains as comparable forecasting tasks. We predict log returns or directional movements for stocks and  deviation between the final settlement and implied probabilities for polymarket.

### ML Algorithms

- **Principal Component Analysis:** We use PCA to reduce dimensionality and identify intrinsic dimensionality differences between stock and Polymarket markets.
- **Linear Regression:** We apply linear regression and could use Ridge, Lasso, and Elastic Net regularization to prevent overfitting, with alpha selected via cross-validation.
- **Support Vector Machine:** To capture complex, non-linear market relationships,we use SVM for classification and regression tasks.
- **Long Short-Term Memory network:** Given the temporal nature of market data, we use LSTM networks to model long-term dependencies.

PCA, Linear Regression and SVM can be implemented via scikit-learn; LSTM via PyTorch.

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
