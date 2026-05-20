# Results and Discussions

This section presents the empirical findings obtained from the **Machine Learning Feature Selection Benchmarking Platform**. It details the experimental setup, analyzes the specific features selected by each paradigm, evaluates the downstream classifier performances, and discusses the trade-offs between predictive accuracy, model complexity, and computational speed.

---

## 1. Experimental Setup and Preprocessing

The benchmarking pipeline was evaluated using the standardized **Hotel Booking Demand Dataset** (originally published by António, de Almeida, & Nunes, 2019), representing a real-world, high-dimensional tabular schema in the tourism and hospitality sector. 

### 1.1 Dataset Summary
*   **Original Observations**: 119,390 rows
*   **Target Variable**: `is_canceled` (Binary: `0` for checked-in bookings, `1` for canceled bookings)
*   **Original Features**: 31 columns (containing numeric, categorical, and temporal data)

### 1.2 Automated Preprocessing Pipeline
To eliminate manual intervention, the platform's backend automatically executed the following cleaning and transformation steps:
1.  **Missing Value Imputation**: Continuous variables (e.g., `children`) were imputed using the median, while categorical variables (e.g., `country`, `agent`) were imputed using the mode.
2.  **Noise Filtering**: Invalid records, such as bookings indicating zero total guests (sum of adults, children, and babies is zero), were pruned from the dataset.
3.  **Categorical Encoding**: High-cardinality nominal features (such as `country` and `agent`) and low-cardinality nominal features (such as `deposit_type`, `market_segment`, and `assigned_room_type`) were one-hot encoded.
4.  **Feature Scaling**: Min-Max scaling was applied to shift continuous feature boundaries between `[0, 1]`, ensuring compatibility for distance-based estimators.

Following one-hot encoding, the feature space expanded from **31 raw features to 110+ processed columns**. The preprocessed data was subsequently partitioned into an **80% training set** and a **20% test set** for classification and benchmarking.

---

## 2. Qualitative Analysis of Selected Features

The top $K = 20$ features selected by each algorithm were analyzed to understand their underlying selection logic. The results reveal distinct behavioral differences between the paradigms:

### 2.1 Filter Methods (ANOVA & Chi-Square)
*   **Selected Features**: Both methods selected continuous variables like `lead_time`, along with key categorical features such as `required_car_parking_spaces`, `total_of_special_requests`, specific countries of origin (`country_PRT`, `country_GBR`, `country_FRA`, `country_DEU`), market segments (`market_segment_Groups`, `market_segment_Direct`, `market_segment_Corporate`), and the deposit type (`deposit_type_Non Refund`).
*   **Selection Logic**: Filter methods analyze each feature individually, ignoring feature dependencies. Features with a strong statistical variance (univariate relationship) relative to the target variable are selected. For example, `deposit_type_Non Refund` and `lead_time` exhibit massive univariate correlation with cancellation rates, making them clear statistical choices.

### 2.2 Wrapper (RFE) & Embedded (Lasso) Methods
*   **Selected Features**: Both algorithms selected a high proportion of sparse, categorical variables—specifically low-frequency room types (`reserved_room_type_C`, `reserved_room_type_G`, `assigned_room_type_I`, `assigned_room_type_K`) and minor countries of origin (`country_ARE`, `country_HKG`, `country_SAU`, `country_MAC`, `country_PAN`).
*   **Selection Logic**: Recursive Feature Elimination (RFE) and Lasso (L1 regularization) rely on linear models (Logistic Regression) to calculate feature coefficients. In highly dimensional and multicollinear spaces (e.g., room assignments and country codes), linear coefficients can become unstable. Lasso and RFE arbitrarily selected single, sparse categorical features over highly informative continuous features (like `lead_time` or `adr`), resulting in severe information loss for downstream non-linear classifiers.

### 2.3 Embedded Method (Random Forest Feature Importance)
*   **Selected Features**: Random Forest selected key continuous variables (`lead_time`, `adr` [Average Daily Rate], `stays_in_week_nights`, `agent` ID), alongside critical operational indicators (`previous_cancellations`, `booking_changes`, `required_car_parking_spaces`, `total_of_special_requests`), deposit types, and primary market segments (`market_segment_Online TA`, `market_segment_Offline TA/TO`).
*   **Selection Logic**: As an ensemble tree-based algorithm, Random Forest evaluates feature importance based on Mean Decrease in Gini Impurity. It naturally captures non-linear interactions and is highly robust to multicollinearity. By selecting continuous features that control booking behavior (`lead_time`, `adr`) and past customer behaviors (`previous_cancellations`, `booking_changes`), the method preserves the core information needed for high-accuracy predictions.

---

## 3. Quantitative Downstream Evaluation

To evaluate feature selection efficacy, two downstream classifiers—**Logistic Regression (Linear)** and **XGBoost (Non-linear)**—were trained on the full dataset (Baseline) and on the 20-feature subsets selected by each method.

### 3.1 Downstream Performance Matrix

| Feature Selection Method | Selected Dimension ($K$) | Downstream Classifier | Predictive Accuracy | $F_1$ Score | Execution Time (s) | Speedup Factor |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Baseline (All Features)** | 110+ | Logistic Regression | 81.90% | 73.32% | 3.93s | *1.0x (Ref)* |
| **Baseline (All Features)** | 110+ | XGBoost | **87.95%** | **83.38%** | 6.78s | *1.0x (Ref)* |
| **ANOVA (Filter)** | 20 | Logistic Regression | 78.57% | 65.98% | 0.59s | **6.6x** |
| **ANOVA (Filter)** | 20 | XGBoost | 83.25% | 75.67% | 0.98s | **6.9x** |
| **Chi-Square (Filter)** | 20 | Logistic Regression | 77.79% | 64.67% | 0.61s | **6.4x** |
| **Chi-Square (Filter)** | 20 | XGBoost | 82.56% | 74.84% | 0.92s | **7.3x** |
| **RFE (Wrapper)** | 20 | Logistic Regression | 76.65% | 54.94% | 0.56s | **7.0x** |
| **RFE (Wrapper)** | 20 | XGBoost | 76.76% | 55.03% | 0.97s | **7.0x** |
| **Lasso (Embedded)** | 20 | Logistic Regression | 76.37% | 54.43% | 0.45s | **8.7x** |
| **Lasso (Embedded)** | 20 | XGBoost | 76.56% | 54.57% | 1.29s | **5.3x** |
| **RF Importance (Embedded)**| 20 | Logistic Regression | 80.13% | 69.67% | 0.87s | **4.5x** |
| **RF Importance (Embedded)**| 20 | XGBoost | **86.84%** | **81.47%** | 1.44s | **4.7x** |

---

## 4. Key Discussions and Architectural Trade-Offs

Analyzing the quantitative results highlights three major trade-offs in automated machine learning pipelines:

### 4.1 The Optimality of Random Forest Importance + XGBoost
The combination of **Random Forest Feature Importance** (for dimensionality reduction) and **XGBoost** (for downstream prediction) proved to be the most optimal configuration. 
*   **Accuracy Retention**: It maintained **86.84% accuracy** and **81.47% $F_1$-score**, suffering a negligible accuracy drop of only **1.11%** and a minor $F_1$ drop of **1.91%** compared to the baseline using all 110+ features.
*   **Latency Optimization**: By shrinking the feature space by **over 80%**, the training and inference time dropped from 6.78 seconds to just 1.44 seconds—yielding a **4.7x speedup**. This confirms that the Gini impurity metric successfully retained highly informative variables while shedding redundant noise.

### 4.2 Why Linear Selection (RFE & Lasso) Failed on Tabular Schemas
Both **RFE** (using a Logistic Regression estimator) and **Lasso** performed poorly, with downstream accuracies hovering around **76.5%** and $F_1$-scores plunging to **~55%**. 
*   This performance collapse occurs because linear coefficient selection assumes independent, linear relationships. 
*   In tabular dataset schemas where features are multicollinear (e.g., booking changes, reserved vs. assigned room types, lead time, and pricing), linear feature selectors arbitrarily zero-out valuable continuous variables in favor of sparse categorical variables. Consequently, downstream non-linear models (like XGBoost) are starved of critical predictive signals, resulting in poor decision trees.

### 4.3 Filter Methods as High-Efficiency Baselines
While **ANOVA** and **Chi-Square** underperformed Random Forest Importance by about **3.5% in accuracy**, they delivered outstanding execution speeds. 
*   Training XGBoost on ANOVA-selected features took only **0.98 seconds** (a **6.9x speedup**), while Chi-Square took **0.92 seconds** (a **7.3x speedup**).
*   Because filter methods are calculated via simple statistical tests (F-statistic, Chi-squared statistic) independent of model training, they compute almost instantly. 
*   This makes filter methods highly attractive for high-throughput streaming systems or CPU-constrained environments where the computational cost of running a Random Forest to select features is prohibitive.

> [!IMPORTANT]
> The empirical results demonstrate that **automated feature selection is not a one-size-fits-all process**. For applications prioritizing maximum predictive power, **Embedded Tree-based Selection** is required. Conversely, for systems operating under strict latency constraints or web-response deadlines, **Filter-based Selection (ANOVA)** offers the best compromise, providing rapid calculations with acceptable accuracy trade-offs.

---

## 5. Visualized Benchmarks

The platform generates graphical plots to visualize these trade-offs, which are rendered on the user interface:
1.  **Accuracy Comparison Chart (`accuracy_comparison.png`)**: Visually contrasts how much classification accuracy is retained by each feature selection paradigm relative to the baseline.
2.  **F1-Score Comparison Chart (`f1_comparison.png`)**: Highlights the drop in minority-class predictive capability for RFE and Lasso due to sub-optimal categorical feature selection.
3.  **Execution Time Comparison Chart (`time_comparison.png`)**: Illustrates the dramatic computational speedups (up to 8.7x) achieved by reducing the input feature space.
