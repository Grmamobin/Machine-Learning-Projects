# Machine Learning Projects 🧠
 This is the collection of my Machine Learning course projects in University! 

Let's look at the summary of each project step by step:

<details>
<summary>📐 <strong>Project 1</strong></summary>

1. Exploratory Data Analysis (EDA)
2. Hypothesis Testing
</details>

<details>
<summary>📚 <strong>Project 2</strong></summary>
1. Data Preprocessing and Exploratory Analysis

2. Baseline Linear Regression
    * Train a standard Linear Regression model to prediction
    * Evaluate the model using the following metrics:
        * Mean Absolute Error (MAE)
        * Mean Squared Error (MSE)
        * Coefficient of Determination (R2)

3. Model Variants and Nonlinear Extensions
    * Apply regularized linear models such as Ridge (L2) and Lasso (L1) regression to improve

    * Model generalization and reduce overfitting. Use cross-validation to find optimal regularization parameters and compare results with the baseline model.

    * Explore polynomial regression by introducing higher-degree terms (e.g., degree 2 or 3) to capture nonlinear relationships in the data.

4. Model Enhancement through Data Preparation

    * Explore additional techniques to improve performance using feature engineering and data transformations. Examples may include:
        * Creating interaction features between variables
        * Binning or encoding categorical values differently
        * Identifying and filtering noisy or low-quality samples
</details>

<details>
<summary>📊 <strong>Project 3</strong></summary>

1. Data Preprocessing and Exploratory Data Analysis (EDA)

2. Model Development and Evaluation
    * Models to implement:
        * Logistic Regression
        * Naive Bayes
        * Linear Discriminant Analysis (LDA)
    * Evaluation Metrics:
        * Accuracy
        * Precision, Recall, F1-score
        * ROC-AUC
    * Interpretation & Visualization:
        * Confusion Matrix
        * ROC Curves
        * Precision-Recall Curves

3. Performance Enhancement
    * Apply techniques to improve performance. Possible approaches include:
       * Feature Engineering
       * Regularization (e.g., L1, L2)
       * Cross-Validation
</details>
<details>
<summary>🧩 <strong>Project 4 (Kaggle Competiotion)
</strong></summary>

1. Data Preprocessing and EDA

2. BaseLine Model

    * Train a Logistic Regression (with and without class weights).
    * Evaluate using: Precision, Recall, F1-score, ROC-AUC
    * Plot the ROC curve and confusion matrix.

3. Advanced Models

    * Support Vector Machine (SVM) with different kernels (linear, RBF).
    * Random Forest or Gradient Boosting (XGBoost, LightGBM).
    * k-Nearest Neighbors or Naïve Bayes (for contrast).

4. Handling Imbalanced Data

    * Implement at least two different imbalance-handling techniques, such as:
        - Random undersampling of the majority class
        - Random oversampling or SMOTE for the minority class
        - Class weighting in your estimator's objective function
        - Cost-sensitive learning or ensemble approaches like EasyEnsemble
    * Compare their effect on performance metrics (precision, recall, F1, ROC-AUC) and discuss trade-offs.
</details>

<details>
<summary>🚀 <strong>Project 5: Clustering & Dimensionality Reduction</strong></summary>

1. Data Preprocessing and Exploratory Data Analysis (EDA)
    * Load and Clean

    * Feature Selection
    
    * Descriptive Statistics & Visualization

    * Dimensionality Reduction: Consider applying dimensionality reduction techniques
        * like Principal Component Analysis (PCA) or t-Distributed Stochastic Neighbor
        Embedding (t-SNE) to visualize the data in lower dimensions before clustering, to
        get a sense of inherent groupings.

    * Scaling

2. Clustering Model Development and Evaluation

    * Clustering Models to Implement
        * K-Means Clustering

        * Hierarchical Clustering

        * DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
    * Evaluation Metrics:
        * Silhouette Scoredefined
        clusters.

        * Davies-Bouldin Index

3. Interpretation & Visualization
    * Cluster Visualization
    * Insights and Interpretation
        * Based on your visualizations and the average trait scores, interpret what defines
        each cluster.

        * Compare the results from different clustering algorithms and discuss which algo-
        rithm performed best.    
</details>