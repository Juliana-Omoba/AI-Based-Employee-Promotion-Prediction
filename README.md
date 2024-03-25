# AI-Based Employee Promotion Prediction


### Project Oveview

This machine learning project aims to provide insights into the prediction of employee promotions. The prediction of promotions entails binary classification, as the target dataset comprises distinct categories of "Promoted" or "Non-Promoted" instances. Through a comprehensive analysis of diverse data facets, our objective is to discern underlying patterns, address the challenge of imbalanced data, attain a deeper comprehension of factors influencing employee promotion, and employ various supervised machine learning algorithms for classification. Ultimately, the aim is to determine the most suitable algorithm for the dataset. By developing a meticulously tuned and trained model for predicting employee promotions, we aspire to facilitate HR departments in expediting their promotion procedures and illustrate the potential applicability of these methodologies to analogous processes.

### Data Sources

The primary dataset utilized for this project was obtained from the Analytics Vidhya learning platform, accessible via the following link: https://datahack.analyticsvidhya.com/contest/wns-analytics-hackathon-2018-1/. This dataset comprises comprehensive information regarding attributes influencing employee promotion.

### Aims 
Exploratory data analysis  involved the employee dataset to answer key questions, such as:

1. To identify the data balancing technique that will optimize the model performance.
2. To identify the factor/features that affect the eligibility of employees getting promoted.
3. To identify machine learning algorithm models that are suitable for users to predict employees’promotion and lastly.
4. To identify performance metrics suitable for evaluating the models’ performance.

### Tools

- Machine Learning Algorithms:
    - Support Vector Machine (SVM)
    - Decision Tree (DT)
    - Generative Adversarial Network (GAN)

- Programming Language:
   - Python [Download here](https://www.python.org/)

### Data Cleaning/Preparation

During the preliminary stages of data preparation, the following tasks are executed:
- Data Loading and Inspection.
- Data Exploration / Analysis / Visualization
    - Explore Missing Values
    - Exploration of Individual Attributes: Verify the distribution of each attribute.
    - Outlier Identification
    - Explore Feature Correlation

### Data Preprocessing
- Drop Unwanted Features
- Discretization
- Capping Outliers
- Normalizing with Standard Scaler

### Feature Engineering and Selection

### Selection of Balancing Techniques
Five distinct data balancing methodologies were chosen for evaluation, encompassing three oversampling techniques and two undersampling techniques. The selected data balancing techniques are delineated as follows:
- Random Oversampling
- Random Undersampling
- Synthetic Minority Over-sampling Technique (SMOTE)
- Adaptive Synthetic Sampling (ADASYN)
- Neighborhood Cleaning Rule Undersampling (edited nearest neighbor (ENN)).

### Algorithm Selection
Five different machine learning algorithms were selected for evaluation. These are:
Logistic Regression
Gaussian Naive Bayes
Decision Tree Classifier
Random Forest Classifier
Support Vector Classifier

### Hyperparameter Tuning

Via the conducted grid searches, optimal parameters were identified for each combination of algorithm and data subset. 
