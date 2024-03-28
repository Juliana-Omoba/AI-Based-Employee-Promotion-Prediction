# AI-Based Employee Promotion Prediction


### Project Oveview

This study aims to investigate notable challenges encountered by HR teams in leveraging analytics to facilitate decision-making processes, particularly in the context of making employee promotion decisions. These challenges encompass subjectivity, bias, scarcity of data-driven insights, and imbalanced data. The objective of this project is to furnish a comprehensive predictive analytics framework that leverages historical employee data, performance metrics, and diverse machine learning algorithms to effectively mitigate these challenges and precisely forecast employee promotions.


### Data Sources

The primary dataset utilized for this project was obtained from the Analytics Vidhya learning platform, accessible via the following link: https://datahack.analyticsvidhya.com/contest/wns-analytics-hackathon-2018-1/. This dataset comprises comprehensive information regarding attributes influencing employee promotion.

### Aims 
Exploratory data analysis was conducted on the employee dataset to address pivotal inquiries, including:

1. Determining the most effective data balancing technique to enhance model performance.
2. Identifying factors or features influencing employees' eligibility for promotion.
3. Evaluating machine learning algorithm models suitable for predicting employee promotions.
4. Identifying appropriate performance metrics for assessing model effectiveness.

### Tools

- Machine Learning Algorithms:
    - Support Vector Machine (SVM)
    - Decision Tree (DT)
    - Logistic Regression
    - Gaussian Naive Bayes
    - Random Forest Classifier


- Programming Language:
   - Python [Download here](https://www.python.org/)


### Process

This represents a binary classification problem wherein the dataset exhibits significant class imbalance, with approximately 91.5% of instances denoting promotions and 8.5% indicating non-promotions. The dataset encompasses various attributes pertaining to employee performance, demographics, and a binary "Target" variable indicating promotion status. Preliminary examination of the dataset reveals the following:

- The dataset comprises 54,808 distinct entities, featuring 13 attribute columns and 1 target column.
- Notably, the dataset manifests an imbalanced distribution, with 50,140 instances (91.5%) labeled as positive target values (Promoted) and 4,668 instances (8.5%) categorized as negative target values (Not Promoted).
- Instances of missing values are observed in 2 distinct columns. Specifically, the attribute 'education' contains 2,409 missing values, while 'previous_year_rating' contains 4,124 missing values.
- Attributes encompass a combination of character, varchar, float, and integer data types.


### Data Exploration / Analysis / Visualization

1. Exploration of Individual Attributes: In examining the distributions of each attribute and assessing their influence on promotions through the utilization of the pandas crosstab function, significant insights were gleaned from the dataset.
   
    - Gender: Analysis revealed a notable discrepancy in promotion distribution between male and female employees. Specifically, a higher proportion of promotions were awarded to male employees, constituting 68.57% of all promotions, whereas only 31.43% were attributed to female employees. Moreover, the dataset exhibited a higher representation of male employees (70.24%) compared to females (29.76%). Notably, the percentage of promotions within both genders remained relatively similar, with 8.99% of all females and 8.32% of all males being promoted. Consequently, gender alone appears to exert minimal influence on promotion outcomes.
      
    - Recruitment Channel: Examination of the recruitment channels revealed a disparity in promotion rates among employees recruited through diverse channels. While the number of referrals was comparatively minimal in comparison to other recruitment methods, employees recruited through this channel demonstrated a higher promotion rate (12.1%) than those sourced through alternative channels (8.5% for sourcing and 8.4% for other). This discrepancy suggests that the recruitment channel does indeed impact promotion outcomes.
      
    - Department Analysis: Upon investigation, it became evident that the department in which an employee works plays a discernible role in their likelihood of promotion. Notably, a lower proportion of employees in the Legal and HR departments, approximately 5-6%, were observed to receive promotions, whereas a notably higher promotion rate of 10.8% was observed among employees in the Technology department.

    - Previous Year Rating & Average Training Score Examination: Analysis of previous year ratings and average training scores indicated a discernible influence on the likelihood of promotion. Employees with higher ratings and superior training scores exhibited an increased probability of promotion.

    - KPIs Met & Awards Won Evaluation: Evaluation of the binary indicators denoting whether key performance indicators (KPIs) were met (>80%) and whether awards were won revealed their significance in predicting promotion outcomes. The promotion rate notably escalated from 4.0% to 16.9% when employees met more than 80% of their KPIs, and surged from 7.7% to 44.0% when employees had received awards. Consequently, these performance metrics are deemed crucial features to consider in predicting promotions.



2. Additional Insights from Individual Feature Analysis:

    - Trainings Attended Distribution: Analysis revealed that the majority of employees, approximately 81%, have undergone only one training session. Subsequent trainings show a significant decline in participation, with merely 0.4% of employees attending five or more sessions. Interestingly, contrary to expectations, a higher number of trainings does not correlate with an increased probability of promotion; rather, it appears to have an inverse relationship.

    - Educational Attainment Distribution: Predominantly, employees possess a Bachelor's degree (71.3%), whereas only a small fraction holds educational qualifications below the secondary level (1.5%).

    - Age and Promotion Probability: Examination of age distribution revealed a slight peak in promotion probability among employees in their thirties.

    - Length of Service Distribution: The distribution of length of service exhibits a right-skewed pattern, indicating that while the majority of employees have relatively shorter tenures, a minority have remained employed for a considerably longer duration.

    - Feature Importance Ranking: Individual feature analysis suggests that the features exert varying degrees of influence on promotion likelihood. Notably, the order of significance appears to be as follows: average training score, awards won, previous year rating, KPIs met >80, region, number of trainings, department, recruitment channel, length of service, age, education, and gender. Among these, the top three features demonstrate the most pronounced effect on promotion outcomes.


3. Outlier Identification: Numerical features, including Number of Trainings, Age, Length of Service, and Average Training Score, were scrutinized to identify outliers. Outliers, which may signify experimental errors, measurement variability, or anomalies, were assessed as follows:

    - Number of Trainings: Outliers were detected within the Number of Trainings feature. Specifically, any data point deviating from the norm of one training session was flagged as an outlier. This determination was made considering that, as previously established, approximately 81% of employees had undergone only one training session.

    - Age: Outliers were identified for age values surpassing the upper inner fence, calculated as the 75th percentile plus 1.5 times the interquartile range (age > 54). The median age was observed to be 33 years, with the highest recorded age reaching 60.

    - Average Training Score: No outliers were detected within the average training score feature.


4. Exploration of Feature Correlation: Correlation analysis was conducted on quantitative features utilizing Pearson correlation coefficients. The analysis revealed notable findings:

    - Strong Correlation between Age and Length of Service: A strong positive correlation was observed between age and length of service. This correlation is not unexpected, as individuals with more years of service tend to be older.
    - Weak Correlation between Age and Education Level: Conversely, a weak correlation was identified between age and education level. This finding aligns with expectations, as individuals with higher levels of education may not necessarily be older.


5. Data Preprocessing: The preprocessing methods employed are contingent upon the algorithm utilized and the features involved. Consequently, rather than effecting explicit alterations to the data, Python functions were developed to execute diverse preprocessing tasks as necessitated.

    - Handling Missing Values:

        - Previous Year Rating: Missing values within the previous year rating feature were imputed with the median. This decision was motivated by the numerical and ordinal nature of the feature.
        - Education: Missing values in the education feature were imputed with the mode. This decision was based on the categorical nature of the feature, although it possesses ordinal characteristics. However, it's noteworthy that in this context, the median would yield the same result as the mode.
    - Removal of Unnecessary Features:
        - Unnamed: 0 and Employee_ID: The feature columns "Unnamed: 0" (serving as an index) and "Employee_ID" (unique for each data point) were dropped as they are deemed unnecessary for the analysis.

    - Function creations:

        - Ordinal Encoding: A custom function was developed to facilitate ordinal encoding of specified features within the feature dataframe. This approach is particularly relevant when categorical data possesses ordinal characteristics, necessitating conversion into numerical format.
          
        - One-Hot Encoding: Another custom function was devised to perform one-hot encoding on designated features. This method is commonly employed to handle categorical variables where no ordinal relationship exists between categories.
          
        - Discretization: Additionally, a function was constructed to discretize feature data using the KBinsDiscretizer module. Discretization is advantageous when continuous variables need to be transformed into discrete intervals, aiding in simplifying analysis or addressing algorithmic requirements.
          
        - Outlier Capping: An outlier capping function was developed to manage outliers within the feature dataframe. This approach proves useful when the objective is to constrain or mitigate outlier values of features within the dataset, without necessitating the complete removal of affected samples.
          
        - Standard Scaling for Normalization: Another function was devised to normalize features using the standard scaler method. This function is invoked when the features in the dataset require normalization to achieve a normally distributed or similar scale.
          
        - MinMax Scaling for Normalization: Additionally, a function was crafted to normalize features utilizing the MinMax scaler method. This function proves beneficial when there is a need to normalize and transform numerical data into a specific range, typically between 0 and 1. 


6. Feature Engineering and Selection

Efforts were undertaken to discern the dependence of the target variable (is_promoted) on individual features. Features were meticulously examined in isolation, with particular attention directed towards analyzing promotion rates across different categories or ranges within each feature. Significant variations in promotion rates across these categories or ranges suggested heightened importance of the respective feature in predicting promotions. Conversely, consistent promotion rates across categories or ranges indicated lesser significance of the feature in promotion prediction.

Based on the exploratory analysis of the data and the investigation into feature importance, four distinct variations or subsets of data were selected for utilization. The following delineates each subset, specifying the included features and the corresponding data preprocessing steps applied:

    Subset 1: This subset encompasses of all the features and undergoes appropriate encoding, outlier capping, and normalization procedures

    Subset 2: This subset includes most features, but differs in that certain features are dropped.

    Subset 3: The third data subset is nearly the same as Subset 2 with the addition of feature categorization of length_of_service

    Subset 4: Building upon Subset 3, this subset further incorporates feature categorization based on promotion rate for the "region" feature.

These variations entail meticulous feature selection, engineering, and transformation, aiming to enhance the model's capacity to discern patterns. By evaluating and comparing models employing each variation, valuable insights can be garnered regarding the significance of various features in predicting the target variable, as well as the impacts of different preprocessing strategies on model efficacy.


### Selection of Balancing Techniques

To address the dataset's imbalance, a careful selection of five distinct data balancing methodologies was made, covering three oversampling techniques and two undersampling techniques. The chosen data balancing techniques are outlined as follows:

- Random Oversampling
- Random Undersampling
- Synthetic Minority Over-sampling Technique (SMOTE)
- Adaptive Synthetic Sampling (ADASYN)
- Neighborhood Cleaning Rule Undersampling (edited nearest neighbor (ENN)).

### Algorithm Selection

Five distinct machine learning algorithms were chosen for evaluation, encompassing a variety of methodologies.

### Hyperparameter Tuning

Following the conducted grid searches, optimal parameters were determined for each combination of algorithm and data subset. These optimal parameters for each algorithm/data subset pairing are succinctly summarized in Table 2.


![Optimum Parameters](https://github.com/Juliana-Omoba/test/assets/71232282/42a398b4-29a5-4da7-b8ae-3749f73e5a51)


### Results/ Findings

Training was conducted for each unique combination of algorithm, balancing technique, and data subset. Subsequently, results were obtained for each algorithm, and the best result was selected based on the evaluation metrics employed. 
Given the utilization of various algorithms, distinct subsets of data, and diverse data balancing techniques, multiple result tables are generated.

![](images/SupportVector.png)


The table below presents a comparison of the best-performing algorithm, along with its corresponding F1 score, the optimal resampling technique employed, and the most effective feature data variation utilized: 

![Optimum Parameters](https://github.com/Juliana-Omoba/test/assets/71232282/fbb0d86f-dfdd-4fa0-acf6-ce7045e873de)




### Discussion
This study successfully addresses the research objectives outlined in Aim:

a. Data Balancing Technique: The Neighborhood Cleaning Rule (NCR) proves effective in optimizing model performance across various scenarios and datasets.

b. Factors Influencing Promotion Eligibility: Performance features such as Average Training Score and Previous Year Rating exhibit significant effects on promotion eligibility, while demographic factors like Gender, Age, and Education have minimal impact.

c. Suitable Machine Learning Algorithms: Decision Tree models demonstrate robust performance without resampling techniques. Additionally, Support Vector Machines (SVM) paired with Neighborhood Cleaning Rule undersampling enhance prediction accuracy, emphasizing the importance of algorithm choice and resampling strategy.

d. Performance Metrics: The F1 Score emerges as the preferred metric for evaluating model performance, balancing precision and recall effectively. Further examination of the confusion matrix, precision, and recall provides additional insights into the model's performance and potential benefits.


### Challenges

Many functions required extensive runtime, often spanning days, posing significant challenges. To address this, we must refactor the functions to execute within hours while maintaining effectiveness.

### Recommendation

Additional enhancements for future iterations of the project may involve:

1. Intensive Parameter Tuning: Future iterations of the project should include more exhaustive parameter tuning. This entails thoroughly exploring the hyperparameter space of selected machine learning algorithms, possibly using techniques like grid search or random search to optimize model performance. This comprehensive approach compensates for resource limitations encountered during the initial search, ensuring the discovery of optimal hyperparameter configurations.
   
2. Analyzing company goals and conducting cost analysis for false positives and false negatives is crucial. Engaging stakeholders ensures alignment with organizational objectives, while assessing associated costs helps select suitable evaluation metrics and refine the model.

3. Exploring more machine learning algorithms is vital for refining the model. Ensemble methods, advanced models like gradient boosting, XGBoost, or neural networks should be considered. Incorporating diverse algorithms helps uncover various strengths and weaknesses, potentially enhancing the model's ability to capture intricate data patterns.


4. Model Interpretability: Improving model interpretability is crucial for understanding prediction factors. Techniques like SHAP values or LIME can enhance transparency. Interpretable models build stakeholder trust, and understanding feature contributions aids decision-making.
   





