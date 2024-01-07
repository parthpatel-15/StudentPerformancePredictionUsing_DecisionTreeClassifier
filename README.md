# StudentPerformancePredictionUsing_DecisionTreeClassifier
This Python script employs a Decision Tree classifier to predict student performance based on various features.
It goes through essential steps such as loading the dataset, creating a target variable, separating features, preprocessing categorical data, constructing a decision tree classifier, evaluating model performance, visualizing the decision tree, and fine-tuning the model for improved predictions.

# Detailed Steps:

- Data Loading and Initial Investigations:
  The script begins by loading a dataset containing student performance information. Initial investigations include examining data types, identifying missing values, obtaining summary statistics, and exploring unique values in specific columns.
- Target Variable Creation:
A new target variable, 'pass_parth,' is created based on the sum of grades (G1, G2, G3). This binary variable indicates whether a student passed or failed.
- Feature Extraction:
Features and the target variable are separated from the dataset to facilitate model training.
- Categorical Data Preprocessing:
Categorical data, such as 'address,' 'famsize,' and 'school,' is identified and processed using one-hot encoding to prepare it for machine learning models.
- Decision Tree Model:
A Decision Tree classifier is constructed with a specified criterion (entropy) and maximum depth.
- Pipeline Creation:
A pipeline is established, integrating preprocessing steps (such as one-hot encoding) and the Decision Tree classifier. Pipelines streamline the workflow and enhance code readability.
- Model Training and Evaluation:
The model is trained on the training set, and its accuracy is evaluated on both the training and testing sets. Cross-validation is employed to assess the model's generalization performance.
- Decision Tree Visualization:
The script visualizes the decision tree using Graphviz, offering insights into how the model makes predictions based on the input features.
- Model Fine-Tuning:
Fine-tuning is performed through a randomized search for hyperparameter optimization. The script prints the best hyperparameters, the corresponding score, and the best estimator. The tuned model's performance is then evaluated.
- Model Saving:
The best model and the entire pipeline are saved using Joblib. Saving models allows for future use without the need to retrain.

# Usage:

- Execute the script to predict student performance using the Decision Tree classifier.
- Explore the decision tree visualization and review model evaluation metrics.
- Experiment with different hyperparameters for fine-tuning the model.
