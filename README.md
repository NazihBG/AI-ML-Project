# Gamma vs. Hadron Event Classification – Project Overview
Project Objective:
The goal of this project is to classify Gamma and Hadron illnesses based on specific medical features extracted from patient data. By using machine learning techniques, particularly K-Nearest Neighbors (KNN), we aim to develop a model that can accurately differentiate between the two illness types.

This classification can assist medical professionals in diagnosis and decision-making, ensuring that patients receive the appropriate treatment based on their illness category.

Dataset Description:
The dataset contains several numerical features representing various medical and biological attributes of patients. These features are essential indicators that help in distinguishing between Gamma illnesses and Hadron illnesses.

Each record in the dataset includes:
fLength – A specific measurement related to the illness progression.
fWidth – A width-related characteristic of the illness.
fSize – A measure of severity based on medical data.
fConc – Concentration of a certain biomarker.
fAsym – Asymmetry of a detected pattern in the illness.
fM3long, fM3trans – Higher-order medical characteristics.
fAlpha – An angular measurement related to the illness.
fDist – Distance from a reference medical point.
class – The target variable, where:
1 represents Gamma illness.
0 represents Hadron illness.


Project Workflow:
1. Data Preprocessing
Before training the model, the dataset undergoes several preprocessing steps to ensure quality and consistency.
a. Handling Class Imbalance
Medical datasets often have imbalanced data, meaning one illness type may be more common than the other. To correct this, Random OverSampling is applied, which increases the number of samples in the minority class to ensure the model learns equally from both classes.
b. Feature Scaling
Since the dataset consists of numerical medical features with different ranges, they are standardized using StandardScaler to bring all values onto a similar scale. This improves the model’s ability to learn patterns without being biased by large numerical values.
c. Splitting the Dataset
To properly train and evaluate the model, the dataset is split into three parts:
Training Set (60%) – Used to train the model.
Validation Set (20%) – Used to fine-tune the model parameters.
Test Set (20%) – Used to assess final model performance.

2. Exploratory Data Analysis (EDA)
Before training the model, it’s important to analyze the dataset to gain insights into how Gamma and Hadron illnesses differ.
Feature Distribution Analysis – Histograms are plotted to visualize the probability distributions of each feature for both illness types. This helps in understanding which features contribute most to the classification.
Correlation Analysis – Checking for relationships between features to identify the most relevant medical indicators.
EDA allows us to make informed decisions about which features might be most useful for classification.

3. Model Selection and Training
The K-Nearest Neighbors (KNN) algorithm is chosen for classification due to its simplicity and effectiveness. The model works as follows:
Stores all training examples.
When a new patient’s data is given, it finds the k nearest cases in the dataset.
The illness type of the majority of these nearest neighbors is assigned to the new case.
The model is trained using different values of k to determine the optimal number of neighbors for classification.

4. Model Evaluation
After training, the model is tested using validation and test data. Several evaluation metrics are used to measure performance:
Accuracy – The percentage of correctly classified illness cases.
Precision – Measures how many of the predicted Gamma (or Hadron) illness cases were actually correct.
Recall – Measures how many actual Gamma (or Hadron) illness cases were correctly identified.
F1-Score – A balanced metric that considers both precision and recall.
These metrics help assess whether the model is reliable enough for real-world medical applications.



Conclusion:
This project demonstrates how machine learning can be applied in the medical field to classify illnesses based on extracted features. By preprocessing the data, balancing the classes, and applying KNN classification, the model successfully distinguishes between Gamma and Hadron illnesses.

Future improvements could include testing advanced models such as Support Vector Machines (SVM), Random Forests, or Neural Networks to achieve even higher accuracy and reliability in classification.

