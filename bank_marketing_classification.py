import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
dataset = pd.read_csv(url, sep=';')

# Display the first few rows
print(dataset.head())

# Display dataset information
print(dataset.info())

# Check for missing values
print(f"Missing values: {dataset.isnull().sum().sum()}")

# Summary statistics
print(dataset.describe())

# Visualize the distribution of the target variable
sns.countplot(x='y', data=dataset)
plt.title('Distribution of Target Variable')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Encode categorical features
dataset = pd.get_dummies(dataset, drop_first=True)

# Normalize numerical features
scaler = MinMaxScaler()
numerical_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

# Split the data into train and test sets
X = dataset.drop('y_yes', axis=1)
y = dataset['y_yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Display the shapes of the train and test sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

def model_scores(y_test, y_pred, y_pred_proba, show_plot=True):
    # Evaluation metrics
    print('Confusion matrix :\n', confusion_matrix(y_test, y_pred))
    print('Accuracy :', accuracy_score(y_test, y_pred))
    print('AUC score :', roc_auc_score(y_test, y_pred_proba))
    print('F1-score :', f1_score(y_test, y_pred))
    print('Precision score :', precision_score(y_test, y_pred))
    print('Recall score :', recall_score(y_test, y_pred))
    # ROC
    fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_pred_proba)
    if show_plot:
        plt.figure()
        plt.plot(fpr_test, tpr_test)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC-AUC')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()

def model_evaluation(model, X_train, y_train, X_test, y_test, trained=False):
    if not trained:
        model.fit(X_train, y_train)
    # Predict on train set
    y_pred_train = model.predict(X_train)
    y_pred_proba_train = model.predict_proba(X_train)[::, 1]
    print('Train dataset :')
    model_scores(y_train, y_pred_train, y_pred_proba_train, show_plot=False)
    # Predict on test set
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[::, 1]
    print('\nTest dataset :')
    model_scores(y_test, y_pred_test, y_pred_proba_test, show_plot=True)

# Logistic Regression
print("\n##### LOGISTIC REGRESSION #####")
log_reg = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000)
model_evaluation(log_reg, X_train, y_train, X_test, y_test, trained=False)

# Neural Network
print("\n##### NEURAL NETWORKS #####")
mlp = MLPClassifier(max_iter=500, alpha=0.0001, hidden_layer_sizes=(100,), learning_rate='adaptive', random_state=42)
model_evaluation(mlp, X_train, y_train, X_test, y_test, trained=False)

# Applying SMOTE for oversampling
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Logistic Regression with oversampled train
print("\n##### LOGISTIC REGRESSION (oversampled data) #####")
model_evaluation(log_reg, X_train_sm, y_train_sm, X_test, y_test, trained=False)

# Neural Network with oversampled train
print("\n##### NEURAL NETWORKS (oversampled data) #####")
mlp = MLPClassifier(max_iter=500, alpha=0.0001, hidden_layer_sizes=(100,), learning_rate='adaptive', random_state=42)
model_evaluation(mlp, X_train_sm, y_train_sm, X_test, y_test, trained=False)