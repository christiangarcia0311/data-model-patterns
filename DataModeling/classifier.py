# classification Model 
# Classifiers
# Data Modeling
# Python Library
# Author: Christian Garcia

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

# Zero-R Classifier Model
class ZeroR:
  def __init__(self):
    self.majority_class = None
  
  def fit(self, data, target_column):
    self.majority_class = data[target_column].mode()[0]
  
  # __runnable
  # Predict the target 
  def predict(self, data):
    
    # Check if the model is been fitted 
    if self.majority_class is None:
      raise ValueError('model has not been fitted yet.')
    return pd.Series([self.majority_class] * len(data))
  
  # __runnable
  # Calculate the model accuracy 
  def score(self, data, target_column):
    predict_data = self.predict(data)
    model_accuracy = (predict_data == data[target_column]).mean().round(3)
    return model_accuracy
  
  # __runnable
  # Show target summary 
  def summary(self, data, target_column):
    value_count = data[target_column].value_counts() 
    overall_count = len(data)
    overall_percent = (overall_count / overall_count) * 100
    max_count = value_count.max()
    max_percent = (max_count / overall_count) * 100
    min_count = value_count.min()
    min_percent = (min_count / overall_count) * 100 
    
    data_results = pd.DataFrame({
      'Instance Type': ['Correctly Classified Instances', 'Incorrectly Classified Instances', 'Total Number of Instances'],
      'Count': [max_count, min_count, overall_count],
      'Percent': [f'{max_percent:.2f}%', f'{min_percent:.2f}%', f'{overall_percent:.1f}%']
    })
    
    return data_results 

# One-R Classifier Model
class OneR:
  def __init__(self):
    self.selected_attribute = None
    self.selected_rule = None 
    self.model_accuracy = 0 
    
  def fit(self, data, target_column):
    
    # Iterate the predictors and exclude the target
    for predictors in data.columns.drop(target_column):
      value_count = defaultdict(lambda: defaultdict(int))
      
      # Iterate through data rows 
      for index, rows in data.iterrows():
        value_count[rows[predictors]][rows[target_column]] += 1 
      
      # Create a rule for selected attribute
      rule = {}
      total_correct = 0 
      
      for value, target_count in value_count.items():
        # Find the most common class for each attribute value 
        common_class = max(target_count, key=target_count.get)
        rule[value] = common_class
        total_correct += target_count[common_class]
      
      # Calculate model accuracy 
      accuracy = total_correct / len(data)
      
      if accuracy > self.model_accuracy:
        self.model_accuracy = accuracy
        self.selected_attribute = predictors 
        self.selected_rule = rule 
  
  # __runnable
  # Predict data target value
  def predict(self, data):
    
    # Empty list for storing predictions 
    predictions = []
    
    # Iterate over data rows
    for index, rows in data.iterrows():
      value = rows[self.selected_attribute]
      prediction = self.selected_rule[value]
      predictions.append(prediction)
    
    return predictions
  
  # __runnable
  # Show data best predictor 
  def best_predictor(self):
    return self.selected_attribute, self.selected_rule, self.model_accuracy
  
  # __runnable  
  # Show target summary 
  def summary(self, data, target_column):
    value_count = data[target_column].value_counts() 
    overall_count = len(data)
    overall_percent = (overall_count / overall_count) * 100
    max_count = value_count.max()
    max_percent = (max_count / overall_count) * 100
    min_count = value_count.min()
    min_percent = (min_count / overall_count) * 100 
    
    data_results = pd.DataFrame({
      'Instance Type': ['Correctly Classified Instances', 'Incorrectly Classified Instances', 'Total Number of Instances'],
      'Count': [max_count, min_count, overall_count],
      'Percent': [f'{max_percent:.2f}%', f'{min_percent:.2f}%', f'{overall_percent:.1f}%']
    })
    
    return data_results 
  
# Gaussian Naive Bayes Classifier 
class NaiveBayesian:
  def __init__(self):
    self.model = GaussianNB()
    self.X_train = None 
    self.X_test = None 
    self.y_train = None 
    self.y_test = None 
    self.y_pred = None 
    self.model_accuracy = None 
    
  def fit(self, X, y, test_size=0.3, random_state=42):
    # Split data to test and train sets 
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
      
    # Train the data and predict the test data
    self.model.fit(self.X_train, self.y_train)
    self.y_pred = self.model.predict(self.X_test)
      
    # Calculate model accuracy 
    self.model_accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
    
    return self
  
  # __runnable  
  # Return accuracy score
  def score(self):
    return self.model_accuracy
  
  # __runnable  
  # Return confusion matrix
  def confusionmatrix(self):
    matrix = metrics.confusion_matrix(self.y_test, self.y_pred) 
    labels = self.model.classes_
    matrix_table = pd.DataFrame(matrix, index=labels, columns=labels)
      
    return matrix_table
  
  # __runnable
  # Return classification report 
  def report(self):
    return metrics.classification_report(self.y_test, self.y_pred)

# Decision Tree Classifier 
class DecisionTree:
  def __init__(self):
    self.model = DecisionTreeClassifier()
    self.X_train = None 
    self.X_test = None 
    self.y_train = None 
    self.y_test = None 
    self.y_pred = None 
    self.model_accuracy = None
    
  def fit(self, X, y, test_size=0.3, random_state=42):
    # Split data to test and train sets 
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
      
    # Train the data and predict the test data
    self.model.fit(self.X_train, self.y_train)
    self.y_pred = self.model.predict(self.X_test)
    
    # Calculate model accuracy 
    self.model_accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
    
    return self 
  
  # __runnable  
  # Return model accuracy 
  def score(self):
    return self.model_accuracy
  
  # __runnable
  # Return confusion matrix 
  def confusionmatrix(self):
    matrix = metrics.confusion_matrix(self.y_test, self.y_pred) 
    labels = self.model.classes_
    matrix_table = pd.DataFrame(matrix, index=labels, columns=labels)
      
    return matrix_table
  
  # __runnable
  # Return classification report 
  def report(self):
    return metrics.classification_report(self.y_test, self.y_pred)
  
  # __runnable
  # Return tree rules 
  def tree_rules(self, feature_name, style='fivethirtyeight'):
    plt.style.use(style)
    plt.figure(figsize=(20,10))
    plot_tree(self.model, feature_names=feature_name, class_names=self.model.classes_, filled=True, rounded=True)
    plt.show()