# Regression Model 
# Data Modeling 
# Python Library 
# Author: Christian Garcia

# Import required libraries 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

# Decision tree regressor model
class DecisionTree:
  def __init__(self):
    self.model = DecisionTreeRegressor()
    self.X_train = None
    self.X_test = None 
    self.y_train = None 
    self.y_test = None 
    self.y_pred = None 
    self.mse = None 
    self.mae = None 
    self.r_score = None 
  
  def fit(self, X, y, test_size=0.3, random_state=42):
    # Split data to test and train sets 
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
      
    # Train the data and predict the test data
    self.model.fit(self.X_train, self.y_train)
    self.y_pred = self.model.predict(self.X_test)
    
    # Calculate evaluation metrics 
    self.mse = metrics.mean_squared_error(self.y_test, self.y_pred)
    self.mae = metrics.mean_absolute_error(self.y_test, self.y_pred)
    self.r_score = metrics.r2_score(self.y_test, self.y_pred)
    
    return self
  
  # __runnable
  # Return evaluation metrics 
  def evaluation_metrics(self):
    return pd.DataFrame({
      'MSE': [f'{self.mse:.3f}'],
      'MAE': [f'{self.mae:.3f}'],
      'R-Squared': [f'{self.r_score:.3f}']
    })
  
  # __runnable
  # Return tree rules 
  def tree_rules(self, feature_name, style='fivethirtyeight'):
    plt.style.use(style)
    plt.figure(figsize=(20, 10))
    plot_tree(self.model, feature_names=feature_name, filled=True, rounded=True)
    plt.show()

class MultipleLinearRegression:
  def __init__(self):
    self.model = LinearRegression()
    self.X_train = None 
    self.X_test = None 
    self.y_train = None 
    self.y_test = None 
    self.y_pred = None 
    self.coef = None 
    self.intercept = None 
    self.mse = None 
    self.mae = None 
    self.r_score = None 
    
  def fit(self, X, y, test_size=0.3, random_state=42):
    # Split data to test and train sets 
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
      
    # Train the data and predict the test data
    self.model.fit(self.X_train, self.y_train)
    self.y_pred = self.model.predict(self.X_test)
    
    # Get coefficient and intercept 
    self.coef = self.model.coef_.round(2)
    self.intercept = self.model.intercept_.round(2)
    # Calculate evaluation metrics 
    self.mse = metrics.mean_squared_error(self.y_test, self.y_pred)
    self.mae = metrics.mean_absolute_error(self.y_test, self.y_pred)
    self.r_score = metrics.r2_score(self.y_test, self.y_pred)
    
    return self 
  
  # __runnable  
  # Return evaluation metrics 
  def evaluation_metrics(self):
    return pd.DataFrame({
      'MSE': [f'{self.mse:.3f}'],
      'MAE': [f'{self.mae:.3f}'],
      'R-Squared': [f'{self.r_score:.3f}']
    })
  
  # __runnable
  # Return coefficient and intercept 
  def coef_intercept(self):
    return pd.DataFrame({
      'Coefficient': [self.coef],
      'intercept': [self.intercept]
    })
  
# KNN regressor model 
class KNearestNeighbor:
  def __init__(self, n_neighbors=5):
    self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
    self.X_train = None
    self.X_test = None 
    self.y_train = None 
    self.y_test = None 
    self.y_pred = None 
    self.mse = None 
    self.mae = None 
    self.r_score = None
  
  def fit(self, X, y, test_size=0.3, random_state=42):
    # Split data to test and train sets 
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
      
    # Train the data and predict the test data
    self.model.fit(self.X_train, self.y_train)
    self.y_pred = self.model.predict(self.X_test)
    
    # Calculate evaluation metrics 
    self.mse = metrics.mean_squared_error(self.y_test, self.y_pred)
    self.mae = metrics.mean_absolute_error(self.y_test, self.y_pred)
    self.r_score = metrics.r2_score(self.y_test, self.y_pred)
    
    return self
  
  # __runnable
  # Return evaluation metrics
  def evaluation_metrics(self):
    return pd.DataFrame({
      'MSE': [f'{self.mse:.3f}'],
      'MAE': [f'{self.mae:.3f}'],
      'R-Squared': [f'{self.r_score:.3f}']
    })
    
