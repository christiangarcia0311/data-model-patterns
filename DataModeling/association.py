# Association Rule Model 
# Apriori
# Data Modeling 
# Python Library 
# Author: Christian Garcia

# Import required libraries 
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Association Rules 
# Apriori 
class AssociationRules:
  def __init__(self, min_support=0.1, min_confidence=0.2, min_lift=1.0):
    self.min_support = min_support
    self.min_confidence = min_confidence
    self.min_lift = min_lift
    self.freq_itemset = None 
    self.rules = None 
  
  def fit(self, data):
    
    # Generate frequent itemset
    self.freq_itemset = apriori(data, min_support=self.min_support, use_colnames=True)
    
    # Generate and filter rules 
    self.rules = association_rules(self.freq_itemset, metric='confidence', min_threshold=self.min_confidence)
    self.rules = self.rules[self.rules['lift'] >= self.min_lift]
    
    return self 
  
  # __runnable
  # Get frequent itemset
  def frequent(self):
    return self.freq_itemset
  
  # __runnable 
  # Get association rules 
  def associationrules(self):
    return self.rules 
    