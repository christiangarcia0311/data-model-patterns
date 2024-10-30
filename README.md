![Static Badge](https://img.shields.io/badge/pypi-v.3.1.0-blue)
![Static Badge](https://img.shields.io/badge/data_modeling-green)
![Static Badge](https://img.shields.io/badge/classification-model-red)
![Static Badge](https://img.shields.io/badge/regression-blue)
![Static Badge](https://img.shields.io/badge/clustering-orange)
![Static Badge](https://img.shields.io/badge/association-rules-brown)


<h1 align="center"> Data Modeling</h1

![logo](images/logo.jpg)

[Reference...](#documentation-reference)

A **mining model** is created by applying an algorithm to _data_, but it is more than an algorithm or a metadata container: it is a set of _data_, _statistics_, and _patterns_ that can be applied to new data to generate predictions and make inferences about relationships. 

## Modeling

**Predictive modeling** is the process by which a model is created to predict an outcome. If the outcome is _categorical_ it is called **classification** and if the outcome is _numerical_ it is called **regression**. **Descriptive modeling** or **clustering** is the assignment of observations into clusters so that observations in the same cluster are similar. Finally, **association rules** can find interesting associations amongst observations.

### Installation 

> install data modeling with pip.

CLI:

```bash
    pip install data-model-patterns
```

```bash
    pip install https://github.com/christiangarcia0311/data-exploration-analysis/raw/main/dist/data_model_patterns-3.1.0.tar.gz
```

## Classification

> Classification is a data mining task of predicting the value of a categorical variable (target or class) by building a model based on one or more numerical and/or categorical variables (predictors or attributes). 

```python
    # import classifier model
    from DataModeling import classifier
```

### Dataset
> Using a sample data (weather nominal dataset).

```python
    # import pandas library
    import pandas as pd
    
    # sample data (weather nominal dataset)
    data = pd.DataFrame({
                 'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
                 'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
                 'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
                 'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'True'],
                 'Play Golf': ['N', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N']
    })
```

### Zero-R

> `ZeroR` is the simplest classification method which relies on the target and ignores all predictors. ZeroR classifier simply predicts the majority category (class). Although there is no predictability power in ZeroR, it is useful for determining a baseline performance as a benchmark for other classification methods.

**Sample/Usage**:

> python code

```python
    """
    Perform Zero-R Classifier
    
    Parameters ->
     - data (DataFrame): the dataset containing the predictor variables and target column.
     - target_column (str): the name of the target column in the dataset.
    
    """
```

> zero-r classifier:

```python
    # initialize the zero-r classifier
    zero_r = classifier.ZeroR()
```

```python
    # fit the classifier
    zero_r.fit(data, 'Play Golf')
```

> results/output:

```python
    # predict the most frequent value
    zero_r.predict(data)
    
    #  calculate the model accuracy
    zero_r.score(data, 'Play Golf')
    
    # get the data summary
    zero_r.summary(data, 'Play Golf')
```

### One-R

> `OneR`, short for "One Rule", is a simple, yet accurate, classification algorithm that generates one rule for each predictor in the data, then selects the rule with the smallest total error as its "one rule". To create a rule for a predictor, we construct a frequency table for each predictor against the target. It has been shown that OneR produces rules only slightly less accurate than state-of-theart classification algorithms while producing rules that are simple for humans to interpret.

**Sample/Usage**:

> python code


```python
    """
    Perform One-R Classifier
    
    Parameters ->
     - data (DataFrame): the dataset containing the predictor variables and target column.
     - target_column (str): the name of the target column in the dataset.
    
    """
```

> one-r classifier:

```python
    # initialize the zero-r classifier
    one_r = classifier.OneR()
```

```python
    # fit the classifier
    one_r.fit(data, 'Play Golf')
```

> results/output:

```python
    # data prediction returns as list
    one_r.predict(data)
    
    # show data best predictor and accuracy
    attribute, rule, accuracy = one_r.best_predictor()
    
    # get the data summary
    one_r.summary(data, 'Play Golf')
```

### Naive Bayesian

> The `Naive Bayesian` classifier is based on Bayes’ theorem with independence assumptions between predictors. A Naive Bayesian model is easy to build, with no complicated iterative parameter estimation which makes it particularly useful for very large datasets. Despite its simplicity, the Naive Bayesian classifier often does surprisingly well and is widely used because it often outperforms more sophisticated classification methods.

**Sample/Usage**:

> python code

```python
    """
    Perform naive bayes classifier
    
    Parameters ->
     - X (matrix): the feature column in dataset.
     - y (array): the target column in dataset.
    
    """
```

> naive bayesian classifier:

```python
    # initialize the naive bayes classifier
    nb_classifier = classifier.NaiveBayesian()
```

```python
    #  - before train the model
    #  - convert categorical values to numerical
    #  - since we have categorical values / ignore it if the data contains numerical values
    data = pd.get_dummies(data, columns=['Outlook', 'Temperature', 'Humidity', 'Windy'])
    
    # separate feature and target values
    X = data.drop('Play Golf', axis=1)
    y = data['Play Golf']
```

```python
    # fit the model with the dataset
    nb_classifier.fit(X, y)
```

> results/output:

```python
    # get model accuracy score
    nb_classifier.score()
    
    # get model confusion matrix
    nb_classifier.confusionmatrix()
    
    # get classification report
    print(nb_classifier.report())
```

### Decision Tree - Classification

> `Decision tree` builds classification or regression models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. 

**Sample/Usage**:

> python code

```python
    """
    Perform decision tree
    
    Parameters ->
     - X (matrix): the feature column in dataset.
     - y (array): the target column in dataset.
     - feature_name (str): the feature column names to apply rules.
       
    """
```

> decision tree:

```python
    # initialize the decision tree classifier
    decisiontree = classifier.DecisionTree()
```

```python
    #  - before train the model
    #  - convert categorical values to numerical
    #  - since we have categorical values / ignore it if the data contains numerical values
    data = pd.get_dummies(data, columns=['Outlook', 'Temperature', 'Humidity', 'Windy'])
    
    # separate feature and target values
    X = data.drop('Play Golf', axis=1)
    y = data['Play Golf']
```

```python
    # fit the model with the dataset
    decisiontree.fit(X, y)
```

> results/output:

```python
    # get model accuracy score
    decisiontree.score()
    
    # get model confusion matrix
    decisiontree.confusionmatrix()
    
    # get classification report
    decisiontree.report()
    
    # visualize tree rules
    decisiontree.tree_rules(feature_name=X.columns)
```

## Regression

> Regression is a data science task of predicting the value of target (numerical variable) by building a model based on one or more predictors (numerical and categorical variables).

```python
    # import regression model
    from DataModeling import regression
```

### Dataset

We create a sample data to be use, from _Dataset_ class in _data exploration analysis_ module.

Full documentation and sample: [Github](https://github.com/christiangarcia0311/data-exploration-analysis/tree/main#dataset-generator)

> install library

```bash
    pip install data-exploration-analysis
```

> generate dataset:

```python
    # import data exploration library
    from DataExploration import analysis
    
    # initialize the dataset class
    dataset = analysis.Dataset()
```

```python
    # create a sample data
    sample_data = {
          'Feature1': ('float', 1, 10),
          'Feature2': ('float', 18, 23),
          'Target': ('float', 1, 3)
          }
          
    # generate the sample data
    data = dataset.make_dataset(sample_data, n_instance=10)
```

### Decision Tree - Regression

> `Decision tree` builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. 

**Sample/Usage**:

> python code

```python
    """
    Perform decision tree
    
    Parameters ->
     - X (matrix): the feature column in dataset.
     - y (array): the target column in dataset.
     - feature_name (str): the feature column names to apply rules.
       
    """
```

> decision tree:

```python
    # initialize the decision tree regressor
    decisiontree = regression.DecisionTree()
```

```python
    # train the model
    X = data.drop('Target', axis=1)
    y = data['Target']
```

```python
    # fit the model with the dataset
    decisiontree.fit(X, y)
```

> results/output:

```python
    # get model evaluation metrics
    decisiontree.evaluation_metrics()
    
    # visualize data tree rules
    decisiontree.tree_rules(feature_name=X.columns)
```

### Multiple Linear Regression

> `Multiple linear regression (MLR)` is a method used to model the linear relationship between a dependent variable (target) and one or more independent variables (predictors).

**Sample/Usage**:

> python code

```python
    """
    Perform multi linear regression
    
    Parameters ->
     - X (matrix): the feature column in dataset.
     - y (array): the target column in dataset.
     
    """
```

> MLR

```python
    # initialize the mlr model 
    mlr = regression.MultipleLinearRegression()
```

```python
    # fit the model with the dataset
    mlr.fit(X, y)
```

> results/output:

```python
    # get model evaluation metrics
    mlr.evaluation_metrics()
    
    # get coefficient and intercept of the model
    mlr.coef_intercept()
```

### K-Nearest Neighbor

> `K nearest neighbors` is a simple algorithm that stores all available cases and predict the numerical target based on a similarity measure (e.g., distance functions). KNN has been used in statistical estimation and pattern recognition already in the beginning of 1970’s as a non-parametric technique.

**Sample/Usage**:

> python code

```python
    """
    Perform KNN
    
    Parameters ->
     - X (matrix): the feature column in dataset.
     - y (array): the target column in dataset.
     
    """
```

> KNN

```python
    # initialize knn model
    knn = regression.KNearestNeighbor()
```

```python
    # fit the model with the dataset
    knn.fit(X, y)
```

> results/output

```python
    # get model evaluation metrics
    knn.evaluation_metrics()
```

## Clustering

>  cluster is a subset of data which are similar. Clustering (also called unsupervised learning) is the process of dividing a dataset into groups such that the members of each group are as similar (close) as possible to one another, and different groups are as dissimilar (far) as possible from one another. Clustering can uncover previously undetected relationships in a dataset.

```python
    # import cluster model
    from DataModeling import clustering
```

### Dataset

We create a sample data to be use, from _Dataset_ class in _data exploration analysis_ module.

Full documentation and sample: [Github](https://github.com/christiangarcia0311/data-exploration-analysis/tree/main#dataset-generator)

> install library

```bash
    pip install data-exploration-analysis
```

> generate dataset:

```python
    # import data exploration library
    from DataExploration import analysis
    
    # initilize the dataset class
    dataset = analysis.Dataset()
```

```python
    # create a sample data representing pixel height/width
    sample_data = {
          'px_height': ('float', 0.0, 1000.0),
          'px_width': ('float', 0.0, 1000.0)
          }
          
    # generate the sample data
    data = dataset.make_dataset(sample_data, n_instance=10)
```

### Heirarchical Clustering

> `Hierarchical` clustering involves creating clusters that have a predetermined ordering from top to bottom. 

**Sample/Usage**:

> python code

```python
    """
    Perform Heirarchical
    
    Parameters ->
     - X (DataFrame): the selected column in dataset to be cluster.
     
    """
```

> Heirarchical clustering:

```python
    # initialize heirarchical clustering model
    heirarchical = clustering.HeirarchicalClustering()
```

```python
    # fit the model with dataset
    heirarchical.fit(X)
```

> results/output:

```python
    # visualize cluster dendrogram
    heirarchical.dendrogram()
    
    # get cluster labels
    heirarchical.labels()
```

### K-Means Clustering

> `K-Means` clustering intends to partition n objects into k clusters in which each object belongs to the cluster with the nearest mean. This method produces exactly k different clusters of greatest possible distinction. The best number of clusters k leading to the greatest separation (distance) is not known as a priori and must be computed from the data. 

**Sample/Usage**:

> python code

```python
    """
    Perform Kmeans
    
    Parameters ->
     - X (DataFrame): the selected column/feature in dataset to be cluster.
     - n_clusters (int): initialize the number of clusters.
     - feature_names (list): column name converted to list.
     - range_length (int):  clusters length to show scores.
     
    """
```

> KMeans

```python
    # initialize the kmeans clustering model
    # default number of cluster -> 3
    kmeans = clustering.KMeansClustering(n_clusters=3)
```

```python
    # - before train the model
    # - extract the feature column names to list
    col_names = data.columns.tolist()
    
    # - convert the dataframe to numpy array
    X = data.values
```

```python
    # fit the model with the dataset converted to array
    kmeans.fit(X)
```

> results/output:

```python
    # get cluster centers
    kmeans.centers(col_names)
    
    # get kmeans model inertia
    kmeans.inertia()
    
    # get cluster labels
    kmeans.labels()
    
    # visualize the cluster data points
    kmeans.plot_clusters(X, xlabel='px_height', ylabel='px_width')
    
    # get model silhouette score default length -> 10
    kmeans.score(X, range_length=10) 
```

## Association Rules

> Association Rules find all sets of items (itemsets) that have support greater than the minimum support and then using the large itemsets to generate the desired rules that have confidence greater than the minimum confidence. The lift of a rule is the ratio of the observed support to that expected if X and 
Y were independent.

```python
    from DataModeling import association
```

### Dataset

> Using a sample data (transaction dataset). each column represent an item and row represent as binary (1 -> if item is present and 0 -> if not).

```python
    data = pd.DataFrame({
                'Milk': [1, 1, 1, 0, 0],
                'Bread': [1, 0, 1, 0, 1],
                'Butter': [0, 1, 0, 1, 1],
                'Beer': [0, 1, 1, 1, 1],
                'Eggs': [1, 0, 0, 1, 0]
    })
```
### Association Rules - Apriori

> The `Apriori` algorithm takes advantage of the fact that any subset of a frequent itemset is also a frequent itemset. The algorithm can therefore, reduce the number of candidates being considered by only exploring the itemsets whose support count is greater than the minimum support count. All infrequent itemsets can be pruned if it has an infrequent subset.

**Sample/Usage**:

> python code

```python
    """
    Perform Association Rule
    
    Parameters ->
     - data (DataFrame): the dataset being used.
     - min_support (float): minimum support threshold for an itemset. (to be considered frequent)
     - min_confidence (float): minimum confidence for a rule. (to be considered strong)
     - min_lift (float): minimum lift for a rule. (to be considered interesting)
     
    """
```

> Apriori - Association Rule
```python
    # initialize association rule model
    ar_model = association.AssociationRules()
```

```python
    # fit the model with dataset
    ar_model.fit(data)
```

> results/output:

```python
    # get frequent itemsets
    print(ar_model.frequent())
    
    # get association rules
    print(ar_model.associationrules())
```

## Plot Configuration

| Arguments | Value |
|-----------|--------|
| `style` | ggplot, bmh, dark_background, fivethirtyeight, grayscale |
| `xlabel` | label name in X-axis |
| `ylabel` | label name in Y-axis |

# License
![Static Badge](https://img.shields.io/badge/MIT-License-blue)

# Author
![Static Badge](https://img.shields.io/badge/Christian-Garcia-orange?link=https%3A%2F%2Fgithub.com%2Fchristiangarcia0311)

# Documentation Reference

[Data Mining](https://www.saedsayad.com/data_mining.htm)

[Data Modeling](https://www.saedsayad.com/modeling.htm)

Feel free to contribute to this library by submitting issues or pull requests to the repository.

