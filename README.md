# OctopusML: A workflow for AutoML

[![status dev](https://img.shields.io/badge/status-dev-sucess.svg)](https://github.com/sebasjp/octopus-ml) [![dev python](https://img.shields.io/badge/python-v3.7-informational.svg)](https://github.com/sebasjp/octopus-ml)

OctopusML is a Machine Learning Workflow proposed in Udacity's Machine Learning Engineer Nanodegree. One task of a Machine Learning Engineer is to design and build applications that automate the execution of predictive models. The goal of this project is to implement a machine learning workflow to increase the efficiency in supervised learning specifically in binary classification problems. The idea is that it can be seen like a baseline for any Data Analyst or Data Scientist.

![Octopus](https://github.com/sebasjp/octopus-ml/blob/master/octopusimages.png)

# Requirements

```
python >= 3.7
pip install -i https://test.pypi.org/simple/ bran
bayesian-optimization == 1.2.0
matplotlib  == 3.2.2
numpy == 1.20.1
pandas == 1.2.4
scipy == 1.5.3
seaborn == 0.11.1
sklearn == 0.23.2
statsmodels == 0.11.1
xgboost == 1.3.3
```
# What can OctopusML do?

![OctopusML](https://github.com/sebasjp/octopus-ml/blob/master/OctopusML_complete.png)

### Octopus Process 
It's the component that does the preprocessing and the statistical analysis tasks.

* Split data in train and test sets.
* **Remove categorical features:** 
   + It's going to allow a maximum of 10 categories in a feature. If one feature has more than 10 and it has 9 categories with more than 75% of data, the remaining categories are going to be categorized like `others`.
   + If its first 9 categories don’t collect at least of the 75% data, the feature will be removed.
   + Also, if the feature has just one category, it will be removed.
   + If just one category collects more than 99% of records, the feature will be removed.
* **Remove numerical features:** If just one value collects more than 99% of records, the feature will be removed.
* **Check the missing values:** If one feature has more than a x% of missing values, it’s going to be removed. The x% value can be given by the analyst.
* **Outlier detection:** Three methods are available.
   + *Adjusted boxplot (Hubert, et al. 2008):* It’s an univariate method used for skewed distributions. If one sample has at least one feature like outlier, that one will be removed.
   + *Local Outlier Factor (LOF) (Breuning, et al. 2000):* It’s an unsupervised and multivariate method based on distances, which computes the local density deviation between each point and its neighbors.
   + *Isolation Forest (Liu, et al. 2008):* It’s an unsupervised method based on decision trees and works on the principle of isolating anomalies.
* **Statistical analysis:**
   + *Bootstrap Confidence Intervals:* Given a significance level, the app computes the confidence intervals for the mean with the goal to compare the two classes that we are trying to predict. It allows us to identify the features that can explain our target.
   + *Chi-square Test:* For the categorical features the Chi-square test for independence is used to identify those that aren’t independent regarding to the target.
These results are just informative, but in the future these ones can be added in a functionality to select features for instance.

### Octopus Prepare
It's the component that basically builds the pipeline with all transformations that are going to be used in the train process, like you can see in the figure.

### Octopus Train 
It's the component that trains 4 kinds of models proposed, like you can see in the figure. The hyperparameter tuning for Regularized Logistic Regression, the Grid Search Cross Validation is used. For Random Forest and XGBoost models, the Bayesian Optimization with Cross Validation is used.

### Octopus Evaluate 
It's the component that evaluates the tuned models in Octopus Train. For a metric given, this component calculates the cross-validation performance and chooses the best model. Then, all metrics are computed with the test set.

# How to use OctopusML?
To use OctopusML you need to clone this repo and then specify some configurations in one jupyter notebook:
1. To specify all numerical an categorical variables in a dictionary:
```
# X features names
features_type = {'qualitative': ['feature_1', ..., 'feature_k'],
                'quantitative': ['feature_2', ..., 'feature_p']}

# target name
y_name = 'name_target'

# create a output folder where you wish
path_output = 'data/output/project/'
```
2. To specify the following arguments:
```
config = {}
config['test_size']        = 0.25
config['min_missing']      = 0.25
config['outliers_method']  = 'lof' # can be 'adjbox' or 'isolation_forest' as well
config['alpha_sta']        = 0.05
config['strategy_missing'] = 'median' # can be 'mean' as well
config['method_scale']     = 'standard' # can be 'robust' as well
config['metric_train']     = 'roc_auc' # can be any metric in sklearn
config['seed']             = 42
config['njobs']            = -1
```
3. Now, you can execute it!
```
from octopus import OctopusML
OctoML = OctopusML(
                test_size        = config['test_size'],
                min_missing      = config['min_missing'],
                outliers_method  = config['outliers_method'],
                alpha_sta        = config['alpha_sta'],
                strategy_missing = config['strategy_missing'],
                method_scale     = config['method_scale'],
                metric_train     = config['metric_train'],
                njobs            = config['njobs'],
                seed             = config['seed'])
                
results = OctoML.autoML(
                   data          = data,
                   y_name        = y_name,
                   features_type = features_type,
                   path_output   = path_output)
```

* In this notebook you can see an example about how to use OctopusML.
* You can execute each step separately as well. In this notebook you can see it.

# Contributing ![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

If you wish to collaborate, here one list about the things that are missing:

* Improve function's documentation.
* More options to handle missing values (imputer).
* To add different posibilities to recognize missing values (nan, none, ?, anything specified by the user).
* Methods to deal with imbalanced datasets.
* To add at HTML the metrics results in evaluate.
* Add feature importances.
* Feature selection; some ideas:
   * Based on statistical analysis.
   * Based on feature importances.
* Recommend if the model is overfitted or not.
* Interpretable Machine Learning.
* Unit test to functions.
* To add more logs to follow better the code.

# License [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/sebasjp/octopus-ml/blob/master/LICENSE)

This project is built under the MIT license, for more information visit [here](https://github.com/sebasjp/octopus-ml/blob/master/LICENSE).

# References

* XIN, Doris, et al. Whither AutoML? Understanding the Role of Automation in Machine Learning Workflows. arXiv preprint arXiv:2101.04834, 2021.

* BREUNIG, Markus M., et al. LOF: identifying density-based local outliers. En Proceedings of the 2000 ACM SIGMOD international conference on Management of data. 2000. p. 93-104.

* HUBERT, Mia; VANDERVIEREN, Ellen. An adjusted boxplot for skewed distributions. Computational statistics & data analysis, 2008, vol. 52, no 12, p. 5186-5201.

* LIU, Fei Tony; TING, Kai Ming; ZHOU, Zhi-Hua. Isolation forest. En 2008 eighth ieee international conference on data mining. IEEE, 2008. p. 413-422.

* https://github.com/fmfn/BayesianOptimization
