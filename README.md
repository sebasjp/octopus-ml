# OctopusML: A workflow for AutoML

OctopusML is a Machine Learning Workflow proposed in Udacity's Machine Learning Engineer Nanodegree. One task of a Machine Learning Engineer is to design and build applications that automate the execution of predictive models. The goal of this project is to implement a machine learning workflow to increase the efficiency in supervised learning specifically in binary classification problems. The idea is that it can be seen like a baseline for any Data Analyst or Data Scientist.

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
* Statistical analysis:
   + *Bootstrap Confidence Intervals:* Given a significance level, the app computes the confidence intervals for the mean with the goal to compare the two classes that we are trying to predict. It allows us to identify the features that can explain our target.
   + *Chi-square Test:* For the categorical features the Chi-square test for independence is used to identify those that aren’t independent regarding to the target.
These results are just informative, but in the future these ones can be added in a functionality to select features for instance.

