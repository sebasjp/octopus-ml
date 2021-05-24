from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np

# =============================================================
# Modeling tools for cross validation
# Reference: https://github.com/fmfn/BayesianOptimization/blob/master/examples/sklearn_example.py
# =============================================================

# ===================
# Random Forest
# ===================
def rfc_cv(n_estimators, 
           max_depth,
           min_samples_split,
           min_samples_leaf,
           max_features,
           metric,
           X,
           y,
           preparessor):
    """
    Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, max_depth, min_samples_leaf and max_features. Combined with X and
    y this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split,
    max_depth, min_samples_leaf and max_featues that maximizes the metric
    """
    preprocessor = preparessor
    
    estimator = RandomForestClassifier(
        n_estimators      = n_estimators,
        max_depth         = max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf  = min_samples_leaf,
        max_features      = max_features,
        random_state      = 42
    )
    
    # Append classifier to preparing pipeline. Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', estimator)])
    
    cval = cross_val_score(clf, 
                           X,
                           y,
                           scoring = metric,
                           cv = 5)
    return cval.mean()

# ===================
# XGBoost
# ===================

def xgb_cv(n_estimators,
           max_depth,
           colsample_bytree,
           learning_rate,
           metric,
           X,
           y,
           preparessor):
    """
    XGBoost cross validation.
    This function will instantiate a XGBoost classifier this will perform 
    cross validation. The result of cross validation is returned.
    Our goal is to find combinations that maximizes the metric
    """
    
    preprocessor = preparessor
    
    PARAM_SCALE_POS = np.ceil( len(y[y == 0]) / len(y[y == 1]) )
    
    estimator = xgb.XGBClassifier(
        n_estimators      = n_estimators,
        max_depth         = max_depth,
        colsample_bytree  = colsample_bytree,
        learning_rate     = learning_rate,
        objective         = 'binary:logistic',
        scale_pos_weight  = PARAM_SCALE_POS,
        random_state      = 42,
        verbosity         = 0
    )
    
    # Append classifier to preparing pipeline. Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', estimator)])
    
    cval = cross_val_score(clf, 
                           X,
                           y,
                           scoring = metric,
                           cv = 5)
    return cval.mean()
