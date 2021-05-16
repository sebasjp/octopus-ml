import os
import logging

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Compute ROC curve and ROC area 
from sklearn.metrics import roc_curve, auc
import numpy as np

def log(path_output, filelogs):
    """
    This function create a log file to record the logs and
    specify the logger to print in console as well
    
    Args
    ----
    path_output (str): generic path where the output will be saved
    dirname (str): folder name where the output will be saved
    filelogs (str): file name where the logs will be recorded
    
    Return
    ------
    logger (logger): logger object configured to use
    """
    # check if the directories and file exist
    log_file = os.path.join(path_output, filelogs)
    
    if not os.path.exists(path_output):
        os.mkdir(path)
    
    #if not os.path.exists(path_dir):
    #    os.mkdir(dir_name)
    
    if not os.path.isfile(log_file):
        open(log_file, "w+").close()
    
    #set the format of the log records and the logging level to INFO
    logging_format = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format = logging_format, 
                        level = logging.INFO)
    logger = logging.getLogger()
    
    # create a file handler for output file
    handler = logging.FileHandler(log_file)
    # set the logging level for log file
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter(logging_format)
    handler.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.addHandler(handler)

    return logger

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
           y):
    """
    Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, max_depth, min_samples_leaf and max_features. Combined with X and
    y this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split,
    max_depth, min_samples_leaf and max_featues that maximizes the metric
    """
    estimator = RandomForestClassifier(
        n_estimators      = n_estimators,
        max_depth         = max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf  = min_samples_leaf,
        max_features      = max_features,
        random_state      = 42
    )
    cval = cross_val_score(estimator, 
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
           y):
    """
    XGBoost cross validation.
    This function will instantiate a XGBoost classifier this will perform 
    cross validation. The result of cross validation is returned.
    Our goal is to find combinations that maximizes the metric
    """
    
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
    cval = cross_val_score(estimator, 
                           X,
                           y,
                           scoring = metric,
                           cv = 5)
    return cval.mean()


# =============================================================
# Evaluation function
# =============================================================

def evaluate(X, y_true, model):
    """
    This function compute the performance's metrics for one model
    
    Args:
    -----
    y_true (pd.Series): True labels
    y_pred (pd.Series): Predicted labels
    y_proba (array): Probability predicted
    
    Return:
    -------
    dict_metrics (dict): dictionary that contains the recall, precision,
                         accuracy, f1-score and AUC
    """
    
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:,1]
    
    vp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    vn = np.sum((y_true == 0) & (y_pred == 0))
    
    # calcular el recall
    recall = vp / (vp + fn)
    
    # calcular el precision  
    precision = vp / (vp + fp)
    
    # accuracy
    acc = (vp + vn) / (vp + fn + fp + vn)
    
    # f1-score
    f1 = 2 * precision * recall / (precision + recall)
    
    # AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    dict_metrics = {'recall' : recall, 
                    'precision' : precision,
                    'f1' : f1,
                    'accuracy' : acc,
                    'auc' : roc_auc}
    
    return dict_metrics