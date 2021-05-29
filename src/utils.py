import os
import logging

from sklearn.model_selection import train_test_split

# Compute ROC curve and ROC area 
from sklearn.metrics import roc_curve, auc
import numpy as np

def split_data(data, y_name, test_size, SEED):
    """
    This function splits the data in train and test sets, but before, it performs one hot encoding for categorical features


    Parameters
    ----------
    X : pd.DataFrame
        Pandas DataFrame to use. This one contains just X features
    y : pd.Series
        Variable of interest
    test_size : float
        Number between 0 and 1 that indicate the proportion of records in test set
    SEED: int
        Seed used to do reproducible the execution


    Return
    ------
    X_train : pd.DataFrame
        Pandas DataFrame to use in trainig. This one contains just X features
    X_test : pd.DataFrame
        Pandas DataFrame to use in test. This one contains just X features
    y_train : pd.Series
        Pandas Series to use in trainig. This one contains just the interest feature
    y_test : pd.Series
        Pandas Series to use in test. This one contains just the interest feature
    """
    
    X = data.drop(columns = y_name).copy()
    y = data[y_name]
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = test_size,
                                                        random_state = SEED
                                                       )

    return X_train, X_test, y_train, y_test


# =============================================================
# cast features
# =============================================================

def cast(data, features_type):
    
    X = data.copy()
    
    for xname in features_type['qualitative']:
        X[xname] = [str(x) if x is not None else None for x in X[xname]]
    
    for xname in features_type['quantitative']:
        X[xname] = [float(x) if x is not None else None for x in X[xname]]
    
    return X

# =============================================================
# Evaluation function
# =============================================================

def compute_metrics(X, y_true, model):
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

def renderize_html(html, path_html):
    """
    To renderize the HTML
    
    Parameters
    ----------
    html : str
        Information in HTML language.
    path_html : str
        Path where the HTML page will be saved.
    """
    html += "<br></body></html>"

    with open(path_html, 'w') as out:
        out.write(html)