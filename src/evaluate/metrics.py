# Compute ROC curve and ROC area 
from sklearn.metrics import roc_curve, auc
import numpy as np

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
