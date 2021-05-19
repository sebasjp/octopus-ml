# Modeling
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV

from modeling.models_cv import rfc_cv
from modeling.models_cv import xgb_cv

class octopus_train:
    
    def __init__(self, seed, metric, njobs):
        
        self.SEED = seed
        self.metric = metric
        self.njobs = njobs
    
    def logistic_regression(self, X_train, y_train):
        """
        Logistic Regression without regularization
        """
        lr = LogisticRegression(C = 1e6,
                                max_iter = 500,
                                random_state = self.SEED)
        lr.fit(X_train, y_train)
        
        return lr
    
    def regularized_logistic_regression(self, X_train, y_train):
        """
        Logistic Regression with regularization
        """

        lrr = LogisticRegression(max_iter = 500,
                                 random_state = self.SEED)        

        grid = {'C': [1000, 100, 10, 1, 0.1, 0.08, 0.02, 0.001, 0.0001, 0.015, 0.0001]}

        grid_search = GridSearchCV(estimator = lrr, 
                                   param_grid = grid, 
                                   n_jobs = self.njobs,
                                   scoring = self.metric
                                   )

        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_
    
    # Definition optimizer for random Forest
    def optimize_rfc(self,
                     X_train, 
                     y_train,
                     metric
                    ):
        """
        Apply Bayesian Optimization to Random Forest parameters.
        """
        def rfc_crossval(n_estimators, 
                         max_depth,
                         min_samples_split,
                         min_samples_leaf,
                         max_features):
            """
            Wrapper of RandomForest cross validation.
            Notice how we ensure n_estimators and min_samples_split are casted
            to integer before we pass them along. Moreover, to avoid max_features
            taking values outside the (0, 1) range, we also ensure it is capped
            accordingly.
            """
            val = rfc_cv(
                        n_estimators      = int(round(n_estimators)),
                        max_depth         = int(round(max_depth)),
                        min_samples_split = int(round(min_samples_split)),
                        min_samples_leaf  = int(round(min_samples_leaf)),
                        max_features      = max(min(max_features, 0.999), 1e-3),
                        metric = metric,
                        X = X_train,
                        y = y_train,
                       )
            return val

        hyp_space = {"n_estimators" : (50, 500),
                     "max_depth"    : (2, 10),
                     "min_samples_split": (15, 100),
                     "min_samples_leaf" : (5, 50),
                     "max_features": (0.1, 0.999)
                    }
                        
        optimizer = BayesianOptimization(
                            f            = rfc_crossval,
                            pbounds      = hyp_space,
                            random_state = self.SEED,
                            verbose      = 0)

        optimizer.maximize(init_points = 20, n_iter = 50)

        return optimizer.max
    
    # Definition Optimizer for XGBoost
    def optimize_xgb(self,
                     X_train, 
                     y_train,
                     metric
                    ):
        """
        Apply Bayesian Optimization to Random Forest parameters.
        """
        def xgb_crossval(n_estimators,
                         max_depth,
                         colsample_bytree,
                         eta):
            """
            Wrapper of XGBoost cross validation.
            Notice how we ensure some parameters are casted o integer before we pass them along. 
            Moreover, to avoid others taking values outside the (0, 1) range, 
            we also ensure it is capped accordingly.
            """
            val = xgb_cv(
                        n_estimators      = int(round(n_estimators)),
                        max_depth         = int(round(max_depth)),
                        colsample_bytree  = max(min(colsample_bytree, 0.999), 1e-3),
                        learning_rate     = max(min(eta, 0.999), 1e-3),
                        metric = metric,
                        X      = X_train,
                        y      = y_train,
                       )
            return val

        hyp_space = {"n_estimators" : (50, 500),
                     "max_depth"    : (2, 10),
                     "colsample_bytree": (0.1, 0.999),
                     "eta" : (0.001, 0.4)}
                        
        optimizer = BayesianOptimization(
                            f            = xgb_crossval,
                            pbounds      = hyp_space,
                            random_state = self.SEED,
                            verbose      = 0)

        optimizer.maximize(init_points = 20, n_iter = 50)

        return optimizer.max
    
    def run(self, X_train, y_train):
        
        lr  = self.logistic_regression(X_train, y_train)
        lrr = self.regularized_logistic_regression(X_train, y_train)
        
        # Random Forest tuning
        opt_rf = self.optimize_rfc(X_train,
                                   y_train,
                                   metric = self.metric)        
        best_hyp_rf = {
               'max_depth'         : int(round(opt_rf['params']['max_depth'])),
               'max_features'      : opt_rf['params']['max_features'],
               'min_samples_leaf'  : int(round(opt_rf['params']['min_samples_leaf'])),
               'min_samples_split' : int(round(opt_rf['params']['min_samples_split'])),
               'n_estimators'      : int(round(opt_rf['params']['n_estimators'])),
               'random_state'      : self.SEED
              }
        rf = RandomForestClassifier(**best_hyp_rf)
        rf.fit(X_train, y_train)
        # Finish tuning Random Forest
        
        # XGBoost classifier tuning       
        opt_xgb = self.optimize_xgb(X_train,
                                    y_train,
                                    metric = self.metric)
        
        PARAM_SCALE_POS = np.ceil( len(y_train[y_train == 0]) / len(y_train[y_train == 1]) )
        
        best_hyp_xgb = {
               'max_depth'        : int(round(opt_xgb['params']['max_depth'])),
               'colsample_bytree' : opt_xgb['params']['colsample_bytree'],
               'n_estimators'     : int(round(opt_xgb['params']['n_estimators'])),
               'learning_rate'    : max(min(opt_xgb['params']['eta'], 0.999), 1e-3),
               'objective'        : 'binary:logistic',
               'scale_pos_weight' : PARAM_SCALE_POS,
               'random_state'     : self.SEED,
               'verbosity'        : 0
              }
        xgb_cl = xgb.XGBClassifier(**best_hyp_xgb)
        xgb_cl.fit(X_train, y_train)
        # Finish tuning XGBoost classifier
        
        models = [('LR', lr),
                  ('LRR', lrr),
                  ('RF', rf),
                  ('XGB', xgb_cl)
                 ]
        
        return models