import numpy as np
import os
import sys
sys.path.append('../src/')
from datetime import datetime
from utils import log

# Octopus process
from utils import split_data
from process.preprocessing import RemoveFeatures
from process.preprocessing import OutlierDetection
from process.analysis import StatsAnalysis

# Octopus Prepare
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Modeling
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from modeling.models_cv import rfc_cv
from modeling.models_cv import xgb_cv

# Evaluate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import compute_metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from utils import renderize_html

class OctopusProcess:
    """
    A class to process the data. It has implemented process_data method. This one remove features, detect outliers and does the statistical analysis.
    
    ...    
    
    Attributes
    ----------
    test_size : float
        Test size
    min_missing_values : float
        Min proportion of missing values allowed
    outliers_method: str
        Method name of outliers detection to use. It can be "adjbox" to use adjusted boxplot; "lof" to use Local Outlier Factor or "isolation_forest" to use Isolation Forest method.
    alpha : float
        Significance value to evaluate the hyphotesis in the statistical analysis.
    seed : float
        seed to use
    
    
    Methods
    -------
    process_data
       Executes the remove features, detect outliers and does the statistical analysis. 
    """
    
    def __init__(self,
                 test_size,
                 min_missing_values,
                 outliers_method,
                 alpha_sta,
                 seed
                ):
        
        self.test_size       = test_size
        self.min_missing     = min_missing_values
        self.outliers_method = outliers_method
        self.alpha           = alpha_sta
        self.SEED            = seed
        
        self.html = """<html><head>"""
        self.html += """<link rel = "stylesheet" href = "style.css"/>"""
        self.html += """</head><body><h1><center>Processing Report</center></h1>"""
        self.path_html = None
        
                
    def process_data(self, data, y_name, features_type, path_output):
        """
        This function executes the preprocessing of data, detect outliers and does the statistical analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Pandas DataFrame to use. This one contains both X features and y variable.
        y_name : str
            Name of variable of interest contained in data.
        features_type : dict[str : list[str]]
            Dictionary that contains two keys: qualitatives and quantitatives. The values are the list of features names respectively.
        path_output: str
            Path where the logs and report.html will be saved.
        
        Return
        ------
        (X_train, y_train) : (pd.DataFrame, pd.Series)
            Data train
        (X_test, y_test) : (pd.DataFrame, pd.Series)
            Data test
        features_type : dict[str : list[str]]
            Features type updated
        html : str
        """
        
        logger = log(path_output, 'logs.txt')
        
        # ====================
        # Train and test split
        X_train, X_test, y_train, y_test = split_data(data,
                                                      y_name,
                                                      self.test_size,
                                                      self.SEED)
        # ====================
        # Remove features
        rm_features = RemoveFeatures(html = self.html, logger = logger)

        X_train, X_test, features_type = rm_features.consistency(X_train,
                                                                 X_test,
                                                                 features_type)

        X_train, X_test, features_type, html = rm_features.check_high_missing_values(
                                                                               X_train, 
                                                                               X_test, 
                                                                               features_type,
                                                                               min_missing = self.min_missing)
        self.html = html
        # =====================
        # Outlier detection
        outlier = OutlierDetection(method = self.outliers_method,
                                   html = self.html,
                                   logger = logger,
                                   seed = self.SEED)

        X_train, X_test, y_train, y_test, html  = outlier.detect_outliers(
                                                                    X_train,
                                                                    X_test,
                                                                    y_train, 
                                                                    y_test,
                                                                    features_type)
        self.html = html
        # ======================
        # statistical analysis
        sta = StatsAnalysis(alpha = self.alpha, 
                            html  = self.html,
                            path_output = path_output,
                            logger = logger)

        self.html = sta.stats_analysis(X_train, y_train, y_name, features_type)
               
        return (X_train, y_train), (X_test, y_test), features_type, html

# ==================================================================================
# PREPARE DATA FOR MODEL PROCESS (IMPUTER, SCALER, ENCODING)
# ==================================================================================

class OctopusPrepare:
    
    def __init__(self, strategy_missing, method_scale):
        
        self.strategy_missing = strategy_missing
        self.method_scale = method_scale
        
    
    def define_imputer(self):
        """
        This function defines the imputer to use, based on the strategy specified. So far, SimpleImputer is available.
        
        Return
        ------
        imputer : sklearn.impute._base.SimpleImputer
        """
        imputer = SimpleImputer(strategy = self.strategy_missing)
        
        return imputer
    
    def define_scaler(self):
        """
        This function defines the scaler to use, based on the method scale specified. So far, StandardScaler and RobustScaler are available.
        
        Return
        ------
        scaler : sklearn.preprocessing._data
        """
        # prepare numeric features
        if self.method_scale == 'standard':
            scaler = StandardScaler()
        elif self.method_scale == 'robust':
            scaler = RobustScaler()
        else:
            print('Invalid method scaler')
            
        return scaler
    
    def prepare_pipeline(self, features_type):
        """
        This function builds the Pipeline that will be used in order to prepare the data for the model process.
        
        Parameters
        ----------
        features_type : dict[str : list[str]]
            Dictionary that contains two keys: qualitatives and quantitatives. The values are the list of features names respectively.
            
        Return
        ------
        preparessor : ColumnTransformer Object
        """
        imputer = self.define_imputer()
        scaler  = self.define_scaler()

        numeric_features = features_type['quantitative']
        numeric_transformer = Pipeline(steps = [('imputer', imputer),
                                                ('scaler', scaler)])

        # prepare categorical features
        categorical_features = features_type['qualitative']
        categorical_transformer = OneHotEncoder(handle_unknown = 'ignore')


        preparessor = ColumnTransformer(transformers = [
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)])
        
        return preparessor
    
# ==================================================================================
# TRAIN DIFERENT MODELS
# ==================================================================================

class OctopusTrain:
    
    def __init__(self, metric, seed, njobs):
        
        self.metric = metric
        self.SEED = seed
        self.njobs = njobs
    
    def logistic_regression(self, X_train, y_train, preparessor):
        """
        Logistic Regression without regularization
        """
        # Pipeline that prepare data for modeling process
        preprocessor = preparessor
        
        lr = LogisticRegression(C = 1e6,
                                max_iter = 500,
                                random_state = self.SEED)
        
        # Append classifier to preparing pipeline. Now we have a full prediction pipeline.
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', lr)])

        clf.fit(X_train, y_train)
        
        return clf
    
    def regularized_logistic_regression(self, X_train, y_train, preparessor):
        """
        Logistic Regression with regularization
        """
        
        # Pipeline that prepare data for modeling process
        preprocessor = preparessor

        lrr = LogisticRegression(max_iter = 500,
                                 random_state = self.SEED)        

        # Append classifier to preparing pipeline. Now we have a full prediction pipeline.
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', lrr)])
        
        grid = {'classifier__C': [1000, 100, 10, 1, 0.1, 0.08, 0.02, 0.001, 0.0001, 0.015, 0.0001]}

        grid_search = GridSearchCV(estimator = clf, 
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
                     metric,
                     preparessor
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
                        preparessor = preparessor
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
                     metric,
                     preparessor
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
                        preparessor = preparessor
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
    
    def train(self, X_train, y_train, preparessor):
        
        
        print('Logistic Regression is going to be fitted')
        lr  = self.logistic_regression(X_train, y_train, preparessor)
        print('Logistic Regression fitted')
        
        print('Regularized Logistic Regression is going to be fitted')
        lrr = self.regularized_logistic_regression(X_train, y_train, preparessor)
        print('Regularized Logistic Regression fitted')
        # ================================================================
        # Random Forest tuning
        
        print('Random Forest is going to be fitted')
        
        opt_rf = self.optimize_rfc(X_train,
                                   y_train,
                                   metric = self.metric,
                                   preparessor = preparessor)
        best_hyp_rf = {
               'max_depth'         : int(round(opt_rf['params']['max_depth'])),
               'max_features'      : opt_rf['params']['max_features'],
               'min_samples_leaf'  : int(round(opt_rf['params']['min_samples_leaf'])),
               'min_samples_split' : int(round(opt_rf['params']['min_samples_split'])),
               'n_estimators'      : int(round(opt_rf['params']['n_estimators'])),
               'random_state'      : self.SEED
              }
        
        rf = RandomForestClassifier(**best_hyp_rf)
        
        # Pipeline that prepare data for modeling process
        preprocessor_rf = preparessor
        
        # Append classifier to preparing pipeline. Now we have a full prediction pipeline.
        clf_rf = Pipeline(steps=[('preprocessor', preprocessor_rf),
                                 ('classifier', rf)])
        
        clf_rf.fit(X_train, y_train)
        
        print('Random Forest fitted')
        # Finish tuning Random Forest
        # ================================================================
        
        # XGBoost classifier tuning
        print('XGBoost is going to be fitted')
        
        opt_xgb = self.optimize_xgb(X_train,
                                    y_train,
                                    metric = self.metric,
                                    preparessor = preparessor)
        
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
        
        # Pipeline that prepare data for modeling process
        preprocessor_xgb = preparessor
        
        # Append classifier to preparing pipeline. Now we have a full prediction pipeline.
        clf_xgb = Pipeline(steps=[('preprocessor', preprocessor_xgb),
                                  ('classifier', xgb_cl)])
        
        clf_xgb.fit(X_train, y_train)
        
        print('XGBoost fitted')
        # Finish tuning XGBoost classifier
        # ================================================================
        
        models = [('LR', lr),
                  ('LRR', lrr),
                  ('RF', clf_rf),
                  ('XGB', clf_xgb)
                 ]
        
        return models
    
# ==================================================================================
# EVALUATE MODELS
# ==================================================================================

class OctopusEvaluate:
    
    def __init__(self, metric, seed):
        
        self.metric = metric
        self.SEED   = seed
    
    def compare_models(self, 
                       X_train,
                       y_train,
                       models_trained,
                       path_images
                      ):
        
        # evaluate each model in turn
        results = []
        results_avg = []
        results_std = []
        names = []

        for name, model in models_trained:
            
            kfold = KFold(n_splits = 10, 
                          shuffle = True,
                          random_state = self.SEED)
            
            cv_results = cross_val_score(model, 
                                         X_train, 
                                         y_train, 
                                         cv = kfold, 
                                         scoring = self.metric)
            results.append(cv_results)
            names.append(name)
            
            results_avg.append(cv_results.mean())
            results_std.append(cv_results.std())
            
            msg = "%s model: %s = %f (%f)" % (name, self.metric, cv_results.mean(), cv_results.std())
            print(msg)

        # Best model
        ixmax = np.array(results_avg).argmax()
        best_name = names[ixmax]
        
        # boxplot algorithm comparison
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        plt.title('Using the ' + self.metric + ' metric, the best model is ' + best_name)
        plt.ylabel(self.metric.capitalize())
        ax.set_xticklabels(names)
            
        # save fig
        namefig = 'model_comparison.png'
        plt.savefig(path_images + namefig)
        plt.clf()
                
        return models_trained[ixmax]
    
    def compute_test_metrics(self, X_test, y_test, models_trained):
        
        metrics = {}
        for name, model in models_trained:
            
            metrics[name] = compute_metrics(X_test, y_test, model)
        
        return metrics
    
    def evaluate(self, X_train, y_train, X_test, y_test, models_trained, path_output):
        
        # To save images
        path_images = path_output + 'images/'
        
        if not os.path.exists(path_images):
            os.mkdir(path_images)
            
        # model comparison
        best_model = self.compare_models(X_train, 
                                         y_train,
                                         models_trained,
                                         path_images)        
        # evaluate with test dataset
        test_metrics = self.compute_test_metrics(X_test, 
                                                 y_test,
                                                 models_trained)        
        test_metrics = pd.DataFrame(test_metrics)
        test_metrics = test_metrics.reset_index()
        test_metrics = test_metrics.melt(id_vars = 'index')
        test_metrics = test_metrics.rename(columns = {'index': 'metric'})
        
        # plot metrics test dataset
        g = sns.FacetGrid(test_metrics,
                  col = "metric",
                  height = 4,
                  aspect = .5)
        g.map(sns.barplot, 
              "variable",
              "value")
        g.set(xlabel = '')
        # save figure
        namefig = 'test_metrics.png'
        g.savefig(path_images + namefig)
        plt.clf()
        
        return best_model, test_metrics


class OctopusML:
    
    def __init__(self, 
                 test_size,
                 min_missing,
                 outliers_method,
                 alpha_sta,
                 strategy_missing,
                 method_scale,
                 metric_train,
                 njobs,
                 seed
                ):
        
        self.test_size       = test_size
        self.min_missing     = min_missing
        self.outliers_method = outliers_method
        self.alpha_sta       = alpha_sta
        
        self.strategy_missing = strategy_missing
        self.method_scale = method_scale
        
        self.metric_train = metric_train
        self.njobs        = njobs
        self.SEED         = seed
        
    def autoML(self, data, y_name, features_type, path_output):
        """
        This function runs all process
        """
        # =======================================================================
        # process, cleaning data
        # =======================================================================
        octoProcess = OctopusProcess(self.test_size, 
                                     self.min_missing,
                                     self.outliers_method,
                                     self.alpha_sta,
                                     self.SEED)

        train, test, features_type, html = octoProcess.process_data(data, 
                                                                    y_name,
                                                                    features_type,
                                                                    path_output)
        X_train, y_train = train
        X_test, y_test = test
        
        # =======================================================================
        # data preparation for model
        # =======================================================================
        octoPrepare = OctopusPrepare(self.strategy_missing, 
                                     self.method_scale)
        
        preparessor = octoPrepare.prepare_pipeline(features_type)
        
        # =======================================================================
        # Modeling proces
        # =======================================================================
        start = datetime.now()

        octoTrain = OctopusTrain(metric = self.metric_train, 
                                 seed   = self.SEED,
                                 njobs  = self.njobs)
        
        models_trained = octoTrain.train(X_train, y_train, preparessor)

        finish = datetime.now()
        print('Time execution training models:', finish - start)
        
        # =======================================================================
        # Evaluate proces
        # =======================================================================
        OctoEval = OctopusEvaluate(metric = self.metric_train,
                                   seed = self.SEED)

        best_model, metrics_df = OctoEval.evaluate(X_train, 
                                                   y_train,
                                                   X_test,
                                                   y_test,
                                                   models_trained,
                                                   path_output)        
        results = {}
        results['train'] = (X_train, y_train)
        results['test'] = (X_test, y_test)
        results['models_trained'] = models_trained
        results['best_model'] = best_model
        results['metrics'] = metrics_df

        # renderize HTML
        path_html = os.path.join(path_output, 'report.html')
        renderize_html(html, path_html)
        
        return results