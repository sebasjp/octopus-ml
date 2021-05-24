import pandas as pd
import numpy as np
import sys

# outlier detection
from statsmodels.stats.stattools import medcouple
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

sys.path.append('../src/')
from utils import log

class RemoveFeatures:
    """
    A class to preprocess the data. It has implemented two methods related with data preparation and one method that consolidate all.
    
    ...    
    
    Attributes
    ----------
    features_type : dict[str : list[str]]
        Dictionary that contains two keys: qualitatives and quantitatives. The values are the list of features names respectively.
    html : str
        Object where useful information is going to be stored in html code
    logger : logging.RootLogger
        Logger object to do the logging.
    
    
    Methods
    -------
    consistency
        This method check the consistency of the features
    check_missing_values
        This method handles the missing values based on the method specified
        mean and median are supported
    """
    
    def __init__(self, 
                 html,
                 logger
                 ):
        
        self.html   = html
        self.logger = logger
        
    def consistency(self, X_train, X_test, features_type):
        """
        This function check the consistency of the features in sense of qualitative variables with many categories, just one category or a high proportion of records in one category. Regarding the quantitative variables, It just check if there is any value with a high proportion of records. These features will be removed.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Train set information
        X_test : pd.DataFrame
            Test set information
        features_type : dict[str : list[str]]
            Dictionary that contains two keys: qualitatives and quantitatives. The values are the list of features names respectively.
        
        Return
        ------
        X_train : pd.DataFrame
            Train set information
        X_test : pd.DataFrame
            Test set information
        features_type : dict[str : list[str]]
            Dictionary that contains two keys: qualitatives and quantitatives. The values are the list of features names respectively.
        """

        X_train_c = X_train.copy()
        X_test_c = X_test.copy()

        if not self.html:
            self.html = """<html><head>"""
            #self.html += """<link rel = "stylesheet" href = "style.css"/>"""
            self.html += """</head><body><h1><center>Processing Report</center></h1>"""
        
        if not self.logger:
            self.logger = log('../data/output/', 'logs.txt')
            
        self.html += "<h2><center>Features' Consistency:</center></h2>"
        
        # max categories to keep in features with many categories
        max_cat = 10
        vars_remove = []
        vars_remove_quali  = []
        vars_remove_quanti = []

        self.logger.info('Started to check the features consistency')

        self.html += "<h3>Qualitative features removed:</h3>"

        for x in features_type['qualitative']:

            freq = X_train_c[x].value_counts(normalize = True)
            freq_acum = np.cumsum(freq)

            # features with many categories
            if len(freq_acum) > max_cat:
                # can we select the first max_cat - 1 categories
                # the other categories will be recodified in 'other'
                if freq_acum.iloc[max_cat - 1] >= 0.75:
                    keep_cat = freq_acum.iloc[max_cat - 1].index
                    df[x] = np.where(X_train_c[x].isin(keep_cat), df[x], 'other')

                    self.logger.info('feature: ' + x + 're-categorized')

                else:
                    vars_remove_quali.append(x)

                    freq_acum = pd.DataFrame(freq_acum)
                    freq_acum = freq_acum.iloc[:5, :]
                    freq_acum = freq_acum.reset_index()
                    freq_acum.columns = [x, 'relative_frecuency']
                    
                    self.html += "<b>" + x + "</b>"
                    self.html += " removed because has more than {} categories <br>".format(max_cat)
                    self.html += "Top 5 categories"
                    self.html += freq_acum.to_html(buf = None, 
                                                   justify = 'center', 
                                                   classes = 'mystyle',
                                                   index = False)
                    self.html += "<br>"

            # features with just 1 category
            elif len(freq_acum) == 1:
                vars_remove_quali.append(x)

                freq_acum = pd.DataFrame(freq_acum)
                freq_acum = freq_acum.reset_index()
                freq_acum.columns = [x, 'relative_frecuency']
                
                self.html += "<b>" + x + "</b>"
                self.html += " removed because has 1 category <br>"
                self.html += freq_acum.to_html(buf = None, 
                                               justify = 'center',
                                               index = False)
                self.html += "<br>"
            # features with a high proportion of records in one category
            elif freq_acum.iloc[0] >= 0.99:
                vars_remove_quali.append(x)

                freq_acum = pd.DataFrame(freq_acum)
                freq_acum = freq_acum.reset_index()
                freq_acum.columns = [x, 'relative_frecuency']
                
                self.html += "<b>" + x + "</b>"
                self.html += " removed because has a high proportion in one category <br>"
                self.html += freq_acum.to_html(buf = None, 
                                               justify = 'center',
                                               index = False)
                self.html += "<br>"
            else:
                pass

        if len(vars_remove_quali) == 0:
            self.html += """None qualitative feature was removed"""


        self.html += "<h3>Quantitative features removed:</h3>"

        for x in features_type['quantitative']:

            # features with a high proportion of records in one value
            prop_values = X_train_c[x].value_counts(normalize = True)
            if prop_values.iloc[0] >= 0.99:
                vars_remove_quanti.append(x)

                self.html += "<b>" + x + "</b>"
                self.html += " removed because has a high proportion in one number<br>"        

        if len(vars_remove_quanti) == 0:
            self.html += """None quantitative feature was removed"""

        # finally, we remove that features
        vars_remove = vars_remove_quali + vars_remove_quanti
        X_train_c = X_train_c.drop(columns = vars_remove)
        X_test_c  = X_test_c.drop(columns = vars_remove)

        quali_vars  = features_type['qualitative']
        quanti_vars = features_type['quantitative']

        features_type_new = {}
        features_type_new['qualitative']  = [x for x in quali_vars if x not in vars_remove]
        features_type_new['quantitative'] = [x for x in quanti_vars if x not in vars_remove]

        self.logger.info('Features: ' + str(vars_remove) + ' were removed because its distribution')
        
        self.logger.info('Consistency values finished!')
        
        return X_train_c, X_test_c, features_type_new

    # ============================================================================= #
    def check_high_missing_values(self, X_train, X_test, features_type, min_missing):
        """
        This function check the features regarding the missing values based. In qualitative features the will be filled with the word 'other'. This apply for features with less than 'min_missing' of missing values, otherwise the feature will be removed.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Train set information
        X_test : pd.DataFrame
            Test set information
        features_type : dict[str : list[str]]
            Dictionary that contains two keys: qualitatives and quantitatives. The values are the list of features names respectively.
        min_missing : float
            Min proportion of missing allowed for don't remove a feature.
        
        Return
        ------
        X_train : pd.DataFrame
            Train set information
        X_test : pd.DataFrame
            Test set information
        features_type : dict[str : list[str]]
            Dictionary that contains two keys: qualitatives and quantitatives. The values are the list of features names respectively.
        """
    
        X_train_c = X_train.copy()
        X_test_c = X_test.copy()
        
        if not self.html:
            self.html = """<html><head>"""
            #self.html += """<link rel = "stylesheet" href = "style.css"/>"""
            self.html += """</head><body><h1><center>Processing Report</center></h1>"""
        
        if not self.logger:
            self.logger = log('../data/output/', 'logs.txt')
            
        self.html += "<h2><center>Check missing values:</center></h2>"
        
        vars_remove = []

        # Computing proportion of nulls
        prop_missing = X_train_c.isnull().mean()

        for x in np.hstack(list(features_type.values())):

            # if the feature has missing values, but this its proportion
            # is less than min_missing, then the values will be imputed, otherwise
            # the feature will be removed
            if 0 < prop_missing.loc[x] <= min_missing:
                
                if x in features_type['qualitative']:
                    
                    val_imputer = 'other'
                    X_train_c[x] = X_train_c[x].fillna(val_imputer)
                    X_test_c[x] = X_test_c[x].fillna(val_imputer)

                    str_ = 'Feature ' + x + ' was imputer with "' + val_imputer + '"' 
                    self.logger.info(str_)
                    self.html += str_ + '<br>'                

            elif prop_missing.loc[x] > min_missing:
                vars_remove.append(x)
            else:
                pass
        
        if len(vars_remove) == 0:
            self.html += """None feature was removed"""
            self.logger.info('None feature were removed because the missing values')
        else:
            X_train_c = X_train_c.drop(columns = vars_remove)
            X_test_c = X_test_c.drop(columns = vars_remove)

            quali_vars  = features_type['qualitative']
            quanti_vars = features_type['quantitative']

            features_type = {}
            features_type['qualitative']  = [x for x in quali_vars if x not in vars_remove]
            features_type['quantitative'] = [x for x in quanti_vars if x not in vars_remove]

            self.logger.info('Features: ' + str(vars_remove) + ' were removed because the missing values')

        self.logger.info('Check the missing values finished!')
        
        return X_train_c, X_test_c, features_type, self.html

# ===================================================================
#                        OUTLIER DETECTION
# ===================================================================

class OutlierDetection:
    """
    A class to detect outliers. It has implemented three methods to detect outliers and one method that consolidate all.
    
    ...
    
    Attributes
    ----------
    method: str
        Method name of outliers detection to use. It can be "adjbox" to use adjusted boxplot "lof" to use Local Outlier Factor or "isolation_forest" to use Isolation Forest method.
    logger : logging.RootLogger
        Logger object to do the logging.
    seed : float
        Seed
        
        
    Methods
    -------
    adjusted_boxplot
        To compute the lower and upper boundaries for a single variable.
    run_adjusted_boxplot
        To find the outliers based on Adjusted Boxplot for all features.
    run_lof
        To find the outliers based on Local Outlier Factor.
    run_isolation_forest
        To find the outliers based on Isolation Forest.
    detect_outliers
        Run the specified method and return an array with booleans 
        that indicates if the point is an outlier
    """
    def __init__(self, method, seed, html, logger):
        
        self.method = method
        self.SEED   = seed
        self.html   = html
        self.logger = logger
        
        
    # univariate method
    def adjusted_boxplot(self, x):
        """
        An Adjusted Boxplot for Skewed Distributions
        
        Parameters
        ----------
        x : numpy.array
            Array that contains the values of a feature quantitative
        
        Return
        ------
        li : float
            lower boundary until where a normal point is considered
        ls : float
            upper boundary until where a normal point is considered
        """
        MC = medcouple(x)
        q1 = np.quantile(x, q = 0.25)
        q3 = np.quantile(x, q = 0.75)
        iqr = q3 - q1
        
        # compute medcouple
        if MC >= 0:
            f1 = 1.5 * np.exp(-4 * MC)
            f2 = 1.5 * np.exp(3 * MC)
        else:
            f1 = 1.5 * np.exp(-3 * MC)
            f2 = 1.5 * np.exp(4 * MC)

        li = q1 - f1 * iqr
        ls = q3 + f2 * iqr

        return li, ls

    def run_adjusted_boxplot(self, X_train, X_test, features_type):
        """
        This function runs adjusted bloxplot for all features
        
        Return
        ------
        outliers : numpy.array(boolean)
            Boolean's array that indicates if a point is outlier
        """
        X_train_c = X_train.copy()
        X_test_c  = X_test.copy()
        
        out_train = []
        out_test  = []

        for var in features_type['quantitative']:

            x = X_train_c.loc[~X_train_c[var].isnull(), var].values
            li, ls = self.adjusted_boxplot(x)
                        
            # Outliers in train set
            x_orig_train = X_train_c[var].values
            
            outliers_var_train = (x_orig_train > ls) | (x_orig_train < li)
            outliers_var_train = outliers_var_train.tolist()
            
            # Outliers in test set
            x_orig_test  = X_test_c[var].values
            
            outliers_var_test = (x_orig_test > ls) | (x_orig_test < li)
            outliers_var_test = outliers_var_test.tolist()

            out_train.append(outliers_var_train)
            out_test.append(outliers_var_test)
        
        outliers_train = np.any(out_train, axis = 0)
        outliers_test  = np.any(out_test, axis = 0)

        return outliers_train, outliers_test
    
    def run_lof(self, X_train, X_test, features_type):
        """
        LOF: Identifying Density-Based Local Outliers
        
        Return
        ------
        outliers : numpy.array(boolean)
            Boolean's array that indicates if a point is outlier
        """
        X_train_c = X_train.copy()
        X_test_c  = X_test.copy()
        
        X_train_c = X_train_c[features_type['quantitative']]
        X_train_c = X_train_c.dropna()
        
        X_test_c = X_test_c[features_type['quantitative']]
        X_test_c = X_test_c.dropna()

        # normalize data because that can be in different
        # scale and it affects the distance measure
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_c)
        X_test_scaled  = scaler.transform(X_test_c)
        
        # specify lof
        lof = LocalOutlierFactor(novelty = True)
        
        # fit and predict
        lof.fit(X_train_scaled)
        
        outpredict_train = lof.predict(X_train_scaled)
        outpredict_test  = lof.predict(X_test_scaled)
        
        outix_train = X_train_c[outpredict_train == -1].index
        outix_test  = X_test_c[outpredict_test == -1].index
        
        outliers_train = X_train.index.isin(outix_train)
        outliers_test  = X_test.index.isin(outix_test)
        
        return outliers_train, outliers_test
    
    def run_isolation_forest(self, X_train, X_test, features_type):
        """
        Isolation Forest
        
        Return
        ------
        outliers_train, outliers_test : numpy.array(boolean)
            Boolean's array that indicates if a point is outlier
        """
        X_train_c = X_train.copy()
        X_test_c  = X_test.copy()

        X_train_c = X_train_c[features_type['quantitative']]
        X_train_c = X_train_c.dropna()
        
        X_test_c = X_test_c[features_type['quantitative']]
        X_test_c = X_test_c.dropna()
        #df = pd.get_dummies(df,
        #                    columns = features_type['qualitative'],
        #                    drop_first = True)
        
        modelo_isof = IsolationForest(
                                      max_samples = X_train_c.shape[0],
                                      bootstrap = True,
                                      random_state  = self.SEED)

        modelo_isof.fit(X = X_train_c)

        # score that determine the strength of the outliers
        score_out_train = modelo_isof.score_samples(X_train_c)
        score_out_test  = modelo_isof.score_samples(X_test_c)

        # identify the outliers from the interquartil range
        q1, q3 = np.quantile(score_out_train, q = [0.25, 0.75])

        iqr = q3 - q1
        thr = q1 - 1.5 * iqr
        
        # identify outliers
        outix_train = X_train_c[score_out_train < thr].index
        outix_test  = X_test_c[score_out_test < thr].index

        outliers_train = X_train.index.isin(outix_train)
        outliers_test  = X_test.index.isin(outix_test)

        return outliers_train, outliers_test
    
    def detect_outliers(self, X_train, X_test, y_train, y_test, features_type):
        """
        Run the specified method and return an array with booleans 
        that indicates if the point is an outlier
        
        Return
        ------
        X_train : pd.DataFrame
            Train set without outliers
        X_test : pd.DataFrame
            Test set without outliers
        y_train : pd.DataFrame
            Train target without outliers
        y_test : pd.DataFrame
            Test target without outliers
        """
        X_train_c = X_train.copy()
        X_test_c  = X_test.copy()
        
        y_train_c = y_train.copy()
        y_test_c  = y_test.copy()
        
        if not self.html:
            self.html = """<html><head>"""
            self.html += """</head><body><h1><center>Processing Report</center></h1>"""
            
        if not self.logger:
            self.logger = log('../data/output/', 'logs.txt')
            
        self.logger.info('Detect outliers started')
        
        if self.method == 'adjbox':
            self.logger.info('Adjusted Boxplot method selected')
            outliers_train, outliers_test = self.run_adjusted_boxplot(X_train_c, X_test_c, features_type)
            
        elif self.method == 'lof':
            self.logger.info('Local Outlier Factor method selected')
            outliers_train, outliers_test = self.run_lof(X_train_c, X_test_c, features_type)
            
        elif self.method == 'isolation_forest':
            self.logger.info('Isolation Forest method selected')
            outliers_train, outliers_test = self.run_isolation_forest(X_train_c, X_test_c, features_type)
        
        total_outliers_detected = outliers_train.sum() +  outliers_test.sum()
        self.logger.info('Detected ' + str(total_outliers_detected) + ' outliers')
        self.logger.info('Detect outliers finished')
        
        # HTML report about outliers
        if self.method == 'adjbox':
            name = 'Adjusted Boxplot for skewed distribution'
        elif self.method == 'lof':
            name = 'Local Outlier Factor (LOF)'
        elif self.method == 'isolation_forest':
            name = 'Isolarion Forest'
            
        str_ = name + " method used<br>Total outliers found: " + str(total_outliers_detected)
        self.html += "<h2><center>Outlier detection:</center></h2>"
        self.html += str_

        # Finally, Remove outliers
        X_train_c = X_train_c[~outliers_train]
        y_train_c = y_train_c[~outliers_train]
        
        X_test_c = X_test_c[~outliers_test]
        y_test_c = y_test_c[~outliers_test]
        
        return X_train_c, X_test_c, y_train_c, y_test_c, self.html