import pandas as pd
import numpy as np

# outlier detection
from statsmodels.stats.stattools import medcouple
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

class detect_outliers:
    """
    A class to detect outliers. It has implemented three methods to detect outliers
    and one method that consolidate all.
    
    ...
    
    Attributes
    ----------
    X : pd.DataFrame
        Pandas DataFrame to use. This one contains just X features
    features_type : dict[str : list[str]]
        Dictionary that contains two keys: qualitatives and quantitatives. The values
        are the list of features names respectively.
    method: str
        Method name of outliers detection to use. It can be "adjbox" to use adjusted boxplot
        "lof" to use Local Outlier Factor or "isolation_forest" to use Isolation Forest method.
    logger : logging.RootLogger
        Logger object to do the logging.
        
        
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
    run
        Run the specified method and return an array with booleans 
        that indicates if the point is an outlier
    """
    def __init__(self, X, features_type, method, logger):
        
        self.data          = X.copy()
        self.features_type = features_type.copy()
        self.method        = method
        
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

    def run_adjusted_boxplot(self):
        """
        This function runs adjusted bloxplot for all features
        
        Return
        ------
        outliers : numpy.array(boolean)
            Boolean's array that indicates if a point is outlier
        """
        df = self.data.copy()
        features_type = self.features_type.copy()
        outliers = []

        for var in features_type['quantitative']:

            x_orig = df[var].values
            x = df.loc[~df[var].isnull(), var].values

            li, ls = self.adjusted_boxplot(x)

            outliers_var = (x_orig > ls) | (x_orig < li)
            outliers_var = outliers_var.tolist()

            outliers.append(outliers_var)
        
        outliers = np.any(outliers, axis = 0)

        return outliers
    
    def run_lof(self):
        """
        LOF: Identifying Density-Based Local Outliers
        
        Return
        ------
        outliers : numpy.array(boolean)
            Boolean's array that indicates if a point is outlier
        """
        df = self.data.copy()
        features_type = self.features_type.copy()
        
        df = df[features_type['quantitative']]
        df = df.dropna()

        # normalize data because that can be in different
        # scale and it affects the distance measure
        scaler = StandardScaler()
        X = scaler.fit_transform(df)
        
        # execute lof
        lof = LocalOutlierFactor()
        outliers_rows = lof.fit_predict(X)
        outix = df[outliers_rows == -1].index
        
        outliers = self.data.index.isin(outix)
        
        return outliers
    
    def run_isolation_forest(self):
        """
        Isolation Forest
        
        Return
        ------
        outliers : numpy.array(boolean)
            Boolean's array that indicates if a point is outlier
        """
        df = self.data.copy()
        features_type = self.features_type.copy()

        df = df.dropna()
        df = pd.get_dummies(df,
                            columns = features_type['qualitative'],
                            drop_first = True)
        
        SEED = 42
        modelo_isof = IsolationForest(
                                      max_samples = df.shape[0],
                                      bootstrap = True,
                                      n_jobs        = -1,
                                      random_state  = SEED)

        modelo_isof.fit(X = df)

        # score that determine the strength of the outliers
        score_out = modelo_isof.score_samples(df)

        # identify the outliers from the interquartil range
        q1, q3 = np.quantile(score_out, q = [0.25, 0.75])

        iqr = q3 - q1
        thr = q1 - 1.5 * iqr

        outix = df[score_out < thr].index

        outliers = self.data.index.isin(outix)

        return outliers
    
    def run(self):
        """
        Run the specified method and return an array with booleans 
        that indicates if the point is an outlier
        
        Return
        ------
        outliers : numpy.array(boolean)
            Boolean's array that indicates if a point is outlier
        """
        
        self.logger.info('Detect outliers started')
        
        if self.method == 'adjbox':
            self.logger.info('Adjusted Boxplot method selected')
            outliers = self.run_adjusted_boxplot()
            
        elif self.method == 'lof':
            self.logger.info('Local Outlier Factor method selected')
            outliers = self.run_lof()
            
        elif self.method == 'isolation_forest':
            self.logger.info('Isolation Forest method selected')
            outliers = self.run_isolation_forest()
        
        total_outliers_detected = outliers.sum()
        self.logger.info('Detected ' + str(total_outliers_detected) + ' outliers')
        self.logger.info('Detect outliers finished')
        
        return outliers