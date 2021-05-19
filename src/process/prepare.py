import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

class octopus_prepare:
    """
    A class to perform the data preparation. It has implemented three methods,
    two of them for split and transform data and one method that consolidate all.
    
    ...
    
    
    Attributes
    ----------
    seed : float
        Seed to be used in train and test split. Controls the shuffling applied 
        to the data before applying the split.
    method_scale : str
        Method to be used to transform or scale the data. Two methods are implemented
        "standard" for StandardScaler and "robust" for RobustScaler
    
    Methods
    -------
    split_data(X, y, features_type)
        Split the data in train and test sets, but before, it performs one hot encoding 
        for categorical features.
    scale_data(X_train, X_test, features_type)
        Transform the data in order with the method specified. 
        Two methods are supported StandardScaler and RobustScaler.
    run(X, y, features_type)
        Runs the split and scale data functions.
    """
    
    def __init__(self, seed, method_scale):
        
        self.SEED   = seed
        self.method = method_scale       
        
    def split_data(self, X, y, features_type):
        """
        This function splits the data in train and test sets, but before, it performs
        one hot encoding for categorical features
        
        Parameters
        ----------
        X : pd.DataFrame
            Pandas DataFrame to use. This one contains just X features
        y : pd.Series
            Variable of interest
        features_type : dict[str : list[str]]
            Dictionary that contains two keys: qualitatives and quantitatives. The values
            are the list of features names respectively.
            
        
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
        
        X = pd.get_dummies(X, 
                           columns    = features_type['qualitative'],
                           drop_first = True)

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size = 0.25,
                                                            random_state = self.SEED)
        
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test, features_type):
        """
        This function transform the data in order with the method specified.
        Two methods are supported StandardScaler and RobustScaler
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Pandas DataFrame to use in trainig. This one contains just X features
        X_test : pd.DataFrame
            Pandas DataFrame to use in test. This one contains just X features
        features_type : dict[str : list[str]]
            Dictionary that contains two keys: qualitatives and quantitatives. The values
            are the list of features names respectively.
        
        Return
        ------
        X_train : pd.DataFrame
            Training set scaled
        X_test : pd.DataFrame
            Test set scaled
        """
        
        if self.method == 'standard':
            scaler = StandardScaler()
        elif self.method == 'robust':
            scaler = RobustScaler()
        else:
            print('Invalid method scaler')
        
        X_train[features_type['quantitative']] = scaler.fit_transform(X_train[features_type['quantitative']])
        X_test[features_type['quantitative']] = scaler.transform(X_test[features_type['quantitative']])
        
        return X_train, X_test
    
    def run(self, X, y, features_type):
        """
        This function runs the split and scale data functions.
        
        split_data: split the data in train and test sets, but before, it performs
        one hot encoding for categorical features.
        
        scale_data: transform the data in order with the method specified.
        Two methods are supported (standard) StandardScaler and (robust) RobustScaler
        """
        X_train, X_test, y_train, y_test = self.split_data(X = X,
                                                           y = y,
                                                           features_type = features_type
                                                           )
        
        X_train, X_test = self.scale_data(X_train, X_test, features_type)
        
        return X_train, X_test, y_train, y_test