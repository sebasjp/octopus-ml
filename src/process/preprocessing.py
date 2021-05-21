import pandas as pd
import numpy as np
import sys

sys.path.append('../src/')
from utils import log

class preprocess_data:
    """
    A class to preprocess the data. It has implemented two methods related with data
    preparation and one method that consolidate all.
    
    ...    
    
    Attributes
    ----------
    data : pd.DataFrame
        Pandas DataFrame to use. This one contains both X features and y variable.
    y_name : str
        Name of variable of interest contained in data.
    features_type : dict[str : list[str]]
        Dictionary that contains two keys: qualitatives and quantitatives. The values
        are the list of features names respectively.
    html : str
        Object where useful information is going to be stored in html code
    logger : logging.RootLogger
        Logger object to do the logging.
    
    
    Methods
    -------
    consistency
        This method check the consistency of the features
    handle_missing_values
        This method handles the missing values based on the method specified
        mean and median are supported
    run
        Run all methods consolidated and return the clean data and features type updated
    """
    
    def __init__(self, 
                 data,
                 y_name,
                 features_type,
                 method_missing_quanti,
                 html,
                 logger
                 ):
        
        self.X             = data.drop(columns = y_name).copy()
        self.features_type = features_type.copy()
        self.y             = data[y_name]
        self.method_missing_quanti = method_missing_quanti
        
        self.html   = html
        self.logger = logger
        
    def consistency(self):
        """
        This function check the consistency of the features in sense
        of qualitative variables with many categories, just one category
        or a high proportion of records in one category. Regarding the 
        quantitative variables, It just check if there is any value with 
        a high proportion of records. These features will be removed.
        """
        features_type = self.features_type
        df = self.X.copy()

        self.html += "<h2><center>Features' Consistency:</center></h2>"
        
        # max categories to keep in features with many categories
        max_cat = 10
        vars_remove = []
        vars_remove_quali  = []
        vars_remove_quanti = []

        self.logger.info('Started to check the features consistency')

        self.html += "<h3>Qualitative features removed:</h3>"

        for x in features_type['qualitative']:

            freq = df[x].value_counts(normalize = True)
            freq_acum = np.cumsum(freq)

            # features with many categories
            if len(freq_acum) > max_cat:
                # can we select the first max_cat - 1 categories
                # the other categories will be recodified in 'other'
                if freq_acum.iloc[max_cat - 1] >= 0.75:
                    keep_cat = freq_acum.iloc[max_cat - 1].index
                    df[x] = np.where(df[x].isin(keep_cat), df[x], 'other')

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
            prop_values = df[x].value_counts(normalize = True)
            if prop_values.iloc[0] >= 0.99:
                vars_remove_quanti.append(x)

                self.html += "<b>" + x + "</b>"
                self.html += " removed because has a high proportion in one number<br>"        

        if len(vars_remove_quanti) == 0:
            self.html += """None quantitative feature was removed"""

        # finally, we remove that features
        vars_remove = vars_remove_quali + vars_remove_quanti
        df = df.drop(columns = vars_remove)

        quali_vars  = features_type['qualitative']
        quanti_vars = features_type['quantitative']

        features_type_new = {}
        features_type_new['qualitative']  = [x for x in quali_vars if x not in vars_remove]
        features_type_new['quantitative'] = [x for x in quanti_vars if x not in vars_remove]

        self.logger.info('Features: ' + str(vars_remove) + ' were removed because its distribution')

        self.X             = df
        self.features_type = features_type_new
        
        self.logger.info('Consistency values finished!')
        
        return None

    # ============================================================================= #
    
    def handle_missing_values(self):
        """
        This function handles the missing values based on the method specified
        for quantitatives features. In qualitative features the will be filled
        with the word 'other'. This apply for features with less than 20% of 
        missing values, otherwise the feature will be removed.
        """
        features_type = self.features_type
        df = self.X.copy()    
    
        self.html += "<h2><center>Handle missing values:</center></h2>"
        
        vars_remove = []
        vars_remove_quanti = []
        vars_remove_quali  = []

        # Computing proportion of nulls
        prop_missing = df.isnull().mean()

        # Possible imputer methods
        imputer_methods = {'median': np.median,
                           'mean'  : np.mean}
        # Imputer method specified
        imputer = imputer_methods[self.method_missing_quanti]

        for x in features_type['quantitative']:

            # if the feature has missing values, but this its proportion
            # is less than 0.2, then the values will be imputed, otherwise
            # the feature will be removed
            if 0 < prop_missing.loc[x] < 0.2:

                val_imputer = imputer(df[x].dropna())
                df[x] = df[x].fillna(val_imputer)
                
                str_ = 'Feature ' + x + ' was imputer with the method ' + self.method_missing_quanti + \
                        ' value = ' + str(val_imputer)
                self.logger.info(str_)
                self.html += str_ + '<br>'

            elif prop_missing.loc[x] >= 0.2:
                vars_remove_quanti.append(x)
            else:
                pass

        for x in features_type['qualitative']:

            if 0 < prop_missing.loc[x] < 0.2:
                val_imputer = 'other'
                df[x] = df[x].fillna(val_imputer)
                
                str_ = 'Feature ' + x + ' was imputer with "' + val_imputer + '"' 
                self.logger.info(str_)
                self.html += str_ + '<br>'
                
            elif prop_missing.loc[x] >= 0.2:
                vars_remove_quali.append(x)
            else:
                pass
            
        vars_remove = vars_remove_quali + vars_remove_quanti
        
        if len(vars_remove) == 0:
            self.html += """None feature was removed"""
            self.logger.info('None feature were removed because the missing values')
        else:
            df = df.drop(columns = vars_remove)

            quali_vars  = features_type['qualitative']
            quanti_vars = features_type['quantitative']

            features_type = {}
            features_type['qualitative']  = [x for x in quali_vars if x not in vars_remove]
            features_type['quantitative'] = [x for x in quanti_vars if x not in vars_remove]

            self.logger.info('Features: ' + str(vars_remove) + ' were removed because the missing values')

        self.X             = df
        self.features_type = features_type
        
        self.logger.info('Handle missing values finished!')
        
        return None
    
    def run(self, 
            check_consistency = True,
            check_missing_values = True
            ):
        """
        This function run two methos:
        1. Consistency: This method check the consistency of the features in sense
        of qualitative variables with many categories, just one category
        or a high proportion of records in one category. Regarding the 
        quantitative variables, It just check if there is any value with 
        a high proportion of records. These features will be removed.
        
        2. This method handles the missing values based on the method specified
        for quantitatives features. In qualitative features the will be filled
        with the word 'other'. This apply for features with less than 20% of 
        missing values, otherwise the feature will be removed.
        
        Parameters
        ----------
        check_consistency : boolean; (default = True)
            It indicates if the consistency method is going to run
        check_missing_values : boolean; (default = True)
            It indicates if the missing values method is going to run
            
        Return
        ------
        X : pd.DataFrame
            Data clean
        y : pd.Series
            Variable of interest
        features_type : dict[str : list[str]]
            Features type updated
        html : str
            html code with useful information
        """
        if check_consistency:
            self.consistency()
        if check_missing_values:
            self.handle_missing_values()
        
        return self.X, self.y, self.features_type, self.html