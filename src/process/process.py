import os
from utils import log
from process.preprocessing import preprocess_data
from process.outlier_detection import detect_outliers
from process.statistical_analysis import statistical_analysis

class octopus_process:
    """
    A class to process the data. It has implemented two methods: run and renderize_html. The first one preprocess data, detect outliers and does the statistical analysis. The second one renderize the html file with information given by the method before.
    
    ...    
    
    Attributes
    ----------
    method_missing_quanti: str
        Method to deal with the missing values in the quantitative features. Imputation with "mean" and "median" are supported.
    outliers_method: str
        Method name of outliers detection to use. It can be "adjbox" to use adjusted boxplot; "lof" to use Local Outlier Factor or "isolation_forest" to use Isolation Forest method.
    alpha : float
        Significance value to evaluate the hyphotesis in the statistical analysis.
    
    
    Methods
    -------
    run
       Executes the preprocessing of data, detect outliers and does the statistical analysis. 
    """
    
    def __init__(self,
                 method_missing_quanti,
                 outliers_method,
                 alpha_sta,
                ):
        self.method_missing_quanti = method_missing_quanti
        self.outliers_method = outliers_method
        self.alpha           = alpha_sta
        
        self.html = """<html><head>"""
        self.html += """<link rel = "stylesheet" href = "style.css"/>"""
        self.html += """</head><body><h1><center>Processing Report</center></h1>"""
        self.path_html = None
    
    def renderize_html(self):
        
        self.html += "<br></body></html>"

        with open(self.path_html, 'w') as out:
            out.write(self.html)
                
    def run(self, 
            data,
            y_name,
            features_type,
            path_output):
        """
        This function executes the preprocessing of data, detect outliers and does the statistical analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Pandas DataFrame to use. This one contains both X features and y variable.
        y_name : str
            Name of variable of interest contained in data.
        features_type : dict[str : list[str]]
            Dictionary that contains two keys: qualitatives and quantitatives. The values
            are the list of features names respectively.
        path_output: str
            Path where the logs and report.html will be saved.
        
        Return
        ------
        X : pd.DataFrame
            Data clean
        y : pd.Series
            Variable of interest
        features_type : dict[str : list[str]]
            Features type updated         
        """
        self.path_html = os.path.join(path_output, 'report.html')
        logger = log(path_output, 'logs.txt')
        
        # Preprocess data        
        preprocess = preprocess_data(data           = data,
                                     y_name         = y_name,
                                     features_type  = features_type,
                                     method_missing_quanti = self.method_missing_quanti,
                                     html           = self.html,
                                     logger         = logger)

        X, y, features_type, html = preprocess.run()
        self.html = html
        
        # =================
        # Outlier detection
        detect_out = detect_outliers(X             = X,
                                     features_type = features_type,
                                     method        = self.outliers_method,
                                     logger        = logger)
        
        outliers = detect_out.run() 
        X = X[~outliers]
        y = y[~outliers]
        
        # HTML report about outliers
        if self.outliers_method == 'adjbox':
            name = 'Adjusted Boxplot for skewed distribution'
        elif self.outliers_method == 'lof':
            name = 'Local Outlier Factor (LOF)'
        elif self.outliers_method == 'isolation_forest':
            name = 'Isolarion Forest'
            
        str_ = name + " method used<br>Total outliers found: " + str(outliers.sum())
        self.html += "<h2><center>Outlier detection:</center></h2>"
        self.html += str_
        
        # =================
        # statistical analysis
        self.html += "<h2><center>Statistical Analysis:</center></h2>"
        
        sta = statistical_analysis(
                           X      = X, 
                           y      = y,
                           y_name = y_name,
                           features_type = features_type,
                           alpha  = self.alpha,
                           html   = self.html,
                           path_output = path_output,
                           logger = logger
                          )

        self.html = sta.run()
        # =================
        
        # Make the HTML file
        self.renderize_html()
        
        return X, y, features_type