import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bran import BootstrapCI
from scipy.stats import chi2_contingency
import os
from utils import log

class StatsAnalysis:
    """
    A class to perform the statistical analysis. It has implemented three methods,
    two of them for features analysis and one method that consolidate all.
    
    ...
    
    Attributes
    ----------
    alpha : float
        Significance value to evaluate the hyphotesis
    html : str
        Object where useful information is going to be stored in html code
    path_output: str
        Path where the images are going to be stored
    logger : logging.RootLogger
        Logger object to do the logging.
        
        
    Methods
    -------
    quantitative analysis
        To perform boxplot per feature and class compare them using Bootstrap
        Confidence Intervals
    qualitative analysis
        To perform countplot per feature and class compare them using Chi-square test
    run
        Run all methods consolidated and return a HTML code with the plots
    """
    def __init__(self,
                 alpha,
                 html,
                 path_output,
                 logger
                 ):
        
        self.alpha  = alpha
        self.html   = html
        self.logger =  logger
        self.path_output = path_output
    
    def quantitative_analysis(self, X_train, y_train, y_name, features_type):
        """
        This function performs the analysis for quantitative features. It builds boxplots and computes the bootstrap confidence intervals
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Pandas DataFrame to use. This one contains just X features
        y_train : pd.Series
            Variable of interest
        y_name : str
            Name of variable of interest contained in data.
        features_type : dict[str : list[str]]
            Dictionary that contains two keys: qualitatives and quantitatives. The values are the list of features names respectively.
            
        Return
        ------
        None
        """
        df = pd.concat([X_train, y_train], axis = 1)
        
        # To compute confidence intervals by classes for all features
        classes = y_train.unique()
        nclass  = len(classes)
        alpha_corrected = self.alpha / nclass

        results = []
        
        for var_x in features_type['quantitative']:
            for classk in classes:        

                bootstrap = BootstrapCI(alpha = alpha_corrected)

                x = X_train.loc[y_train == classk, var_x]
                x = x[~np.isnan(x)]
                li, ls = bootstrap.calculate_ci(x)
                m = x.mean()

                result_k =  (var_x, 'class ' + str(classk), m, m - li, ls - m)

                results.append(result_k)

        colnames = ['variable', 'class', 'm', 'errli', 'errls']
        results  = pd.DataFrame(results, columns = colnames)
        
        # To save images
        path_images = self.path_output + 'images/'
        if not os.path.exists(path_images):
            os.mkdir(path_images)
        
        # Build and save plots
        for var_x in features_type['quantitative']:

            # distribution plots
            g = sns.boxplot(x = y_name, y = var_x, data = df)
            #g = sns.kdeplot(x = var_x, hue = self.y_name, data = df)
            g.set_title(var_x + ' Distribution by class')

            # save fig
            gfigure = g.get_figure()
            namefig1 = 'dist_' + var_x + '_vs_' + y_name + '.png'
            gfigure.savefig(path_images + namefig1)
            plt.clf()
            
            # confidence intervals plots
            res_x = results.loc[results['variable'] == var_x]
            # lower and upper limits
            dy = np.array([res_x['errli'].tolist(), res_x['errls'].tolist()])

            plt.errorbar(x = res_x['class'].tolist(),
                             y = res_x['m'].values,
                             yerr = dy,
                             fmt = '.k')
            plt.ylabel(var_x)
            plt.title('Confidence intervals at ' + str(int((1 - self.alpha) * 100)) + '%')

            # save fig
            namefig2 = 'ci_' + var_x + '_vs_' + y_name + '.png'
            plt.savefig(path_images + namefig2)
            plt.clf()
            
            # to html
            str_1 = """<div style="width:900px; margin:0 auto;"><img src = "images/{}">""".format(namefig1)
            str_2 = """<img src = "images/{}"></div>""".format(namefig2)
            
            self.html += str_1 + str_2
        
        hm = sns.heatmap(X_train[features_type['quantitative']].corr(),
                         vmin = -1,
                         vmax = 1,
                         annot = True)

        hm.set_title('Correlation Heatmap', 
                     fontdict = {'fontsize': 12},
                     pad = 12)
        
        # save fig
        gfigure = hm.get_figure()
        namefig = 'correlation_heatmap.png'
        gfigure.savefig(path_images + namefig)
        plt.clf()
        
        str_1 = """<div style="width:600px; margin:0 auto;"><img src = "images/{}"></div>""".format(namefig)
        self.html += str_1 + "<br>"
        
        return None
    
    def qualitative_analysis(self, X_train, y_train, y_name, features_type):
        """
        This function performs the analysis for qualitative features. It builds countplots and performs the hyphotesis test based on chi-square test
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Pandas DataFrame to use. This one contains just X features
        y_train : pd.Series
            Variable of interest
        y_name : str
            Name of variable of interest contained in data.
        features_type : dict[str : list[str]]
            Dictionary that contains two keys: qualitatives and quantitatives. The values are the list of features names respectively.
            
        Return
        ------
        None
        """
        
        df = pd.concat([X_train, y_train], axis = 1)
        
        # Build and save plots
        path_images = self.path_output + 'images/'
        
        for var_x in features_type['qualitative']:

            # test hyphotesis chi-square
            table = pd.crosstab(X_train[var_x], y_train).values
            stat, pvalue, dof, expected = chi2_contingency(table)            
            if pvalue <= self.alpha:
                conclusion = "There is significant difference at "
            else:
                conclusion = "There isn't significant difference at "
            
            conclusion += str(int((1 - self.alpha) * 100)) + '% confidence'
            
            # distribution plots
            g = sns.countplot(x = y_name, hue = var_x, data = df)
            title = var_x + ' Distribution by class\n' + conclusion
            g.set_title(title)

            # save fig
            gfigure = g.get_figure()
            namefig1 = 'dist_' + var_x + '_vs_' + y_name + '.png'
            gfigure.savefig(path_images + namefig1)
            plt.clf()
            
            str_1 = """<div style="width:600px; margin:0 auto;"><img src = "images/{}"></div>""".format(namefig1)
            self.html += str_1 + "<br>"
        
        return None
        
    def stats_analysis(self, X_train, y_train, y_name, features_type):
        """
        This function run two methos:
        1. quantitative_analysis:
            performs the analysis for quantitative features. It builds boxplots and computes the bootstrap confidence intervals
        2. qualitative_analysis:
            performs the analysis for qualitative features. It builds countplots and performs the hyphotesis test based on chi-square test
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Pandas DataFrame to use. This one contains just X features
        y_train : pd.Series
            Variable of interest
        y_name : str
            Name of variable of interest contained in data.
        features_type : dict[str : list[str]]
            Dictionary that contains two keys: qualitatives and quantitatives. The values are the list of features names respectively.
            
        Return
        ------
        html : str
            html code with all plots
        """
        
        if not self.html:
            self.html = """<html><head>"""
            self.html += """</head><body><h1><center>Processing Report</center></h1>"""
        
        if not self.logger:
            self.logger = log(self.path_output, 'logs.txt')
        
        self.html += "<h2><center>Statistical Analysis:</center></h2>"
        
        self.quantitative_analysis(X_train, y_train, y_name, features_type)
        self.qualitative_analysis(X_train, y_train, y_name, features_type)
        
        return self.html