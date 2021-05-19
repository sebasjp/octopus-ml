import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bran import BootstrapCI
from scipy.stats import chi2_contingency
import os

class statistical_analysis:
    """
    A class to perform the statistical analysis. It has implemented three methods,
    two of them for features analysis and one method that consolidate all.
    
    ...
    
    Attributes
    ----------
    X : pd.DataFrame
        Pandas DataFrame to use. This one contains just X features
    y : pd.Series
        Variable of interest
    y_name : str
        Name of variable of interest contained in data.
    features_type : dict[str : list[str]]
        Dictionary that contains two keys: qualitatives and quantitatives. The values
        are the list of features names respectively.
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
                 X,
                 y,
                 y_name,
                 features_type,
                 alpha,
                 html,
                 path_output,
                 logger
                 ):
        
        self.X      = X
        self.y      = y
        self.y_name = y_name
        self.features_type = features_type
        self.alpha  = alpha
        self.html   = html
        self.path_output = path_output
    
    def quantitative_analysis(self):
        """
        This function performs the analysis for quantitative features. It builds boxplots
        and computes the bootstrap confidence intervals
        """
        df = pd.concat([self.X, self.y], axis = 1)
        
        # To compute confidence intervals by classes for all features
        classes = self.y.unique()
        nclass  = len(classes)
        alpha_corrected = self.alpha / nclass

        results = []
        
        for var_x in self.features_type['quantitative']:
            for classk in classes:        

                bootstrap = BootstrapCI(alpha = alpha_corrected)

                x = self.X.loc[self.y == classk, var_x]
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
        for var_x in self.features_type['quantitative']:

            # distribution plots
            g = sns.boxplot(x = self.y_name, y = var_x, data = df)
            #g = sns.kdeplot(x = var_x, hue = self.y_name, data = df)
            g.set_title(var_x + ' Distribution by class')

            # save fig
            gfigure = g.get_figure()
            namefig1 = 'dist_' + var_x + '_vs_' + self.y_name + '.png'
            gfigure.savefig(path_images + namefig1)
            plt.clf()
            
            # confidence intervals plots
            res_x = results.loc[results['variable'] == var_x ]
            # lower and upper limits
            dy = np.array([res_x['errli'].tolist(), res_x['errls'].tolist()])

            plt.errorbar(x = res_x['class'].tolist(),
                             y = res_x['m'].values,
                             yerr = dy,
                             fmt = '.k')
            plt.ylabel(var_x)
            plt.title('Confidence intervals at ' + str(int((1 - self.alpha) * 100)) + '%')

            # save fig
            namefig2 = 'ci_' + var_x + '_vs_' + self.y_name + '.png'
            plt.savefig(path_images + namefig2)
            plt.clf()
            
            # to html
            str_1 = """<div style="width:900px; margin:0 auto;"><img src = "images/{}">""".format(namefig1)
            str_2 = """<img src = "images/{}"></div>""".format(namefig2)
            
            self.html += str_1 + str_2
        
        hm = sns.heatmap(self.X[self.features_type['quantitative']].corr(),
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
    
    def qualitative_analysis(self):
        """
        This function performs the analysis for qualitative features. It builds countplots
        and performs the hyphotesis test based on chi-square test
        """
        
        df = pd.concat([self.X, self.y], axis = 1)
        
        # Build and save plots
        path_images = self.path_output + 'images/'
        
        for var_x in self.features_type['qualitative']:

            # test hyphotesis chi-square
            table = pd.crosstab(self.X[var_x], self.y).values
            stat, pvalue, dof, expected = chi2_contingency(table)            
            if pvalue <= self.alpha:
                conclusion = "There is significant difference at "
            else:
                conclusion = "There isn't significant difference at "
            
            conclusion += str(int((1 - self.alpha) * 100)) + '% confidence'
            
            # distribution plots
            g = sns.countplot(x = self.y_name, hue = var_x, data = df)
            title = var_x + ' Distribution by class\n' + conclusion
            g.set_title(title)

            # save fig
            gfigure = g.get_figure()
            namefig1 = 'dist_' + var_x + '_vs_' + self.y_name + '.png'
            gfigure.savefig(path_images + namefig1)
            plt.clf()
            
            str_1 = """<div style="width:600px; margin:0 auto;"><img src = "images/{}"></div>""".format(namefig1)
            self.html += str_1 + "<br>"
        
    def run(self):
        """
        This function run two methos:
        1. quantitative_analysis:
            performs the analysis for quantitative features. It builds boxplots and
            computes the bootstrap confidence intervals
        2. qualitative_analysis:
            performs the analysis for qualitative features. It builds countplots and
            performs the hyphotesis test based on chi-square test
            
        Return
        ------
        html : str
            html code with all plots
        """
        self.quantitative_analysis()
        self.qualitative_analysis()
        
        return self.html