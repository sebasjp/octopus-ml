# evaluate
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from evaluate.metrics import compute_metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

class octopus_evaluate:
    
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
    
    def run(self, X_train, y_train, X_test, y_test, models_trained, path_output):
        
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
