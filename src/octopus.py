import sys
sys.path.append('../src/')

from process.process import octopus_process
from process.prepare import octopus_prepare
from modeling.train  import octopus_train
from evaluate.evaluate import octopus_evaluate

class octopus_ml:
    
    def __init__(self, 
                 method_missing_quanti,
                 outliers_method,
                 alpha_sta,
                 method_scale,
                 metric_train,
                 njobs,
                 seed
                ):
        
        self.method_missing_quanti = method_missing_quanti
        self.outliers_method = outliers_method
        self.alpha_sta    = alpha_sta
        self.method_scale = method_scale
        self.metric_train = metric_train
        self.njobs        = njobs
        self.SEED = seed
        
    def run(self, data, y_name, features_type, path_output):
        """
        This function runs all process
        """
        # process data, data cleaning
        octo_process = octopus_process(method_missing_quanti = self.method_missing_quanti,
                                       outliers_method = self.outliers_method,
                                       alpha_sta       = self.alpha_sta)

        X, y, features_type = octo_process.run(
                                            data          = data,
                                            y_name        = y_name,
                                            features_type = features_type,
                                            path_output   = path_output)
        
        # data preparation for model
        octo_prepare = octopus_prepare(seed = self.SEED,
                                       method_scale = self.method_scale)

        X_train, X_test, y_train, y_test = octo_prepare.run(X = X,
                                                            y = y,
                                                            features_type = features_type)
        
        # modeling
        octo_train = octopus_train(seed = self.SEED,
                                   metric = self.metric_train,
                                   njobs = self.njobs)

        models_trained = octo_train.run(X_train, y_train)
        
        # evaluate
        octo_eval = octopus_evaluate(metric = self.metric_train,
                                     seed = self.SEED)

        best_model, metrics_df = octo_eval.run(X_train, 
                                               y_train,
                                               X_test,
                                               y_test,
                                               models_trained,
                                               path_output)
        
        results = {}
        results['data_train'] = (X_train, y_train)
        results['data_test'] = (X_test, y_test)
        results['models_trained'] = models_trained
        results['best_model'] = best_model
        results['metrics'] = metrics_df
        
        return results