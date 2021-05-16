import os
import logging

def log(path_output, filelogs):
    """
    This function create a log file to record the logs and
    specify the logger to print in console as well
    
    Args
    ----
    path_output (str): generic path where the output will be saved
    dirname (str): folder name where the output will be saved
    filelogs (str): file name where the logs will be recorded
    
    Return
    ------
    logger (logger): logger object configured to use
    """
    # check if the directories and file exist
    log_file = os.path.join(path_output, filelogs)
    
    if not os.path.exists(path_output):
        os.mkdir(path)
    
    #if not os.path.exists(path_dir):
    #    os.mkdir(dir_name)
    
    if not os.path.isfile(log_file):
        open(log_file, "w+").close()
    
    #set the format of the log records and the logging level to INFO
    logging_format = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format = logging_format, 
                        level = logging.INFO)
    logger = logging.getLogger()
    
    # create a file handler for output file
    handler = logging.FileHandler(log_file)
    # set the logging level for log file
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter(logging_format)
    handler.setFormatter(formatter)
    
    # add the handlers to the logger
    logger.addHandler(handler)

    return logger
