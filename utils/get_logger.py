import logging
import os
import time
def create_logger(log_dir, model_name, phase="train"):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(model_name, time_str, phase)
    logging.basicConfig(filename=os.path.join(log_dir, log_file+'.txt'), format=' %(message)s')
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    # file_handler = logging.FileHandler(os.path.join(log_dir, log_file+'.txt'))
    # file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console_handler)
    return logger