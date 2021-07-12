from datetime import time
from os import path
import timeit
import numpy as np
import pandas as pd
import os
from mpi4py import MPI

from PredictiveModel import PredictiveModel
from utils import  data_split,import_st_model_train_configuration, set_train_path

if __name__ == "__main__":

    config = import_st_model_train_configuration(config_file='st_model_training_setting.ini')
    path = set_train_path(config['models_path_name'])
    

    adjacency_matrix = [[0, 750, 750, 750, 750],
                        [750, 0, 0, 0, 0],
                        [750, 0, 0, 0, 0],
                        [750, 0, 0, 0, 0],
                        [750, 0, 0, 0, 0],]
    
    model = PredictiveModel(input_shape=(config['batch_size'],1,8), 
                            adjacency_matrix=adjacency_matrix, 
                            batch_size=config['batch_size'], 
                            prediction_steps=config['prediction_steps'])
                        
    start_time = timeit.default_timer()

    for index in range(100):
        ### Reading Traffic information ###
        print('----- Reading From traffic features %i -----' % (index)) 
        file_name = 'traffic_features' + str(index) + '.csv'
        traffic_features = pd.read_csv(os.path.join('TrafficFeatures',file_name))

        ### Splitting Data ###
        print(' spliting data...')
        x_train, y_train, x_val,  y_val = data_split(traffic_features.iloc[:,1:], 
                                                     config['train_split'], 
                                                     config['batch_size'], 
                                                     config['prediction_steps'])
        print('input_shape', x_train.shape)
        print(' training model on features %i...' %(index))
        model.train_model(x_train, y_train, x_val,  y_val,epochs=config['training_epochs'])
    
    print(' Total Training Time: ', str(timeit.default_timer() - start_time))
    model.save_model(path)






        
        




