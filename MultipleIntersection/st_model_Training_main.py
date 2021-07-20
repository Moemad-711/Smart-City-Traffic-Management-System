from datetime import time
from os import path
from shutil import copyfile
import timeit
from traceback import print_tb
import numpy as np
import pandas as pd
import os

from PredictiveModel import PredictiveModel
from utils import  data_split,import_st_model_train_configuration, set_train_path

if __name__ == "__main__":

    config = import_st_model_train_configuration(config_file='st_model_training_setting.ini')
    path = set_train_path(config['models_path_name'])
    folder_num = config['traffic_feature_folder_num']
                    ###  UW   LN   TL1  RN   UE   TL2  LE   RS   TL4  LS   LW   TL3 ###
    adjacency_matrix = [[0,   0,   800, 0,   0,   0,   0,   0,   0,   0,   0,   0],   # UW
                        [0,   0,   800, 0,   0,   0,   0,   0,   0,   0,   0,   0],   # LN 
                        [800, 800, 0,   0,   800, 800, 0,   0,   0,   0,   0,   800], # TL1 
                        [0,   0,   0,   0,   0,   800, 0,   0,   0,   0,   0,   0],   # RN
                        [0,   0,   0,   0,   0,   800, 0,   0,   0,   0,   0,   0],   # UE 
                        [0,   0,   800, 800, 800, 0,   0,   0,   800, 0,   0,   0],   # TL2 
                        [0,   0,   0,   0,   0,   0,   0,   0,   800, 0,   0,   0],   # LE
                        [0,   0,   0,   0,   0,   0,   0,   0,   800, 0,   0,   0],   # RS
                        [0,   0,   0,   0,   0,   800, 800, 800, 0,   0,   0,   800], # TL4
                        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   800], # LS
                        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   800], # LW
                        [0,   0,   800, 0,   0,   0,   0,   0,   800, 800, 800, 0]]   # TL3
    
    model = PredictiveModel(input_shape=(config['batch_size'],4,8), 
                            adjacency_matrix=adjacency_matrix,  
                            batch_size=config['batch_size'], 
                            prediction_steps=config['prediction_steps'])
                        
    start_time = timeit.default_timer()
    total_episodes = 99

    copyfile(src='st_model_training_setting.ini', dst=os.path.join(path, 'st_model_training_setting.ini'))

    for index in range(total_episodes):
        ### Reading Traffic information ###
        print('----- Reading From traffic features %i -----' % (index)) 
        file_name = 'traffic_features' + str(index) + '.csv'
        traffic_features = pd.read_csv(os.path.join('TrafficFeatures','traffic_features_%i' %(folder_num),file_name))

        ### Splitting Data ###
        print(' spliting data...')
        x_train, y_train, x_val,  y_val = data_split(traffic_features.iloc[:,:], 
                                                     config['train_split'], 
                                                     config['batch_size'], 
                                                     config['prediction_steps'])
        print('input_shape', x_train.shape)
        print('val_shape', x_val.shape)
        print(' training model on features %i...' %(index))
        model.train_model(x_train, y_train, x_val,  y_val,epochs=config['training_epochs'])

    
    print(' Total Training Time: ', str(timeit.default_timer() - start_time))
    model.save_model(path)
    print('Data Saved at: ', path)






        
        




