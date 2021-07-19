import configparser
import datetime
import timeit
import numpy as np

import pandas as pd
from tensorflow.python.keras.backend import random_uniform
from sumolib import checkBinary
import os
import sys
import xml.etree.ElementTree as et 

def import_st_model_train_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['batch_size'] = content['model'].getint('batch_size')
    config['prediction_steps'] = content['model'].getint('prediction_steps')
    config['train_split'] = content['model'].getfloat('train_split')
    config['training_epochs'] = content['model'].getint('training_epochs')
    config['traffic_feature_folder_num'] = content['dir'].getint('traffic_feature_folder_num')
    config['models_path_name'] = content['dir']['models_path_name']
    return config

def import_static_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['path_name'] = content['dir']['path_name']
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    return config
 
def import_generate_data_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    print(config_file)
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['traffic_data_path_name'] = content['dir']['traffic_data_path_name']
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    return config

def import_train_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['num_layers'] = content['model'].getint('num_layers')
    config['width_layers'] = content['model'].getint('width_layers')
    config['batch_size'] = content['model'].getint('batch_size')
    config['learning_rate'] = content['model'].getfloat('learning_rate')
    config['training_epochs'] = content['model'].getint('training_epochs')
    config['memory_size_min'] = content['memory'].getint('memory_size_min')
    config['memory_size_max'] = content['memory'].getint('memory_size_max')
    config['st_memory_size'] = content['memory'].getint('st_memory_size')
    config['num_states'] = content['agent'].getint('num_states')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['gamma'] = content['agent'].getfloat('gamma')
    config['models_path_name'] = content['dir']['models_path_name']
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    return config


def import_test_configuration(config_file):
    """
    Read the config file regarding the testing and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['gui'] = content['simulation'].getboolean('gui')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint(
        'n_cars_generated')
    config['episode_seed'] = content['simulation'].getint('episode_seed')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['num_states'] = content['agent'].getint('num_states')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    config['models_path_name'] = content['dir']['models_path_name']
    config['model_to_test'] = content['dir'].getint('model_to_test')
    return config


def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", os.path.join('intersection', sumocfg_file_name),
                "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

    return sumo_cmd

def set_raw_traffic_data_path(traffic_data_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    traffic_data_path = os.path.join(os.getcwd(), traffic_data_path_name, '')
    os.makedirs(os.path.dirname(traffic_data_path), exist_ok=True)

    dir_content = os.listdir(traffic_data_path)
    if dir_content:
        previous_versions = [int(name.split("_")[2]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(traffic_data_path, 'traffic_data_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path

def set_traffic_features_path(traffic_features_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    traffic_features_path = os.path.join(os.getcwd(), traffic_features_path_name, '')
    os.makedirs(os.path.dirname(traffic_features_path), exist_ok=True)

    dir_content = os.listdir(traffic_features_path)
    if dir_content:
        previous_versions = [int(name.split("_")[2]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(traffic_features_path, 'traffic_features_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path

def set_static_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'static_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path

def set_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path


def set_test_path(models_path_name, model_n):
    """
    Returns a model path that identifies the model number provided as argument and a newly created 'test' path
    """
    model_folder_path = os.path.join(
        os.getcwd(), models_path_name, 'model_'+str(model_n), '')

    if os.path.isdir(model_folder_path):
        plot_path = os.path.join(model_folder_path, 'test', '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else:
        sys.exit('The model number specified does not exist in the models folder')


def data_split(dataset: pd.DataFrame, train_split, batch_size: int, prediction_steps: int):
    """
    Splits The Dataset into Train and Validate sets based on train_split
    """
    # write code time*node*features(8) 3D
    dataset.drop('Unnamed: 0',axis=1 ,inplace=True) # drop index column // generated by mistake
    # data.shape (30000, 10)
    data = np.array(list(dataset.groupby('Node').apply(pd.DataFrame.to_numpy))) # group by Node column as a 3d matrix
    #a3d.shape (4, 7500, 10) # we still have the node and the time stamp columns in the main features matrix
    data_reshaped = data[:,:,2:]  # data_reshaped.shape (4, 7500, 8)
    data_reshaped = data_reshaped.reshape(data_reshaped.shape[1],data_reshaped.shape[0],data_reshaped.shape[2])
    print('data_reshaped', data_reshaped.shape)

    #prediction_steps = int(batch_size/kernel_size)
    batch_count = int(data_reshaped.shape[0]/batch_size)
    batch_count_train = int(int(data_reshaped.shape[0]/batch_size)*train_split)

    x = np.zeros((batch_count, batch_size, 4, data_reshaped.shape[2]))
    y = np.zeros((batch_count, prediction_steps, 4, data_reshaped.shape[2]))

    batch_num = 0
    for i in range(len(data_reshaped)):
        if(batch_num < batch_count-1):
            batch_start = i * batch_size
            x[batch_num, :, :, :] = data_reshaped[batch_start:batch_start+batch_size, :, :]
            #print("batch_start: ",batch_start)
            #print("batch_start+batch_size: ",batch_start+batch_size)
            y[batch_num, :, :, :] = data_reshaped[batch_start +
                                                  batch_size:batch_start + batch_size + prediction_steps, :, :]
        else:
            break
        batch_num = batch_num + 1
        i = i + batch_size-1

    x_train = x[:batch_count_train, :, :, :]
    y_train = y[:batch_count_train, :, :, :]
    x_val = x[batch_count_train+1:, :, :, :]
    y_val = y[batch_count_train+1:, :, :, :]
  
    return x_train, y_train, x_val, y_val

def read_data_from_xml(file_neme,episode):
    print(' reading file... ')
    path = set_traffic_features_path('TrafficFeatures')
    xml_data = open(os.path.join(file_neme),'r').read()
    root =  et.XML(xml_data)

    db_colomns = ['time_step','speed','edge']
    raws = []
    ### Reading data from file ###
    for node in root:
        time_step = node.attrib.get("time")
        for childnode in node:
            vehicle_speed = childnode.attrib.get("speed")
            vehicle_edge = childnode.attrib.get("edge")

            if vehicle_edge in ['E2TL', 'W2TL', 'N2TL', 'S2TL']:
                raws.append({'time_step':time_step, 
                             'speed':vehicle_speed, 
                             'edge':vehicle_edge})

    ### Putting data in a dataframe ###                            
    data = pd.DataFrame(raws, columns=db_colomns)
    print(' raw data: \n', data)
    convert_dict = {'time_step':float,'speed':float,'edge':str}
    data = data.astype(convert_dict)

    ### Creating an empty dataframe for proccessed data ###
    #nodes_features_coloumns = [ 'east_speed','east_count',
    #                           'west_speed','west_count',
    #                            'north_speed','north_count',
    #                           'south_speed','south_count']
    #nodes_features = None
    traffic_features = pd.DataFrame(columns=['time_step','east_speed','east_count',
                                'west_speed','west_count',
                                'north_speed','north_count',
                               'south_speed','south_count'])

    current_step = 0
    #temp = []
    east_speed = west_speed = north_speed = south_speed = 0.0
    east_count = west_count = north_count = south_count = 0.0

    ### Processing Data ###
    print(' processing data....')
    start_time = timeit.default_timer()
    raw_data_length = len(data)
    for i in range(raw_data_length):
        if int(data.iloc[i]['time_step']) - int(current_step) > 1:
            for k in range(int(current_step), int(data.iloc[i]['time_step'])):
                #temp.append({'east_speed': 0.0,'east_count': 0.0,
                #         'west_speed': 0.0,'west_count': 0.0,
                #         'north_speed': 0.0,'north_count': 0.0,
                #         'south_speed': 0.0,'south_count': 0.0})

                #nodes_features = pd.DataFrame(temp, columns=nodes_features_coloumns, index=['TL'])

                traffic_features = traffic_features.append({'time_step':current_step, 
                                                            'east_speed': 0.0,'east_count': 0.0,
                                                            'west_speed': 0.0,'west_count': 0.0,
                                                            'north_speed': 0.0,'north_count': 0.0,
                                                            'south_speed': 0.0,'south_count': 0.0},
                                                            ignore_index= True)
                #temp = []
                east_speed = west_speed = north_speed = south_speed = 0.0
                east_count = west_count = north_count = south_count = 0.0
                current_step += 1
                if data.iloc[i]['time_step'] - current_step == 1:
                    break
            
        if data.iloc[i]['time_step'] == current_step:

            if data.iloc[i]['edge'] == "E2TL":    
                east_speed += data.iloc[i]['speed']
                east_count += 1

            if data.iloc[i]['edge'] =="W2TL":
                west_speed += data.iloc[i]['speed']
                west_count += 1

            if data.iloc[i]['edge'] =="N2TL":
                north_speed += data.iloc[i]['speed']
                north_count += 1

            if data.iloc[i]['edge'] =="S2TL":
                south_speed += data.iloc[i]['speed']
                south_count += 1
            
        else:
            avg_east_speed = avg_west_speed = avg_north_speed = avg_south_speed = 0.0
                
            if east_count != 0:
                avg_east_speed = east_speed/east_count
            
            if west_count != 0:
                avg_west_speed = west_speed/west_count
            
            if north_count != 0:
                avg_north_speed = north_speed/north_count

            if south_count != 0:
                avg_south_speed = south_speed/south_count
            
            #temp.append({'east_speed': avg_east_speed,'east_count': east_count,
            #             'west_speed': avg_west_speed,'west_count': west_count,
            #             'north_speed': avg_north_speed,'north_count': north_count,
            #             'south_speed': avg_south_speed,'south_count': south_count})

            #nodes_features = pd.DataFrame(temp, columns=nodes_features_coloumns, index=['TL'])

            traffic_features = traffic_features.append({'time_step':current_step, 
                                                        'east_speed': avg_east_speed,'east_count': east_count,
                                                        'west_speed': avg_west_speed,'west_count': west_count,
                                                        'north_speed': avg_north_speed,'north_count': north_count,
                                                        'south_speed': avg_south_speed,'south_count': south_count},
                                                        ignore_index= True)

            #temp = []
            east_speed = west_speed = north_speed = south_speed = 0.0
            east_count = west_count = north_count = south_count = 0.0
            current_step+=1

            if data.iloc[i]['edge'] == "E2TL":    
                east_speed += data.iloc[i]['speed']
                east_count += 1

            if data.iloc[i]['edge'] =="W2TL":
                west_speed += data.iloc[i]['speed']
                west_count += 1

            if data.iloc[i]['edge'] =="N2TL":
                north_speed += data.iloc[i]['speed']
                north_count += 1

            if data.iloc[i]['edge'] =="S2TL":
                south_speed += data.iloc[i]['speed']
                south_count += 1
                

    print(' processing time: ', str(timeit.default_timer() - start_time))
    print(' processed data: ', traffic_features)
    traffic_features.to_csv(os.path.join(path,'traffic_features'+str(episode)+'.csv'), index=False)
    return traffic_features

def read_data_from_xml_mpi(file_neme, episode, output_path):
    print(' reading file... ')
    xml_data = open(os.path.join(file_neme),'r').read()
    root =  et.XML(xml_data)

    db_colomns = ['time_step','speed','edge']
    raws = []
    ### Reading data from file ###
    for node in root:
        time_step = node.attrib.get("time")
        for childnode in node:
            vehicle_speed = childnode.attrib.get("speed")
            vehicle_edge = childnode.attrib.get("edge")

            if vehicle_edge in ['E2TL', 'W2TL', 'N2TL', 'S2TL']:
                raws.append({'time_step':time_step, 
                             'speed':vehicle_speed, 
                             'edge':vehicle_edge})

    ### Putting data in a dataframe ###                            
    data = pd.DataFrame(raws, columns=db_colomns)
    print(' raw data: \n', data)
    convert_dict = {'time_step':float,'speed':float,'edge':str}
    data = data.astype(convert_dict)

    ### Creating an empty dataframe for proccessed data ###
    #nodes_features_coloumns = [ 'east_speed','east_count',
    #                           'west_speed','west_count',
    #                            'north_speed','north_count',
    #                           'south_speed','south_count']
    #nodes_features = None
    traffic_features = pd.DataFrame(columns=['time_step','east_speed','east_count',
                                'west_speed','west_count',
                                'north_speed','north_count',
                               'south_speed','south_count'])

    current_step = 0
    #temp = []
    east_speed = west_speed = north_speed = south_speed = 0.0
    east_count = west_count = north_count = south_count = 0.0

    ### Processing Data ###
    print(' processing data....')
    start_time = timeit.default_timer()
    raw_data_length = len(data)
    for i in range(raw_data_length):
        if int(data.iloc[i]['time_step']) - int(current_step) > 1:
            for k in range(int(current_step), int(data.iloc[i]['time_step'])):
                #temp.append({'east_speed': 0.0,'east_count': 0.0,
                #         'west_speed': 0.0,'west_count': 0.0,
                #         'north_speed': 0.0,'north_count': 0.0,
                #         'south_speed': 0.0,'south_count': 0.0})

                #nodes_features = pd.DataFrame(temp, columns=nodes_features_coloumns, index=['TL'])

                traffic_features = traffic_features.append({'time_step':current_step, 
                                                            'east_speed': 0.0,'east_count': 0.0,
                                                            'west_speed': 0.0,'west_count': 0.0,
                                                            'north_speed': 0.0,'north_count': 0.0,
                                                            'south_speed': 0.0,'south_count': 0.0},
                                                            ignore_index= True)
                #temp = []
                east_speed = west_speed = north_speed = south_speed = 0.0
                east_count = west_count = north_count = south_count = 0.0
                current_step += 1
                if data.iloc[i]['time_step'] - current_step == 1:
                    break
            
        if data.iloc[i]['time_step'] == current_step:

            if data.iloc[i]['edge'] == "E2TL":    
                east_speed += data.iloc[i]['speed']
                east_count += 1

            if data.iloc[i]['edge'] =="W2TL":
                west_speed += data.iloc[i]['speed']
                west_count += 1

            if data.iloc[i]['edge'] =="N2TL":
                north_speed += data.iloc[i]['speed']
                north_count += 1

            if data.iloc[i]['edge'] =="S2TL":
                south_speed += data.iloc[i]['speed']
                south_count += 1
            
        else:
            avg_east_speed = avg_west_speed = avg_north_speed = avg_south_speed = 0.0
                
            if east_count != 0:
                avg_east_speed = east_speed/east_count
            
            if west_count != 0:
                avg_west_speed = west_speed/west_count
            
            if north_count != 0:
                avg_north_speed = north_speed/north_count

            if south_count != 0:
                avg_south_speed = south_speed/south_count
            
            #temp.append({'east_speed': avg_east_speed,'east_count': east_count,
            #             'west_speed': avg_west_speed,'west_count': west_count,
            #             'north_speed': avg_north_speed,'north_count': north_count,
            #             'south_speed': avg_south_speed,'south_count': south_count})

            #nodes_features = pd.DataFrame(temp, columns=nodes_features_coloumns, index=['TL'])

            traffic_features = traffic_features.append({'time_step':current_step, 
                                                        'east_speed': avg_east_speed,'east_count': east_count,
                                                        'west_speed': avg_west_speed,'west_count': west_count,
                                                        'north_speed': avg_north_speed,'north_count': north_count,
                                                        'south_speed': avg_south_speed,'south_count': south_count},
                                                        ignore_index= True)

            #temp = []
            east_speed = west_speed = north_speed = south_speed = 0.0
            east_count = west_count = north_count = south_count = 0.0
            current_step+=1

            if data.iloc[i]['edge'] == "E2TL":    
                east_speed += data.iloc[i]['speed']
                east_count += 1

            if data.iloc[i]['edge'] =="W2TL":
                west_speed += data.iloc[i]['speed']
                west_count += 1

            if data.iloc[i]['edge'] =="N2TL":
                north_speed += data.iloc[i]['speed']
                north_count += 1

            if data.iloc[i]['edge'] =="S2TL":
                south_speed += data.iloc[i]['speed']
                south_count += 1
                

    print(' processing time: ', str(timeit.default_timer() - start_time))
    print(' processed data: ', traffic_features)
    traffic_features.to_csv(os.path.join(output_path,'traffic_features'+str(episode)+'.csv'), index=False)
    return traffic_features