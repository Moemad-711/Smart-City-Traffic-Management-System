from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile
from PredictiveModel import PredictiveModel, TestPredictiveModel
from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory, ST_Memory
#from model_cnn import TrainModel
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path


if __name__ == "__main__":

    config = import_train_configuration(config_file='training_settings.ini')
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
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    Models={'TL1':TrainModel(config['num_layers'], 
                             config['width_layers'], 
                             config['batch_size'], 
                             config['learning_rate'], 
                             input_dim=config['num_states'], 
                             output_dim=config['num_actions']),
            'TL2':TrainModel(config['num_layers'], 
                             config['width_layers'], 
                             config['batch_size'], 
                             config['learning_rate'], 
                             input_dim=config['num_states'], 
                             output_dim=config['num_actions']),
            'TL3':TrainModel(config['num_layers'], 
                            config['width_layers'], 
                            config['batch_size'], 
                            config['learning_rate'], 
                            input_dim=config['num_states'], 
                            output_dim=config['num_actions']),
            'TL4':TrainModel(config['num_layers'], 
                            config['width_layers'], 
                            config['batch_size'], 
                            config['learning_rate'], 
                            input_dim=config['num_states'], 
                            output_dim=config['num_actions'])}
    
    #Create an ST_Model(GNN) Object 
    st_model = TestPredictiveModel(
        adjacency_matrix, 
        os.path.join('st_models', 'model_2'))
   
    Memories={  'TL1':Memory(config['memory_size_max'], 
                             config['memory_size_min']),
                'TL2':Memory(config['memory_size_max'], 
                             config['memory_size_min']),            
                'TL3':Memory(config['memory_size_max'], 
                             config['memory_size_min']),
                'TL4':Memory(config['memory_size_max'], 
                             config['memory_size_min'])}

    #Create A memory for the ST_Model(GNN)
    st_memory=ST_Memory((5,4,8))

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )
    
    Visualization = Visualization(
        path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Models,
        st_model,
        Memories,
        st_memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs']
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon=1.0
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        
        simulation_time, training_time = Simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    for TL in Simulation._TL_list:
        Models[TL].save_model(path, TL)
        Visualization.save_data_and_plot(data=Simulation.reward_store[TL], filename='reward_%s' %(TL), xlabel='Episode', ylabel='Cumulative negative reward')
        Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store[TL], filename='delay_%s' %(TL), xlabel='Episode', ylabel='Cumulative delay (s)')
        Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store[TL], filename='queue_%s' %(TL), xlabel='Episode', ylabel='Average queue length (vehicles)')

    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store['all'], filename='delay_all', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store['all'], filename='queue_all', xlabel='Episode', ylabel='Average queue length (vehicles)')
    
        