from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile
from PredictiveModel import TestPredictiveModel
from memory import ST_Memory

from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path


if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

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

    Models={'TL1':TestModel(input_dim=config['num_states'],
                            model_path=model_path,
                            TL = 'TL1'
                            ),
            'TL2':TestModel(input_dim=config['num_states'],
                            model_path=model_path,
                            TL = 'TL2'
                            ),
            'TL3':TestModel(input_dim=config['num_states'],
                            model_path=model_path,
                            TL = 'TL3'
                            ),
            'TL4':TestModel(input_dim=config['num_states'],
                            model_path=model_path,
                            TL = 'TL4'
                            )}
    
    st_model=TestPredictiveModel(
        adjacency_matrix, 
        os.path.join('st_models', 'model_2')
    )

    st_memory=ST_Memory((5,4,8))


    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Models,
        st_model,
        st_memory,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions']
    )

    print('\n----- Test episode')
    simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
    print('Simulation time:', simulation_time, 's')

    print("----- Testing info saved at:", plot_path)

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    for TL in Simulation._TL_list:
        Visualization.save_data_and_plot(data=Simulation.reward_episode[TL], filename='reward_%s' %(TL), xlabel='Episode', ylabel='Cumulative negative reward')
        Visualization.save_data_and_plot(data=Simulation.queue_length_episode[TL], filename='queue_%s' %(TL), xlabel='Episode', ylabel='Average queue length (vehicles)')

    Visualization.save_data_and_plot(data=Simulation.queue_length_episode['all'], filename='queue_all', xlabel='Episode', ylabel='Average queue length (vehicles)')
