from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile
from static_simulation import Simulation
from generator import TrafficGenerator
from visualization import Visualization
from utils import import_static_configuration, import_train_configuration, set_static_path, set_sumo, set_train_path


if __name__ == "__main__":

    config = import_static_configuration(config_file='static_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_static_path(config['path_name'])
    print('config done')

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )
    
    Visualization = Visualization(
        path, 
        dpi=96
    )
        
    Simulation = Simulation(
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration']
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    while episode < 1: #config['total_episodes']
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))        
        simulation_time= Simulation.run(10000)  # run the simulation  episode
        print('Simulation time:', simulation_time)
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    copyfile(src='static_settings.ini', dst=os.path.join(path, 'static_settings.ini'))
    for TL in Simulation._TL_list:
        Visualization.save_data_and_plot(data=Simulation.reward_store[TL], filename='reward_%s' %(TL), xlabel='Episode', ylabel='Cumulative negative reward')
        Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store[TL], filename='delay_%s' %(TL), xlabel='Episode', ylabel='Cumulative delay (s)')
        Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store[TL], filename='queue_%s' %(TL), xlabel='Episode', ylabel='Average queue length (vehicles)')

    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store['all'], filename='delay_all', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store['all'], filename='queue_all', xlabel='Episode', ylabel='Average queue length (vehicles)')
    