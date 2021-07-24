from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile
from static_simulation import Simulation
from generator import TrafficGenerator
from visualization import Visualization
from utils import import_static_configuration, set_sumo, set_static_path


if __name__ == "__main__":

    config = import_static_configuration(config_file='static_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_static_path(config['path_name'])

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
    
    while episode < 1: #1 config['total_episodes']
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        simulation_time= Simulation.run(10000)  # run the simulation #10000 episode 
        print('Simulation time:', simulation_time)
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'static_settings.ini'))

    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')
    