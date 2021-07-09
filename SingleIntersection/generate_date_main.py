from __future__ import absolute_import
from __future__ import print_function
import datetime

import os
import traci
from generate_data_smulation import Simulation
from generator import TrafficGenerator
from utils import import_generate_data_configuration, set_sumo


if __name__ == "__main__":

    config = import_generate_data_configuration(config_file='generate_data_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

        
    Simulation = Simulation(
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        simulation_time = Simulation.run(episode)  # run the Generator data
        print('Simulation time:', simulation_time)
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
