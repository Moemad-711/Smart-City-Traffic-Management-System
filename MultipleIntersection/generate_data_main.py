from __future__ import absolute_import
from __future__ import print_function
import datetime
import math
import os

from mpi4py import MPI
from generate_data_smulation import Simulation
from generator import TrafficGenerator
from utils import import_generate_data_configuration, set_sumo, set_traffic_features_path


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        config = import_generate_data_configuration(config_file='generate_data_settings.ini')
        #sumo_cmd = set_sumo(config['gui'], os.path.join('routes','sumo_config_%i.sumocfg' %(rank)), config['max_steps'])
        #print('SUMO_CMD: ', sumo_cmd)
        path = set_traffic_features_path(config['traffic_data_path_name'])
    
    else:
        config = None
        #sumo_cmd = None
        path = None

    config = comm.bcast(config, root=0)
    sumo_cmd = set_sumo(config['gui'], os.path.join('routes','sumo_config_%i.sumocfg' %(rank)), config['max_steps'])
    print('SUMO_CMD: ', sumo_cmd)
    path = comm.bcast(path, root=0)
    
    

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
    
    for episode in range(rank * math.ceil(config['total_episodes']/size), (rank*math.ceil(config['total_episodes']/size)) + math.ceil(config['total_episodes']/size)):
        if episode >= config['total_episodes']:
            break
        else:
            print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']),'-----')
            simulation_time = Simulation.run_mpi(episode, path, rank, sumo_cmd)  # run the Generator data
            print('Simulation time:', simulation_time)

    #while episode < config['total_episodes']:
    #    print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
    #    simulation_time = Simulation.run(episode, path)  # run the Generator data
    #    print('Simulation time:', simulation_time)
    #    episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
