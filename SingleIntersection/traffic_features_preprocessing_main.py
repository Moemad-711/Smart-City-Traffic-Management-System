from os import path
import numpy as np
import pandas as pd
import os
from mpi4py import MPI
import math

from utils import  read_data_from_xml, read_data_from_xml_mpi


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    episode_count = 100
    dataset_num = 2

    if rank == 0:
        path = os.path.join('TrafficData','traffic_data_%i' % (dataset_num))
    else: 
        path = None
    
    path = comm.bcast(path, root=0)

    for index in range(rank * math.ceil(episode_count/size), (rank * math.ceil(episode_count/size)) + math.ceil(episode_count/size)):
        #reading data form files
        if index >= episode_count:
            break
        print('----- Readinding from data%i -----' % (index))
        
        file_name = 'data' + str(index) + '.xml'
        traffic_data = []
        traffic_data.append(read_data_from_xml_mpi(os.path.join(path, file_name), index, comm, rank))
        #traffic_data.append(read_data_from_xml(os.path.join(path, file_name), index))
    
    



