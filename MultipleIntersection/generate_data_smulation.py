from libsumo.libsumo import edge, vehicle
import traci
import numpy as np
import random
import timeit
import os
    

class Simulation:
    def __init__(self,TrafficGen, sumo_cmd, max_steps):
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._step = 0

      
    def run(self, episode, path):
       
        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile(seed=episode)
        self._step = 0
        TL_Nodes = ['tl1','tl2','tl3','tl4']

        traci.start(self._sumo_cmd)
        
        print("Simulating...")
        file_name =path+'/traffic_features'+str(episode)+'.xml'
        
        while self._step < self._max_steps:
            traci.simulationStep()
            edge_list = traci.edge.getIDList()
            for edge in edge_list:
                node = str(edge).split()[1]
                if node in TL_Nodes:
                    avg_speed = traci.edge.getLastStepMeanSpeed(edge)
                    vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
                    ## Data ---> Dataframe? ----> numpy array(batch_size(time_steps),4,8) 
            self._step += 1
        
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time

    def run_mpi(self, episode, path, rank, sumo_cmd):
       
        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile_mpi(seed=episode, rank=rank)
        self._step = 0
        TL_Nodes = ['tl1','tl2','tl3','tl4']
        
        traci.start(sumo_cmd)
        
        print("Simulating...")
        file_name =path+'/traffic_features'+str(episode)+'.xml'
        
        while self._step < self._max_steps:
            traci.simulationStep()
            edge_list = traci.edge.getIDList()
            for edge in edge_list:
                node = str(edge).split()[1]
                if node in TL_Nodes:
                    avg_speed = traci.edge.getLastStepMeanSpeed(edge)
                    vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
                    ## Data ---> Dataframe? ----> numpy array(batch_size(time_steps),4,8) 
            self._step += 1
        
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        
        return simulation_time
