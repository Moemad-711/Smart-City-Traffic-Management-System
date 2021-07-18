from pandas._libs.tslibs.timestamps import Timestamp
from libsumo.libsumo import edge, vehicle
import traci
import numpy as np
import pandas as pd
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
        columnsNames = ['TimeStamp','Node','east_speed','east_count','west_speed','west_count','north_speed','north_count','south_speed','south_count']
        allDataList = []

        traci.start(self._sumo_cmd)
        
        print("Simulating...")
        file_name =path+'/traffic_features'+str(episode)+'.csv'

        nodes = {'tl1':{'east_speed':0,'east_count':0,'west_speed':0,'west_count':0,'north_speed':0,'north_count':0,'south_speed':0,'south_count':0},
                'tl2' :{'east_speed':0,'east_count':0,'west_speed':0,'west_count':0,'north_speed':0,'north_count':0,'south_speed':0,'south_count':0},
                'tl3' :{'east_speed':0,'east_count':0,'west_speed':0,'west_count':0,'north_speed':0,'north_count':0,'south_speed':0,'south_count':0},
                'tl4' :{'east_speed':0,'east_count':0,'west_speed':0,'west_count':0,'north_speed':0,'north_count':0,'south_speed':0,'south_count':0}
                }
        edges_direction={    'west':['uw_tl1','tl1_tl2','lw_tl3','tl3_tl4'],
                             'east':['tl2_tl1','ue_tl2','tl4_tl3','le_tl4'],
                             'south':['tl3_tl1','tl4_tl2','ls_tl3','rs_tl4'],
                             'north':['ln_tl1','rn_tl2','tl1_tl3','tl2_tl4'] }
        while self._step < self._max_steps:
            traci.simulationStep()
            edge_list = traci.edge.getIDList()
            for edge in edge_list:
                node = str(edge).split('_')[1]
                if node in TL_Nodes:
                    avg_speed = traci.edge.getLastStepMeanSpeed(edge)
                    vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
                    ## Data ---> Dataframe? ----> numpy array(batch_size(time_steps),4,8) 
                    Timestamp = self._step
                    Node = node
                    if edge in edges_direction['west']:
                        nodes[node]['west_speed'] = avg_speed
                        nodes[node]['west_count'] = vehicle_count
                    if edge in edges_direction['east']:
                        nodes[node]['east_speed'] = avg_speed
                        nodes[node]['east_count'] = vehicle_count
                    if edge in edges_direction['south']:
                        nodes[node]['south_speed'] = avg_speed
                        nodes[node]['south_count'] = vehicle_count
                    if edge in edges_direction['north']:
                        nodes[node]['north_speed'] = avg_speed
                        nodes[node]['north_count'] = vehicle_count
                    

            for tl in TL_Nodes:      
                    current_data = [self._step,tl,nodes[node]['east_speed'],nodes[node]['east_count'],nodes[node]['west_speed'],nodes[node]['west_count']
                    ,nodes[node]['north_speed'],nodes[node]['north_count'],nodes[node]['south_speed'],nodes[node]['south_count']]
                    allDataList.append(current_data)
            self._step += 1
     
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
         
        MultipleIntersection_df  = pd.DataFrame(allDataList,columns=columnsNames)
        MultipleIntersection_df.to_csv(file_name)

        return simulation_time

    def run_mpi(self, episode, path, rank, sumo_cmd):
       
        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile_mpi(seed=episode, rank=rank)
        self._step = 0
        TL_Nodes = ['tl1','tl2','tl3','tl4']
        columnsNames = ['TimeStamp','Node','east_speed','east_count','west_speed','west_count','north_speed','north_count','south_speed','south_count']
        allDataList = []
        
        traci.start(sumo_cmd)
        
        print("Simulating...")
        

        file_name =path+'/traffic_features'+str(episode)+'.csv'

        nodes = {'tl1':{'east_speed':0,'east_count':0,'west_speed':0,'west_count':0,'north_speed':0,'north_count':0,'south_speed':0,'south_count':0},
                'tl2' :{'east_speed':0,'east_count':0,'west_speed':0,'west_count':0,'north_speed':0,'north_count':0,'south_speed':0,'south_count':0},
                'tl3' :{'east_speed':0,'east_count':0,'west_speed':0,'west_count':0,'north_speed':0,'north_count':0,'south_speed':0,'south_count':0},
                'tl4' :{'east_speed':0,'east_count':0,'west_speed':0,'west_count':0,'north_speed':0,'north_count':0,'south_speed':0,'south_count':0}
                }
        edges_direction={'west':['uw_tl1','tl1_tl2','lw_tl3','tl3_tl4'],
                         'east':['tl2_tl1','ue_tl2','tl4_tl3','le_tl4'],
                         'south':['tl3_tl1','tl4_tl2','ls_tl3','rs_tl4'],
                         'north':['ln_tl1','rn_tl2','tl1_tl3','tl2_tl4']}
        
        while self._step < self._max_steps:
            traci.simulationStep()
            edge_list = traci.edge.getIDList()
            for edge in edge_list:
                node = str(edge).split('_')[1]
                if node in TL_Nodes:
                    avg_speed = traci.edge.getLastStepMeanSpeed(edge)
                    vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
                    ## Data ---> Dataframe? ----> numpy array(batch_size(time_steps),4,8) 
                    Timestamp = self._step
                    Node = node
                    if edge in edges_direction['west']:
                        nodes[node]['west_speed'] = avg_speed
                        nodes[node]['west_count'] = vehicle_count
                    if edge in edges_direction['east']:
                        nodes[node]['east_speed'] = avg_speed
                        nodes[node]['east_count'] = vehicle_count
                    if edge in edges_direction['south']:
                        nodes[node]['south_speed'] = avg_speed
                        nodes[node]['south_count'] = vehicle_count
                    if edge in edges_direction['north']:
                        nodes[node]['north_speed'] = avg_speed
                        nodes[node]['north_count'] = vehicle_count
                    

            for tl in TL_Nodes:      
                    current_data = [self._step,tl,nodes[node]['east_speed'],nodes[node]['east_count'],nodes[node]['west_speed'],nodes[node]['west_count']
                    ,nodes[node]['north_speed'],nodes[node]['north_count'],nodes[node]['south_speed'],nodes[node]['south_count']]
                    allDataList.append(current_data)
            self._step += 1
        
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        MultipleIntersection_df  = pd.DataFrame(allDataList,columns=columnsNames)
        MultipleIntersection_df.to_csv(file_name)
        
        return simulation_time
