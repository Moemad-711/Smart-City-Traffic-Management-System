from numpy.core.numeric import NaN
from numpy.lib.arraysetops import intersect1d
from numpy.lib.function_base import average
import pandas as pd
from libsumo.libsumo import edge, edge_getIDList, vehicle
import traci
import numpy as np
import random
import timeit
import math
import os

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration):
        self._TrafficGen = TrafficGen
        self._step = 0
        self._action = {'TL1':0,'TL2':0,'TL3':0,'TL3':0}
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._reward_store = {'TL1':[], 'TL2':[], 'TL3':[], 'TL4':[]}
        self._cumulative_wait_store = {'TL1':[], 'TL2':[], 'TL3':[], 'TL4':[]}
        self._avg_queue_length_store = {'TL1':[], 'TL2':[], 'TL3':[], 'TL4':[]}
        self._current_phase_duration = {'TL1':0, 'TL2':0, 'TL3':0, 'TL4':0}
        self._TL_list = ['TL1','TL2','TL3','TL4']
        self._action = {'TL1':-1,'TL2':-1,'TL3':-1,'TL4':-1}



    def run(self, episode):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {'TL1':{},'TL2':{},'TL3':{},'TL4':{}}
        self._sum_neg_reward = {'TL1':0,'TL2':0,'TL3':0,'TL4':0}
        self._sum_queue_length = {'TL1':0,'TL2':0,'TL3':0,'TL4':0}
        self._sum_waiting_time = {'TL1':0,'TL2':0,'TL3':0,'TL4':0}
        old_total_wait = {'TL1':0,'TL2':0,'TL3':0,'TL4':0}
        current_total_wait = {'TL1':0,'TL2':0,'TL3':0,'TL4':0}
        is_phase_green = {'TL1':True, 'TL2':True, 'TL3':True, 'TL4':True}
        reward = {'TL1':0,'TL2':0,'TL3':0,'TL4':0}
        old_action = {'TL1':-1,'TL2':-1,'TL3':-1,'TL4':-1}

        while self._step < self._max_steps:
            for TL in self._TL_list:
                if self._current_phase_duration[TL] ==0:
                    if is_phase_green[TL]== True:
                        if self._step != 0:
                            old_total_wait[TL] = current_total_wait[TL]
                            old_action[TL] = self._action[TL]

                            # saving only the meaningful reward to better see if the agent is behaving correctly
                            if reward[TL] < 0:
                                self._sum_neg_reward[TL] += reward[TL]           
                        # calculate reward of previous action: (change in cumulative waiting time between actions)
                        # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
                        current_total_wait[TL] = self._collect_waiting_times(TL)
                        reward[TL] = old_total_wait[TL] - current_total_wait[TL]
            
                        # choose the light phase to activate, based on the current state of the intersection
                        self._action[TL] = self._choose_action(TL)
                        # if the chosen phase is different from the last phase, activate the yellow phase
                        if self._step != 0:
                            self._set_yellow_phase(old_action[TL] , TL)
                            is_phase_green[TL] = False
                            self._current_phase_duration[TL] = self._yellow_duration
                        else: 
                            self._set_green_phase(self._action[TL], TL)
                            is_phase_green[TL] = True
                            self._current_phase_duration[TL] = self._green_duration
                    else:
                        # execute the phase selected before
                        self._set_green_phase(self._action[TL], TL)
                        is_phase_green[TL] = True
                        self._current_phase_duration[TL] = self._green_duration

            self._simulate() 
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time


    def _simulate(self):
        
        traci.simulationStep()  # simulate 1 step in sumo
        self._step += 1 # update the step counter
        for TL in self._TL_list:
            self._current_phase_duration[TL]-=1
            queue_length = self._get_queue_length(TL)
            self._sum_queue_length[TL] += queue_length
            self._sum_waiting_time[TL] += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds
       
    def _collect_waiting_times(self,TL):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        
        incoming_roads ={'TL1':['uw_tl1','tl3_tl1','tl2_tl1','ln_tl1'],
                         'TL2':['rn_tl2','tl1_tl2','tl4_tl2','ue_tl2'],
                         'TL3':['tl4_tl3','tl1_tl3','lw_tl3','ls_tl3'],
                         'TL4':['le_tl4','rs_tl4','tl2_tl4','tl3_tl4']} 
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads[TL]:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[TL][car_id] = wait_time
            else:
                if car_id in self._waiting_times[TL]: # a car that was tracked has cleared the intersection
                    del self._waiting_times[TL][car_id] 
        total_waiting_time = sum(self._waiting_times[TL].values())
        return total_waiting_time

    def _choose_action(self,TL):
        return (self._action[TL]+1)%4
    #Write method to get the Greenlight Time
    
    def _set_yellow_phase(self, old_action,TL):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase(TL, yellow_phase_code)

    def _set_green_phase(self, action_number,TL):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase(TL, PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase(TL, PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase(TL, PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase(TL, PHASE_EWL_GREEN)

    def _get_queue_length(self,TL):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        incoming_roads ={'TL1':['uw_tl1','tl3_tl1','tl2_tl1','ln_tl1'],
                         'TL2':['rn_tl2','tl1_tl2','tl4_tl2','ue_tl2'],
                         'TL3':['tl4_tl3','tl1_tl3','lw_tl3','ls_tl3'],
                         'TL4':['le_tl4','rs_tl4','tl2_tl4','tl3_tl4']} 
        result=incoming_roads[TL]
        halt_N = traci.edge.getLastStepHaltingNumber(result[0])
        halt_S = traci.edge.getLastStepHaltingNumber(result[1])
        halt_E = traci.edge.getLastStepHaltingNumber(result[2])
        halt_W = traci.edge.getLastStepHaltingNumber(result[3])
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        for TL in self._TL_list:
            self._reward_store.append(self._sum_neg_reward[TL])  # how much negative reward in this episode
            self._cumulative_wait_store.append(self._sum_waiting_time[TL])  # total number of seconds waited by cars in this episode
            self._avg_queue_length_store.append(self._sum_queue_length[TL] / self._max_steps)  # average number of queued cars per step, in this episode


    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

