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

# Lanes List
N_Straight={'TL1':['ln_tl1_0', 'ln_tl1_1', 'ln_tl1_2'],
            'TL2':['rn_tl2_0', 'rn_tl2_1', 'rn_tl2_2'],
            'TL3':['tl1_tl3_0', 'tl1_tl3_1', 'tl1_tl3_2'],
            'TL4':['tl2_tl4_0', 'tl2_tl4_1', 'tl2_tl4_2']}
N_left={'TL1':['ln_tl1_3'],
        'TL2':['rn_tl2_3'],
        'TL3':['tl1_tl3_3'],
        'TL4':['tl2_tl4_3']}
S_Straight={'TL1':['tl3_tl1_0', 'tl3_tl1_1', 'tl3_tl1_2'],
            'TL2':['tl4_tl2_0', 'tl4_tl2_1', 'tl4_tl2_2'],
            'TL3':['ls_tl3_0', 'ls_tl3_1', 'ls_tl3_2'],
            'TL4':['rs_tl4_0', 'rs_tl4_1', 'rs_tl4_2']}
S_left={'TL1':['tl3_tl1_3'],
        'TL2':['tl4_tl2_3'],
        'TL3':['ls_tl3_3'],
        'TL4':['rs_tl4_3']}
W_Straight={'TL1':['uw_tl1_0', 'uw_tl1_1', 'uw_tl1_2'],
            'TL2':['tl1_tl2_0', 'tl1_tl2_1', 'tl1_tl2_2'],
            'TL3':['lw_tl3_0', 'lw_tl3_1', 'lw_tl3_2'],
            'TL4':['tl3_tl4_0', 'tl3_tl4_1', 'tl3_tl4_2']} 
W_left={'TL1':['uw_tl1_3'],
        'TL2':['tl1_tl2_3'],
        'TL3':['lw_tl3_3'],
        'TL4':['tl3_tl4_3']}
E_Straight={'TL1':['tl2_tl1_0', 'tl2_tl1_1', 'tl2_tl1_2'],
            'TL2':['ue_tl2_0', 'ue_tl2_1', 'ue_tl2_2'],
            'TL3':['tl4_tl3_0', 'tl4_tl3_1', 'tl4_tl3_2'],
            'TL4':['le_tl4_0', 'le_tl4_1', 'le_tl4_2']} 
E_left={'TL1':['tl2_tl1_3'],
        'TL2':['ue_tl2_3'],
        'TL3':['tl4_tl3_3'],
        'TL4':['le_tl4_3']}

class Simulation:
    def __init__(self, Models,st_model,Memories, st_meomry,TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Models = Models
        self._st_model=st_model
        self._Memories = Memories
        self._st_meomry=st_meomry
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = {'TL1':[], 'TL2':[], 'TL3':[], 'TL4':[]}
        self._cumulative_wait_store = {'TL1':[], 'TL2':[], 'TL3':[], 'TL4':[], 'all':[]}
        self._avg_queue_length_store = {'TL1':[], 'TL2':[], 'TL3':[], 'TL4':[], 'all':[]}
        self._training_epochs = training_epochs
        self._current_phase_duration = {'TL1':0, 'TL2':0, 'TL3':0, 'TL4':0}
        self._TL_list = ['TL1','TL2','TL3','TL4']



    def run(self, episode, epsilon):
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
        old_state = {'TL1':-1,'TL2':-1,'TL3':-1,'TL4':-1}
        old_action = {'TL1':-1,'TL2':-1,'TL3':-1,'TL4':-1}
        action = {'TL1':-1,'TL2':-1,'TL3':-1,'TL4':-1}
        is_phase_green = {'TL1':True, 'TL2':True, 'TL3':True, 'TL4':True}
        reward = {'TL1':0,'TL2':0,'TL3':0,'TL4':0}
        current_state = {'TL1':-1,'TL2':-1,'TL3':-1,'TL4':-1}

        while self._step < self._max_steps:
            for TL in self._TL_list:
                if self._current_phase_duration[TL] ==0:
                    if is_phase_green[TL]== True:
                        if self._step != 0:
                            # saving variables for later & accumulate reward
                            old_state[TL] = current_state[TL]
                            old_action[TL] = action[TL]
                            old_total_wait[TL] = current_total_wait[TL]
                            # saving only the meaningful reward to better see if the agent is behaving correctly
                            if reward[TL] < 0:
                                self._sum_neg_reward[TL] += reward[TL]

                        # get current state of the intersection
                        current_state[TL] = self._get_state(TL)
           

                        # calculate reward of previous action: (change in cumulative waiting time between actions)
                        # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
                        current_total_wait[TL] = self._collect_waiting_times(TL)
                        reward[TL] = old_total_wait[TL] - current_total_wait[TL]
            
                        # saving the data into the memory
                        if self._step != 0:
                            self._Memories[TL].add_sample((old_state[TL], old_action[TL], reward[TL], current_state[TL]))

                        # choose the light phase to activate, based on the current state of the intersection
                        action[TL] = self._choose_action(current_state[TL], epsilon,TL)

                        # if the chosen phase is different from the last phase, activate the yellow phase
                        if self._step !=0 and old_action[TL] != action[TL]:
                            self._set_yellow_phase(old_action[TL] , TL)
                            is_phase_green[TL] = False
                            self._current_phase_duration[TL] = self._yellow_duration
                        else:
                            self._set_green_phase(action[TL], TL)
                            is_phase_green[TL] = True
                            greenlight_durations= self._get_green_duration(action[TL], TL)
                            greenlight_durations =[x for x in greenlight_durations if x>0 and x<= self._green_duration]
                            if len(greenlight_durations) > 0:
                                greenlight_duration = math.ceil(min(greenlight_durations))
                                self._current_phase_duration[TL] = greenlight_duration
                            else:
                                self._current_phase_duration[TL] = self._green_duration
                    else:
                        # execute the phase selected before
                        self._set_green_phase(action[TL], TL)
                        is_phase_green[TL] = True
                        greenlight_durations= self._get_green_duration(action[TL], TL)
                        greenlight_durations =[x for x in greenlight_durations if x>0 and x<= self._green_duration]
                        if len(greenlight_durations) > 0:
                            greenlight_duration = math.ceil(min(greenlight_durations))
                            self._current_phase_duration[TL] = greenlight_duration
                        else:
                            self._current_phase_duration[TL] = self._green_duration

            self._simulate()

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        #Train the model
        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time


    def _simulate(self):
        """
        Execute steps in sumo while gathering statistics
        """
        #5400 if todo = 5 and step = 5397 --> 5402 --> todo = 3 
        
        #print(' - Step:', self._step)
        #traci.simulationStep()  # simulate 1 step in sumo

        ### saving traffic features to st_memory ###
        TL_Nodes = ['tl1','tl2','tl3','tl4']
        nodes = {'tl1':{'east_speed':0,'east_count':0,'west_speed':0,'west_count':0,'north_speed':0,'north_count':0,'south_speed':0,'south_count':0},
                 'tl2' :{'east_speed':0,'east_count':0,'west_speed':0,'west_count':0,'north_speed':0,'north_count':0,'south_speed':0,'south_count':0},
                 'tl3' :{'east_speed':0,'east_count':0,'west_speed':0,'west_count':0,'north_speed':0,'north_count':0,'south_speed':0,'south_count':0},
                 'tl4' :{'east_speed':0,'east_count':0,'west_speed':0,'west_count':0,'north_speed':0,'north_count':0,'south_speed':0,'south_count':0}
                }
        edges_direction={'west':['uw_tl1','tl1_tl2','lw_tl3','tl3_tl4'],
                         'east':['tl2_tl1','ue_tl2','tl4_tl3','le_tl4'],
                         'south':['tl3_tl1','tl4_tl2','ls_tl3','rs_tl4'],
                         'north':['ln_tl1','rn_tl2','tl1_tl3','tl2_tl4']}
        current_node_features = pd.DataFrame(columns=['east_speed','east_count',
                                                      'west_speed','west_count',
                                                      'north_speed','north_count',
                                                      'south_speed','south_count'])                                           
        traci.simulationStep()
        edge_list = traci.edge.getIDList()
        for edge in edge_list:
            node = str(edge).split('_')[1]
            if node in TL_Nodes:
                vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
                if vehicle_count == 0: 
                    avg_speed = 0
                else: 
                    avg_speed = traci.edge.getLastStepMeanSpeed(edge)

                if edge in edges_direction['west']:
                    nodes[node]['west_speed'] = avg_speed
                    nodes[node]['west_count'] = vehicle_count

                elif edge in edges_direction['east']:
                    nodes[node]['east_speed'] = avg_speed
                    nodes[node]['east_count'] = vehicle_count

                elif edge in edges_direction['south']:
                    nodes[node]['south_speed'] = avg_speed
                    nodes[node]['south_count'] = vehicle_count

                elif edge in edges_direction['north']:
                    nodes[node]['north_speed'] = avg_speed
                    nodes[node]['north_count'] = vehicle_count
        
        for tl in TL_Nodes:      
            current_node_features = current_node_features.append({'east_speed': nodes[tl]['east_speed'],'east_count':nodes[tl]['east_count'],
                                                                  'west_speed': nodes[tl]['west_speed'],'west_count':nodes[tl]['west_count'],
                                                                  'north_speed': nodes[tl]['north_speed'],'north_count':nodes[tl]['north_count'],
                                                                  'south_speed':nodes[tl]['south_speed'],'south_count':nodes[tl]['south_count']}, ignore_index=True)
                
        sample = current_node_features.to_numpy()
        #print(sample)
        self._st_meomry.add_sample(sample)
        #sample = sample_dict.to_numpy()
        #self._st_meomry.add_sample(sample)

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


    def _choose_action(self, state, epsilon,TL):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Models[TL].predict_one(state)) # the best action given the current state

    #Write method to get the Greenlight Time
    def _get_green_duration(self,action,TL): 
        """
        Returns The Minimum of current demand and future demand Greenlight Times
        """
        #current_demand_green_duration = None
        #future_demand_green_duration = None
        green_duration = []

        ##### Get Current Demand Green Duration #####
        if action == 0:
            intersection_length = 33.60
            ### N2TL Duration ###
            N_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in N_Straight[TL]])/3
            N_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in N_Straight[TL]])
            if N_vehicle_count != 0:
                N_single_car_time = -N_avg_speed + math.sqrt((N_avg_speed*N_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(N_single_car_time * N_vehicle_count)
            
            ### S2TL Duration ###
            S_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in S_Straight[TL]])/3
            S_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in S_Straight[TL]])
            if S_vehicle_count != 0:
                S_single_car_time = -S_avg_speed + math.sqrt((S_avg_speed*S_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(S_single_car_time * S_vehicle_count)

        elif action == 1:
            intersection_length = 29.67
            ### N2TL Duration ###
            N_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in N_left[TL]])
            N_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in N_left[TL]])
            if N_vehicle_count != 0:
                N_single_car_time = -N_avg_speed + math.sqrt((N_avg_speed*N_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(N_single_car_time * N_vehicle_count)

            ### S2TL Duration ###
            S_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in S_left[TL]])
            S_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in S_left[TL]])
            if S_vehicle_count != 0:    
                S_single_car_time = -S_avg_speed + math.sqrt((S_avg_speed*S_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(S_single_car_time * S_vehicle_count)

        elif action == 2:
            intersection_length = 33.60
            ### W2TL Duration ###
            W_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in W_Straight[TL]])/3
            W_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in W_Straight[TL]])
            if W_vehicle_count != 0:
                W_single_car_time = -W_avg_speed + math.sqrt((W_avg_speed*W_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(W_single_car_time * W_vehicle_count)

            ### E2TL Duration ###
            E_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in E_Straight[TL]])/3
            E_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in E_Straight[TL]])
            if E_vehicle_count != 0:
                E_single_car_time = -E_avg_speed + math.sqrt((E_avg_speed*E_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(E_single_car_time * E_vehicle_count)

        elif action == 3:
            intersection_length = 29.67
            ### W2TL Duration ###
            W_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in W_left[TL]])
            W_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in W_left[TL]])
            if W_vehicle_count != 0:
                W_single_car_time = -W_avg_speed + math.sqrt((W_avg_speed*W_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(W_single_car_time * W_vehicle_count)

            ### E2TL Duration ###
            E_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in E_left[TL]])
            E_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in E_left[TL]])
            if E_vehicle_count != 0:
                E_single_car_time = -E_avg_speed+ math.sqrt((E_avg_speed*E_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(E_single_car_time * E_vehicle_count)
        ##### End Get Current Demand #####

        ##### Get Future Demand #####
        
        st_model_input =  self._st_meomry.get_samples()
        #print('st_samples: ',st_model_input)
        #print('st_samples_size: ',len(st_model_input))
        
        tl_loc = {'TL1':0, 'TL2':1, 'TL3':2,'TL4':3}
        
        if len(st_model_input) == 0 :
            return green_duration
        #print('     predicting traffic...')
        st_model_output =  self._st_model.predict_one(st_model_input[:,:,:])
        future_traffic = pd.DataFrame(st_model_output[0,-1,:,:], columns=['east_speed','east_count',
                                                                        'west_speed','west_count',
                                                                        'north_speed','north_count',
                                                                        'south_speed','south_count'])
        #print('     future_traffic:\n', future_traffic)
        if action == 0:
            intersection_length = 33.60
            #N_single_car_time = intersection_length/future_traffic.iloc[tl_loc[TL]]['north_speed']
            N_single_car_time = (-future_traffic.iloc[tl_loc[TL]]['north_speed'] 
                                + math.sqrt((future_traffic.iloc[tl_loc[TL]]['north_speed']*future_traffic.iloc[tl_loc[TL]]['north_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[tl_loc[TL]]['north_count']  > 0 and not  math.isnan(future_traffic.iloc[tl_loc[TL]]['north_count']): 
                green_duration.append(N_single_car_time* future_traffic.iloc[tl_loc[TL]]['north_count'] * 3/4)

            #S_single_car_time = intersection_length/future_traffic.iloc[tl_loc[TL]]['south_speed']
            S_single_car_time = (-future_traffic.iloc[tl_loc[TL]]['south_speed'] 
                                + math.sqrt((future_traffic.iloc[tl_loc[TL]]['south_speed']*future_traffic.iloc[tl_loc[TL]]['south_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[tl_loc[TL]]['south_count']  > 0 and not  math.isnan(future_traffic.iloc[tl_loc[TL]]['south_count']):
                green_duration.append(S_single_car_time* future_traffic.iloc[tl_loc[TL]]['south_count'] * 3/4)

        elif action == 1:
            intersection_length = 29.67
            #N_single_car_time = intersection_length/future_traffic.iloc[tl_loc[TL]]['north_speed']
            N_single_car_time = (-future_traffic.iloc[tl_loc[TL]]['north_speed'] 
                                + math.sqrt((future_traffic.iloc[tl_loc[TL]]['north_speed']*future_traffic.iloc[tl_loc[TL]]['north_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[tl_loc[TL]]['north_count']  > 0 and not  math.isnan(future_traffic.iloc[tl_loc[TL]]['north_count']): 
                green_duration.append(N_single_car_time* future_traffic.iloc[tl_loc[TL]]['north_count'] * 1/4)

            #S_single_car_time = intersection_length/future_traffic.iloc[tl_loc[TL]]['south_speed']
            S_single_car_time = (-future_traffic.iloc[tl_loc[TL]]['south_speed'] 
                                + math.sqrt((future_traffic.iloc[tl_loc[TL]]['south_speed']*future_traffic.iloc[tl_loc[TL]]['south_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[tl_loc[TL]]['south_count']  > 0 and not math.isnan(future_traffic.iloc[tl_loc[TL]]['south_count']):
                green_duration.append(S_single_car_time* future_traffic.iloc[tl_loc[TL]]['south_count'] * 1/4)

        elif action == 2:
            intersection_length = 33.60
            #W_single_car_time = intersection_length/future_traffic.iloc[tl_loc[TL]]['west_speed']
            W_single_car_time = (-future_traffic.iloc[tl_loc[TL]]['west_speed'] 
                                + math.sqrt((future_traffic.iloc[tl_loc[TL]]['west_speed']*future_traffic.iloc[tl_loc[TL]]['west_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[tl_loc[TL]]['west_count']  > 0 and not math.isnan(future_traffic.iloc[tl_loc[TL]]['west_count']):
                green_duration.append(W_single_car_time* future_traffic.iloc[tl_loc[TL]]['west_count'] * 3/4)

            #E_single_car_time = intersection_length/future_traffic.iloc[tl_loc[TL]]['east_speed']
            E_single_car_time = (-future_traffic.iloc[tl_loc[TL]]['east_speed'] 
                                + math.sqrt((future_traffic.iloc[tl_loc[TL]]['east_speed']*future_traffic.iloc[tl_loc[TL]]['east_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[tl_loc[TL]]['east_count']  > 0 and not math.isnan(future_traffic.iloc[tl_loc[TL]]['east_count']):    
                green_duration.append(E_single_car_time* future_traffic.iloc[tl_loc[TL]]['east_count'] * 3/4)

        elif action == 3:
            intersection_length = 29.67
            #W_single_car_time = intersection_length/future_traffic.iloc[tl_loc[TL]]['west_speed']
            W_single_car_time = (-future_traffic.iloc[tl_loc[TL]]['west_speed'] 
                                + math.sqrt((future_traffic.iloc[tl_loc[TL]]['west_speed']*future_traffic.iloc[tl_loc[TL]]['west_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[tl_loc[TL]]['west_count']  > 0 and not math.isnan(future_traffic.iloc[tl_loc[TL]]['west_count']):    
                green_duration.append(W_single_car_time* math.ceil(future_traffic.iloc[tl_loc[TL]]['west_count']) * 1/4)

            #E_single_car_time = intersection_length/future_traffic.iloc[tl_loc[TL]]['east_speed']
            E_single_car_time = (-future_traffic.iloc[tl_loc[TL]]['east_speed'] 
                                + math.sqrt((future_traffic.iloc[tl_loc[TL]]['east_speed']*future_traffic.iloc[tl_loc[TL]]['east_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[tl_loc[TL]]['east_count']  > 0 and not math.isnan(future_traffic.iloc[tl_loc[TL]]['east_count']):     
                green_duration.append(E_single_car_time* math.ceil(future_traffic.iloc[tl_loc[TL]]['east_count']) * 1/4)
        #print('     green_duration:', green_duration)
        return green_duration

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


    def _get_state(self, TL):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        straights={'weast':['uw_tl1_0','uw_tl1_1','uw_tl1_2','tl1_tl2_0','tl1_tl2_1','tl1_tl2_2','lw_tl3_0','lw_tl3_1','lw_tl3_2','tl3_tl4_0','tl3_tl4_1','tl3_tl4_2'],
                    'east':['tl2_tl1_0','tl2_tl1_1','tl2_tl1_2','ue_tl2_0','ue_tl2_1','ue_tl2_2','tl4_tl3_0','tl4_tl3_1','tl4_tl3_2','le_tl4_0','le_tl4_1','le_tl4_2'],
                    'south':['tl3_tl1_0','tl3_tl1_1','tl3_tl1_2','tl4_tl2_0','tl4_tl2_1','tl4_tl2_2','ls_tl3_0','ls_tl3_1','ls_tl3_2','rs_tl4_0','rs_tl4_1','rs_tl4_2'],
                    'north':['ln_tl1_0','ln_tl1_1','ln_tl1_2','rn_tl2_0','rn_tl2_1','rn_tl2_2','tl1_tl3_0','tl1_tl3_1','tl1_tl3_2','tl2_tl4_0','tl2_tl4_1','tl2_tl4_2']}
        
        turns={     'weast':['uw_tl1_3','tl1_tl2_3','lw_tl3_3','tl3_tl4_3'],
                    'east':['tl2_tl1_3','ue_tl2_3','tl4_tl3_3','le_tl4_3'],
                    'south':['tl3_tl1_3','tl4_tl2_3','ls_tl3_3','rs_tl4_3'],
                    'north':['ln_tl1_3','rn_tl2_3','tl1_tl3_3','tl2_tl4_3']}
        lane_groups=[straights['weast'],turns['weast'],straights['east'],turns['east'],
                    straights['south'],turns['south'],straights['north'],turns['north']]
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()
        incoming_roads ={'TL1':['uw_tl1','tl3_tl1','tl2_tl1','ln_tl1'],
                         'TL2':['rn_tl2','tl1_tl2','tl4_tl2','ue_tl2'],
                         'TL3':['tl4_tl3','tl1_tl3','lw_tl3','ls_tl3'],
                         'TL4':['le_tl4','rs_tl4','tl2_tl4','tl3_tl4']} 

        for car_id in car_list:
            edge_id = traci.vehicle.getRoadID(car_id)
            if edge_id in incoming_roads[TL]:
                lane_id = traci.vehicle.getLaneID(car_id)
                lane_pos = traci.vehicle.getLanePosition(car_id)
                lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road
                # distance in meters from the traffic light -> mapping into cells
                if lane_pos < 7:
                    lane_cell = 0
                elif lane_pos < 14:
                    lane_cell = 1
                elif lane_pos < 21:
                    lane_cell = 2
                elif lane_pos < 28:
                    lane_cell = 3
                elif lane_pos < 40:
                    lane_cell = 4
                elif lane_pos < 60:
                    lane_cell = 5
                elif lane_pos < 100:
                    lane_cell = 6
                elif lane_pos < 160:
                    lane_cell = 7
                elif lane_pos < 400:
                    lane_cell = 8
                elif lane_pos <= 750:
                    lane_cell = 9

        
                # finding the lane where the car is located 
                # x2TL_3 are the "turn left only" lanes
                if lane_id in lane_groups[0]:
                    lane_group = 0
                elif lane_id in lane_groups[1]:
                    lane_group = 1
                elif lane_id in lane_groups[2]:
                    lane_group = 2
                elif lane_id in lane_groups[3]:
                    lane_group = 3
                elif lane_id in lane_groups[4] :
                    lane_group = 4
                elif lane_id in lane_groups[5]:
                    lane_group = 5
                elif lane_id in lane_groups[6]:
                    lane_group = 6
                elif lane_id in lane_groups[7]: 
                    lane_group = 7
                else:
                    lane_group = -1
                #state: is an array 0 --> 79 if cell = 4 and lane group 7 --> 74
                #lane: 0 --> 7 means that the car in incoming lanes with respect to TL if not range means 
                if lane_group >= 1 and lane_group <= 7:
                    car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                    valid_car = True
                elif lane_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

                if valid_car:
                    state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state


    def _replay(self):
    
        #Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        for TL in self._TL_list:
            batch = self._Memories[TL].get_samples(self._Models[TL].batch_size)
            if len(batch) > 0:  # if the memory is full enough
                states = np.array([val[0] for val in batch])  # extract states from the batch
                next_states = np.array([val[3] for val in batch])  # extract next states from the batch

                # prediction
                q_s_a = self._Models[TL].predict_batch(states)  # predict Q(state), for every sample
                q_s_a_d = self._Models[TL].predict_batch(next_states)  # predict Q(next_state), for every sample

                # setup training arrays
                x = np.zeros((len(batch), self._num_states))
                y = np.zeros((len(batch), self._num_actions))

                # DQN State,Q-Value--> how corrent is the action --> table {State0: Q-Value(action1),Q-Value(actoin2),...}
                #in DQN --> NN calc. Q-Value  
                for i, b in enumerate(batch): # i--> index of sample, b --> i-th sample , sample --> (old state, old action, reward, current_state[TL])
                    state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                    current_q = q_s_a[i]  # get the Q(state) predicted before
                    current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                    x[i] = state
                    y[i] = current_q  # Q(state) that includes the updated action value

            self._Models[TL].train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        for TL in self._TL_list:
            self._reward_store[TL].append(self._sum_neg_reward[TL])  # how much negative reward in this episode
            self._cumulative_wait_store[TL].append(self._sum_waiting_time[TL])  # total number of seconds waited by cars in this episode
            self._avg_queue_length_store[TL].append(self._sum_queue_length[TL] / self._max_steps)  # average number of queued cars per step, in this episode
            self._cumulative_wait_store['all'].append(self._sum_waiting_time[TL])
            self._avg_queue_length_store['all'].append(self._sum_queue_length[TL] / self._max_steps)


    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

