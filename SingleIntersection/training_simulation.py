from numpy.core.numeric import NaN
from numpy.lib.arraysetops import intersect1d
from numpy.lib.function_base import average
import pandas as pd
from libsumo.libsumo import edge_getIDList, vehicle
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
N_Straight = ['N2TL_0', 'N2TL_1', 'N2TL_2']
N_left = ['N2TL_3']
S_Straight = ['S2TL_0', 'S2TL_1', 'S2TL_2']
S_left = ['S2TL_3']
W_Straight = ['W2TL_0', 'W2TL_1', 'W2TL_2']
W_left = ['W2TL_3']
E_Straight = ['E2TL_0', 'E2TL_1', 'E2TL_2']
E_left = ['E2TL_3']

class Simulation:
    def __init__(self, Model,st_model,Memory, st_meomry,TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._st_model=st_model
        self._Memory = Memory
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
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs


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
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait
            
            # saving the data into the memory
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            
            # TODO: Call Method to Evaluate Greenlight Time
            greenlight_durations = self.get_green_duration(action=action)
            greenlight_durations =[x for x in greenlight_durations if x>0 and x<= 30]
            print(' greenlight_durations: ', greenlight_durations)
            if len(greenlight_durations) > 0:
                greenlight_duration = math.ceil(min(greenlight_durations))
                print(' green_duration: ',greenlight_duration )
                self._simulate(greenlight_duration)
            else:
                self._simulate(self._green_duration)
            
            #self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

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


    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        #5400 if todo = 5 and step = 5397 --> 5402 --> todo = 3 
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            #print(' - Step:', self._step)
            traci.simulationStep()  # simulate 1 step in sumo

            # saving traffic features to st_memory                
            north_speed = traci.edge.getLastStepMeanSpeed('N2TL')
            north_count = traci.edge.getLastStepVehicleNumber('N2TL')

            south_speed = traci.edge.getLastStepMeanSpeed('S2TL')
            south_count = traci.edge.getLastStepVehicleNumber('S2TL')

            west_speed = traci.edge.getLastStepMeanSpeed('W2TL')
            west_count = traci.edge.getLastStepVehicleNumber('W2TL')

            east_speed = traci.edge.getLastStepMeanSpeed('E2TL')
            east_count = traci.edge.getLastStepVehicleNumber('E2TL')
            
            sample_dict = pd.DataFrame(columns=['east_speed','east_count',
                                                'west_speed','west_count',
                                                'north_speed','north_count',
                                                'south_speed','south_count'])
            sample_dict = sample_dict.append({'east_speed': west_speed,'east_count': east_count,
                                              'west_speed': east_speed,'west_count': west_count,
                                              'north_speed': north_speed,'north_count': north_count,
                                              'south_speed': south_speed,'south_count': south_count},
                                              ignore_index=True)                                                
            
            sample = sample_dict.to_numpy()
            self._st_meomry.add_sample(sample)

            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state

    #TODO: Write method to get the Greenlight Time
    def get_green_duration(self,action): 
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
            N_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in N_Straight])/3
            N_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in N_Straight])
            if N_vehicle_count != 0:
                N_single_car_time = -N_avg_speed+ math.sqrt((N_avg_speed*N_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(N_single_car_time * N_vehicle_count)
            
            ### S2TL Duration ###
            S_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in S_Straight])/3
            S_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in S_Straight])
            if S_vehicle_count != 0:
                S_single_car_time = -S_avg_speed + math.sqrt((S_avg_speed*S_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(S_single_car_time * S_vehicle_count)

        elif action == 1:
            intersection_length = 29.67
            ### N2TL Duration ###
            N_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in N_left])
            N_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in N_left])
            if N_vehicle_count != 0:
                N_single_car_time = -N_avg_speed + math.sqrt((N_avg_speed*N_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(N_single_car_time * N_vehicle_count)

            ### S2TL Duration ###
            S_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in S_left])
            S_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in S_left])
            if S_vehicle_count != 0:    
                S_single_car_time = -S_avg_speed + math.sqrt((S_avg_speed*S_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(S_single_car_time * S_vehicle_count)

        elif action == 2:
            intersection_length = 33.60
            ### W2TL Duration ###
            W_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in W_Straight])/3
            W_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in W_Straight])
            if W_vehicle_count != 0:
                W_single_car_time = -W_avg_speed + math.sqrt((W_avg_speed*W_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(W_single_car_time * W_vehicle_count)

            ### E2TL Duration ###
            E_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in E_Straight])/3
            E_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in E_Straight])
            if E_vehicle_count != 0:
                E_single_car_time = -E_avg_speed + math.sqrt((E_avg_speed*E_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(E_single_car_time * E_vehicle_count)

        elif action == 3:
            intersection_length = 29.67
            ### W2TL Duration ###
            W_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in W_left])
            W_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in W_left])
            if W_vehicle_count != 0:
                W_single_car_time = -W_avg_speed + math.sqrt((W_avg_speed*W_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(W_single_car_time * W_vehicle_count)

            ### E2TL Duration ###
            E_avg_speed = sum([traci.lane.getLastStepMeanSpeed(lane) for lane in E_left])
            E_vehicle_count = sum([traci.lane.getLastStepVehicleNumber(lane) for lane in E_left])
            if E_vehicle_count != 0:
                E_single_car_time = -E_avg_speed+ math.sqrt((E_avg_speed*E_avg_speed) - (4*.5*-intersection_length))
                green_duration.append(E_single_car_time * E_vehicle_count)
        ##### End Get Current Demand #####

        ##### Get Future Demand #####
        
        st_model_input =  self._st_meomry.get_samples()
        #print('st_samples: ',st_model_input)
        #print('st_samples_size: ',len(st_model_input))
        
        if len(st_model_input) == 0 :
            return green_duration
        #print('     predicting traffic...')
        st_model_output =  self._st_model.predict_one(st_model_input[:,:,:])
        future_traffic = pd.DataFrame(st_model_output[0,0,:,:], columns=[ 'east_speed','east_count',
                                                                        'west_speed','west_count',
                                                                        'north_speed','north_count',
                                                                        'south_speed','south_count'])
        #print('     future_traffic:\n', future_traffic)
        if action == 0:
            intersection_length = 33.60
            #N_single_car_time = intersection_length/future_traffic.iloc[0]['north_speed']
            N_single_car_time = (-future_traffic.iloc[0]['north_speed'] 
                                + math.sqrt((future_traffic.iloc[0]['north_speed']*future_traffic.iloc[0]['north_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[0]['north_count']  > 0 and not  math.isnan(future_traffic.iloc[0]['north_count']): 
                green_duration.append(N_single_car_time* future_traffic.iloc[0]['north_count'] * 3/4)

            #S_single_car_time = intersection_length/future_traffic.iloc[0]['south_speed']
            S_single_car_time = (-future_traffic.iloc[0]['south_speed'] 
                                + math.sqrt((future_traffic.iloc[0]['south_speed']*future_traffic.iloc[0]['south_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[0]['south_count']  > 0 and not  math.isnan(future_traffic.iloc[0]['south_count']):
                green_duration.append(S_single_car_time* future_traffic.iloc[0]['south_count'] * 3/4)

        elif action == 1:
            intersection_length = 29.67
            #N_single_car_time = intersection_length/future_traffic.iloc[0]['north_speed']
            N_single_car_time = (-future_traffic.iloc[0]['north_speed'] 
                                + math.sqrt((future_traffic.iloc[0]['north_speed']*future_traffic.iloc[0]['north_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[0]['north_count']  > 0 and not  math.isnan(future_traffic.iloc[0]['north_count']): 
                green_duration.append(N_single_car_time* future_traffic.iloc[0]['north_count'] * 1/4)

            #S_single_car_time = intersection_length/future_traffic.iloc[0]['south_speed']
            S_single_car_time = (-future_traffic.iloc[0]['south_speed'] 
                                + math.sqrt((future_traffic.iloc[0]['south_speed']*future_traffic.iloc[0]['south_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[0]['south_count']  > 0 and not math.isnan(future_traffic.iloc[0]['south_count']):
                green_duration.append(S_single_car_time* future_traffic.iloc[0]['south_count'] * 1/4)

        elif action == 2:
            intersection_length = 33.60
            #W_single_car_time = intersection_length/future_traffic.iloc[0]['west_speed']
            W_single_car_time = (-future_traffic.iloc[0]['west_speed'] 
                                + math.sqrt((future_traffic.iloc[0]['west_speed']*future_traffic.iloc[0]['west_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[0]['west_count']  > 0 and not math.isnan(future_traffic.iloc[0]['west_count']):
                green_duration.append(W_single_car_time* future_traffic.iloc[0]['west_count'] * 3/4)

            #E_single_car_time = intersection_length/future_traffic.iloc[0]['east_speed']
            E_single_car_time = (-future_traffic.iloc[0]['east_speed'] 
                                + math.sqrt((future_traffic.iloc[0]['east_speed']*future_traffic.iloc[0]['east_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[0]['east_count']  > 0 and not math.isnan(future_traffic.iloc[0]['east_count']):    
                green_duration.append(E_single_car_time* future_traffic.iloc[0]['east_count'] * 3/4)

        elif action == 3:
            intersection_length = 29.67
            #W_single_car_time = intersection_length/future_traffic.iloc[0]['west_speed']
            W_single_car_time = (-future_traffic.iloc[0]['west_speed'] 
                                + math.sqrt((future_traffic.iloc[0]['west_speed']*future_traffic.iloc[0]['west_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[0]['west_count']  > 0 and not math.isnan(future_traffic.iloc[0]['west_count']):    
                green_duration.append(W_single_car_time* math.ceil(future_traffic.iloc[0]['west_count']) * 1/4)

            #E_single_car_time = intersection_length/future_traffic.iloc[0]['east_speed']
            E_single_car_time = (-future_traffic.iloc[0]['east_speed'] 
                                + math.sqrt((future_traffic.iloc[0]['east_speed']*future_traffic.iloc[0]['east_speed']) 
                                            - (4*.5*-intersection_length)))
            if future_traffic.iloc[0]['east_count']  > 0 and not math.isnan(future_traffic.iloc[0]['east_count']):     
                green_duration.append(E_single_car_time* math.ceil(future_traffic.iloc[0]['east_count']) * 1/4)
        #print('     green_duration:', green_duration)
        return green_duration

    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length


    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
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
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
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
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            # DQN State,Q-Value--> how corrent is the action --> table {State0: Q-Value(action1),Q-Value(actoin2),...}
            #in DQN --> NN calc. Q-Value  
            for i, b in enumerate(batch): # i--> index of sample, b --> i-th sample , sample --> (old state, old action, reward, current_state)
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode


    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

