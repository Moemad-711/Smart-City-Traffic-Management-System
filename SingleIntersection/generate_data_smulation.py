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

      
    def run(self, episode):
       
        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        
        print("Simulating...")
        file_name ='./TrafficData/data'+str(episode)+'.xml'
        
        with open(os.path.join(file_name),"w") as data:
            print("""<fcd-export xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/fcd_file.xsd">""",file=data)

            while self._step < self._max_steps:
                traci.simulationStep()
                print('<timestep time="%d">' % (self._step),file=data)
                car_list = traci.vehicle.getIDList()
                for car_id in car_list:
                    #print(car_id)
                    x,y = traci.vehicle.getPosition(car_id)
                    angle=traci.vehicle.getAngle(car_id)                  
                    vehicle_type=traci.vehicle.getTypeID(car_id)                  
                    speed=traci.vehicle.getSpeed(car_id)
                    lane_pos=traci.vehicle.getLanePosition(car_id)                   
                    edge=traci.vehicle.getRoadID(car_id)
                    lane_id=traci.vehicle.getLaneID(car_id)
                    print('    <vehicle id="%s" x="%d" y="%d" angle="%d" type="%s" speed="%d" pos="%d" edge="%s" lane="%s" />' % (car_id, x, y, angle, vehicle_type, speed, lane_pos, edge, lane_id), file=data)
                print('</timestep>',file=data)
                self._step += 1
            print('</fcd-export>',file=data)   
        
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
   
   

        return simulation_time
