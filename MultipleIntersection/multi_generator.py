import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps
    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />
            <route id="UW_LN" edges="uw_tl1 tl1_ln"/>
            <route id="UW_RN" edges="uw_tl1 tl1_tl2 tl2_rn"/>
            <route id="UW_UE" edges="uw_tl1 tl1_tl2 tl2_ue"/>
            <route id="UW_LE0" edges="uw_tl1 tl1_tl2 tl2_tl4 tl4_le"/>
            <route id="UW_LE1" edges="uw_tl1 tl1_tl3 tl3_tl4 tl4_le"/>
            <route id="UW_RS0" edges="uw_tl1 tl1_tl2 tl2_tl4 tl4_rs"/>
            <route id="UW_RS1" edges="uw_tl1 tl1_tl3 tl3_tl4 tl4_rs"/>
            <route id="UW_LS" edges="uw_tl1 tl1_tl3 tl3_ls"/>
            <route id="UW_LW" edges="uw_tl1 tl1_tl3 tl3_lw"/>
            <route id="LW_LS" edges="lw_tl3 tl3_ls"/>
            <route id="LW_RS" edges="lw_tl3 tl3_tl4 tl4_rs"/>
            <route id="LW_LE" edges="lw_tl3 tl3_tl4 tl4_le"/>
            <route id="LW_UE0" edges="lw_tl3 tl3_tl4 tl4_tl2 tl2_ue"/>
            <route id="LW_UE1" edges="lw_tl3 tl3_tl1 tl1_tl2 tl2_ue"/>
            <route id="LW_LN0" edges="lw_tl3 tl3_tl4 tl4_tl2 tl2_ln"/>
            <route id="LW_LN1" edges="lw_tl3 tl3_tl1 tl1_tl2 tl2_ln"/>
            <route id="LW_RN" edges="lw_tl3 tl3_tl1 tl1_rn"/>
            <route id="LW_UW" edges="lw_tl3 tl3_tl1 tl1_uw"/>
            <route id="LS_RS" edges="ls_tl3 tl3_tl4 tl4_rs"/>
            <route id="LS_LE" edges="ls_tl3 tl3_tl4 tl4_le"/>
            <route id="LS_UE0" edges="ls_tl3 tl3_tl4 tl4_tl2 tl2_ue"/>
            <route id="LS_UE1" edges="ls_tl3 tl3_tl1 tl1_tl2 tl2_ue"/>
            <route id="LS_RN0" edges="ls_tl3 tl3_tl4 tl4_tl2 tl2_rn"/>
            <route id="LS_RN1" edges="ls_tl3 tl3_tl1 tl1_tl2 tl2_rn"/>
            <route id="LS_LN" edges="ls_tl3 tl3_tl1 tl1_ln"/>
            <route id="LS_UW" edges="ls_tl3 tl3_tl1 tl1_uw"/>
            <route id="LS_LW" edges="ls_tl3 tl3_lw"/>
            <route id="RS_LE" edges="rs_tl4 tl4_le"/>
            <route id="RS_UE" edges="rs_tl4 tl4_tl2 tl2_ue"/>
            <route id="RS_RN" edges="rs_tl4 tl4_tl2 tl2_rn"/>
            <route id="RS_LN0" edges="rs_tl4 tl4_tl2 tl2_tl1 tl1_ln"/>
            <route id="RS_LN1" edges="rs_tl4 tl4_tl3 tl3_tl1 tl1_ln"/>
            <route id="RS_UW0" edges="rs_tl4 tl4_tl2 tl2_tl1 tl1_uw"/>
            <route id="RS_UW1" edges="rs_tl4 tl4_tl3 tl3_tl1 tl1_um"/>
            <route id="RS_LW" edges="rs_tl4 tl4_tl3 tl3_lw"/>
            <route id="RS_LS" edges="rs_tl4 tl4_tl3 tl3_ls"/>
            <route id="LE_UE" edges="le_tl4 tl4_tl2 tl3_ue"/>
            <route id="LE_RN" edges="le_tl4 tl4_tl2 tl2_rn"/>
            <route id="LE_LN0" edges="le_tl4 tl4_tl2 tl2_tl1 tl1_ln"/>
            <route id="LE_LN1" edges="le_tl4 tl4_tl3 tl3_tl1 tl1_ln"/>
            <route id="LE_UW0" edges="le_tl4 tl4_tl2 tl2_tl1 tl1_uw"/>
            <route id="LE_UW1" edges="le_tl4 tl4_tl3 tl3_tl1 tl1_uw"/>
            <route id="LE_LE" edges="le_tl4 tl4_tl3 tl3_LE"/>
            <route id="LE_LS" edges="le_tl4 tl4_tl3 tl3_ls"/>
            <route id="LE_RS" edges="le_tl4 tl4_RS"/>
            <route id="UE_RN" edges="ue_tl2 tl2_rn"/>
            <route id="UE_LN" edges="ue_tl2 tl2_tl1 tl1_rn"/>
            <route id="UE_UW" edges="ue_tl2 tl2_tl1 tl1_uw"/>
            <route id="UE_LW0" edges="ue_tl2 tl2_tl1 tl1_tl3 tl3_lw"/>
            <route id="UE_LW1" edges="ue_tl2 tl2_tl4 tl4_tl3 tl3_lw"/>
            <route id="UE_LS0" edges="ue_tl2 tl2_tl1 tl1_tl3 tl3_ls"/>
            <route id="UE_LS1" edges="ue_tl2 tl2_tl4 tl4_tl3 tl3_ls"/>
            <route id="UE_RS" edges="ue_tl2 tl2_tl4 tl4_rs"/>
            <route id="UE_LE" edges="ue_tl2 tl2_tl4 tl4_le"/>
            <route id="RN_LN" edges="rn_tl2 tl2_tl1 tl1_ln"/>
            <route id="RN_UW" edges="rn_tl2 tl2_tl1 tl1_uw"/>
            <route id="RN_LW0" edges="rn_tl2 tl2_tl1 tl1_tl3 tl3_lw"/>
            <route id="RN_LW1" edges="rn_tl2 tl2_tl4 tl4_tl3 tl3_lw"/>
            <route id="RN_LS0" edges="rn_tl2 tl2_tl1 tl1_tl3 tl3_ls"/>
            <route id="RN_LS1" edges="rn_tl2 tl2_tl4 tl4_tl3 tl3_ls"/>
            <route id="RN_rs" edges="rn_tl2 tl2_tl4 tl4_rs"/>
            <route id="RN_LE" edges="rn_tl2 tl2_tl4 tl4_le"/>
            <route id="RN_UE" edges="rn_tl2 tl2_ue"/>
            <route id="LN_UW" edges="ln_tl1 tl1_uw"/>
            <route id="LN_LW" edges="ln_tl1 tl1_tl3 tl3_lw"/>
            <route id="LN_LS" edges="ln_tl1 tl1_tl3 tl3_ls"/>
            <route id="LN_RS0" edges="ln_tl1 tl1_tl3 tl3_tl4 tl4_rs"/>
            <route id="LN_RS1" edges="ln_tl1 tl1_tl2 tl2_tl4 tl4_rs"/>
            <route id="LN_LE0" edges="ln_tl1 tl1_tl3 tl3_tl4 tl4_le"/>
            <route id="LN_LE1" edges="ln_tl1 tl1_tl2 tl2_tl4 tl4_le"/>
            <route id="LN_UE" edges="ln_tl1 tl1_tl2 tl2_ue"/>
            <route id="LN_RN" edges="ln_tl1 tl1_tl2 tl2_rn"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 41)  # choose a random source & destination
                     if route_straight == 1:
                        print('    <vehicle id="UW_RN_%i" type="standard_car" route="UW_RN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="UW_UE_%i" type="standard_car" route="UW_UE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="UW_LS_%i" type="standard_car" route="UW_LS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 4:
                        print('    <vehicle id="UW_LW_%i" type="standard_car" route="UW_LW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 5:
                        print('    <vehicle id="UW_LN_%i" type="standard_car" route="UW_LN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 6:
                        print('    <vehicle id="LW_RS_%i" type="standard_car" route="LW_RS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 7:
                        print('    <vehicle id="LW_LE_%i" type="standard_car" route="LW_LE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 8:
                        print('    <vehicle id="LW_RN_%i" type="standard_car" route="LW_RN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 9:
                        print('    <vehicle id="LW_UW_%i" type="standard_car" route="LW_UW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 10:
                        print('    <vehicle id="LW_LS_%i" type="standard_car" route="LW_LS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 11:
                        print('    <vehicle id="LS_RS_%i" type="standard_car" route="LS_RS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 12:
                        print('    <vehicle id="LS_LE_%i" type="standard_car" route="LS_LE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 13:
                        print('    <vehicle id="LS_LN_%i" type="standard_car" route="LS_LN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 14:
                        print('    <vehicle id="LS_UW_%i" type="standard_car" route="LS_UW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 15:
                        print('    <vehicle id="LS_LW_%i" type="standard_car" route="LS_LW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 16:
                        print('    <vehicle id="RS_UE_%i" type="standard_car" route="RS_UE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 17:
                        print('    <vehicle id="RS_RN_%i" type="standard_car" route="RS_RN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 18:
                        print('    <vehicle id="RS_LW_%i" type="standard_car" route="RS_LW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 19:
                        print('    <vehicle id="RS_LS_%i" type="standard_car" route="RS_LS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 20:
                        print('    <vehicle id="RS_LE_%i" type="standard_car" route="RS_LE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 21:
                        print('    <vehicle id="LE_UE_%i" type="standard_car" route="LE_UE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 22:
                        print('    <vehicle id="LE_RN_%i" type="standard_car" route="LE_RN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 23:
                        print('    <vehicle id="LE_LE_%i" type="standard_car" route="LE_LE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 24:
                        print('    <vehicle id="LE_LS_%i" type="standard_car" route="LE_LS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 25:
                        print('    <vehicle id="LE_RS_%i" type="standard_car" route="LE_RS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 26:
                        print('    <vehicle id="UE_LN_%i" type="standard_car" route="UE_LN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 27:
                        print('    <vehicle id="UE_UW_%i" type="standard_car" route="UE_UW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 28:
                        print('    <vehicle id="UE_RS_%i" type="standard_car" route="UE_RS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 29:
                        print('    <vehicle id="UE_LE_%i" type="standard_car" route="UE_LE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 30:
                        print('    <vehicle id="UE_RN_%i" type="standard_car" route="UE_RN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 31:
                        print('    <vehicle id="RN_LN_%i" type="standard_car" route="RN_LN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 32:
                        print('    <vehicle id="RN_UW_%i" type="standard_car" route="RN_UW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 33:
                        print('    <vehicle id="RN_rs_%i" type="standard_car" route="RN_rs" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 34:
                        print('    <vehicle id="RN_LE_%i" type="standard_car" route="RN_LE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 35:
                        print('    <vehicle id="RN_UE_%i" type="standard_car" route="RN_UE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 36:
                        print('    <vehicle id="LN_LW_%i" type="standard_car" route="LN_LW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 37:
                        print('    <vehicle id="LN_LS_%i" type="standard_car" route="LN_LS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 38:
                        print('    <vehicle id="LN_UE_%i" type="standard_car" route="LN_UE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 39:
                        print('    <vehicle id="LN_RN_%i" type="standard_car" route="LN_RN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else: 
                        print('    <vehicle id="LN_UW_%i" type="standard_car" route="LN_UW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
       
          else:  # car that turn -25% of the time the car turns
                    route_turn = np.random.randint(1, 33)  # choose random source source & destination
                    if route_turn == 1:
                   print('    <vehicle id="UW_LE0_%i" type="standard_car" route="UW_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                   print('    <vehicle id="UW_LE1_%i" type="standard_car" route="UW_LE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 3:
                   print('    <vehicle id="UW_RS0_%i" type="standard_car" route="UW_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 4:
                   print('    <vehicle id="UW_RS1_%i" type="standard_car" route="UW_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 5:
                   print('    <vehicle id="LW_UE0_%i" type="standard_car" route="LW_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 6:
                   print('    <vehicle id="LW_UE1_%i" type="standard_car" route="LW_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 7:
                   print('    <vehicle id="LW_LN0_%i" type="standard_car" route="LW_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 8:
                   print('    <vehicle id="LW_LN1_%i" type="standard_car" route="LW_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 9:
                   print('    <vehicle id="LS_UE0_%i" type="standard_car" route="LS_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 10:
                   print('    <vehicle id="LS_UE1_%i" type="standard_car" route="LS_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 11:
                   print('    <vehicle id="LS_RN0_%i" type="standard_car" route="LS_RN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 12:
                   print('    <vehicle id="LS_RN1_%i" type="standard_car" route="LS_RN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 13:
                   print('    <vehicle id="RS_LN0_%i" type="standard_car" route="RS_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 14:
                   print('    <vehicle id="RS_LN1_%i" type="standard_car" route="RS_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 15:
                   print('    <vehicle id="RS_UW0_%i" type="standard_car" route="RS_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 16:
                   print('    <vehicle id="RS_UW1_%i" type="standard_car" route="RS_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 17:
                   print('    <vehicle id="LE_LN0_%i" type="standard_car" route="LE_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 18:
                   print('    <vehicle id="LE_LN1_%i" type="standard_car" route="LE_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 19:
                   print('    <vehicle id="LE_UW0_%i" type="standard_car" route="LE_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 20:
                   print('    <vehicle id="LE_UW1_%i" type="standard_car" route="LE_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 21:
                   print('    <vehicle id="UE_LW0_%i" type="standard_car" route="UE_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 22:
                   print('    <vehicle id="UE_LW1_%i" type="standard_car" route="UE_LW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 23:
                   print('    <vehicle id="UE_LS0_%i" type="standard_car" route="UE_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 24:
                   print('    <vehicle id="UE_LS1_%i" type="standard_car" route="UE_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 25:
                   print('    <vehicle id="RN_LW0_%i" type="standard_car" route="RN_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 26:
                   print('    <vehicle id="RN_LW1_%i" type="standard_car" route="RN_LW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 27:
                   print('    <vehicle id="RN_LS0_%i" type="standard_car" route="RN_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 28:
                   print('    <vehicle id="RN_LS1_%i" type="standard_car" route="RN_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 29:
                   print('    <vehicle id="LN_RS0_%i" type="standard_car" route="LN_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 30:
                   print('    <vehicle id="LN_RS1_%i" type="standard_car" route="LN_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 31:
                   print('    <vehicle id="LN_LE0_%i" type="standard_car" route="LN_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 32:
                   print('    <vehicle id="LN_LE1_%i" type="standard_car" route="" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                                                               
            print("</routes>", file=routes)

    
    def generate_routefile_mpi(self, seed, rank):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/routes/episode_routes_%i.rou.xml" %(rank), "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />
            <route id="UW_LN" edges="uw_tl1 tl1_ln"/>
            <route id="UW_RN" edges="uw_tl1 tl1_tl2 tl2_rn"/>
            <route id="UW_UE" edges="uw_tl1 tl1_tl2 tl2_ue"/>
            <route id="UW_LE0" edges="uw_tl1 tl1_tl2 tl2_tl4 tl4_le"/>
            <route id="UW_LE1" edges="uw_tl1 tl1_tl3 tl3_tl4 tl4_le"/>
            <route id="UW_RS0" edges="uw_tl1 tl1_tl2 tl2_tl4 tl4_rs"/>
            <route id="UW_RS1" edges="uw_tl1 tl1_tl3 tl3_tl4 tl4_rs"/>
            <route id="UW_LS" edges="uw_tl1 tl1_tl3 tl3_ls"/>
            <route id="UW_LW" edges="uw_tl1 tl1_tl3 tl3_lw"/>
            <route id="LW_LS" edges="lw_tl3 tl3_ls"/>
            <route id="LW_RS" edges="lw_tl3 tl3_tl4 tl4_rs"/>
            <route id="LW_LE" edges="lw_tl3 tl3_tl4 tl4_le"/>
            <route id="LW_UE0" edges="lw_tl3 tl3_tl4 tl4_tl2 tl2_ue"/>
            <route id="LW_UE1" edges="lw_tl3 tl3_tl1 tl1_tl2 tl2_ue"/>
            <route id="LW_LN0" edges="lw_tl3 tl3_tl4 tl4_tl2 tl2_ln"/>
            <route id="LW_LN1" edges="lw_tl3 tl3_tl1 tl1_tl2 tl2_ln"/>
            <route id="LW_RN" edges="lw_tl3 tl3_tl1 tl1_rn"/>
            <route id="LW_UW" edges="lw_tl3 tl3_tl1 tl1_uw"/>
            <route id="LS_RS" edges="ls_tl3 tl3_tl4 tl4_rs"/>
            <route id="LS_LE" edges="ls_tl3 tl3_tl4 tl4_le"/>
            <route id="LS_UE0" edges="ls_tl3 tl3_tl4 tl4_tl2 tl2_ue"/>
            <route id="LS_UE1" edges="ls_tl3 tl3_tl1 tl1_tl2 tl2_ue"/>
            <route id="LS_RN0" edges="ls_tl3 tl3_tl4 tl4_tl2 tl2_rn"/>
            <route id="LS_RN1" edges="ls_tl3 tl3_tl1 tl1_tl2 tl2_rn"/>
            <route id="LS_LN" edges="ls_tl3 tl3_tl1 tl1_ln"/>
            <route id="LS_UW" edges="ls_tl3 tl3_tl1 tl1_uw"/>
            <route id="LS_LW" edges="ls_tl3 tl3_lw"/>
            <route id="RS_LE" edges="rs_tl4 tl4_le"/>
            <route id="RS_UE" edges="rs_tl4 tl4_tl2 tl2_ue"/>
            <route id="RS_RN" edges="rs_tl4 tl4_tl2 tl2_rn"/>
            <route id="RS_LN0" edges="rs_tl4 tl4_tl2 tl2_tl1 tl1_ln"/>
            <route id="RS_LN1" edges="rs_tl4 tl4_tl3 tl3_tl1 tl1_ln"/>
            <route id="RS_UW0" edges="rs_tl4 tl4_tl2 tl2_tl1 tl1_uw"/>
            <route id="RS_UW1" edges="rs_tl4 tl4_tl3 tl3_tl1 tl1_um"/>
            <route id="RS_LW" edges="rs_tl4 tl4_tl3 tl3_lw"/>
            <route id="RS_LS" edges="rs_tl4 tl4_tl3 tl3_ls"/>
            <route id="LE_UE" edges="le_tl4 tl4_tl2 tl3_ue"/>
            <route id="LE_RN" edges="le_tl4 tl4_tl2 tl2_rn"/>
            <route id="LE_LN0" edges="le_tl4 tl4_tl2 tl2_tl1 tl1_ln"/>
            <route id="LE_LN1" edges="le_tl4 tl4_tl3 tl3_tl1 tl1_ln"/>
            <route id="LE_UW0" edges="le_tl4 tl4_tl2 tl2_tl1 tl1_uw"/>
            <route id="LE_UW1" edges="le_tl4 tl4_tl3 tl3_tl1 tl1_uw"/>
            <route id="LE_LE" edges="le_tl4 tl4_tl3 tl3_LE"/>
            <route id="LE_LS" edges="le_tl4 tl4_tl3 tl3_ls"/>
            <route id="LE_RS" edges="le_tl4 tl4_RS"/>
            <route id="UE_RN" edges="ue_tl2 tl2_rn"/>
            <route id="UE_LN" edges="ue_tl2 tl2_tl1 tl1_rn"/>
            <route id="UE_UW" edges="ue_tl2 tl2_tl1 tl1_uw"/>
            <route id="UE_LW0" edges="ue_tl2 tl2_tl1 tl1_tl3 tl3_lw"/>
            <route id="UE_LW1" edges="ue_tl2 tl2_tl4 tl4_tl3 tl3_lw"/>
            <route id="UE_LS0" edges="ue_tl2 tl2_tl1 tl1_tl3 tl3_ls"/>
            <route id="UE_LS1" edges="ue_tl2 tl2_tl4 tl4_tl3 tl3_ls"/>
            <route id="UE_RS" edges="ue_tl2 tl2_tl4 tl4_rs"/>
            <route id="UE_LE" edges="ue_tl2 tl2_tl4 tl4_le"/>
            <route id="RN_LN" edges="rn_tl2 tl2_tl1 tl1_ln"/>
            <route id="RN_UW" edges="rn_tl2 tl2_tl1 tl1_uw"/>
            <route id="RN_LW0" edges="rn_tl2 tl2_tl1 tl1_tl3 tl3_lw"/>
            <route id="RN_LW1" edges="rn_tl2 tl2_tl4 tl4_tl3 tl3_lw"/>
            <route id="RN_LS0" edges="rn_tl2 tl2_tl1 tl1_tl3 tl3_ls"/>
            <route id="RN_LS1" edges="rn_tl2 tl2_tl4 tl4_tl3 tl3_ls"/>
            <route id="RN_rs" edges="rn_tl2 tl2_tl4 tl4_rs"/>
            <route id="RN_LE" edges="rn_tl2 tl2_tl4 tl4_le"/>
            <route id="RN_UE" edges="rn_tl2 tl2_ue"/>
            <route id="LN_UW" edges="ln_tl1 tl1_uw"/>
            <route id="LN_LW" edges="ln_tl1 tl1_tl3 tl3_lw"/>
            <route id="LN_LS" edges="ln_tl1 tl1_tl3 tl3_ls"/>
            <route id="LN_RS0" edges="ln_tl1 tl1_tl3 tl3_tl4 tl4_rs"/>
            <route id="LN_RS1" edges="ln_tl1 tl1_tl2 tl2_tl4 tl4_rs"/>
            <route id="LN_LE0" edges="ln_tl1 tl1_tl3 tl3_tl4 tl4_le"/>
            <route id="LN_LE1" edges="ln_tl1 tl1_tl2 tl2_tl4 tl4_le"/>
            <route id="LN_UE" edges="ln_tl1 tl1_tl2 tl2_ue"/>
            <route id="LN_RN" edges="ln_tl1 tl1_tl2 tl2_rn"/>           
         """, file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 41)  # choose a random source & destination
                    if route_straight == 1:
                        print('    <vehicle id="UW_RN_%i" type="standard_car" route="UW_RN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="UW_UE_%i" type="standard_car" route="UW_UE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="UW_LS_%i" type="standard_car" route="UW_LS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 4:
                        print('    <vehicle id="UW_LW_%i" type="standard_car" route="UW_LW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 5:
                        print('    <vehicle id="UW_LN_%i" type="standard_car" route="UW_LN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 6:
                        print('    <vehicle id="LW_RS_%i" type="standard_car" route="LW_RS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 7:
                        print('    <vehicle id="LW_LE_%i" type="standard_car" route="LW_LE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 8:
                        print('    <vehicle id="LW_RN_%i" type="standard_car" route="LW_RN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 9:
                        print('    <vehicle id="LW_UW_%i" type="standard_car" route="LW_UW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 10:
                        print('    <vehicle id="LW_LS_%i" type="standard_car" route="LW_LS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 11:
                        print('    <vehicle id="LS_RS_%i" type="standard_car" route="LS_RS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 12:
                        print('    <vehicle id="LS_LE_%i" type="standard_car" route="LS_LE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 13:
                        print('    <vehicle id="LS_LN_%i" type="standard_car" route="LS_LN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 14:
                        print('    <vehicle id="LS_UW_%i" type="standard_car" route="LS_UW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 15:
                        print('    <vehicle id="LS_LW_%i" type="standard_car" route="LS_LW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 16:
                        print('    <vehicle id="RS_UE_%i" type="standard_car" route="RS_UE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 17:
                        print('    <vehicle id="RS_RN_%i" type="standard_car" route="RS_RN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 18:
                        print('    <vehicle id="RS_LW_%i" type="standard_car" route="RS_LW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 19:
                        print('    <vehicle id="RS_LS_%i" type="standard_car" route="RS_LS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 20:
                        print('    <vehicle id="RS_LE_%i" type="standard_car" route="RS_LE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 21:
                        print('    <vehicle id="LE_UE_%i" type="standard_car" route="LE_UE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 22:
                        print('    <vehicle id="LE_RN_%i" type="standard_car" route="LE_RN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 23:
                        print('    <vehicle id="LE_LE_%i" type="standard_car" route="LE_LE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 24:
                        print('    <vehicle id="LE_LS_%i" type="standard_car" route="LE_LS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 25:
                        print('    <vehicle id="LE_RS_%i" type="standard_car" route="LE_RS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 26:
                        print('    <vehicle id="UE_LN_%i" type="standard_car" route="UE_LN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 27:
                        print('    <vehicle id="UE_UW_%i" type="standard_car" route="UE_UW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 28:
                        print('    <vehicle id="UE_RS_%i" type="standard_car" route="UE_RS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 29:
                        print('    <vehicle id="UE_LE_%i" type="standard_car" route="UE_LE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 30:
                        print('    <vehicle id="UE_RN_%i" type="standard_car" route="UE_RN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 31:
                        print('    <vehicle id="RN_LN_%i" type="standard_car" route="RN_LN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 32:
                        print('    <vehicle id="RN_UW_%i" type="standard_car" route="RN_UW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 33:
                        print('    <vehicle id="RN_rs_%i" type="standard_car" route="RN_rs" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 34:
                        print('    <vehicle id="RN_LE_%i" type="standard_car" route="RN_LE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 35:
                        print('    <vehicle id="RN_UE_%i" type="standard_car" route="RN_UE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 36:
                        print('    <vehicle id="LN_LW_%i" type="standard_car" route="LN_LW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 37:
                        print('    <vehicle id="LN_LS_%i" type="standard_car" route="LN_LS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 38:
                        print('    <vehicle id="LN_UE_%i" type="standard_car" route="LN_UE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 39:
                        print('    <vehicle id="LN_RN_%i" type="standard_car" route="LN_RN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else: 
                        print('    <vehicle id="LN_UW_%i" type="standard_car" route="LN_UW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                        
                
                
                else:  # car that turn -25% of the time the car turns
                    route_turn = np.random.randint(1, 33)  # choose random source source & destination      
                    if route_turn == 1:
                   print('    <vehicle id="UW_LE0_%i" type="standard_car" route="UW_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                   print('    <vehicle id="UW_LE1_%i" type="standard_car" route="UW_LE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 3:
                   print('    <vehicle id="UW_RS0_%i" type="standard_car" route="UW_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 4:
                   print('    <vehicle id="UW_RS1_%i" type="standard_car" route="UW_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 5:
                   print('    <vehicle id="LW_UE0_%i" type="standard_car" route="LW_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 6:
                   print('    <vehicle id="LW_UE1_%i" type="standard_car" route="LW_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 7:
                   print('    <vehicle id="LW_LN0_%i" type="standard_car" route="LW_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 8:
                   print('    <vehicle id="LW_LN1_%i" type="standard_car" route="LW_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 9:
                   print('    <vehicle id="LS_UE0_%i" type="standard_car" route="LS_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 10:
                   print('    <vehicle id="LS_UE1_%i" type="standard_car" route="LS_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 11:
                   print('    <vehicle id="LS_RN0_%i" type="standard_car" route="LS_RN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 12:
                   print('    <vehicle id="LS_RN1_%i" type="standard_car" route="LS_RN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 13:
                   print('    <vehicle id="RS_LN0_%i" type="standard_car" route="RS_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 14:
                   print('    <vehicle id="RS_LN1_%i" type="standard_car" route="RS_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 15:
                   print('    <vehicle id="RS_UW0_%i" type="standard_car" route="RS_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 16:
                   print('    <vehicle id="RS_UW1_%i" type="standard_car" route="RS_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 17:
                   print('    <vehicle id="LE_LN0_%i" type="standard_car" route="LE_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 18:
                   print('    <vehicle id="LE_LN1_%i" type="standard_car" route="LE_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 19:
                   print('    <vehicle id="LE_UW0_%i" type="standard_car" route="LE_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 20:
                   print('    <vehicle id="LE_UW1_%i" type="standard_car" route="LE_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 21:
                   print('    <vehicle id="UE_LW0_%i" type="standard_car" route="UE_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 22:
                   print('    <vehicle id="UE_LW1_%i" type="standard_car" route="UE_LW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 23:
                   print('    <vehicle id="UE_LS0_%i" type="standard_car" route="UE_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 24:
                   print('    <vehicle id="UE_LS1_%i" type="standard_car" route="UE_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 25:
                   print('    <vehicle id="RN_LW0_%i" type="standard_car" route="RN_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 26:
                   print('    <vehicle id="RN_LW1_%i" type="standard_car" route="RN_LW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 27:
                   print('    <vehicle id="RN_LS0_%i" type="standard_car" route="RN_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 28:
                   print('    <vehicle id="RN_LS1_%i" type="standard_car" route="RN_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 29:
                   print('    <vehicle id="LN_RS0_%i" type="standard_car" route="LN_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 30:
                   print('    <vehicle id="LN_RS1_%i" type="standard_car" route="LN_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 31:
                   print('    <vehicle id="LN_LE0_%i" type="standard_car" route="LN_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 32:
                   print('    <vehicle id="LN_LE1_%i" type="standard_car" route="" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                                                               
            print("</routes>", file=routes)
        
        # Produce Sumocfg file for the route files
        with open("intersection/routes/sumo_config_%i.sumocfg" %(rank), "w") as cfgs:
            print("""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="..\\environment.net.xml"/>
        <route-files value="episode_routes_%i.rou.xml"/>
    </input>
    <time>
    <begin value="0"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>""" %(rank), file=cfgs)