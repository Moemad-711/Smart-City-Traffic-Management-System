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
            <route id="UW_RN0" edges="uw_tl1 tl1_tl2 tl2_rn"/>
            <route id="UW_RN1" edges="uw_tl1 tl1_tl3 tl3_tl4 tl4_tl2 tl2_rn"/>
            <route id="UW_UE0" edges="uw_tl1 tl1_tl2 tl2_ue"/>
            <route id="UW_UE1" edges="uw_tl1 tl1_tl3 tl3_tl4 tl4_tl2 tl2_ue"/>
            <route id="UW_LE0" edges="uw_tl1 tl1_tl2 tl2_tl4 tl4_le"/>
            <route id="UW_LE1" edges="uw_tl1 tl1_tl3 tl3_tl4 tl4_le"/>
            <route id="UW_RS0" edges="uw_tl1 tl1_tl2 tl2_tl4 tl4_rs"/>
            <route id="UW_RS1" edges="uw_tl1 tl1_tl3 tl3_tl4 tl4_rs"/>
            <route id="UW_LS0" edges="uw_tl1 tl1_tl3 tl3_ls"/>
            <route id="UW_LS1" edges="uw_tl1 tl1_tl2 tl2_tl4 tl4_tl3 tl3_ls"/>
            <route id="UW_LW0" edges="uw_tl1 tl1_tl3 tl3_lw"/>
            <route id="UW_LW1" edges="uw_tl1 tl1_tl2 tl2_tl4 tl4_tl3 tl3_lw"/>
            <route id="LW_LS" edges="lw_tl3 tl3_ls"/>
            <route id="LW_RS0" edges="lw_tl3 tl3_tl4 tl4_rs"/>
            <route id="LW_RS1" edges="lw_tl3 tl3_tl1 tl1_tl2 tl2_tl4 tl4_rs"/>
            <route id="LW_LE0" edges="lw_tl3 tl3_tl4 tl4_le"/>
            <route id="LW_LE1" edges="lw_tl3 tl3_tl1 tl1_tl2 tl2_tl4 tl4_le"/>
            <route id="LW_UE0" edges="lw_tl3 tl3_tl4 tl4_tl2 tl2_ue"/>
            <route id="LW_UE1" edges="lw_tl3 tl3_tl1 tl1_tl2 tl2_ue"/>
            <route id="LW_LN0" edges="lw_tl3 tl3_tl4 tl4_tl2 tl2_tl1 tl1_ln"/>
            <route id="LW_LN1" edges="lw_tl3 tl3_tl1 tl1_ln"/>
            <route id="LW_RN0" edges="lw_tl3 tl3_tl4 tl4_tl2 tl2_rn"/>
            <route id="LW_RN1" edges="lw_tl3 tl3_tl1 tl1_tl2 tl2_rn"/>
            <route id="LW_UW0" edges="lw_tl3 tl3_tl1 tl1_uw"/>
            <route id="LW_UW1" edges="lw_tl3 tl3_tl4 tl4_tl2 tl2_tl1 tl1_uw"/>
            <route id="LS_RS0" edges="ls_tl3 tl3_tl4 tl4_rs"/>
            <route id="LS_RS1" edges="ls_tl3 tl3_tl1 tl1_tl2 tl2_tl4 tl4_rs"/>
            <route id="LS_LE0" edges="ls_tl3 tl3_tl4 tl4_le"/>
            <route id="LS_LE1" edges="ls_tl3 tl3_tl1 tl1_tl2 tl2_tl4 tl4_le"/>
            <route id="LS_UE0" edges="ls_tl3 tl3_tl4 tl4_tl2 tl2_ue"/>
            <route id="LS_UE1" edges="ls_tl3 tl3_tl1 tl1_tl2 tl2_ue"/>
            <route id="LS_RN0" edges="ls_tl3 tl3_tl4 tl4_tl2 tl2_rn"/>
            <route id="LS_RN1" edges="ls_tl3 tl3_tl1 tl1_tl2 tl2_rn"/>
            <route id="LS_LN0" edges="ls_tl3 tl3_tl1 tl1_ln"/>
            <route id="LS_LN1" edges="ls_tl3 tl3_tl4 tl4_tl2 tl2_tl1 tl1_ln"/>
            <route id="LS_UW0" edges="ls_tl3 tl3_tl4 tl4_tl2 tl2_tl1 tl1_uw"/>
            <route id="LS_UW1" edges="ls_tl3 tl3_tl1 tl1_uw"/>
            <route id="LS_LW" edges="ls_tl3 tl3_lw"/>
            <route id="RS_LE" edges="rs_tl4 tl4_le"/>
            <route id="RS_UE0" edges="rs_tl4 tl4_tl2 tl2_ue"/>
            <route id="RS_UE1" edges="rs_tl4 tl4_tl3 tl3_tl1 tl1_tl2 tl2_ue"/>
            <route id="RS_RN0" edges="rs_tl4 tl4_tl2 tl2_rn"/>
            <route id="RS_RN1" edges="rs_tl4 tl4_tl3 tl3_tl1 tl1_tl2 tl2_rn"/>
            <route id="RS_LN0" edges="rs_tl4 tl4_tl2 tl2_tl1 tl1_ln"/>
            <route id="RS_LN1" edges="rs_tl4 tl4_tl3 tl3_tl1 tl1_ln"/>
            <route id="RS_UW0" edges="rs_tl4 tl4_tl2 tl2_tl1 tl1_uw"/> 
            <route id="RS_UW1" edges="rs_tl4 tl4_tl3 tl3_tl1 tl1_uw"/>
            <route id="RS_LW0" edges="rs_tl4 tl4_tl3 tl3_lw"/>
            <route id="RS_LW1" edges="rs_tl4 tl4_tl2 tl2_tl1 tl1_tl3 tl3_lw"/> 
            <route id="RS_LS0" edges="rs_tl4 tl4_tl3 tl3_ls"/>
            <route id="RS_LS1" edges="rs_tl4 tl4_tl2 tl2_tl1 tl1_tl3 tl3_ls"/>
            <route id="LE_UE0" edges="le_tl4 tl4_tl2 tl2_ue"/>
            <route id="LE_UE1" edges="le_tl4 tl4_tl3 tl3_tl1 tl1_tl2 tl2_ue"/>
            <route id="LE_RN0" edges="le_tl4 tl4_tl2 tl2_rn"/>
            <route id="LE_RN1" edges="le_tl4 tl4_tl3 tl3_tl1 tl1_tl2 tl2_rn"/>
            <route id="LE_LN0" edges="le_tl4 tl4_tl2 tl2_tl1 tl1_ln"/>
            <route id="LE_LN1" edges="le_tl4 tl4_tl3 tl3_tl1 tl1_ln"/>
            <route id="LE_UW0" edges="le_tl4 tl4_tl2 tl2_tl1 tl1_uw"/>
            <route id="LE_UW1" edges="le_tl4 tl4_tl3 tl3_tl1 tl1_uw"/>
            <route id="LE_LW0" edges="le_tl4 tl4_tl3 tl3_lw"/>
            <route id="LE_LW1" edges="le_tl4 tl4_tl2 tl2_tl1 tl1_tl3 tl3_lw"/>
            <route id="LE_LS0" edges="le_tl4 tl4_tl3 tl3_ls"/>
            <route id="LE_LS1" edges="le_tl4 tl4_tl2 tl2_tl1 tl1_tl3 tl3_ls"/>
            <route id="LE_RS" edges="le_tl4 tl4_rs"/>
            <route id="UE_RN" edges="ue_tl2 tl2_rn"/>
            <route id="UE_LN0" edges="ue_tl2 tl2_tl1 tl1_ln"/>
            <route id="UE_LN1" edges="ue_tl2 tl2_tl4 tl4_tl3 tl3_tl1 tl1_ln"/>
            <route id="UE_UW0" edges="ue_tl2 tl2_tl1 tl1_uw"/>
            <route id="UE_UW1" edges="ue_tl2 tl2_tl4 tl4_tl3 tl3_tl1 tl1_uw"/>
            <route id="UE_LW0" edges="ue_tl2 tl2_tl1 tl1_tl3 tl3_lw"/>
            <route id="UE_LW1" edges="ue_tl2 tl2_tl4 tl4_tl3 tl3_lw"/>
            <route id="UE_LS0" edges="ue_tl2 tl2_tl1 tl1_tl3 tl3_ls"/>
            <route id="UE_LS1" edges="ue_tl2 tl2_tl4 tl4_tl3 tl3_ls"/>
            <route id="UE_RS0" edges="ue_tl2 tl2_tl4 tl4_rs"/>
            <route id="UE_RS1" edges="ue_tl2 tl2_tl1 tl1_tl3 tl3_tl4 tl4_rs"/>
            <route id="UE_LE0" edges="ue_tl2 tl2_tl4 tl4_le"/>
            <route id="UE_LE1" edges="ue_tl2 tl2_tl1 tl1_tl3 tl3_tl4 tl4_le"/>
            <route id="RN_LN0" edges="rn_tl2 tl2_tl1 tl1_ln"/>
            <route id="RN_LN1" edges="rn_tl2 tl2_tl4 tl4_tl3 tl3_tl1 tl1_ln"/>
            <route id="RN_UW0" edges="rn_tl2 tl2_tl1 tl1_uw"/>
            <route id="RN_UW1" edges="rn_tl2 tl2_tl4 tl4_tl3 tl3_tl1 tl1_uw"/>
            <route id="RN_LW0" edges="rn_tl2 tl2_tl1 tl1_tl3 tl3_lw"/>
            <route id="RN_LW1" edges="rn_tl2 tl2_tl4 tl4_tl3 tl3_lw"/>
            <route id="RN_LS0" edges="rn_tl2 tl2_tl1 tl1_tl3 tl3_ls"/>
            <route id="RN_LS1" edges="rn_tl2 tl2_tl4 tl4_tl3 tl3_ls"/>
            <route id="RN_RS0" edges="rn_tl2 tl2_tl4 tl4_rs"/>
            <route id="RN_RS1" edges="rn_tl2 tl2_tl1 tl1_tl3 tl3_tl4 tl4_rs"/>
            <route id="RN_LE0" edges="rn_tl2 tl2_tl4 tl4_le"/>
            <route id="RN_LE1" edges="rn_tl2 tl2_tl1 tl1_tl3 tl3_tl4 tl4_le"/>
            <route id="RN_UE" edges="rn_tl2 tl2_ue"/>
            <route id="LN_UW" edges="ln_tl1 tl1_uw"/>
            <route id="LN_LW0" edges="ln_tl1 tl1_tl3 tl3_lw"/>
            <route id="LN_LW1" edges="ln_tl1 tl1_tl2 tl2_tl4 tl4_tl3 tl3_lw"/>
            <route id="LN_LS0" edges="ln_tl1 tl1_tl3 tl3_ls"/>
            <route id="LN_LS1" edges="ln_tl1 tl1_tl2 tl2_tl4 tl4_tl3 tl3_ls"/>
            <route id="LN_RS0" edges="ln_tl1 tl1_tl3 tl3_tl4 tl4_rs"/>
            <route id="LN_RS1" edges="ln_tl1 tl1_tl2 tl2_tl4 tl4_rs"/>
            <route id="LN_LE0" edges="ln_tl1 tl1_tl3 tl3_tl4 tl4_le"/>
            <route id="LN_LE1" edges="ln_tl1 tl1_tl2 tl2_tl4 tl4_le"/>
            <route id="LN_UE0" edges="ln_tl1 tl1_tl2 tl2_ue"/>
            <route id="LN_UE1" edges="ln_tl1 tl1_tl3 tl3_tl4 tl4_tl2 tl2_ue"/>
            <route id="LN_RN0" edges="ln_tl1 tl1_tl2 tl2_rn"/>
            <route id="LN_RN1" edges="ln_tl1 tl1_tl3 tl3_tl4 tl4_tl2 tl2_rn"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                route_prop = np.random.uniform()
                if route_prop < 0.56: # ssf, s--> straight, f--> final
                    random_ssf = np.random.randint(1, 8)
                    if random_ssf == 1:
                        print('    <vehicle id="UW_UE0_%i" type="standard_car" route="UW_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ssf == 2:
                        print('    <vehicle id="UE_UW0_%i" type="standard_car" route="UE_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ssf == 3:
                        print('    <vehicle id="LW_LE0_%i" type="standard_car" route="LW_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ssf == 4:
                        print('    <vehicle id="LE_LW0_%i" type="standard_car" route="LE_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ssf == 5:
                        print('    <vehicle id="LS_LN0_%i" type="standard_car" route="LS_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ssf == 6:
                        print('    <vehicle id="LN_LS0_%i" type="standard_car" route="LN_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ssf == 7:
                        print('    <vehicle id="RN_RS0_%i" type="standard_car" route="RN_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ssf == 8:
                        print('    <vehicle id="RS_RN0_%i" type="standard_car" route="RS_RN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .56 and route_prop < .69: # tf, t-->turn, f--> final
                    random_tf = np.random.randint(1, 8)
                    if random_tf == 1:
                        print('    <vehicle id="UW_LN_%i" type="standard_car" route="UW_LN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tf == 2:
                        print('    <vehicle id="LW_LS_%i" type="standard_car" route="LW_LS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tf == 3:
                        print('    <vehicle id="LS_LW_%i" type="standard_car" route="LS_LW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tf == 4:
                        print('    <vehicle id="RS_LE_%i" type="standard_car" route="RS_LE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tf == 5:
                        print('    <vehicle id="LE_RS_%i" type="standard_car" route="LE_RS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tf == 6:
                        print('    <vehicle id="UE_RN_%i" type="standard_car" route="UE_RN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tf == 7:
                        print('    <vehicle id="LN_UW_%i" type="standard_car" route="LN_UW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tf == 8:
                        print('    <vehicle id="RN_UE_%i" type="standard_car" route="RN_UE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .69 and route_prop < .78: # tsf, s--> straight, t-->turn, f--> final
                    random_tsf = np.random.randint(1, 8)
                    if random_tsf == 1:
                        print('    <vehicle id="UW_LS0_%i" type="standard_car" route="UW_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tsf == 2:
                        print('    <vehicle id="LN_UE0_%i" type="standard_car" route="LN_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tsf == 3:
                        print('    <vehicle id="LW_LN1_%i" type="standard_car" route="LW_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tsf == 4:
                        print('    <vehicle id="LS_LE0_%i" type="standard_car" route="LS_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tsf == 5:
                        print('    <vehicle id="RS_LW0_%i" type="standard_car" route="RS_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tsf == 6:
                        print('    <vehicle id="LE_RN0_%i" type="standard_car" route="LE_RN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tsf == 7:
                        print('    <vehicle id="UE_RS0_%i" type="standard_car" route="UE_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tsf == 8:
                        print('    <vehicle id="RN_UW0_%i" type="standard_car" route="RN_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .78 and route_prop < .87: # stf, s--> straight, t-->turn, f--> final
                    random_stf = np.random.randint(1, 8)
                    if random_stf == 1:
                        print('    <vehicle id="UW_RN0_%i" type="standard_car" route="UW_RN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stf == 2:
                        print('    <vehicle id="LN_LW0_%i" type="standard_car" route="LN_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stf == 3:
                        print('    <vehicle id="LW_RS0_%i" type="standard_car" route="LW_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stf == 4:
                        print('    <vehicle id="LS_UW1_%i" type="standard_car" route="LS_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stf == 5:
                        print('    <vehicle id="RS_UE0_%i" type="standard_car" route="RS_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stf == 6:
                        print('    <vehicle id="LE_LS0_%i" type="standard_car" route="LE_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stf == 7:
                        print('    <vehicle id="UE_LN0_%i" type="standard_car" route="UE_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stf == 8:
                        print('    <vehicle id="RN_LE0_%i" type="standard_car" route="RN_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .87 and route_prop < .94: # stsf, s--> straight, t-->turn, f--> final
                    random_stsf = np.random.randint(1, 8)
                    if random_stsf == 1:
                        print('    <vehicle id="UW_RS0_%i" type="standard_car" route="UW_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stsf == 2:
                        print('    <vehicle id="LN_LE0_%i" type="standard_car" route="LN_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stsf == 3:
                        print('    <vehicle id="LW_RN0_%i" type="standard_car" route="LW_RN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stsf == 4:
                        print('    <vehicle id="LS_UE1_%i" type="standard_car" route="LS_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stsf == 5:
                        print('    <vehicle id="RS_UW0_%i" type="standard_car" route="RS_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stsf == 6:
                        print('    <vehicle id="LE_LN1_%i" type="standard_car" route="LE_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stsf == 7:
                        print('    <vehicle id="UE_LS0_%i" type="standard_car" route="UE_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stsf == 8:
                        print('    <vehicle id="RN_LW1_%i" type="standard_car" route="RN_LW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .94 and route_prop < .96: # ttf, s--> straight, t-->turn, f--> final
                    random_ttf = np.random.randint(1, 8)
                    if random_ttf == 1:
                        print('    <vehicle id="UW_LW0_%i" type="standard_car" route="UW_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttf == 2:
                        print('    <vehicle id="LW_UW0_%i" type="standard_car" route="LW_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttf == 3:
                        print('    <vehicle id="LN_RN0_%i" type="standard_car" route="LN_RN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttf == 4:
                        print('    <vehicle id="RN_LN0_%i" type="standard_car" route="RN_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttf == 5:
                        print('    <vehicle id="UE_LE0_%i" type="standard_car" route="UE_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttf == 6:
                        print('    <vehicle id="LE_UE0_%i" type="standard_car" route="LE_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttf == 7:
                        print('    <vehicle id="RS_LS0_%i" type="standard_car" route="RS_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttf == 8:
                        print('    <vehicle id="LS_RS0_%i" type="standard_car" route="LS_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .96 and route_prop < .97: # sttf, s--> straight, t-->turn, f--> final
                    random_sttf = np.random.randint(1, 8)
                    if random_sttf == 1:
                        print('    <vehicle id="UW_LE0_%i" type="standard_car" route="UW_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttf == 2:
                        print('    <vehicle id="LN_RS0_%i" type="standard_car" route="LN_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttf == 3:
                        print('    <vehicle id="LS_RN1_%i" type="standard_car" route="LS_RN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttf == 4:
                        print('    <vehicle id="LW_UE0_%i" type="standard_car" route="LW_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttf == 5:
                        print('    <vehicle id="RS_LN0_%i" type="standard_car" route="RS_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttf == 6:
                        print('    <vehicle id="RS_UW1_%i" type="standard_car" route="RS_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttf == 7:
                        print('    <vehicle id="UE_LW0_%i" type="standard_car" route="UE_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttf == 8:
                        print('    <vehicle id="RN_LS1_%i" type="standard_car" route="RN_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .97 and route_prop < .98: # sttsf, s--> straight, t-->turn, f--> final
                    random_sttsf = np.random.randint(1, 8)
                    if random_sttsf == 1:
                        print('    <vehicle id="UW_LW1_%i" type="standard_car" route="UW_LW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttsf == 2:
                        print('    <vehicle id="LW_UW1_%i" type="standard_car" route="LW_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttsf == 3:
                        print('    <vehicle id="LN_RN1_%i" type="standard_car" route="LN_RN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttsf == 4:
                        print('    <vehicle id="RN_LN1_%i" type="standard_car" route="RN_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttsf == 5:
                        print('    <vehicle id="UE_LE1_%i" type="standard_car" route="UE_LE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttsf == 6:
                        print('    <vehicle id="LE_UE1_%i" type="standard_car" route="LE_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttsf == 7:
                        print('    <vehicle id="RS_LS1_%i" type="standard_car" route="RS_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttsf == 8:
                        print('    <vehicle id="LS_RS1_%i" type="standard_car" route="LS_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .98 and route_prop < .99: # ttsf, t--> straight, t-->turn, f--> final
                    random_ttsf = np.random.randint(1, 8)
                    if random_ttsf == 1:
                        print('    <vehicle id="UW_LE1_%i" type="standard_car" route="UW_LE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttsf == 2:
                        print('    <vehicle id="LN_RS1_%i" type="standard_car" route="LN_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttsf == 3:
                        print('    <vehicle id="LS_RN0_%i" type="standard_car" route="LS_RN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttsf == 4:
                        print('    <vehicle id="LW_UE1_%i" type="standard_car" route="LW_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttsf == 5:
                        print('    <vehicle id="RS_LN1_%i" type="standard_car" route="RS_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttsf == 6:
                        print('    <vehicle id="LE_UW0_%i" type="standard_car" route="LE_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttsf == 7:
                        print('    <vehicle id="UE_LW1_%i" type="standard_car" route="UE_LW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttsf == 8:
                        print('    <vehicle id="RN_LS0_%i" type="standard_car" route="RN_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else: # rest of 32 routes 
                    random_rest = np.random.randint(1,31)
                    if random_rest == 1:
                        print('    <vehicle id="UW_RN1_%i" type="standard_car" route="UW_RN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 2:
                        print('    <vehicle id="UW_UE1_%i" type="standard_car" route="UW_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 3:
                        print('    <vehicle id="UW_RS1_%i" type="standard_car" route="UW_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 4:
                        print('    <vehicle id="UW_LS1_%i" type="standard_car" route="UW_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 5:
                        print('    <vehicle id="LW_RS1_%i" type="standard_car" route="LW_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 6:
                        print('    <vehicle id="LW_LE1_%i" type="standard_car" route="LW_LE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 7:
                        print('    <vehicle id="LW_LN0_%i" type="standard_car" route="LW_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 8:
                        print('    <vehicle id="LW_RN1_%i" type="standard_car" route="LW_RN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 9:
                        print('    <vehicle id="LS_LE1_%i" type="standard_car" route="LS_LE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 10:
                        print('    <vehicle id="LS_UE0_%i" type="standard_car" route="LS_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 11:
                        print('    <vehicle id="LS_LN1_%i" type="standard_car" route="LS_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 12:
                        print('    <vehicle id="LS_UW0_%i" type="standard_car" route="LS_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 13:
                        print('    <vehicle id="RS_UE1_%i" type="standard_car" route="RS_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 14:
                        print('    <vehicle id="RS_RN1_%i" type="standard_car" route="RS_RN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 15:
                        print('    <vehicle id="RS_LW1_%i" type="standard_car" route="RS_LW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 16:
                        print('    <vehicle id="LE_RN1_%i" type="standard_car" route="LE_RN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 17:
                        print('    <vehicle id="LE_LN0_%i" type="standard_car" route="LE_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 18:
                        print('    <vehicle id="LE_UW1_%i" type="standard_car" route="LE_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 19:
                        print('    <vehicle id="LE_LS1_%i" type="standard_car" route="LE_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 20:
                        print('    <vehicle id="UE_LN1_%i" type="standard_car" route="UE_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 21:
                        print('    <vehicle id="UE_UW1_%i" type="standard_car" route="UE_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 22:
                        print('    <vehicle id="UE_LS1_%i" type="standard_car" route="UE_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 23:
                        print('    <vehicle id="UE_RS1_%i" type="standard_car" route="UE_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 24:
                        print('    <vehicle id="RN_UW1_%i" type="standard_car" route="RN_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 25:
                        print('    <vehicle id="RN_LW0_%i" type="standard_car" route="RN_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 26:
                        print('    <vehicle id="RN_RS1_%i" type="standard_car" route="RN_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 27:
                        print('    <vehicle id="RN_LE1_%i" type="standard_car" route="RN_LE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 28:
                        print('    <vehicle id="LN_LW1_%i" type="standard_car" route="LN_LW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 29:
                        print('    <vehicle id="LN_LS1_%i" type="standard_car" route="LN_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 30:
                        print('    <vehicle id="LN_LE1_%i" type="standard_car" route="LN_LE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 31:
                        print('    <vehicle id="LN_UE1_%i" type="standard_car" route="LN_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                                                               
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
            <route id="UW_RN0" edges="uw_tl1 tl1_tl2 tl2_rn"/>
            <route id="UW_RN1" edges="uw_tl1 tl1_tl3 tl3_tl4 tl4_tl2 tl2_rn"/>
            <route id="UW_UE0" edges="uw_tl1 tl1_tl2 tl2_ue"/>
            <route id="UW_UE1" edges="uw_tl1 tl1_tl3 tl3_tl4 tl4_tl2 tl2_ue"/>
            <route id="UW_LE0" edges="uw_tl1 tl1_tl2 tl2_tl4 tl4_le"/>
            <route id="UW_LE1" edges="uw_tl1 tl1_tl3 tl3_tl4 tl4_le"/>
            <route id="UW_RS0" edges="uw_tl1 tl1_tl2 tl2_tl4 tl4_rs"/>
            <route id="UW_RS1" edges="uw_tl1 tl1_tl3 tl3_tl4 tl4_rs"/>
            <route id="UW_LS0" edges="uw_tl1 tl1_tl3 tl3_ls"/>
            <route id="UW_LS1" edges="uw_tl1 tl1_tl2 tl2_tl4 tl4_tl3 tl3_ls"/>
            <route id="UW_LW0" edges="uw_tl1 tl1_tl3 tl3_lw"/>
            <route id="UW_LW1" edges="uw_tl1 tl1_tl2 tl2_tl4 tl4_tl3 tl3_lw"/>
            <route id="LW_LS" edges="lw_tl3 tl3_ls"/>
            <route id="LW_RS0" edges="lw_tl3 tl3_tl4 tl4_rs"/>
            <route id="LW_RS1" edges="lw_tl3 tl3_tl1 tl1_tl2 tl2_tl4 tl4_rs"/>
            <route id="LW_LE0" edges="lw_tl3 tl3_tl4 tl4_le"/>
            <route id="LW_LE1" edges="lw_tl3 tl3_tl1 tl1_tl2 tl2_tl4 tl4_le"/>
            <route id="LW_UE0" edges="lw_tl3 tl3_tl4 tl4_tl2 tl2_ue"/>
            <route id="LW_UE1" edges="lw_tl3 tl3_tl1 tl1_tl2 tl2_ue"/>
            <route id="LW_LN0" edges="lw_tl3 tl3_tl4 tl4_tl2 tl2_tl1 tl1_ln"/>
            <route id="LW_LN1" edges="lw_tl3 tl3_tl1 tl1_ln"/>
            <route id="LW_RN0" edges="lw_tl3 tl3_tl4 tl4_tl2 tl2_rn"/>
            <route id="LW_RN1" edges="lw_tl3 tl3_tl1 tl1_tl2 tl2_rn"/>
            <route id="LW_UW0" edges="lw_tl3 tl3_tl1 tl1_uw"/>
            <route id="LW_UW1" edges="lw_tl3 tl3_tl4 tl4_tl2 tl2_tl1 tl1_uw"/>
            <route id="LS_RS0" edges="ls_tl3 tl3_tl4 tl4_rs"/>
            <route id="LS_RS1" edges="ls_tl3 tl3_tl1 tl1_tl2 tl2_tl4 tl4_rs"/>
            <route id="LS_LE0" edges="ls_tl3 tl3_tl4 tl4_le"/>
            <route id="LS_LE1" edges="ls_tl3 tl3_tl1 tl1_tl2 tl2_tl4 tl4_le"/>
            <route id="LS_UE0" edges="ls_tl3 tl3_tl4 tl4_tl2 tl2_ue"/>
            <route id="LS_UE1" edges="ls_tl3 tl3_tl1 tl1_tl2 tl2_ue"/>
            <route id="LS_RN0" edges="ls_tl3 tl3_tl4 tl4_tl2 tl2_rn"/>
            <route id="LS_RN1" edges="ls_tl3 tl3_tl1 tl1_tl2 tl2_rn"/>
            <route id="LS_LN0" edges="ls_tl3 tl3_tl1 tl1_ln"/>
            <route id="LS_LN1" edges="ls_tl3 tl3_tl4 tl4_tl2 tl2_tl1 tl1_ln"/>
            <route id="LS_UW0" edges="ls_tl3 tl3_tl4 tl4_tl2 tl2_tl1 tl1_uw"/>
            <route id="LS_UW1" edges="ls_tl3 tl3_tl1 tl1_uw"/>
            <route id="LS_LW" edges="ls_tl3 tl3_lw"/>
            <route id="RS_LE" edges="rs_tl4 tl4_le"/>
            <route id="RS_UE0" edges="rs_tl4 tl4_tl2 tl2_ue"/>
            <route id="RS_UE1" edges="rs_tl4 tl4_tl3 tl3_tl1 tl1_tl2 tl2_ue"/>
            <route id="RS_RN0" edges="rs_tl4 tl4_tl2 tl2_rn"/>
            <route id="RS_RN1" edges="rs_tl4 tl4_tl3 tl3_tl1 tl1_tl2 tl2_rn"/>
            <route id="RS_LN0" edges="rs_tl4 tl4_tl2 tl2_tl1 tl1_ln"/>
            <route id="RS_LN1" edges="rs_tl4 tl4_tl3 tl3_tl1 tl1_ln"/>
            <route id="RS_UW0" edges="rs_tl4 tl4_tl2 tl2_tl1 tl1_uw"/> 
            <route id="RS_UW1" edges="rs_tl4 tl4_tl3 tl3_tl1 tl1_uw"/>
            <route id="RS_LW0" edges="rs_tl4 tl4_tl3 tl3_lw"/>
            <route id="RS_LW1" edges="rs_tl4 tl4_tl2 tl2_tl1 tl1_tl3 tl3_lw"/> 
            <route id="RS_LS0" edges="rs_tl4 tl4_tl3 tl3_ls"/>
            <route id="RS_LS1" edges="rs_tl4 tl4_tl2 tl2_tl1 tl1_tl3 tl3_ls"/>
            <route id="LE_UE0" edges="le_tl4 tl4_tl2 tl2_ue"/>
            <route id="LE_UE1" edges="le_tl4 tl4_tl3 tl3_tl1 tl1_tl2 tl2_ue"/>
            <route id="LE_RN0" edges="le_tl4 tl4_tl2 tl2_rn"/>
            <route id="LE_RN1" edges="le_tl4 tl4_tl3 tl3_tl1 tl1_tl2 tl2_rn"/>
            <route id="LE_LN0" edges="le_tl4 tl4_tl2 tl2_tl1 tl1_ln"/>
            <route id="LE_LN1" edges="le_tl4 tl4_tl3 tl3_tl1 tl1_ln"/>
            <route id="LE_UW0" edges="le_tl4 tl4_tl2 tl2_tl1 tl1_uw"/>
            <route id="LE_UW1" edges="le_tl4 tl4_tl3 tl3_tl1 tl1_uw"/>
            <route id="LE_LW0" edges="le_tl4 tl4_tl3 tl3_lw"/>
            <route id="LE_LW1" edges="le_tl4 tl4_tl2 tl2_tl1 tl1_tl3 tl3_lw"/>
            <route id="LE_LS0" edges="le_tl4 tl4_tl3 tl3_ls"/>
            <route id="LE_LS1" edges="le_tl4 tl4_tl2 tl2_tl1 tl1_tl3 tl3_ls"/>
            <route id="LE_RS" edges="le_tl4 tl4_rs"/>
            <route id="UE_RN" edges="ue_tl2 tl2_rn"/>
            <route id="UE_LN0" edges="ue_tl2 tl2_tl1 tl1_ln"/>
            <route id="UE_LN1" edges="ue_tl2 tl2_tl4 tl4_tl3 tl3_tl1 tl1_ln"/>
            <route id="UE_UW0" edges="ue_tl2 tl2_tl1 tl1_uw"/>
            <route id="UE_UW1" edges="ue_tl2 tl2_tl4 tl4_tl3 tl3_tl1 tl1_uw"/>
            <route id="UE_LW0" edges="ue_tl2 tl2_tl1 tl1_tl3 tl3_lw"/>
            <route id="UE_LW1" edges="ue_tl2 tl2_tl4 tl4_tl3 tl3_lw"/>
            <route id="UE_LS0" edges="ue_tl2 tl2_tl1 tl1_tl3 tl3_ls"/>
            <route id="UE_LS1" edges="ue_tl2 tl2_tl4 tl4_tl3 tl3_ls"/>
            <route id="UE_RS0" edges="ue_tl2 tl2_tl4 tl4_rs"/>
            <route id="UE_RS1" edges="ue_tl2 tl2_tl1 tl1_tl3 tl3_tl4 tl4_rs"/>
            <route id="UE_LE0" edges="ue_tl2 tl2_tl4 tl4_le"/>
            <route id="UE_LE1" edges="ue_tl2 tl2_tl1 tl1_tl3 tl3_tl4 tl4_le"/>
            <route id="RN_LN0" edges="rn_tl2 tl2_tl1 tl1_ln"/>
            <route id="RN_LN1" edges="rn_tl2 tl2_tl4 tl4_tl3 tl3_tl1 tl1_ln"/>
            <route id="RN_UW0" edges="rn_tl2 tl2_tl1 tl1_uw"/>
            <route id="RN_UW1" edges="rn_tl2 tl2_tl4 tl4_tl3 tl3_tl1 tl1_uw"/>
            <route id="RN_LW0" edges="rn_tl2 tl2_tl1 tl1_tl3 tl3_lw"/>
            <route id="RN_LW1" edges="rn_tl2 tl2_tl4 tl4_tl3 tl3_lw"/>
            <route id="RN_LS0" edges="rn_tl2 tl2_tl1 tl1_tl3 tl3_ls"/>
            <route id="RN_LS1" edges="rn_tl2 tl2_tl4 tl4_tl3 tl3_ls"/>
            <route id="RN_RS0" edges="rn_tl2 tl2_tl4 tl4_rs"/>
            <route id="RN_RS1" edges="rn_tl2 tl2_tl1 tl1_tl3 tl3_tl4 tl4_rs"/>
            <route id="RN_LE0" edges="rn_tl2 tl2_tl4 tl4_le"/>
            <route id="RN_LE1" edges="rn_tl2 tl2_tl1 tl1_tl3 tl3_tl4 tl4_le"/>
            <route id="RN_UE" edges="rn_tl2 tl2_ue"/>
            <route id="LN_UW" edges="ln_tl1 tl1_uw"/>
            <route id="LN_LW0" edges="ln_tl1 tl1_tl3 tl3_lw"/>
            <route id="LN_LW1" edges="ln_tl1 tl1_tl2 tl2_tl4 tl4_tl3 tl3_lw"/>
            <route id="LN_LS0" edges="ln_tl1 tl1_tl3 tl3_ls"/>
            <route id="LN_LS1" edges="ln_tl1 tl1_tl2 tl2_tl4 tl4_tl3 tl3_ls"/>
            <route id="LN_RS0" edges="ln_tl1 tl1_tl3 tl3_tl4 tl4_rs"/>
            <route id="LN_RS1" edges="ln_tl1 tl1_tl2 tl2_tl4 tl4_rs"/>
            <route id="LN_LE0" edges="ln_tl1 tl1_tl3 tl3_tl4 tl4_le"/>
            <route id="LN_LE1" edges="ln_tl1 tl1_tl2 tl2_tl4 tl4_le"/>
            <route id="LN_UE0" edges="ln_tl1 tl1_tl2 tl2_ue"/>
            <route id="LN_UE1" edges="ln_tl1 tl1_tl3 tl3_tl4 tl4_tl2 tl2_ue"/>
            <route id="LN_RN0" edges="ln_tl1 tl1_tl2 tl2_rn"/>
            <route id="LN_RN1" edges="ln_tl1 tl1_tl3 tl3_tl4 tl4_tl2 tl2_rn"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                route_prop = np.random.uniform()
                if route_prop < 0.56: # ssf, s--> straight, f--> final
                    random_ssf = np.random.randint(1, 8)
                    if random_ssf == 1:
                        print('    <vehicle id="UW_UE0_%i" type="standard_car" route="UW_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ssf == 2:
                        print('    <vehicle id="UE_UW0_%i" type="standard_car" route="UE_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ssf == 3:
                        print('    <vehicle id="LW_LE0_%i" type="standard_car" route="LW_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ssf == 4:
                        print('    <vehicle id="LE_LW0_%i" type="standard_car" route="LE_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ssf == 5:
                        print('    <vehicle id="LS_LN0_%i" type="standard_car" route="LS_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ssf == 6:
                        print('    <vehicle id="LN_LS0_%i" type="standard_car" route="LN_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ssf == 7:
                        print('    <vehicle id="RN_RS0_%i" type="standard_car" route="RN_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ssf == 8:
                        print('    <vehicle id="RS_RN0_%i" type="standard_car" route="RS_RN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .56 and route_prop < .69: # tf, t-->turn, f--> final
                    random_tf = np.random.randint(1, 8)
                    if random_tf == 1:
                        print('    <vehicle id="UW_LN_%i" type="standard_car" route="UW_LN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tf == 2:
                        print('    <vehicle id="LW_LS_%i" type="standard_car" route="LW_LS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tf == 3:
                        print('    <vehicle id="LS_LW_%i" type="standard_car" route="LS_LW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tf == 4:
                        print('    <vehicle id="RS_LE_%i" type="standard_car" route="RS_LE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tf == 5:
                        print('    <vehicle id="LE_RS_%i" type="standard_car" route="LE_RS" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tf == 6:
                        print('    <vehicle id="UE_RN_%i" type="standard_car" route="UE_RN" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tf == 7:
                        print('    <vehicle id="LN_UW_%i" type="standard_car" route="LN_UW" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tf == 8:
                        print('    <vehicle id="RN_UE_%i" type="standard_car" route="RN_UE" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .69 and route_prop < .78: # tsf, s--> straight, t-->turn, f--> final
                    random_tsf = np.random.randint(1, 8)
                    if random_tsf == 1:
                        print('    <vehicle id="UW_LS0_%i" type="standard_car" route="UW_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tsf == 2:
                        print('    <vehicle id="LN_UE0_%i" type="standard_car" route="LN_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tsf == 3:
                        print('    <vehicle id="LW_LN1_%i" type="standard_car" route="LW_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tsf == 4:
                        print('    <vehicle id="LS_LE0_%i" type="standard_car" route="LS_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tsf == 5:
                        print('    <vehicle id="RS_LW0_%i" type="standard_car" route="RS_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tsf == 6:
                        print('    <vehicle id="LE_RN0_%i" type="standard_car" route="LE_RN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tsf == 7:
                        print('    <vehicle id="UE_RS0_%i" type="standard_car" route="UE_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_tsf == 8:
                        print('    <vehicle id="RN_UW0_%i" type="standard_car" route="RN_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .78 and route_prop < .87: # stf, s--> straight, t-->turn, f--> final
                    random_stf = np.random.randint(1, 8)
                    if random_stf == 1:
                        print('    <vehicle id="UW_RN0_%i" type="standard_car" route="UW_RN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stf == 2:
                        print('    <vehicle id="LN_LW0_%i" type="standard_car" route="LN_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stf == 3:
                        print('    <vehicle id="LW_RS0_%i" type="standard_car" route="LW_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stf == 4:
                        print('    <vehicle id="LS_UW1_%i" type="standard_car" route="LS_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stf == 5:
                        print('    <vehicle id="RS_UE0_%i" type="standard_car" route="RS_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stf == 6:
                        print('    <vehicle id="LE_LS0_%i" type="standard_car" route="LE_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stf == 7:
                        print('    <vehicle id="UE_LN0_%i" type="standard_car" route="UE_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stf == 8:
                        print('    <vehicle id="RN_LE0_%i" type="standard_car" route="RN_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .87 and route_prop < .94: # stsf, s--> straight, t-->turn, f--> final
                    random_stsf = np.random.randint(1, 8)
                    if random_stsf == 1:
                        print('    <vehicle id="UW_RS0_%i" type="standard_car" route="UW_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stsf == 2:
                        print('    <vehicle id="LN_LE0_%i" type="standard_car" route="LN_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stsf == 3:
                        print('    <vehicle id="LW_RN0_%i" type="standard_car" route="LW_RN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stsf == 4:
                        print('    <vehicle id="LS_UE1_%i" type="standard_car" route="LS_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stsf == 5:
                        print('    <vehicle id="RS_UW0_%i" type="standard_car" route="RS_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stsf == 6:
                        print('    <vehicle id="LE_LN1_%i" type="standard_car" route="LE_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stsf == 7:
                        print('    <vehicle id="UE_LS0_%i" type="standard_car" route="UE_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_stsf == 8:
                        print('    <vehicle id="RN_LW1_%i" type="standard_car" route="RN_LW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .94 and route_prop < .96: # ttf, s--> straight, t-->turn, f--> final
                    random_ttf = np.random.randint(1, 8)
                    if random_ttf == 1:
                        print('    <vehicle id="UW_LW0_%i" type="standard_car" route="UW_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttf == 2:
                        print('    <vehicle id="LW_UW0_%i" type="standard_car" route="LW_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttf == 3:
                        print('    <vehicle id="LN_RN0_%i" type="standard_car" route="LN_RN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttf == 4:
                        print('    <vehicle id="RN_LN0_%i" type="standard_car" route="RN_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttf == 5:
                        print('    <vehicle id="UE_LE0_%i" type="standard_car" route="UE_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttf == 6:
                        print('    <vehicle id="LE_UE0_%i" type="standard_car" route="LE_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttf == 7:
                        print('    <vehicle id="RS_LS0_%i" type="standard_car" route="RS_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttf == 8:
                        print('    <vehicle id="LS_RS0_%i" type="standard_car" route="LS_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .96 and route_prop < .97: # sttf, s--> straight, t-->turn, f--> final
                    random_sttf = np.random.randint(1, 8)
                    if random_sttf == 1:
                        print('    <vehicle id="UW_LE0_%i" type="standard_car" route="UW_LE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttf == 2:
                        print('    <vehicle id="LN_RS0_%i" type="standard_car" route="LN_RS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttf == 3:
                        print('    <vehicle id="LS_RN1_%i" type="standard_car" route="LS_RN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttf == 4:
                        print('    <vehicle id="LW_UE0_%i" type="standard_car" route="LW_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttf == 5:
                        print('    <vehicle id="RS_LN0_%i" type="standard_car" route="RS_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttf == 6:
                        print('    <vehicle id="RS_UW1_%i" type="standard_car" route="RS_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttf == 7:
                        print('    <vehicle id="UE_LW0_%i" type="standard_car" route="UE_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttf == 8:
                        print('    <vehicle id="RN_LS1_%i" type="standard_car" route="RN_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .97 and route_prop < .98: # sttsf, s--> straight, t-->turn, f--> final
                    random_sttsf = np.random.randint(1, 8)
                    if random_sttsf == 1:
                        print('    <vehicle id="UW_LW1_%i" type="standard_car" route="UW_LW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttsf == 2:
                        print('    <vehicle id="LW_UW1_%i" type="standard_car" route="LW_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttsf == 3:
                        print('    <vehicle id="LN_RN1_%i" type="standard_car" route="LN_RN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttsf == 4:
                        print('    <vehicle id="RN_LN1_%i" type="standard_car" route="RN_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttsf == 5:
                        print('    <vehicle id="UE_LE1_%i" type="standard_car" route="UE_LE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttsf == 6:
                        print('    <vehicle id="LE_UE1_%i" type="standard_car" route="LE_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttsf == 7:
                        print('    <vehicle id="RS_LS1_%i" type="standard_car" route="RS_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_sttsf == 8:
                        print('    <vehicle id="LS_RS1_%i" type="standard_car" route="LS_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                elif route_prop >= .98 and route_prop < .99: # ttsf, t--> straight, t-->turn, f--> final
                    random_ttsf = np.random.randint(1, 8)
                    if random_ttsf == 1:
                        print('    <vehicle id="UW_LE1_%i" type="standard_car" route="UW_LE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttsf == 2:
                        print('    <vehicle id="LN_RS1_%i" type="standard_car" route="LN_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttsf == 3:
                        print('    <vehicle id="LS_RN0_%i" type="standard_car" route="LS_RN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttsf == 4:
                        print('    <vehicle id="LW_UE1_%i" type="standard_car" route="LW_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttsf == 5:
                        print('    <vehicle id="RS_LN1_%i" type="standard_car" route="RS_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttsf == 6:
                        print('    <vehicle id="LE_UW0_%i" type="standard_car" route="LE_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttsf == 7:
                        print('    <vehicle id="UE_LW1_%i" type="standard_car" route="UE_LW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_ttsf == 8:
                        print('    <vehicle id="RN_LS0_%i" type="standard_car" route="RN_LS0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else: # rest of 32 routes 
                    random_rest = np.random.randint(1,31)
                    if random_rest == 1:
                        print('    <vehicle id="UW_RN1_%i" type="standard_car" route="UW_RN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 2:
                        print('    <vehicle id="UW_UE1_%i" type="standard_car" route="UW_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 3:
                        print('    <vehicle id="UW_RS1_%i" type="standard_car" route="UW_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 4:
                        print('    <vehicle id="UW_LS1_%i" type="standard_car" route="UW_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 5:
                        print('    <vehicle id="LW_RS1_%i" type="standard_car" route="LW_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 6:
                        print('    <vehicle id="LW_LE1_%i" type="standard_car" route="LW_LE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 7:
                        print('    <vehicle id="LW_LN0_%i" type="standard_car" route="LW_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 8:
                        print('    <vehicle id="LW_RN1_%i" type="standard_car" route="LW_RN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 9:
                        print('    <vehicle id="LS_LE1_%i" type="standard_car" route="LS_LE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 10:
                        print('    <vehicle id="LS_UE0_%i" type="standard_car" route="LS_UE0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 11:
                        print('    <vehicle id="LS_LN1_%i" type="standard_car" route="LS_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 12:
                        print('    <vehicle id="LS_UW0_%i" type="standard_car" route="LS_UW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 13:
                        print('    <vehicle id="RS_UE1_%i" type="standard_car" route="RS_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 14:
                        print('    <vehicle id="RS_RN1_%i" type="standard_car" route="RS_RN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 15:
                        print('    <vehicle id="RS_LW1_%i" type="standard_car" route="RS_LW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 16:
                        print('    <vehicle id="LE_RN1_%i" type="standard_car" route="LE_RN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 17:
                        print('    <vehicle id="LE_LN0_%i" type="standard_car" route="LE_LN0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 18:
                        print('    <vehicle id="LE_UW1_%i" type="standard_car" route="LE_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 19:
                        print('    <vehicle id="LE_LS1_%i" type="standard_car" route="LE_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 20:
                        print('    <vehicle id="UE_LN1_%i" type="standard_car" route="UE_LN1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 21:
                        print('    <vehicle id="UE_UW1_%i" type="standard_car" route="UE_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 22:
                        print('    <vehicle id="UE_LS1_%i" type="standard_car" route="UE_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 23:
                        print('    <vehicle id="UE_RS1_%i" type="standard_car" route="UE_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 24:
                        print('    <vehicle id="RN_UW1_%i" type="standard_car" route="RN_UW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 25:
                        print('    <vehicle id="RN_LW0_%i" type="standard_car" route="RN_LW0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 26:
                        print('    <vehicle id="RN_RS1_%i" type="standard_car" route="RN_RS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 27:
                        print('    <vehicle id="RN_LE1_%i" type="standard_car" route="RN_LE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 28:
                        print('    <vehicle id="LN_LW1_%i" type="standard_car" route="LN_LW1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 29:
                        print('    <vehicle id="LN_LS1_%i" type="standard_car" route="LN_LS1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 30:
                        print('    <vehicle id="LN_LE1_%i" type="standard_car" route="LN_LE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif random_rest == 31:
                        print('    <vehicle id="LN_UE1_%i" type="standard_car" route="LN_UE1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    
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