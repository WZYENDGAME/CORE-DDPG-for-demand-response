import torch
import torch.nn as nn
import numpy as np
import math
import yaml
import os
import sys
os.chdir(sys.path[0])

class section_3:
    def __init__(self,
                 config,
                 idx,
                 ):
        
        self.config = config
        self.device = config["device"]
        self.up_bufffer_cap = config["devices"][idx-1]["buffer_capacity"]
        self.own_bufffer_cap = config["devices"][idx]["buffer_capacity"]
        self.pro_margin = config["devices"][idx]["pro_margin"]
        self.max_x = config["devices"][idx]["max_x"]
        self.min_x = config["devices"][idx]["min_x"]
        self.target_pro = config["target_pro"]
        self.constraint_violation_count = 0 
        
        self.death_reward = config["devices"][idx]["death_reward"]
        self.ele_enlarge = config["devices"][idx]["ele_enlarge"]

        self.last_action = 0.5


    def apply_constraints(self, proposed_x, up_buffer, own_buffer, product, remaining_t, this_action, Test):
        if not Test:
            consumed_concentrated = 27.972 * proposed_x
            produced_NaOH = 7.75 * proposed_x
            tolerance = 1e-6
            violation = (
                (up_buffer - consumed_concentrated < -tolerance) |
                (up_buffer - consumed_concentrated > self.up_bufffer_cap + tolerance) |
                (proposed_x < self.min_x - tolerance) |
                (proposed_x > self.max_x + tolerance) 
            )

            return proposed_x, violation  
        
        else:
            safe_x_0 = proposed_x

            #Constraint 1: The amount of NaOH produced should not exceed the allowance. 937.5 = 300/0.32
            max_pro_x = (937.5 + self.pro_margin - product) / 7.75
            safe_x_1 = min(safe_x_0, max_pro_x)
            safe_x_1 = safe_x_0
            consumed_concentrated = 27.972 * safe_x_1

            #Constraint 2: Ensure that the upstream is not negative after subtraction
            if up_buffer - consumed_concentrated < -(1e-6):
                safe_x_2 = min((up_buffer / 27.972), safe_x_1)
            else:
                safe_x_2 = safe_x_1

            consumed_concentrated = 27.972 * safe_x_2
            #Constraint 3: Ensure that the upstream will not be full after the reduction
            if up_buffer - consumed_concentrated > self.up_bufffer_cap:
                safe_x_3 = max((up_buffer - self.up_bufffer_cap) / 27.972 , safe_x_2)
            else:
                safe_x_3 = safe_x_2

            consumed_concentrated = 27.972 * safe_x_3
            produced_NaOH = 7.75 * safe_x_3

            # Determine whether a fatal violation (any buffer overflow) is triggered
            tolerance = 1e-6
            violation = (
                (up_buffer - consumed_concentrated < -tolerance) |
                (up_buffer - consumed_concentrated > self.up_bufffer_cap + tolerance) |
                (safe_x_3 < self.min_x - tolerance) |
                (safe_x_3 > self.max_x + tolerance) 
            )

            return safe_x_3, violation  # Returns the action and whether it is violated



    def normalizated_price(self, max_price, min_price):
        if min_price < 0:
            self.ele_price_min = (272.61 * self.max_x * self.max_x + 3937.5 * self.max_x)* min_price
        else:
            self.ele_price_min = (272.61 * self.min_x * self.min_x + 3937.5 * self.min_x)* min_price
        self.ele_price_max = (272.61 * self.max_x * self.max_x + 3937.5 * self.max_x)* max_price


    
    def step(self, up_buffer_ratio, own_buffer_ratio, price, product, remaining_t, Test, training_step, action, x_milp):
        
        up_buffer = up_buffer_ratio * self.up_bufffer_cap
        own_buffer = own_buffer_ratio * self.own_bufffer_cap
        proposed_x = self.min_x + action[1] * (self.max_x - self.min_x)

        # Apply physics constraints
        safe_x, violation  = self.apply_constraints(proposed_x,  up_buffer, own_buffer, product, remaining_t, action[1], Test)
        if violation :
            self.last_action = (proposed_x - self.min_x) / (self.max_x - self.min_x)
            
            #32% NaOH solution produced in this hour:   t/h
            produced_NaOH = 7.75 * proposed_x
            #Concentrated NaCl solution consumed in this hour:   t/h
            consumed_concentrated_NaCl = 27.972 * proposed_x
            #The dilute NaCl solution produced in this hour:   t/h
            generated_dilute_NaCl = 20.126 * proposed_x

            own_buffer +=  generated_dilute_NaCl 
            up_buffer -= consumed_concentrated_NaCl
            product += produced_NaOH
            self.constraint_violation_count += 1  


            up_buffer_ratio = up_buffer / self.up_bufffer_cap
            own_buffer_ratio = own_buffer/self.own_bufffer_cap
            
            return up_buffer_ratio, own_buffer_ratio, product, torch.tensor([[self.death_reward]],dtype=torch.float32).to(self.device), True ,proposed_x, self.last_action

        self.last_action = (safe_x - self.min_x) / (self.max_x - self.min_x)
        #32% NaOH solution produced in this hour:   t/h
        produced_NaOH = 7.75 * safe_x
        #Concentrated NaCl solution consumed in this hour:   t/h
        consumed_concentrated_NaCl = 27.972 * safe_x
        #The dilute NaCl solution produced in this hour:   t/h
        generated_dilute_NaCl = 20.126 * safe_x
        
        
        ele_price = (272.61 * safe_x * safe_x + 3937.5 * safe_x)* price 
        ele_price_normalized = ( 1 - ((ele_price - self.ele_price_min)/(self.ele_price_max - self.ele_price_min))) * self.ele_enlarge  
        assert ele_price_normalized <= self.ele_enlarge + 1e-6 and ele_price_normalized >= -1e-6, f"The ele_price value is abnormal"
        ele_price_normalized = 0

        required_pro = max((self.target_pro - product) / remaining_t , 0)  #0
        pro_penalty = min( (produced_NaOH - required_pro)/(required_pro + 1e-6) , 0) 
        assert pro_penalty <= 1e-6 and pro_penalty >= -1 - 1e-6, f"The pro_penalty value is abnormal"
        pro_penalty = 0



        # Update own buffer
        own_buffer +=  generated_dilute_NaCl 
        up_buffer -= consumed_concentrated_NaCl
        product += produced_NaOH

	    # Constraint Penalty
        constraint_violation = abs(safe_x - proposed_x)  
        penalty = -0.5 * constraint_violation / (self.max_x - self.min_x)
        assert penalty <= 1e-6 and penalty>= -1 - 1e-6 , f"The Penalty value is abnormal! It should be a non-positive number. The actual value: {penalty:.4f}"
        penalty = 0


        if remaining_t <= 5 :
            reward = ele_price_normalized + penalty + pro_penalty * (1 + (6 - remaining_t) * 0.1)  
        else:
            reward = ele_price_normalized + penalty + pro_penalty



        reward = (1 - abs(x_milp[1] - action[1])) * 3
        assert reward <= 3 + 1e-6, f"Reward value is abnormal"

        up_buffer_ratio = up_buffer / self.up_bufffer_cap
        own_buffer_ratio = own_buffer/self.own_bufffer_cap

        return up_buffer_ratio, own_buffer_ratio, product, torch.tensor([[reward]],dtype=torch.float32).to(self.device),  False, safe_x, self.last_action


