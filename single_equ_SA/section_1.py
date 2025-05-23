import torch
import torch.nn as nn
import numpy as np
import math
import yaml
import os
import sys
os.chdir(sys.path[0])

class section_1:
    def __init__(self,
                 config,
                 idx,
                 ):

        self.config = config
        self.device = config["device"]
        self.up_bufffer_cap = config["devices"][idx-1]["buffer_capacity"]
        self.own_bufffer_cap = config["devices"][idx]["buffer_capacity"]
        self.max_x = config["devices"][idx]["max_x"]
        self.min_x = config["devices"][idx]["min_x"]
        self.constraint_violation_count = 0 
        self.death_reward = config["devices"][idx]["death_reward"]
        self.last_action = 0.5
        

    
    def apply_constraints(self, proposed_x, up_buffer, own_buffer, product, this_action, Test):
        if not Test: 
            #To consume dechlorinated NaCL
            consumed_dilute_NaCl_wtCl2 = proposed_x
            tolerance = 1e-6
            violation = (
                (up_buffer - consumed_dilute_NaCl_wtCl2 < -tolerance) |
                (up_buffer - consumed_dilute_NaCl_wtCl2 > self.up_bufffer_cap + tolerance) |
                (proposed_x < self.min_x - tolerance) |
                (proposed_x > self.max_x + tolerance) 
            )

            return proposed_x, violation  
        
        else:
            consumed_dilute_NaCl_wtCl2 = proposed_x
            if up_buffer - consumed_dilute_NaCl_wtCl2 < 0:
                safe_x_1 = min(up_buffer, proposed_x)
            else:
                safe_x_1 = proposed_x

            consumed_dilute_NaCl_wtCl2 = safe_x_1
            if up_buffer - consumed_dilute_NaCl_wtCl2 > self.up_bufffer_cap:
                safe_x_2 = max((up_buffer - self.up_bufffer_cap), safe_x_1)
            else:
                safe_x_2 = safe_x_1

            consumed_dilute_NaCl_wtCl2 = safe_x_2
            tolerance = 1e-6
            violation = (
                (up_buffer - consumed_dilute_NaCl_wtCl2 < -tolerance) |
                (up_buffer - consumed_dilute_NaCl_wtCl2 > self.up_bufffer_cap + tolerance) |
                (safe_x_2 < self.min_x - tolerance) |
                (safe_x_2 > self.max_x + tolerance) 
            )
            return safe_x_2, violation  
        

    def step(self, up_buffer_ratio, own_buffer_ratio, price, product, remaining_t, Test, training_step, action, x_milp):
        
        up_buffer = up_buffer_ratio * self.up_bufffer_cap
        own_buffer = own_buffer_ratio * self.own_bufffer_cap
        proposed_x = self.min_x + action[0] * (self.max_x - self.min_x)

        safe_x, violation  = self.apply_constraints(proposed_x,  up_buffer, own_buffer, product, action[0], Test)
        
        
        if violation :

            self.last_action = (proposed_x - self.min_x) / (self.max_x - self.min_x)

            consumed_dilute_NaCl_wtCl2 = proposed_x
            generated_concentrated_NaCl = (proposed_x / 20.126) * 27.972

            own_buffer +=  generated_concentrated_NaCl 
            up_buffer -= consumed_dilute_NaCl_wtCl2

            self.constraint_violation_count += 1  

            up_buffer_ratio = up_buffer / self.up_bufffer_cap
            own_buffer_ratio = own_buffer/self.own_bufffer_cap

            return up_buffer_ratio, own_buffer_ratio, product, torch.tensor([[self.death_reward]],dtype=torch.float32).to(self.device), True , proposed_x, self.last_action


        self.last_action = (safe_x - self.min_x) / (self.max_x - self.min_x)
        consumed_dilute_NaCl_wtCl2 = safe_x
        generated_concentrated_NaCl = (safe_x/ 20.126) * 27.972
        ele_price = 0

        own_buffer +=  generated_concentrated_NaCl 
        up_buffer -= consumed_dilute_NaCl_wtCl2
    

    
        constraint_violation = abs(safe_x - proposed_x)  
        penalty = -0.5 * constraint_violation / (self.max_x - self.min_x) 
        assert penalty <= 1e-6 and penalty>= -1- 1e-6 , f"The Penalty value is abnormal: {penalty:.4f}"


        reward = (1 - abs(x_milp[0] - action[0])) * 0.5
        assert reward <= 0.5 + 1e-6, f"Reward abnormal！"


        up_buffer_ratio = up_buffer / self.up_bufffer_cap
        own_buffer_ratio = own_buffer/self.own_bufffer_cap

        return up_buffer_ratio, own_buffer_ratio, product, torch.tensor([[reward]],dtype=torch.float32).to(self.device), False, safe_x, self.last_action
    
     



