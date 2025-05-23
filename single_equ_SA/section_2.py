import torch
import torch.nn as nn
import numpy as np
import math
import yaml
import os
import sys
os.chdir(sys.path[0])

class section_2:
    def __init__(self,
                 config,
                 idx,
                 ):
        
        self.config = config
        self.device = config["device"]
        self.up_bufffer_cap = config["devices"][idx-1]["buffer_capacity"]
        self.own_bufffer_cap = config["devices"][idx]["buffer_capacity"]
        self.adjustable_rate = config["devices"][idx]["adjustable_rate"]
        self.fixed_x = (config["target_pro"] / (1-self.adjustable_rate) / 7.75 / 24) * 27.972 
        self.constraint_violation_count = 0 
        self.death_reward = config["devices"][idx]["death_reward"]
        self.last_action = 0

    def apply_constraints(self, proposed_x, up_buffer, own_buffer, product, Test):
        if not Test:
            consumed_Nacl_from_1 = proposed_x
            tolerance = 1e-6
            violation = (
                (up_buffer - consumed_Nacl_from_1 < -tolerance) |
                (up_buffer - consumed_Nacl_from_1 > self.up_bufffer_cap + tolerance)|
                (proposed_x  < self.fixed_x - self.fixed_x * self.adjustable_rate - tolerance) |
                (proposed_x  > self.fixed_x + self.fixed_x * self.adjustable_rate + tolerance) 
            )

            return proposed_x, violation  
        
        else:
            consumed_Nacl_from_1 = proposed_x
            #Constraint 1: Ensure that the upstream is not negative after subtraction
            if up_buffer - consumed_Nacl_from_1 < 0:
                safe_x_1 = min(up_buffer, proposed_x)
            else:
                safe_x_1 = proposed_x

            consumed_Nacl_from_1 = safe_x_1
            # #Constraint 2: Ensure that the upstream will not be full after the reduction
            if up_buffer - consumed_Nacl_from_1 > self.up_bufffer_cap:
                safe_x_2 = max((up_buffer - self.up_bufffer_cap), safe_x_1)
            else:
                safe_x_2 = safe_x_1

            consumed_Nacl_from_1 = safe_x_2

            # Determine whether a fatal violation (any buffer overflow) is triggered
            tolerance = 1e-6
            violation = (
                (up_buffer - consumed_Nacl_from_1 < -tolerance) |
                (up_buffer - consumed_Nacl_from_1 > self.up_bufffer_cap + tolerance)|
                (safe_x_2  < self.fixed_x - self.fixed_x * self.adjustable_rate - tolerance) |
                (safe_x_2  > self.fixed_x + self.fixed_x * self.adjustable_rate + tolerance) 
            )

            return safe_x_2, violation  # Returns the action and whether it is violated
    
    
    def step(self, up_buffer_ratio, own_buffer_ratio, price, product, remaining_t, Test, training_step, action, x_milp):
        
        up_buffer = up_buffer_ratio * self.up_bufffer_cap
        own_buffer = own_buffer_ratio * self.own_bufffer_cap
        proposed_x = self.fixed_x

        safe_x, violation  = self.apply_constraints(proposed_x,  up_buffer, own_buffer, product, Test)


        if violation :
            #The amount of fresh salt water consumed in this hour:   t/h
            consumed_Nacl_from_1 = proposed_x
            #The amount of dechlorinated salt water produced in this hour:   t/h
            generated_NaCl_to_3 = proposed_x

            # Update own buffer
            own_buffer +=  generated_NaCl_to_3 
            up_buffer -= consumed_Nacl_from_1

            self.constraint_violation_count += 1  

            up_buffer_ratio = up_buffer / self.up_bufffer_cap
            own_buffer_ratio = own_buffer/self.own_bufffer_cap

            return up_buffer_ratio, own_buffer_ratio, product, torch.tensor([[self.death_reward]],dtype=torch.float32).to(self.device), True ,proposed_x, self.last_action

        consumed_Nacl_from_1 = safe_x
        generated_NaCl_to_3 = safe_x
        ele_price = 0

        # Update own buffer
        own_buffer +=  generated_NaCl_to_3 
        up_buffer -= consumed_Nacl_from_1
    
        constraint_violation = abs(safe_x - proposed_x)  

        if self.adjustable_rate != 0 :
            penalty = -(constraint_violation / (2 * self.adjustable_rate * self.fixed_x)) * 2 
        else:
            penalty = -constraint_violation * 2 * 0.5
        assert penalty <= 1e-6 and penalty>= -1- 1e-6 , f"The Penalty value is abnormal: {penalty:.4f}"

        # penalty = 0

        reward = ele_price + penalty
        up_buffer_ratio = up_buffer / self.up_bufffer_cap
        own_buffer_ratio = own_buffer/self.own_bufffer_cap
        
        return up_buffer_ratio, own_buffer_ratio, product, torch.tensor([[reward]],dtype=torch.float32).to(self.device), False, safe_x, self.last_action

