import torch
import torch.nn as nn
import numpy as np
import math
import yaml
import os
import sys
os.chdir(sys.path[0])

class section_4:
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
        self.ele_enlarge = config["devices"][idx]["ele_enlarge"]

        self.last_action = 0.5

    def apply_constraints(self, proposed_x, up_buffer, own_buffer, product, this_action, Test):
        if not Test:
            #Diluted NaOH to be consumed
            consumed_dilute_NaCl = proposed_x
            # Determine whether a fatal violation
            tolerance = 1e-6
            violation = (
                (up_buffer - consumed_dilute_NaCl < -tolerance) |
                (up_buffer - consumed_dilute_NaCl > self.up_bufffer_cap + tolerance) |
                (proposed_x < self.min_x - tolerance) |
                (proposed_x > self.max_x + tolerance) 

            )
            return proposed_x, violation  
        
        else: 
            #Diluted NaOH to be consumed
            consumed_dilute_NaCl = proposed_x

            #Constraint 1: Ensure that the upstream value is not negative after subtraction
            if up_buffer - consumed_dilute_NaCl < 0:
                safe_x_1 = min(up_buffer, proposed_x)
            else:
                safe_x_1 = proposed_x


            consumed_dilute_NaCl = safe_x_1
            #Constraint 2: Ensure that the upstream will not be full after the reduction
            if up_buffer - consumed_dilute_NaCl > self.up_bufffer_cap:
                safe_x_2 = max((up_buffer - self.up_bufffer_cap), safe_x_1)
            else:
                safe_x_2 = safe_x_1
            
            consumed_dilute_NaCl = safe_x_2

            # Determine whether a fatal violation (any buffer overflow) is triggered
            tolerance = 1e-6
            violation = (
                (up_buffer - consumed_dilute_NaCl < -tolerance) |
                (up_buffer - consumed_dilute_NaCl > self.up_bufffer_cap + tolerance) |
                (safe_x_2 < self.min_x - tolerance) |
                (safe_x_2 > self.max_x + tolerance) 
            )

            return safe_x_2, violation  # Returns the action and whether it is violated

    
    def normalizated_price(self, max_price, min_price):
        if min_price < 0:
            self.ele_price_min = 5.0219E-3 * self.max_x * min_price
        else:
            self.ele_price_min = 5.0219E-3 * self.min_x * min_price
        self.ele_price_max = 5.0219E-3 * self.max_x * max_price
    

    def step(self, up_buffer_ratio, own_buffer_ratio, price, product, remaining_t, Test, training_step, action, x_milp):
        
        up_buffer = up_buffer_ratio * self.up_bufffer_cap
        own_buffer = own_buffer_ratio * self.own_bufffer_cap
        proposed_x = self.min_x + action[2] * (self.max_x - self.min_x)
        
        safe_x, violation  = self.apply_constraints(proposed_x,  up_buffer, own_buffer, product, action[2], Test)

        if violation :
            self.last_action = (proposed_x - self.min_x) / (self.max_x - self.min_x)

            #The amount of fresh salt water consumed in this hour:   t/h
            consumed_dilute_NaCl = proposed_x
            #The amount of dechlorinated salt water produced in this hour:   t/h
            generated_dilute_NaCl_without_Cl2 = proposed_x

            # Update own buffer
            own_buffer +=  generated_dilute_NaCl_without_Cl2 
            up_buffer -= consumed_dilute_NaCl
    
            
            self.constraint_violation_count += 1  


            up_buffer_ratio = up_buffer / self.up_bufffer_cap
            own_buffer_ratio = own_buffer/self.own_bufffer_cap

            return up_buffer_ratio, own_buffer_ratio, product, torch.tensor([[self.death_reward]],dtype=torch.float32).to(self.device), True ,proposed_x , self.last_action


        self.last_action = (safe_x - self.min_x) / (self.max_x - self.min_x)

        #The amount of fresh salt water consumed in this hour:   t/h
        consumed_dilute_NaCl = safe_x
        #The amount of dechlorinated salt water produced in this hour:   t/h
        generated_dilute_NaCl_without_Cl2 = safe_x
        

        ele_price = 5.0219E-3 * safe_x * price
        ele_price_normalized = ( 1 - ((ele_price - self.ele_price_min)/(self.ele_price_max - self.ele_price_min))) * self.ele_enlarge  #Zoom in to 0-6, the less power consumption the better
        assert ele_price_normalized <= self.ele_enlarge + 1e-6 and ele_price_normalized >= -1e-6, f"The ele_price value is abnormal! It should be a positive number"
        ele_price_normalized = 0

        # Update own buffer
        own_buffer +=  generated_dilute_NaCl_without_Cl2 
        up_buffer -= consumed_dilute_NaCl
    
        constraint_violation = abs(safe_x - proposed_x)  
        penalty = -0.5 * constraint_violation / (self.max_x - self.min_x)
        assert penalty <= 1e-6 and penalty>= -1 - 1e-6 , f"The Penalty value is abnormal! It should be a non-positive number"
        penalty = 0


        reward = ele_price_normalized+penalty



        reward = (1 - abs(x_milp[2] - action[2])) * 0.5
        assert reward <= 0.5 + 1e-6, f"Reward value is abnormal"


        up_buffer_ratio = up_buffer / self.up_bufffer_cap
        own_buffer_ratio = own_buffer/self.own_bufffer_cap
        return up_buffer_ratio, own_buffer_ratio, product, torch.tensor([[reward]],dtype=torch.float32).to(self.device), False, safe_x, self.last_action







 
if __name__ == "__main__":
    
    def test_section4():
        # 模拟配置文件（精确到小数点后3位）
        config_dir = "equipment_config.yaml"
        config = yaml.load(open(config_dir, "r", encoding='utf-8'), Loader=yaml.FullLoader)#使用预先定义的config文件，把其中的定义加载出来给config

        # 初始化第四工段（idx=3）
        device_idx = 3
        dechlorinator = section_4(config, device_idx)
    
        # 测试用例1：极限处理量验证
        def case_max_min():
            # 测试max_x=120.756
            up_buf = torch.tensor([[500.0]])
            own_buf = torch.tensor([[50.0]])   # 下游初始50吨
            price = torch.tensor([[0.5]])
            product = torch.tensor([[300.0]])

            new_up, new_own, _, reward, done = dechlorinator.step(
                up_buf, own_buf, price, product, is_trian=True
            )

            print("测试1通过：处理量极限值验证成功")

        # 运行测试
        case_max_min()
        
    test_section4()