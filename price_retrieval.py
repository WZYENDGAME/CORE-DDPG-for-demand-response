import requests
import datetime
import os
import sys
import random
from collections import defaultdict

os.chdir(sys.path[0])


class PriceQuery:
    """Electricity Price Data Query"""
    def __init__(self):
        self.cache = {}
        
    def get_price(self, year, month, day, hour):
        """Method 1: Query at a precise time point"""
        target_time = datetime.datetime(year, month, day, hour)
        if target_time in self.cache:
            return self.cache[target_time]
        
        file_path = f"price_data/{year}/{month:02d}.csv"
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    dt_str, price = line.strip().split(',')
                    dt = datetime.datetime.fromisoformat(dt_str)
                    if dt == target_time.replace(tzinfo=datetime.timezone.utc):
                        self.cache[dt] = float(price)
                        return float(price)
        return None

    def get_24h_test(self, start_time):
        """Method 2: Obtain 25 consecutive hours of data"""
        sequence = []
        current_time = start_time
        for _ in range(25):
            price = self.get_price(
                current_time.year,
                current_time.month,
                current_time.day,
                current_time.hour
            )
            if price is None:
                raise ValueError(f"without {current_time} ")
            sequence.append(price)
            current_time += datetime.timedelta(hours=1)

            for i in range(len(sequence)):
                if sequence[i] < -5:  # Extremely low price protection
                    sequence[i] = -5
                elif sequence[i] > 20:  # Appropriately relax the upper limit (leave a safety margin)
                    sequence[i] = 20
        return sequence
    

    def generate_random_datetime(self):
        year = 2024  
        current_month = random.randint(1, 12)
        if current_month == 2 or current_month == 12:
            current_day = random.randint(1, 29)
        else:
            current_day = random.randint(1, 30)
        current_hour = random.randint(0, 23)

        current_datetime = datetime.datetime(year, current_month, current_day, current_hour)

        if  current_datetime.month == 1 and current_datetime.day == 1:
            previous_24h_datetime = current_datetime
        else:
            previous_24h_datetime = current_datetime - datetime.timedelta(hours=24)


        return current_datetime , previous_24h_datetime


    def get_24h_train(self):
        """Method 3: Randomly obtain 25 consecutive hours of data"""
        current_sequence = []
        last_sequence = []
        current_time, previous_24h_datetime = self.generate_random_datetime()
        

        for _ in range(25):
            price = self.get_price(
                current_time.year,
                current_time.month,
                current_time.day,
                current_time.hour
            )
            if price is None:
                raise ValueError(f"without {current_time} ")
            current_sequence.append(price)
            current_time += datetime.timedelta(hours=1)

        for i in range(len(current_sequence)):
            if current_sequence[i] < -5:  # Extremely low price protection
                current_sequence[i] = -5
            elif current_sequence[i] > 20:  # Appropriately relax the upper limit (leave a safety margin)
                current_sequence[i] = 20

        for _ in range(24):
            price = self.get_price(
                previous_24h_datetime.year,
                previous_24h_datetime.month,
                previous_24h_datetime.day,
                previous_24h_datetime.hour
            )
            if price is None:
                raise ValueError(f"without {current_time} ")
            last_sequence.append(price)
            previous_24h_datetime += datetime.timedelta(hours=1)

        for i in range(len(last_sequence)):
            if last_sequence[i] < -5:  
                last_sequence[i] = -5
            elif last_sequence[i] > 20:  
                last_sequence[i] = 20
            
        return current_sequence , sum(last_sequence[:24])/24

