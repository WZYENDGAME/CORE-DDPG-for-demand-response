import requests
import datetime
import os
import sys
from collections import defaultdict
# 设置工作目录到脚本所在目录
os.chdir(sys.path[0])



"""
Large-scale saving and reading
Remember around line 71, if you read files from other years, remember to change the folder name
"""


def get_hourly_pricing():
    """Acquire and store electricity price data"""
    # 请求数据
    url = "https://hourlypricing.comed.com/api"
    params = {
        "type": "5minutefeed",
        "datestart": "202301312355",
        "dateend": "202402292355",
        "format": "text"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Request failed, status code:{response.status_code}")
        return

    # Processing raw data
    data_str = response.text.strip()
    groups = [g for g in data_str.split(',') if g]

    # Group by hour
    hourly_prices = defaultdict(list)
    for group in groups:
        try:
            timestamp_str, price_str = group.split(':')
            timestamp_ms = int(timestamp_str)
            timestamp_sec = timestamp_ms / 1000
            dt_utc = datetime.datetime.fromtimestamp(timestamp_sec, datetime.timezone.utc)
            
            # （UTC-4）
            local_time = dt_utc - datetime.timedelta(hours=4)
            hour_key = local_time.replace(minute=0, second=0, microsecond=0)
            hourly_prices[hour_key].append(float(price_str))
        except Exception as e:
            print(f"Data parsing failed:{group} -> {str(e)}")
            continue

    # Calculate hourly averages
    hourly_avg = []
    for hour in sorted(hourly_prices.keys()):
        prices = hourly_prices[hour]
        if len(prices) != 12:
            print(f"warning {hour.isoformat()} The data is incomplete (there should be 12 groups, but the actual {len(prices)} ")
        avg_price = sum(prices) / len(prices)
        hourly_avg.append((hour.isoformat(), avg_price))

    # save
    save_hourly_data(hourly_avg)
    return hourly_avg

def save_hourly_data(hourly_avg):
    """Storing processed data"""
    data_dir = datetime.datetime.now().strftime("data/2025")
    os.makedirs(data_dir, exist_ok=True)
    
    for hour, price in hourly_avg:
        dt = datetime.datetime.fromisoformat(hour)
        monthly_file = f"{data_dir}/{dt.strftime('%m')}.csv"
        latest_file = "data/latest.csv"
        
        if not os.path.exists(monthly_file) or dt > last_recorded_time(monthly_file):
            with open(monthly_file, 'a') as f:
                f.write(f"{hour},{price:.4f}\n")
                

def last_recorded_time(filename):
    """Get the last record time of the file"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].split(',')[0]
                return datetime.datetime.fromisoformat(last_line)
    return datetime.datetime.min

def update_latest(filename, hour, price):
    """Maintain the data files for the last 24 hours"""
    new_line = f"{hour},{price:.4f}\n"
    
    if os.path.exists(filename):
        with open(filename, 'r+') as f:
            lines = f.readlines()
            # Remove duplicates and keep the last 23 items
            lines = [line for line in lines if not line.startswith(hour)]
            lines = lines[-23:] + [new_line]
            f.seek(0)
            f.writelines(lines)
            f.truncate()
    else:
        with open(filename, 'w') as f:
            f.write(new_line)

class PriceQuery:
    """Electricity Price Data Query"""
    def __init__(self):
        self.cache = {}
        
    def get_price(self, year, month, day, hour):
        """Method 1: Query at a precise time point"""
        target_time = datetime.datetime(year, month, day, hour)
        if target_time in self.cache:
            return self.cache[target_time]
        
        file_path = f"data/{year}/{month:02d}.csv"
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    dt_str, price = line.strip().split(',')
                    dt = datetime.datetime.fromisoformat(dt_str)
                    if dt == target_time.replace(tzinfo=datetime.timezone.utc):
                        self.cache[dt] = float(price)
                        return float(price)
        return None

    def get_24h_sequence(self, start_time):
        """Method 2: Obtain 24-hour continuous data"""
        sequence = []
        current_time = start_time
        for _ in range(24):
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
        return sequence

if __name__ == "__main__":
    get_hourly_pricing()
    
    query = PriceQuery()
