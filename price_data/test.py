import os
import csv
from datetime import datetime, timedelta
from dateutil import tz  
import sys
os.chdir(sys.path[0])


"""
It is better to open the file yourself and calculate whether the number of rows at different time points is different. It will be faster to find the answer"""

def check_data_integrity(start_str, end_str, timezone='America/New_York'):
    tz_obj = tz.gettz(timezone)
    
    start_naive = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
    end_naive = datetime.strptime(end_str, "%Y-%m-%d %H:%M")
    start_time = start_naive.replace(tzinfo=tz_obj)
    end_time = end_naive.replace(tzinfo=tz_obj)

    expected = []
    current = start_time
    while current <= end_time:
        expected.append(current)
        current += timedelta(hours=1)

    actual = set()
    for year, month in get_months_range(start_time, end_time):
        file_path = f"data/{year}/{month:02d}.csv"
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 1: continue
                    dt_utc = datetime.fromisoformat(row[0].replace('Z', '+00:00'))
                    local_time = dt_utc.astimezone(tz_obj)
                    if start_time <= local_time <= end_time:
                        actual.add(local_time)

    # 比较差异
    expected_set = set(expected)
    missing = sorted(expected_set - actual)
    
    # 打印报告
    print(f"Number of data points that should exist{len(expected)}")
    print(f"The actual number of data points:{len(actual)}")
    print(f"Number of missing data points:{len(missing)}")
    
    if missing:
        print("\nList of missing time points (local time):：")
        for dt in missing:
            print(dt.strftime("%Y-%m-%d %H:%M"))
    
    return missing

def get_months_range(start, end):
    months = set()
    current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while current <= end:
        months.add((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year+1, month=1)
        else:
            current = current.replace(month=current.month+1)
    return sorted(months)

if __name__ == "__main__":
    missing = check_data_integrity(
        start_str="2024-11-01 00:00",
        end_str="2024-11-30 23:00",
        timezone='America/New_York'
    )