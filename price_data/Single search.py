import requests
import datetime
from collections import defaultdict

"""
Note that this does not have a save function.
This code prefers that after you enter a period of time, it will directly output the time of each hour.
It is suitable for single retrieval of data at a specific time, rather than for large-scale saving.
Note: The 2024 1102 2355 you enter may not be the real 11:55 on the 2nd, because it has some time zone difference that makes people confused.
"""

def get_hourly_pricing():
    # ask for data
    url = "https://hourlypricing.comed.com/api"
    params = {
        "type": "5minutefeed",
        "datestart": "202411022355",
        "dateend": "202411040100",
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
            
            # Generate hour key YYYY-MM-DDTHH:00:00+00:00）
            hour_key = dt_utc.replace(minute=0, second=0, microsecond=0) - datetime.timedelta(hours=4)

            hourly_prices[hour_key].append(float(price_str))
        except Exception as e:
            print(f"Data parsing failed:{group} -> {str(e)}")
            continue

    hourly_avg = []
    for hour in sorted(hourly_prices.keys()):
        prices = hourly_prices[hour]
        if len(prices) != 12:
            print(f"Warning: Hours {hour.isoformat()} The data is incomplete {len(prices)} Group")
        avg_price = sum(prices) / len(prices)
        hourly_avg.append((hour.isoformat(), avg_price))

    # Output
    print("UTC：")
    for hour, price in hourly_avg:
        print(f"{hour} -> {price:.2f} $/MWh")

# Execute function
if __name__ == "__main__":
    get_hourly_pricing()