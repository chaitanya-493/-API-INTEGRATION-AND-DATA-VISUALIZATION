import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Replace with your actual OpenWeatherMap API key
API_KEY = 'f912da9d9329e8d4bdef47e4d96eeee5'
CITY = 'Vijayawada'
URL = f'https://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric'

# Fetch data
response = requests.get(URL)
data = response.json()

# Extract relevant data
dates = []
temperatures = []

for entry in data['list']:
    dt = datetime.strptime(entry['dt_txt'], '%Y-%m-%d %H:%M:%S')
    temp = entry['main']['temp']
    dates.append(dt)
    temperatures.append(temp)

# Visualization
sns.set(style="darkgrid")
plt.figure(figsize=(12, 6))
sns.lineplot(x=dates, y=temperatures, marker='o', color='blue')
plt.title(f'Temperature Forecast for {CITY}', fontsize=16)
plt.xlabel('Date & Time')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
