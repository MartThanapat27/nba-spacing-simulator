import requests
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 1. ดึงข้อมูลจาก FastAPI ของคุณ
print("Fetching data from API...")
response = requests.get("http://127.0.0.1:8000/api/players")
df = pd.DataFrame(response.json())

# 2. กรองข้อมูล Volume การยิง 3 แต้ม (บวก 1 เพื่อกันปัญหาเลข 0 ตามสูตรของเรา)
y = df['fg3a'] + 1

# 3. รัน Box-Cox Transformation
transformed_data, best_lambda = stats.boxcox(y)
print(f"\n======================================")
print(f"Optimal Lambda for FG3A is: {best_lambda:.4f}")
print(f"======================================\n")

# 4. พล็อตกราฟเช็คความสวยงาม
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
stats.boxcox_normplot(y, -2, 2, plot=ax)
ax.axvline(best_lambda, color='r', linestyle='--', label=f'Best Lambda ({best_lambda:.2f})')
plt.title("Box-Cox Transformation for 3PT Volume")
plt.legend()
plt.show()