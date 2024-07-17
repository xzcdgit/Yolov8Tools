import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 生成时间数据（一天内的每小时）
base = datetime.datetime(2023, 5, 1)
time_data = [base + datetime.timedelta(hours=x) for x in range(24)]

# 生成对应的温度数据（摄氏度）
# 这里使用正弦函数模拟一天内的温度变化，加上一些随机波动
y_data = [20 + 5 * np.sin((x - 6) * np.pi / 12) + np.random.normal(0, 0.5) for x in range(24)]

# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制数据
ax.plot(time_data, y_data, marker='o')

# 设置x轴为时间轴
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # 每两小时显示一个刻度

# 设置标题和标签
plt.title('Hourly Temperature Variation (2023-05-01)')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')

# 设置y轴范围，使图表更美观
plt.ylim(15, 30)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 调整布局
plt.gcf().autofmt_xdate()

# 显示图表
plt.tight_layout()
plt.show()

# 打印数据以供参考
for date, temp in zip(time_data, y_data):
    print(f"{date.strftime('%Y-%m-%d %H:%M')}: {temp:.1f}°C")