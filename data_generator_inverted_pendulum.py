import numpy as np
import pandas as pd
import control as ctrl

# **時間軸**
time = np.linspace(0, 10, 201)  # 0~10秒，201個點

# **倒單擺系統參數 (確保滯後但仍收斂)**
Kp = 1.5   # 增益
wn = 3.0   # 自然頻率
zeta = 1.2  # **適度阻尼，確保系統可收斂但不過度滯後**
num = [Kp * wn**2]
den = [1, 2 * zeta * wn, wn**2]
system = ctrl.TransferFunction(num, den)

# **輸入信號 (持續變化的目標)**
input_signal = np.sin(0.5 * time) + 0.2 * np.random.randn(len(time))  # **混合正弦波 + 隨機噪聲**

# **計算系統輸出**
T, output_signal = ctrl.forced_response(system, time, U=input_signal)

# **修正 `actuator`**
error_signal = input_signal - output_signal  # **計算誤差 (系統需要補償的值)**
actuator_signal = Kp * error_signal  # **使用增益來補償誤差，使系統穩定收斂**

# **建立 CSV**
df = pd.DataFrame({
    "time": time,
    "input": input_signal,
    "output": output_signal,
    "actuator": actuator_signal
})

df.to_csv("inverted_pendulum_data_tracking_fixed.csv", index=False)
print("✅ CSV 檔案已生成！輸出將滯後追蹤變動目標，控制行為更合理")
