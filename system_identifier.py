import numpy as np
import pandas as pd
import control as ctrl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.linalg import lstsq

# **讀取 CSV 數據**
def load_data(filename):
    df = pd.read_csv(filename)
    time = df["time"].values.ravel()
    setpoint = df["input"].values.ravel()
    feedback = df["output"].values.ravel()
    actuator = df["actuator"].values.ravel()
    return time, setpoint, feedback, actuator

# **估計時間常數**
def estimate_time_constant(wn, zeta):
    """時間常數 τ = 1 / (ζ * ω_n)"""
    tau = 1 / (zeta * wn) if zeta * wn > 0 else 1  # 避免除零錯誤
    return tau

# **擬合系統參數**
def estimate_system_parameters(setpoint, feedback, actuator, use_actuator):
    """使用最小二乘法擬合 ARX 模型，根據 use_actuator 來決定是否納入 actuator"""
    if use_actuator:
        A = np.vstack([setpoint, actuator, np.ones_like(setpoint)]).T
    else:
        A = np.vstack([setpoint, np.ones_like(setpoint)]).T
        
    B = feedback
    coeffs, _, _, _ = lstsq(A, B)  # 最小二乘法擬合

    Kp = coeffs[0]  # 增益
    tau_raw = 1 / max(abs(coeffs))  # 初步時間常數估算
    wn = 1 / tau_raw  # 自然頻率
    zeta = 0.7  # 阻尼比 (預設值)
    tau = estimate_time_constant(wn, zeta)  # 計算時間常數

    return Kp, wn, zeta, tau

# **建立傳遞函數**
def create_transfer_function(time, setpoint, feedback, actuator, use_actuator):
    """根據 `use_actuator` 來決定是否納入 `actuator` 參考"""
    Kp, wn, zeta, tau = estimate_system_parameters(setpoint, feedback, actuator, use_actuator)

    num = [Kp * wn**2]  
    den = [1, 2 * zeta * wn, wn**2]
    system = ctrl.TransferFunction(num, den)

    return system, Kp, wn, zeta, tau

# **計算步階響應**
def compute_step_response(system):
    time_out, response = ctrl.step_response(system, T=10)
    return time_out, response

# **繪製步階響應**
def plot_step_response(time_out, response, Kp, wn, zeta, tau, use_actuator):
    actuator_str = "with Actuator" if use_actuator else "without Actuator"
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_out, response, label=f'Step Response {actuator_str} (Kp={Kp:.2f}, wn={wn:.2f}, ζ={zeta:.2f}, τ={tau:.2f})', color='blue')
    plt.axhline(y=1, color='r', linestyle='--', label='Target: 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.xlim([0, 10])
    plt.legend()
    plt.title(f'System Identification - Step Response ({actuator_str})')
    plt.show()

# **執行分析**
if __name__ == "__main__":
    filename = "inverted_pendulum_data_tracking_fixed.csv"
    time, setpoint, feedback, actuator = load_data(filename)

    # **不使用 `actuator`**
    system_no_act, Kp_no_act, wn_no_act, zeta_no_act, tau_no_act = create_transfer_function(time, setpoint, feedback, actuator, use_actuator=False)
    time_out, response = compute_step_response(system_no_act)
    plot_step_response(time_out, response, Kp_no_act, wn_no_act, zeta_no_act, tau_no_act, use_actuator=False)

    # **使用 `actuator`**
    system_act, Kp_act, wn_act, zeta_act, tau_act = create_transfer_function(time, setpoint, feedback, actuator, use_actuator=True)
    time_out, response = compute_step_response(system_act)
    plot_step_response(time_out, response, Kp_act, wn_act, zeta_act, tau_act, use_actuator=True)
