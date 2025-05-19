import numpy as np
import pandas as pd
import control as ctrl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.linalg import lstsq
from scipy.signal import savgol_filter
import argparse

# **讀取 CSV 數據**
def load_data(filename, time_col, input_col, output_col, actuator_col):
    df = pd.read_csv(filename)
    time = df[time_col].values.ravel()
    setpoint = df[input_col].values.ravel()
    feedback = df[output_col].values.ravel()
    actuator = df[actuator_col].values.ravel()
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

    # 平滑處理
    feedback = savgol_filter(feedback, window_length=11, polyorder=2)

    # 檢測峰值
    positive_peaks, _ = find_peaks(feedback, height=0.01, distance=10)
    negative_peaks, _ = find_peaks(-feedback, height=0.01, distance=10)
    all_peaks = np.sort(np.concatenate((positive_peaks, negative_peaks)))

    if len(all_peaks) > 1:
        # 基於衰減比計算
        peak_values = feedback[all_peaks]
        print(f"All peaks: {all_peaks}, Peak values: {peak_values}")  # 調試用
        decay_ratio = abs(peak_values[1]) / abs(peak_values[0])
        decay_ratio = max(min(decay_ratio, 0.99), 0.01)  # 限制範圍
        zeta_decay = -np.log(decay_ratio) / np.sqrt(np.pi**2 + np.log(decay_ratio)**2)

        # 基於振盪頻率計算
        time_intervals = np.diff(all_peaks)  # 峰值間距
        if len(time_intervals) > 0:
            avg_period = np.mean(time_intervals)  # 平均周期
            wd = 2 * np.pi / avg_period  # 振盪頻率
            zeta_freq = np.sqrt(1 - (wd / wn)**2) if wd < wn else 0.7  # 確保值合理
        else:
            zeta_freq = 0.7  # 默認值

        # 綜合兩種方法
        zeta = (zeta_decay + zeta_freq) / 2
    elif len(all_peaks) == 1:
        zeta = 0.7  # 假設阻尼比為中等值
    else:
        zeta = 1.0  # 假設系統是過阻尼的

    tau = estimate_time_constant(wn, zeta)  # 計算時間常數
    return Kp, wn, zeta, tau

# **建立傳遞函數**
def create_transfer_function(order, time, setpoint, feedback, actuator, use_actuator):
    """根據指定的階數建立傳遞函數"""
    Kp, wn, zeta, tau = estimate_system_parameters(setpoint, feedback, actuator, use_actuator)

    if order == 1:
        # 一階模型
        num = [Kp]
        den = [tau, 1]
    elif order == 2:
        # 二階模型
        num = [Kp * wn**2]
        den = [1, 2 * zeta * wn, wn**2]
    elif order == 3:
        # 三階模型
        num = [Kp * wn**2]
        den = [tau, 1, 2 * zeta * wn, wn**2]
    else:
        raise ValueError("Unsupported model order. Please choose 1, 2, or 3.")

    system = ctrl.TransferFunction(num, den)
    return system, Kp, wn, zeta, tau

# **計算步階響應**
def compute_step_response(system):
    time_out, response = ctrl.step_response(system, T=np.linspace(0, 10, 201))
    return time_out, response

# **繪製步階響應**
def plot_step_response(time_out, response, Kp, wn, zeta, tau, order, use_actuator):
    actuator_str = "with Actuator" if use_actuator else "without Actuator"
    plt.figure(figsize=(10, 5))
    plt.plot(time_out, response, label=f'Step Response (Order={order}, {actuator_str})\nKp={Kp:.2f}, wn={wn:.2f}, ζ={zeta:.2f}, τ={tau:.2f}', color='blue')
    plt.axhline(y=1, color='r', linestyle='--', label='Target: 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.legend()
    plt.title(f'System Identification - Step Response (Order={order}, {actuator_str})')
    plt.grid()
    plt.show()

# **執行分析**
if __name__ == "__main__":
    # **解析命令列參數**
    parser = argparse.ArgumentParser(description="System Identification with Transfer Function Models")
    parser.add_argument("--filename", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--order", type=int, required=True, choices=[1, 2, 3], help="Order of the transfer function (1, 2, or 3)")
    parser.add_argument("--use_actuator", action="store_true", help="Include actuator in the system identification")
    parser.add_argument("--time_col", type=str, default="time", help="Column name for time (default: time)")
    parser.add_argument("--input_col", type=str, default="input", help="Column name for input/setpoint (default: input)")
    parser.add_argument("--output_col", type=str, default="output", help="Column name for output/feedback (default: output)")
    parser.add_argument("--actuator_col", type=str, default="actuator", help="Column name for actuator (default: actuator)")
    args = parser.parse_args()


    # **讀取數據**
    filename = args.filename
    order = args.order
    use_actuator = args.use_actuator
    time, setpoint, feedback, actuator = load_data(
        args.filename, args.time_col, args.input_col, args.output_col, args.actuator_col
    )

    # **不使用 `actuator`**
    system_no_act, Kp_no_act, wn_no_act, zeta_no_act, tau_no_act = create_transfer_function(order, time, setpoint, feedback, actuator, use_actuator=False)
    time_out, response = compute_step_response(system_no_act)
    plot_step_response(time_out, response, Kp_no_act, wn_no_act, zeta_no_act, tau_no_act, order, use_actuator=False)

    # **使用 `actuator`（僅在指定 --use_actuator 時執行）**
    if use_actuator:
        system_act, Kp_act, wn_act, zeta_act, tau_act = create_transfer_function(order, time, setpoint, feedback, actuator, use_actuator=True)
        time_out, response = compute_step_response(system_act)
        plot_step_response(time_out, response, Kp_act, wn_act, zeta_act, tau_act, order, use_actuator=True)