# PID Analyzer for UAV Rate Control Gain Tuning

This project is a Python-based tool for system identification and PID tuning, specifically designed for UAV rate control gain tuning. Unlike other tools that rely on `.exe` files or MATLAB's PID Toolbox, this project leverages the Python `control` library to provide an open-source and platform-independent solution. The tool is capable of analyzing real-world data and performing qualitative analysis.

## Features
- **System Identification**: Analyze system dynamics using real-world data.
- **PID Tuning**: Generate data for PID tuning with adjustable parameters.
- **CSV Output**: Automatically save results in a structured CSV format.
- **Visualization**: Generate step response plots for system analysis.
- **Log Compatibility**: Supports PX4 `ulog` and Betaflight logs for seamless integration.

## To-Do
- **Validated Analysis**: Perform multiple tests to ensure the reliability of the analysis methods.
- **Flexible Input/Output**: Add more adjustable file input and output mechanisms.
- **Model Estimation**: Implement additional estimation models for system identification.
- **Automated Results**: Enhance automation for saving and packaging results.
- **Advanced Log Integration**: Improve compatibility with PX4 `ulog` and Betaflight logs for advanced use cases.
## Requirements
- Python 3.7+
- Required libraries:
  - `numpy`
  - `pandas`
  - `control`
  - `matplotlib`
  - `scipy`

Install the dependencies using:
```bash
pip install -r requirements.txt
```


## Usage
1. **Generate TEST Data**: Use `data_generator_inverted_pendulum.py` to generate system data.
  ```bash
  python data_generator_inverted_pendulum.py
  ```
  This will create a CSV file named inverted_pendulum_data_tracking_fixed.csv.

2. **Perform System Identification**: Run `system_identifier.py` to analyze the generated or custom data.
  ```bash
  python system_identifier.py --filename <input_csv_file> --order <model_order> [--use_actuator] [--time_col <col>] [--input_col <col>] [--output_col <col>] [--actuator_col <col>]
  ```
  - **Parameters**:
      - `--filename`: Path to the input CSV file (e.g., [inverted_pendulum_data_tracking_fixed.csv](http://_vscodecontentref_/0)).
      - `--order`: Specify the order of the transfer function model (1, 2, or 3).
      - `--use_actuator`: Optional flag to include actuator data in the system identification.
      - `--time_col`: (Optional) Column name for time (default: time)
      - `--input_col`: (Optional) Column name for input/setpoint (default: input)
      - `--output_col`: (Optional) Column name for output/feedback (default: output)
      - `--actuator_col`: (Optional) Column name for actuator (default: actuator)

  - **Example**:
      ```bash
      python system_identifier.py --filename inverted_pendulum_data_tracking_fixed.csv --order 2 --use_actuator
      ```

  - **Output**:
      - Displays step response plots for the specified model order.
      - Provides system parameters such as:
          - Gain (`Kp`)
          - Natural frequency (`wn`)
          - Damping ratio (`ζ`)
          - Time constant (`τ`)
      - Saves results in a structured CSV format for further analysis.

1. **Integrate Logs(Feature)**: Replace the input CSV file with your PX4 `ulog` or Betaflight log data for custom analysis.

## File Structure
```
PID_Analyzer/
├── data_generator_inverted_pendulum.py   # Script to generate system data for testing
├── system_identifier.py                  # Script for system identification and analysis
├── inverted_pendulum_data_tracking_fixed.csv  # Example generated data (output from data_generator_inverted_pendulum.py)
├── requirements.txt                      # List of dependencies for the project
├── materials/                            # Directory containing example plots and additional resources
│   ├── Step_Resp_WO_Actuator.png         # Step response plot without actuator compensation
│   ├── Step_Resp_W_Actuator.png          # Step response plot with actuator compensation
├── README.md                             # Project documentation
├── LICENSE                               # MIT License
```

## Example
The following is an example of a step response plot generated by the tool:

![Step Response Example](/materials/Step_Resp_WO_Actuator.png)

![Step Response Example](/materials/Step_Resp_W_Actuator.png)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Author
Kenny Chan, Microsoft Copilot, Github Copilot

