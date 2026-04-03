import time, dataclasses, enum, signal, os, atexit, socket, can, torch, gc
import multiprocessing as mp
import numpy as np
import pandas as pd
import tensorrt as trt
from scipy.signal import butter, filtfilt
#from epicpower import actuation, utils

# from Header_ICM20948_I2C import ICM20948_I2C_IMUs
from Header_ICM20948_I2C_pcb2 import ICM20948_I2C_IMUs
from Header_Mocap_trigger import Mocap_trigger
import Jetson.GPIO as GPIO

import can
from epicpower_tmotorV3.actuator_group import ActuatorGroup
from epicpower_tmotorV3.tmotor_v3 import TMotorV3


# GPIO 설정 (main 함수에서 수행하도록 이동)
output_pin = 7  # Jetson Board Pin 7

# Trial setting
subject = 'AB11_Ryan'  # Change this for different subjects
trial_name = f'{subject}_FullExperiment'  # Change this for different trials
trial_start_sec = 5
target_duration_sec = 125
target_time_range = 30
exo_ON = True

# Trigger setting
trigger_type = "typing"  # "mocap" or "typing"

# Body mass setting
body_mass_kg = 43.9  # kg

# Model path
trt_engine_path = '/home/metamobility2/Jimin/Trained Models IMUonly_fixed/SemiDEP Models/SI_AB11_Ryan_TL_IMUonly_fixed_SemiDEP/SI_AB11_Ryan_TL_IMUonly_fixed_SemiDEP.trt'

scale_factor_percent = 10
desired_delay_ms = 300

scale_factor = scale_factor_percent/100
delay_factor = int(desired_delay_ms/10 - 7)


# data to be saved (changed to lists for efficient appending)
data_to_save = {
    "timestamp": [],
    "mtr_cmd_L": [], "mtr_cmd_R": [],    "mtr_pos_L": [], "mtr_pos_R": [],    "mtr_vel_L": [], "mtr_vel_R": [],
    "imu_P": np.empty((0, 6)), "imu_L": np.empty((0, 6)), "imu_R": np.empty((0, 6)),
    "model_output_L": [], "model_output_R": [],
    "net_torque_L": [], "net_torque_R": [],
    "bio_torque_L": [], "bio_torque_R": [],
    "scaled_torque_L": [], "scaled_torque_R": [],
    "delayed_torque_L": [], "delayed_torque_R": [],
    "filtered_torque_L": [], "filtered_torque_R": [],
    "applied_torque_L": [], "applied_torque_R": [],
    "actual_torque_L": [], "actual_torque_R": [],
    "gpio_output": []  # GPIO 상태 추가
}


# Global variables
inference_process, input_q, output_q = None, None, None

# GPIO control functions
def init_gpio():
    """Initialize GPIO safely"""
    try:
        # First, try to clean up GPIO
        GPIO.cleanup()
    except:
        pass 
    
    try:
        GPIO.setmode(GPIO.BOARD)  # Jetson board numbering scheme
        GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)
        print("GPIO initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing GPIO: {e}")
        return False

def send_gpio_pulse_start():
    """Start a GPIO pulse by setting pin HIGH"""
    try:
        GPIO.output(output_pin, GPIO.HIGH)
        print("GPIO pulse started (HIGH)")
    except Exception as e:
        print(f"Error starting GPIO pulse: {e}")

def send_gpio_pulse_end():
    """End a GPIO pulse by setting pin LOW"""
    try:
        GPIO.output(output_pin, GPIO.LOW)
        print("GPIO pulse ended (LOW)")
    except Exception as e:
        print(f"Error ending GPIO pulse: {e}")

def get_gpio_output_state():
    """Get current GPIO output pin state (0 or 1)"""
    try:
        return int(GPIO.input(output_pin))
    except:
        # if GPIO not initialized or error occurs
        return 0

def safe_gpio_cleanup():
    """GPIO를 안전하게 정리"""
    try:
        GPIO.cleanup()
        print("GPIO cleaned up successfully")
    except Exception as e:
        print(f"Error during GPIO cleanup: {e}")

class Exo:
    def __init__(self,):

        self.CAN_id_L = 1 # NEED TO FIRST IDENTIFY THIS using display_motor_data.py
        self.CAN_id_R = 2 # NEED TO FIRST IDENTIFY THIS using display_motor_data.py
        self.mtr_type = "AK80-9"
        self.mtr_version = 3 # either 2 or 3
        self.control_freq_Hz = 100
        self.frame_length = 95  # Window size (in frame)
        # self.torque_limit = 10

        # biotorque parameters
        self.scale_factor = scale_factor
        self.delay_factor = delay_factor # Number of frames to delay the torque command

        # note: motors zero themselves when actuation.Motors() runs
        _ = input("Press Enter to initialize motors: ")
        assert self.mtr_version in (2,3)
        if self.mtr_version == 2:
            init_dict = {
                mtr_id: self.mtr_type for mtr_id in [self.CAN_id_L, self.CAN_id_R]
            }
            self.mtr_comms = actuation.Motors(init_dict)
        elif self.mtr_version == 3:
            init_list = [
                TMotorV3(mtr_id, self.mtr_type) for mtr_id in [self.CAN_id_L, self.CAN_id_R]
            ]
            self.mtr_comms = ActuatorGroup(init_list)

        # IMU initialization
        self.imus = ICM20948_I2C_IMUs()  # Back, Left hip, Right hip

        # Specify the CAN interface and channel
        try:
            self.bus = can.Bus(interface='socketcan', channel='can0')  # Replace 'socketcan' and 'can0' with your actual interface and channel
            # print("CAN bus initialized successfully.")
        except Exception as e:
            print(f"Error initializing CAN bus: {e}")
        self.notifier = can.Notifier(self.bus, [])

    def update_readings(self, CAN_id):
        mtr_pos = self.mtr_comms.get_position(CAN_id, degrees=True)
        mtr_vel = self.mtr_comms.get_velocity(CAN_id, degrees=True)
        mtr_torque = self.mtr_comms.get_torque(CAN_id)

        return mtr_pos, mtr_vel, mtr_torque




class lowpass_filter:
    def __init__(self, order=2, cutoff = 4, fs = 100.0):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.nyq = 0.5 * fs
        self.normal_cutoff = cutoff / self.nyq
        self.filter_b, self.filter_a = butter(order, self.normal_cutoff, btype='low', analog=False)

    def apply_lowpass_filter(self, data):
        
        filtered_data = np.zeros_like(data)

        if np.all(data == 0):
            return data

        window_size = data.shape[1]
        if window_size % 2 == 0:
            raise ValueError("Window size should be odd number.")
        half_win = window_size // 2

        for ch in range(data.shape[0]):
            padded_channel = np.pad(data[ch, :], pad_width=(half_win, half_win), mode='reflect')
            filtered_padded = filtfilt(self.filter_b, self.filter_a, padded_channel)
            filtered_data[ch, :] = filtered_padded[half_win:-half_win]

        return filtered_data
    
    def causal_filter(self, data, tau=0.1, dt=0.01, y0=None, return_last=False):
        x = data
        x = np.asarray(x, dtype=float)

        squeeze_1d = False
        if x.ndim == 1:
            x = x[None, :]          # (1, T)
            squeeze_1d = True
        elif x.ndim != 2:
            raise ValueError("data must be 1D or 2D with shape (C, T).")

        C, T = x.shape


        alpha = dt / float(tau)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        # 5) 필터
        y = np.empty_like(x)
        if y0 is None:
            y[:, 0] = x[:, 0]
        else:
            y0 = np.asarray(y0, dtype=float).reshape(-1)
            y[:, 0] = y0.item() if y0.size == 1 else (
                y0 if y0.size == C else
                (_ for _ in ()).throw(ValueError(f"y0 must be scalar or length {C}"))
            )
        for t in range(1, T):
            y[:, t] = y[:, t-1] + alpha * (x[:, t] - y[:, t-1])
        # 6) 반환 모드
        if return_last:
            out = y[:, -1]
            return out.item() if squeeze_1d else out
        return y.squeeze(0) if squeeze_1d else y



# Fast roll function to shift array elements
def fast_roll(arr):
    # For unilateral model
    if len(arr.shape) == 1:
        # 1D arr
        arr[:-1] = arr[1:]
        arr[-1] = 0
    elif len(arr.shape) == 2:
        # 2D arr (e.g. scaled_torque_arr, delayed_torque_arr) (2xframe_length)
        arr[:, :-1] = arr[:, 1:]
        arr[:, -1] = 0
    elif len(arr.shape) == 3:
        # 3D arr (e.g. model_input_arr) (2x14xframe_length)
        arr[:, :, :-1] = arr[:, :, 1:]
        arr[:, :, -1] = 0
    return arr

# TensorRT inference function
def trt_inference(input_data, output_shape, context):
    # Use torch.tensor(...) and torch.empty(...) completely on CUDA:
    d_input = torch.tensor(input_data, dtype=torch.float32, device='cuda')
    d_output = torch.empty(*output_shape, dtype=torch.float32, device='cuda')

    # Prepare bindings
    bindings = [int(d_input.data_ptr()), int(d_output.data_ptr())]

    context.execute_v2(bindings=bindings)

    output = d_output.cpu().numpy()
    return output

# Inference worker function for multiprocessing
def inference_worker(input_q, output_q, trt_engine_path,
                     input_mean_path, input_std_path,
                     label_mean_path, label_std_path,
                     num_input_features, frame_length):
    if torch.cuda.is_available():
        device = torch.device("cuda")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(trt_engine_path, 'rb') as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        print("Worker: Failed to deserialize TensorRT engine.")
        return
    context = engine.create_execution_context()

    input_mean = np.load(input_mean_path).astype(np.float32)
    input_std = np.load(input_std_path).astype(np.float32)
    label_mean = np.load(label_mean_path).astype(np.float32)
    label_std = np.load(label_std_path).astype(np.float32)

    dummy_input_data = np.zeros((1, num_input_features, frame_length), dtype=np.float32)
    dummy_output_shape = (1,)
    for _ in range(10):
        _ = trt_inference(dummy_input_data, dummy_output_shape, context)
    print("TensorRT engine warmed up.\nTrigger the trial to start...")

    while True:
        try:
            data_in = input_q.get()
            if data_in is None:  # Stop signal
                print("Worker: Stop signal received. Exiting.")
                break

            model_input_r_arr, model_input_l_arr = data_in

            # Right Model Inferencing
            output_shape = (1,)  # Assuming scalar output from model
            model_output_r_norm = trt_inference(model_input_r_arr, output_shape, context)
            model_output_r = model_output_r_norm * label_std + label_mean

            # Left Model Inferencing
            model_output_l_norm = trt_inference(model_input_l_arr, output_shape, context)
            model_output_l = model_output_l_norm * label_std + label_mean

            output_q.put((model_output_r, model_output_l))
        except Exception as e:
            print(f"Worker: Error during inference: {e}")
            break
    del context
    del engine
    del runtime
    print("Worker: Exited.")

# Telemetry function for real-time data visualization
def sendTelemetry(name, value):
    now = time.time() * 1000
    msg = name+":"+str(now)+":"+str(value)+"|g"
    sock.sendto(msg.encode(), teleplotAddr)

def sendBatchTelemetry(data_dict):
    now = time.time() * 1000
    try:
        for name, value in data_dict.items():
            msg = name + ":" + str(now) + ":" + str(value) + "|g"
            sock.sendto(msg.encode(), teleplotAddr)

        return True  # Successfully sent
    except Exception as e:
        print(f"Error in sendBatchTelemetry: {e}")
        return False

# Function to save all collected data
def save_data(start_rec_sec=0, trial_time_sec=None):
    global data_to_save

    # Convert lists to NumPy arrays
    data_np = {k: np.array(v) if isinstance(v, list) else v for k, v in data_to_save.items()}

    # Determine minimum length
    min_len = min(len(data_np["timestamp"]), len(data_np["mtr_pos_L"]), len(data_np["mtr_pos_R"]),
                  len(data_np["mtr_vel_L"]), len(data_np["mtr_vel_R"]), data_np["imu_P"].shape[0],
                  data_np["imu_L"].shape[0], data_np["imu_R"].shape[0], len(data_np["model_output_L"]),
                  len(data_np["gpio_output"]) if "gpio_output" in data_np else float('inf'))

    print(f'Total data length collected: {min_len}')
    
    if min_len == 0:
        print("ERROR: No data collected! min_len is 0")
        return
    
    # Calculate start and end indices for slicing
    start_idx = int(start_rec_sec * 100)  # 100 Hz data collection rate
    end_idx = min_len
    
    if trial_time_sec:
        end_idx = min(min_len, int((start_rec_sec + trial_time_sec) * 100))
    
    print(f'Slicing data from {start_rec_sec}s to {(start_rec_sec + (trial_time_sec or (min_len/100 - start_rec_sec)))}s')
    print(f'Index range: {start_idx} to {end_idx}')

    # Slice data with start offset
    timestamp_sliced = [t - start_rec_sec for t in data_np["timestamp"][start_idx:end_idx]]
    sliced_data = {k: v[start_idx:end_idx] if k.startswith('mtr') or k.startswith('model_output') or k.startswith('net_torque') or k.startswith('bio_torque') or k.startswith('scaled_torque') or k.startswith('delayed_torque') or k.startswith('filtered_torque') or k.startswith('applied_torque') or k.startswith('actual_torque') or k == 'gpio_output' else v[start_idx:end_idx, :] if k.startswith('imu') else None for k, v in data_np.items()}
    sliced_data['time'] = timestamp_sliced

    # Create DataFrames and save to CSV
    motor_data_keys = ['time', 'mtr_pos_L', 'mtr_pos_R', 'mtr_vel_L', 'mtr_vel_R']
    if 'gpio_output' in sliced_data and sliced_data['gpio_output'] is not None:
        motor_data_keys.append('gpio_output')
    
    df_mtr = pd.DataFrame({k: sliced_data[k] for k in motor_data_keys})
    df_mtr.to_csv(f'{trial_name}_input_motor.csv', index=False)
    print(f'Motor Data saved to {trial_name}_input_motor.csv')
    print('Dimensions:', df_mtr.shape)

    # Save IMU data
    imu_data = {'time': sliced_data['time'],
                'Pelvis_Acc_X': sliced_data['imu_P'][:, 0], 'Pelvis_Acc_Y': sliced_data['imu_P'][:, 1], 'Pelvis_Acc_Z': sliced_data['imu_P'][:, 2], 
                'Pelvis_Gyr_X': sliced_data['imu_P'][:, 3], 'Pelvis_Gyr_Y': sliced_data['imu_P'][:, 4], 'Pelvis_Gyr_Z': sliced_data['imu_P'][:, 5],
                'Thigh_L_Acc_X': sliced_data['imu_L'][:, 0], 'Thigh_L_Acc_Y': sliced_data['imu_L'][:, 1], 'Thigh_L_Acc_Z': sliced_data['imu_L'][:, 2], 
                'Thigh_L_Gyr_X': sliced_data['imu_L'][:, 3], 'Thigh_L_Gyr_Y': sliced_data['imu_L'][:, 4], 'Thigh_L_Gyr_Z': sliced_data['imu_L'][:, 5],
                'Thigh_R_Acc_X': sliced_data['imu_R'][:, 0], 'Thigh_R_Acc_Y': sliced_data['imu_R'][:, 1], 'Thigh_R_Acc_Z': sliced_data['imu_R'][:, 2], 
                'Thigh_R_Gyr_X': sliced_data['imu_R'][:, 3], 'Thigh_R_Gyr_Y': sliced_data['imu_R'][:, 4], 'Thigh_R_Gyr_Z': sliced_data['imu_R'][:, 5]
                }
    if 'gpio_output' in sliced_data and sliced_data['gpio_output'] is not None:
        imu_data['gpio_output'] = sliced_data['gpio_output']
    
    df_imu = pd.DataFrame(imu_data)
    df_imu.to_csv(f'{trial_name}_input_imu.csv', index=False)
    print(f'IMU Data saved to {trial_name}_input_imu.csv')
    print('Dimensions:', df_imu.shape)

    # Save motor command data
    df_torque = pd.DataFrame({k: sliced_data[k] for k in ['time', 'model_output_L', 'model_output_R', 'net_torque_L', 'net_torque_R', 'bio_torque_L', 'bio_torque_R', 'scaled_torque_L', 'scaled_torque_R', 'delayed_torque_L', 'delayed_torque_R', 'filtered_torque_L', 'filtered_torque_R', 'applied_torque_L', 'applied_torque_R', 'mtr_cmd_L', 'mtr_cmd_R', 'actual_torque_L', 'actual_torque_R', 'gpio_output']})
    df_torque.to_csv(f'{trial_name}_output_torque.csv', index=False)
    print(f'Torque data saved to {trial_name}_output_torque.csv')
    print('Dimensions:', df_torque.shape)
    
# Signal handler for graceful exit
def exit_signal_handler(sig, frame):
    global inference_process, input_q
    print("Signal received, initiating shutdown...")    
    
    # Apply zero torque to the motors
    Exo.mtr_comms.set_torque(Exo.CAN_id_L, 0)
    Exo.mtr_comms.set_torque(Exo.CAN_id_R, 0)

    save_data(trial_start_sec, target_duration_sec)
    cleanup_can(Exo.bus, Exo.notifier)
    safe_gpio_cleanup()  # 안전한 GPIO 정리 함수 사용
    gc.collect()
    torch.cuda.empty_cache()

    print("Exiting program")
    os._exit(0)

# Function to cleanup CAN resources
def cleanup_can(bus, notifier):
    try:
        notifier.stop()
        bus.shutdown()
        print("CAN resources cleaned up successfully")
    except Exception as e:
        print(f"Error during CAN cleanup: {e}")

def main():
    # include global variables that need to be reassigned inside the main function
    global data_to_save, Exo
    global inference_process, input_q, output_q

    # GPIO 초기화 (안전하게)
    if not init_gpio():
        print("Failed to initialize GPIO. Exiting...")
        return

    # load the normalization values
    base_model_path = os.path.dirname(trt_engine_path)
    input_mean_path = os.path.join(base_model_path, 'input_mean.npy')
    input_std_path = os.path.join(base_model_path, 'input_std.npy')
    label_mean_path = os.path.join(base_model_path, 'label_mean.npy')
    label_std_path = os.path.join(base_model_path, 'label_std.npy')

    current_input_mean = np.load(input_mean_path).astype(np.float32)
    current_input_std = np.load(input_std_path).astype(np.float32)
    num_input_features = current_input_mean.shape[0]

    # Initialize the exoskeleton
    Exo = Exo()
    # Initialize lowpass filter
    lpf = lowpass_filter()

    # Initialize queues for multiprocessing
    input_q = mp.Queue()
    output_q = mp.Queue()

    # Start inference_worker process
    inference_process = mp.Process(target=inference_worker,
                                   args=(input_q, output_q, trt_engine_path,
                                         input_mean_path, input_std_path,label_mean_path, label_std_path,
                                         num_input_features, Exo.frame_length))
    inference_process.start()

    # Rolling array initialization of model output array (2xframe_length)
    model_input_arr = np.zeros((2, num_input_features, Exo.frame_length), dtype=np.float32)
    scaled_torque_arr = np.zeros((2, Exo.frame_length), dtype=np.float32)
    delayed_torque_arr = np.zeros((2, Exo.frame_length), dtype=np.float32)
    filtered_torque_arr = np.zeros((2, Exo.frame_length), dtype=np.float32)
    applied_torque_arr = np.zeros((2, Exo.frame_length), dtype=np.float32)

    current_pos_L, current_vel_L = 0.0, 0.0
    current_pos_R, current_vel_R = 0.0, 0.0
    
    local_p_data = np.zeros(6)
    local_l_data = np.zeros(6)
    local_r_data = np.zeros(6)

    last_model_output_r = np.array([0.0], dtype=np.float32)
    last_model_output_l = np.array([0.0], dtype=np.float32)
    model_output_r_val = last_model_output_r
    model_output_l_val = last_model_output_l

    # Setting for the exiting process
    atexit.register(lambda: (cleanup_can(Exo.bus, Exo.notifier), safe_gpio_cleanup()))
    signal.signal(signal.SIGINT, exit_signal_handler)

    # Maria 스타일의 트리거 처리 변수들
    logging_started = False
    first_pulse_sent = False
    first_pulse_end_time = None
    second_pulse_sent = False
    second_pulse_end_time = None
    start_time = None
    start_index = 1

    # Wait for the trigger to start the trial (Maria 스타일로 변경)
    if trigger_type == "mocap":
        print("Wait for the tensorrt to warm up...\n")
        # 트리거 대기는 메인 루프에서 처리
    elif trigger_type == "typing":
        input_trigger = input("Wait for the tensorRT to warm up...\n")
        if input_trigger == "":
            print("Trial started")

    # Main control loop
    while True:
        # Maria 스타일의 트리거 처리
        if trigger_type == "mocap" and not logging_started:
            mocap_trigger.wait_for_trigger()
            print("Mocap trigger received - starting data logging")
            start_time = time.time()
            logging_started = True
        elif trigger_type == "typing" and not logging_started:
            # typing 모드는 위에서 이미 처리됨            
            start_time = time.time()
            logging_started = True

        # 데이터 수집이 시작된 후에만 실행
        if not logging_started:
            continue

        # 1. Read the motor encoder values
        time_1 = time.time()
        
        current_pos_L, current_vel_L, current_torque_L = Exo.update_readings(Exo.CAN_id_L)
        current_pos_R, current_vel_R, current_torque_R = Exo.update_readings(Exo.CAN_id_R)

        ####################
        # so by the nature of the motor, LEFT motor reads Flex. = + & Ext. = -
                                      # RIGHT motor reads Flex. = - & Ext. = +
        # Which means we need to follow LEFT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ###############################################################################################################
        # print(f'{Exo.CAN_id_L} = current_posL: | {current_pos_L:^14.2f} | current_vel_L: {current_vel_L:^16.2f}')
        # print(f'{Exo.CAN_id_R} = current_posR: | {current_pos_R:^14.2f} | current_vel_R: {current_vel_R:^16.2f}')
        ###############################################################################################################
        
        # 2. Read the IMU values
        imu_dict = Exo.imus.read_IMUs()

        local_p_data = imu_dict["IMU_PELVIS"]
        local_l_data = imu_dict["IMU_THIGH_LEFT"]
        local_r_data = imu_dict["IMU_THIGH_RIGHT"]

        data_to_save['mtr_pos_L'].append(current_pos_L); data_to_save['mtr_pos_R'].append(-current_pos_R)
        data_to_save['mtr_vel_L'].append(current_vel_L); data_to_save['mtr_vel_R'].append(-current_vel_R)
        
        data_to_save['imu_P'] = np.vstack([data_to_save['imu_P'], local_p_data.reshape(1, 6)])
        data_to_save['imu_L'] = np.vstack([data_to_save['imu_L'], local_l_data.reshape(1, 6)])
        data_to_save['imu_R'] = np.vstack([data_to_save['imu_R'], local_r_data.reshape(1, 6)])

        # 3. Mirror the left data to the right side
        p_data_reflected = local_p_data.copy()
        p_data_reflected[1] *= -1
        p_data_reflected[3] *= -1
        p_data_reflected[5] *= -1

        l_data_reflected = local_l_data.copy()
        l_data_reflected[1] *= -1
        l_data_reflected[3] *= -1
        l_data_reflected[5] *= -1

        pos_L_reflected = float(current_pos_L) * -1
        vel_L_reflected = float(current_vel_L) * -1
        
        # 4. Prepare the model input data
        # right_data = np.hstack((local_r_data, np.array([current_pos_R]), np.array([current_vel_R])))
        # left_data = np.hstack((l_data_reflected, np.array([pos_L_reflected]), np.array([vel_L_reflected])))
        
        right_data = np.hstack((local_r_data))
        left_data = np.hstack((l_data_reflected))

        right_latest_input = (right_data - current_input_mean) / current_input_std
        left_latest_input = (left_data - current_input_mean) / current_input_std

        model_input_arr = fast_roll(model_input_arr)
        model_input_arr[0, :, -1] = right_latest_input
        model_input_arr[1, :, -1] = left_latest_input

        # 5. TensorRT inference
        time_3 = time.time()

        input_q.put((model_input_arr[0:1, :, :].copy(), model_input_arr[1:2, :, :].copy()))

        try:
            model_output_r_val, model_output_l_val = output_q.get_nowait()
            last_model_output_r, last_model_output_l = model_output_r_val, model_output_l_val
        except mp.queues.Empty:
            model_output_r_val, model_output_l_val = last_model_output_r, last_model_output_l

        model_output_combined = np.hstack((model_output_r_val, model_output_l_val))

        # 6. Calculate the net torque
        time_4 = time.time()

        net_torque_combined = model_output_combined * body_mass_kg

        current_applied_torque = applied_torque_arr[:, -1]
        bio_torque_combined = net_torque_combined - current_applied_torque

        scaled_torque_arr = fast_roll(scaled_torque_arr)
        scaled_torque_arr[:, -1] = bio_torque_combined * Exo.scale_factor

        delayed_torque_arr = fast_roll(delayed_torque_arr)
        delayed_torque_arr[:, -1] = scaled_torque_arr[:, -Exo.delay_factor-1]

        # filtered_torque_arr = lpf.apply_lowpass_filter(delayed_torque_arr)
        filtered_torque_arr = lpf.causal_filter(delayed_torque_arr, tau=0.1)
        
        applied_torque_arr = fast_roll(applied_torque_arr)
        applied_torque_arr[:, -1] = filtered_torque_arr[:, -1]
        
        # 7. Send the torque command to the motors        
        motor_cmd_val_L = filtered_torque_arr[1, -1]
        motor_cmd_val_R = filtered_torque_arr[0, -1]

        if exo_ON == False: motor_cmd_val_L, motor_cmd_val_R = 0.0, 0.0 # use this for Exo off condition

        Exo.mtr_comms.set_torque(Exo.CAN_id_L, motor_cmd_val_L)
        Exo.mtr_comms.set_torque(Exo.CAN_id_R, -motor_cmd_val_R)
        
        actual_motor_torque_L = Exo.mtr_comms.get_torque(Exo.CAN_id_L)
        actual_motor_torque_R = -Exo.mtr_comms.get_torque(Exo.CAN_id_R)


        # 8. Stack the data (that will be saved after the trial)

        data_to_save['mtr_cmd_R'].append(motor_cmd_val_R)        
        data_to_save['mtr_cmd_L'].append(motor_cmd_val_L)
        
        data_to_save['model_output_R'].append(model_output_combined[0])
        data_to_save['model_output_L'].append(model_output_combined[1])

        data_to_save['net_torque_R'].append(net_torque_combined[0])
        data_to_save['net_torque_L'].append(net_torque_combined[1])

        data_to_save['bio_torque_R'].append(bio_torque_combined[0])
        data_to_save['bio_torque_L'].append(bio_torque_combined[1])

        data_to_save['scaled_torque_R'].append(scaled_torque_arr[0, -1])
        data_to_save['scaled_torque_L'].append(scaled_torque_arr[1, -1])

        data_to_save['delayed_torque_R'].append(delayed_torque_arr[0, -1])
        data_to_save['delayed_torque_L'].append(delayed_torque_arr[1, -1])

        data_to_save['filtered_torque_R'].append(filtered_torque_arr[0, -1])
        data_to_save['filtered_torque_L'].append(filtered_torque_arr[1, -1])

        data_to_save['applied_torque_R'].append(applied_torque_arr[0, -1])
        data_to_save['applied_torque_L'].append(applied_torque_arr[1, -1])

        data_to_save['actual_torque_R'].append(actual_motor_torque_R)            
        data_to_save['actual_torque_L'].append(actual_motor_torque_L)


        # GPIO 펄스 로직 추가
        current_time = time.time() - start_time
        
        # 첫 번째 펄스
        if current_time >= (5 + 90 + 2) and not first_pulse_sent:
            send_gpio_pulse_start()
            first_pulse_sent = True
            first_pulse_end_time = current_time + 0.2  # 200ms 펄스 지속시간
            print("First pulse started 2 seconds after mocap trigger")
        
        # 첫 번째 펄스 종료
        if first_pulse_sent and first_pulse_end_time and current_time >= first_pulse_end_time:
            send_gpio_pulse_end()
            first_pulse_end_time = None
            print("First pulse ended")
        
        # 두 번째 펄스
        if current_time >= (5 + 90 + 2 + target_time_range) and not second_pulse_sent:
            send_gpio_pulse_start()
            second_pulse_sent = True
            second_pulse_end_time = current_time + 0.2  # 200ms 펄스 지속시간
            print("Second pulse started after 120 seconds")

        # 두 번째 펄스 종료
        if second_pulse_sent and second_pulse_end_time and current_time >= second_pulse_end_time:
            send_gpio_pulse_end()
            second_pulse_end_time = None
            print("Second pulse ended")

        # GPIO 상태 로깅 (100Hz로)
        data_to_save['gpio_output'].append(get_gpio_output_state())

        # 9. Loop time
        time_0 = time.time()
        loop_time = time_0 - time_1
        
        # 10. Send telemetry data
        telemetry_data = {
            "gpio_output": get_gpio_output_state(),
            
            "pos_R": -current_pos_R,
            "pos_L": current_pos_L,
            
            "output_R": model_output_combined[0],
            "output_L": model_output_combined[1],
            
            "mtr_cmd_R": motor_cmd_val_R,
            "mtr_cmd_L": motor_cmd_val_L,
            
            "actual_torque_R": actual_motor_torque_R,
            "actual_torque_L": actual_motor_torque_L,
            
            'scaled_torque_R': scaled_torque_arr[0, -1],
            'scaled_torque_L': scaled_torque_arr[1, -1],
            
            "delayed_torque_R": delayed_torque_arr[0, -1],
            "delayed_torque_L": delayed_torque_arr[1, -1],
            
            "filtered_torque_R": filtered_torque_arr[0, -1],
            "filtered_torque_L": filtered_torque_arr[1, -1],
            
            # "loop_time": loop_time,
            # "inference_time": time_4 - time_3,
            
            # "imu_L_Acc_X": local_l_data[0],
            # "imu_R_Acc_X": local_r_data[0],
            # "imu_P_Acc_X": local_p_data[0]
        }
        sendBatchTelemetry(telemetry_data)

        # 11. Wait for the time to reach the next clock cycle
        if (time.time() - start_time) > (start_index / Exo.control_freq_Hz):
            print("Loop time exceeded: ", (time.time() - start_time) - (start_index / Exo.control_freq_Hz))
        else:
            while (time.time() - start_time) < (start_index / Exo.control_freq_Hz):
                pass
        data_to_save['timestamp'].append(time.time()-start_time)
        start_index += 1

if __name__ == '__main__':

    # Garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    # Teleplot setting
    os.system('echo nc -u -w0 127.0.0.1 47269')
    teleplotAddr = ("127.0.0.1", 47269)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if trigger_type == "mocap":
        mocap_trigger = Mocap_trigger(server_ip="172.24.44.177", port_number=10)
        mocap_trigger.start_client()
        # trial_name = mocap_trigger.get_trial_info()
    
    mp.set_start_method('spawn', force=True)
    main()