# For allnewk5 controller, helper functions

import Jetson.GPIO as GPIO
import time, dataclasses, enum, signal, os, atexit, socket, can, torch, gc
import multiprocessing as mp
import numpy as np
import pandas as pd
import tensorrt as trt
from scipy.signal import butter, filtfilt
from scipy import signal as sp_signal

output_pin = 7  # Jetson Board Pin 7

# Teleplot setting
os.system('echo nc -u -w0 127.0.0.1 47269')
teleplotAddr = ("127.0.0.1", 47269)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
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
    

class lowpass_filter:
    def __init__(self, order=2, cutoff = 10, fs = 100.0):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.nyq = 0.5 * fs
        self.normal_cutoff = cutoff / self.nyq
        self.filter_b, self.filter_a = butter(order, self.normal_cutoff, btype='low', analog=False)
        self.sos = sp_signal.butter(order, cutoff, btype='low', fs=fs, output='sos')

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
    
    def realtimeButterworth(self, data, zi=None, reset=False):
        """Stateful real-time SOS Butterworth filtering.

        Args:
            data: scalar or 1D array-like input at current step.
            zi: optional SOS filter state for this channel. If None, initialize from first input sample.
            reset: kept for compatibility (no internal shared state is used).

        Returns:
            filtered_data: filtered output with same shape as input.
            zf: final SOS state to carry to next call.
        """
        x = np.asarray(data, dtype=float)
        squeeze_out = False
        if x.ndim == 0:
            x = x.reshape(1)
            squeeze_out = True
        elif x.ndim != 1:
            raise ValueError("data must be scalar or 1D array")

        if zi is None:
            zi = sp_signal.sosfilt_zi(self.sos) * x[0]

        y, zf = sp_signal.sosfilt(self.sos, x, zi=zi)
        if squeeze_out:
            return y[0], zf
        return y, zf


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
    
    









# V2 motor based Exo class
# class Exo:
#     def __init__(self,):

#         self.CAN_id_L = 1 # NEED TO FIRST IDENTIFY THIS using display_motor_data.py
#         self.CAN_id_R = 0 # NEED TO FIRST IDENTIFY THIS using display_motor_data.py
#         self.mtr_type = "AK80-9"
#         self.control_freq_Hz = 100
#         self.frame_length = 95  # Window size (in frame)

#         # biotorque parameters
#         self.scale_factor = scale_factor
#         self.delay_factor = delay_factor # Number of frames to delay the torque command

#         # note: motors zero themselves when actuation.Motors() runs
#         _ = input("Press Enter to initialize motors: ")
#         init_dict = {mtr_id: self.mtr_type for mtr_id in [self.CAN_id_L, self.CAN_id_R]}
#         self.mtr_comms = actuation.Motors(init_dict)

#         # IMU initialization
#         self.imus = ICM20948_I2C_IMUs()  # Back, Left hip, Right hip

#         # Specify the CAN interface and channel
#         try:
#             self.bus = can.Bus(interface='socketcan', channel='can0')  # Replace 'socketcan' and 'can0' with your actual interface and channel
#             # print("CAN bus initialized successfully.")
#         except Exception as e:
#             print(f"Error initializing CAN bus: {e}")
#         self.notifier = can.Notifier(self.bus, [])

#     def update_readings(self, CAN_id):
#         mtr_pos = self.mtr_comms.get_position(CAN_id, degrees=True)
#         mtr_vel = self.mtr_comms.get_velocity(CAN_id, degrees=True)

#         return mtr_pos, mtr_vel