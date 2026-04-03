import time, dataclasses, enum
import numpy as np
import signal
import os
import atexit
import torch
from scipy.signal import butter, filtfilt
from epicpower import actuation, utils
from Header_ICM20948_I2Cpcb2 import ICM20948_I2C_IMUs
from Header_Mocap_trigger import Mocap_trigger
import socket
import pandas as pd
import can
import Jetson.GPIO as GPIO

output_pin = 7  # Jetson Board Pin 7

GPIO.setmode(GPIO.BOARD)  # Jetson board numbering scheme
    # set pin as an output pin with optional initial state of HIGH
GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)

# Teleplot setting
os.system('echo nc -u -w0 127.0.0.1 47269')
teleplotAddr = ("127.0.0.1",47269)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def sendTelemetry(name, value):
	now = time.time() * 1000
	msg = name+":"+str(now)+":"+str(value)+"|g"
	sock.sendto(msg.encode(), teleplotAddr)

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

def send_gpio_pulse(duration_ms=100):
    """Send a high pulse on GPIO pin for specified duration"""
    try:
        GPIO.output(output_pin, GPIO.HIGH)
        time.sleep(duration_ms / 1000.0)  # Convert ms to seconds
        GPIO.output(output_pin, GPIO.LOW)
        print(f"GPIO pulse sent (duration: {duration_ms}ms)")
    except Exception as e:
        print(f"Error sending GPIO pulse: {e}")

def get_gpio_output_state():
    """Get current GPIO output pin state (0 or 1)"""
    try:
        return int(GPIO.input(output_pin))
    except:
        # Fallback: assume LOW if we can't read
        return 0

# Mocap trigger setting
mocap_use = True
if mocap_use:
    mocap_trigger = Mocap_trigger(server_ip="172.24.44.177", port_number=11)
    mocap_trigger.start_client()

# data to be saved
timestamp = np.array([])
mtr_pos_L = np.array([])
mtr_pos_R = np.array([])
mtr_vel_L = np.array([])
mtr_vel_R = np.array([])
imu_P = np.empty((0, 6))
imu_L = np.empty((0, 6))
imu_R = np.empty((0, 6))
gpio_output_state = np.array([])  # Log GPIO output state

class ExoSide(enum.Enum):
    RIGHT = 0
    LEFT = 1

@dataclasses.dataclass
class ControlData:
    CAN_id:int
    mtr_type:str
    exo_side:str
    mtr_comms:actuation.Motors

    # TCN
    model_file:str
    body_mass_kg:float
    max_torque:float

    # time based estimation
    history:int
    phase_shift:int

    time:list = dataclasses.field(default_factory=list)
    pos:list = dataclasses.field(default_factory=list)
    vel:list = dataclasses.field(default_factory=list)
    imu:list = dataclasses.field(default_factory=list)
    mtr_cmd:float = 0

class BaselineControl:
    def __init__(self, ctrl_dat:ControlData):
        self.ctrl_dat = ctrl_dat
        self.t0 = time.time()

    def update_readings(self, log_data=True):
        global mtr_pos_L, mtr_pos_R, mtr_vel_L, mtr_vel_R
        mtr_pos = self.ctrl_dat.mtr_comms.get_position(
            self.ctrl_dat.CAN_id, degrees=True)
        mtr_vel = self.ctrl_dat.mtr_comms.get_velocity(
            self.ctrl_dat.CAN_id, degrees=True)
        self.ctrl_dat.pos.append(mtr_pos)
        self.ctrl_dat.vel.append(mtr_vel)
        self.ctrl_dat.time.append(time.time()-self.t0)

        if self.ctrl_dat.CAN_id == 0:
            sendTelemetry("mtr_pos_R", mtr_pos)
            sendTelemetry("mtr_vel_R", mtr_vel)
            if log_data:
                mtr_pos_R = np.append(mtr_pos_R, mtr_pos)
                mtr_vel_R = np.append(mtr_vel_R, mtr_vel)
        else:
            sendTelemetry("mtr_pos_L", mtr_pos)
            sendTelemetry("mtr_vel_L", mtr_vel)
            if log_data:
                mtr_pos_L = np.append(mtr_pos_L, mtr_pos)
                mtr_vel_L = np.append(mtr_vel_L, mtr_vel)

        if len(self.ctrl_dat.pos) > self.ctrl_dat.history:
            self.ctrl_dat.pos.pop(0)
            self.ctrl_dat.time.pop(0)

def set_up_exo(controller):
    motor_sides_vs_CAN_ids = {ExoSide.LEFT:ExoSide.LEFT.value, ExoSide.RIGHT:ExoSide.RIGHT.value}
    mtr_type = "AK80-9"
    update_rate_hz = 200 # 200/2 = 100 Hz for each motor
    
    model_file = "/home/kaustubh-meta/Documents/mmHipExo/Controls_Testing/AB07.tar"
    
    body_mass_kg = 57 # ASB's weight
    scale_factor = 0.2
    delay_factor = 10  # Number of frames to delay the torque command
    frame_length = 95  # Window size (in frame)
    
    safe_torque = 0 # Nm
    history = 400
    phase_shift = -30

    # start motors per epicpower process
    init_dict = {mtr_id:mtr_type for mtr_id in motor_sides_vs_CAN_ids.values()}
    # note: motors zero themselves when actuation.Motors() runs
    mtr_comms = actuation.Motors(init_dict)

    # Specify the CAN interface and channel
    try:
        bus = can.Bus(interface='socketcan', channel='can0')  # Replace 'socketcan' and 'can0' with your actual interface and channel
        print("CAN bus initialized successfully.")
    except Exception as e:
        print(f"Error initializing CAN bus: {e}")
    notifier = can.Notifier(bus, [])

    # package all relevant data at least semi-neatly
    controllers = [] # 1 per motor: left and right
    for side, CAN_id in motor_sides_vs_CAN_ids.items():
        ctrl_dat = ControlData(
                CAN_id=CAN_id,
                mtr_type=mtr_type,
                exo_side=side,
                mtr_comms=mtr_comms,
                model_file=model_file,
                body_mass_kg=body_mass_kg,
                max_torque=safe_torque,
                history=history,
                phase_shift=phase_shift,
            )
        controllers.append(controller(ctrl_dat))

    # Entire loop runs at 200 Hz including 2 motors
    clocker = utils.clocking.LoopTimer(update_rate_hz) 

    imus = ICM20948_I2C_IMUs() # Back, Left hip, Right hip

    # Keep the return values as they are
    return controllers, clocker, imus, bus, notifier, body_mass_kg, scale_factor, delay_factor, frame_length

# Function to save all collected data
def save_data(trial_time_sec = None):
    global timestamp, mtr_pos_L, mtr_pos_R, mtr_vel_L, mtr_vel_R, imu_P, imu_L, imu_R, gpio_output_state
      # Slice each array/list to min_len
    array_lengths = [len(timestamp), len(mtr_pos_L), len(mtr_pos_R), len(mtr_vel_L), len(mtr_vel_R),
                     imu_P.shape[0], imu_L.shape[0], imu_R.shape[0], len(gpio_output_state)]
    min_len = min(array_lengths)
    
    print(f'Array lengths: timestamp={len(timestamp)}, mtr_pos_L={len(mtr_pos_L)}, mtr_pos_R={len(mtr_pos_R)}, mtr_vel_L={len(mtr_vel_L)}, mtr_vel_R={len(mtr_vel_R)}, imu_P={imu_P.shape[0]}, imu_L={imu_L.shape[0]}, imu_R={imu_R.shape[0]}, gpio={len(gpio_output_state)}')

    if trial_time_sec: # If a trial time is provided, slice the data to that time
        target_len = int(trial_time_sec * 100)  # 100 Hz data collection rate
        min_len = min(min_len, target_len)

    print('Data length:', min_len)
    
    # Ensure all arrays are the same length by trimming to min_len
    timestamp = timestamp[:min_len]
    mtr_pos_L = mtr_pos_L[:min_len]
    mtr_pos_R = mtr_pos_R[:min_len]
    mtr_vel_L = mtr_vel_L[:min_len]
    mtr_vel_R = mtr_vel_R[:min_len]
    imu_P = imu_P[:min_len, :]
    imu_L = imu_L[:min_len, :]
    imu_R = imu_R[:min_len, :]
    gpio_output_state = gpio_output_state[:min_len]
    
    print('After trimming:', timestamp.shape, mtr_pos_L.shape, mtr_pos_R.shape, mtr_vel_L.shape, mtr_vel_R.shape, imu_P.shape, imu_L.shape, imu_R.shape, gpio_output_state.shape)
    # Create a dictionary with the data
    data = {'time': timestamp,
            'mtr_pos_L': mtr_pos_L,
            'mtr_pos_R': mtr_pos_R,
            'mtr_vel_L': mtr_vel_L,
            'mtr_vel_R': mtr_vel_R,
            'gpio_output': gpio_output_state}
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv('motor_positions.csv', index=False)
    print('Data saved to motor_positions.csv')
    print('Dimensions:', df.shape)

    # Save IMU data to CSV
    data = {'time': timestamp,
            'Pelvis_Acc_X': imu_P[:,0], 'Pelvis_Acc_Y': imu_P[:,1], 'Pelvis_Acc_Z': imu_P[:,2], 'Pelvis_Gyr_X': imu_P[:,3], 'Pelvis_Gyr_Y': imu_P[:,4], 'Pelvis_Gyr_Z': imu_P[:,5],
            'Thigh_L_Acc_X': imu_L[:,0], 'Thigh_L_Acc_Y': imu_L[:,1], 'Thigh_L_Acc_Z': imu_L[:,2], 'Thigh_L_Gyr_X': imu_L[:,3], 'Thigh_L_Gyr_Y': imu_L[:,4], 'Thigh_L_Gyr_Z': imu_L[:,5],
            'Thigh_R_Acc_X': imu_R[:,0], 'Thigh_R_Acc_Y': imu_R[:,1], 'Thigh_R_Acc_Z': imu_R[:,2], 'Thigh_R_Gyr_X': imu_R[:,3], 'Thigh_R_Gyr_Y': imu_R[:,4], 'Thigh_R_Gyr_Z': imu_R[:,5]}
    df = pd.DataFrame(data)
    df.to_csv('imu_data.csv', index=False)
    print('Data saved to imu_data.csv')
    print('Dimensions:', df.shape)

# Function to cleanup CAN resources
def cleanup_can(bus, notifier):
    try:
        notifier.stop()
        bus.shutdown()
        print("CAN resources cleaned up successfully")
    except Exception as e:
        print(f"Error during CAN cleanup: {e}")

def fast_roll(arr):
    """
    Rolling along the middle axis (frame sequence, axis=1) of the array
    Input: array to roll (1, frame_length, 2)
    Return: Rolled array
    """
    arr[:, :-1, :] = arr[:, 1:, :]  # Important Here!!!!!! Rolling at (axis=1)
    arr[:, -1, :] = 0
    return arr

def butter_lowpass_filter(data):
    """
    Butterworth LPF     
    Input: data to be filtered (1x95x2 time series motor command array)
    Return: filtered data (same size with the input data array)
    """
    # Filter parameters
    cutoff = 4
    fs = 100.0
    order = 2
    
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    filtered_data = np.zeros_like(data)
    if np.all(data == 0):
        return data  # If all values are 0, skip filtering
        
    # First motor channel filtering (data[0,:,0])
    filtered_data[0,:,0] = filtfilt(b, a, data[0,:,0])
    
    # Second motor channel filtering (data[0,:,1])
    filtered_data[0,:,1] = filtfilt(b, a, data[0,:,1])
    
    return filtered_data


def main():

    global timestamp, mtr_pos_L, mtr_pos_R, mtr_vel_L, mtr_vel_R, imu_P, imu_L, imu_R, gpio_output_state

    controllers, clocker, imus, bus, notifier, body_mass_kg, scale_factor, delay_factor, frame_length = set_up_exo(BaselineControl)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    trial_time_sec = None ##CHANGE - Set to 30 seconds to capture both pulses 
      # Register the cleanup function to be called at exit
    atexit.register(lambda: (cleanup_can(bus, notifier), GPIO.cleanup()))
    # Create a signal handler for clean shutdown
    def signal_handler(sig, frame):
        save_data(trial_time_sec)
        cleanup_can(bus, notifier)
        GPIO.cleanup()
        print("Exiting program")
        os._exit(0)    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    start_time = None
    start_index = 1
    logging_started = False
    first_pulse_sent = False
    first_pulse_end_time = None
    second_pulse_sent = False
    second_pulse_end_time = None

    # Modified main loop
    while imus.IMUs_are_on: # doesn't run if IMU setup failed
        # Check for mocap trigger before starting data logging
        if mocap_use and not logging_started:
            mocap_trigger.wait_for_trigger()
            print("Mocap trigger received - starting data logging")
            start_time = time.time()
            logging_started = True
        elif not mocap_use and not logging_started:
            print("No mocap trigger needed - starting data logging immediately")
            start_time = time.time()
            logging_started = True
        
        # time_3 = time.time()

        # Single IMU read for all attached sensors
        imu_dict = imus.read_IMUs()  # returns { "IMU_BACK": [...], "IMU_THIGH_LEFT": [...], "IMU_THIGH_RIGHT": [...] }

        # Pelvis
        p_data = imu_dict["IMU_PELVIS"]
        sendTelemetry("imu_B_Acc_Z", p_data[2])
        #sendTelemetry("imu_B_Gyr_Y", p_data[4])
        if logging_started:
            imu_P = np.vstack([imu_P, p_data.reshape(1,6)])        # Left thigh
        l_data = imu_dict["IMU_THIGH_LEFT"]
        sendTelemetry("imu_L_Acc_Z", l_data[2])
        sendTelemetry("imu_L_Gyr_Y", l_data[4])
        if logging_started:
            imu_L = np.vstack([imu_L, l_data.reshape(1,6)])

        # Right thigh
        r_data = imu_dict["IMU_THIGH_RIGHT"]
        sendTelemetry("imu_R_Acc_Z", r_data[2])
        sendTelemetry("imu_R_Gyr_Y", r_data[4])
        if logging_started:
            imu_R = np.vstack([imu_R, r_data.reshape(1,6)])

        # This one is necessary for reading the motor positions and velocities
        for controller in controllers:
            controller.ctrl_dat.mtr_comms.set_torque(controller.ctrl_dat.CAN_id, 0)

        # Exo Encoder readings
        for controller in controllers:
            controller.update_readings(log_data=logging_started)
        
        # Only log timing and GPIO data if logging has started
        if logging_started:
            # Wait until we reach the desired time for this iteration
            while time.time() - start_time < start_index * 0.01:
                pass  # Active wait to maintain precise 100Hz timing
            timestamp = np.append(timestamp, time.time()-start_time)            # Send first pulse after first data point is logged
            current_time = time.time() - start_time
            if current_time >= 2.0 and not first_pulse_sent:
                send_gpio_pulse_start()  # Start 200ms pulse at 2 seconds
                first_pulse_sent = True
                first_pulse_end_time = current_time + 0.2  # 200ms pulse duration
                print("First pulse started 2 seconds after mocap trigger")
            
            # End first pulse after 200ms
            if first_pulse_sent and first_pulse_end_time and current_time >= first_pulse_end_time:
                send_gpio_pulse_end()
                first_pulse_end_time = None
                print("First pulse ended")
            
            # Check if 10 seconds have passed and send second pulse
            if current_time >= 10.0 and not second_pulse_sent:
                send_gpio_pulse_start()  # Start 200ms pulse at 10 seconds
                second_pulse_sent = True
                second_pulse_end_time = current_time + 0.2  # 200ms pulse duration
                print("Second pulse started after 10 seconds")
            
            # End second pulse after 200ms
            if second_pulse_sent and second_pulse_end_time and current_time >= second_pulse_end_time:
                send_gpio_pulse_end()
                second_pulse_end_time = None
                print("Second pulse ended")
            
            # Log GPIO output state at 100Hz
            gpio_output_state = np.append(gpio_output_state, get_gpio_output_state())
            
            start_index += 1

if __name__ == '__main__':
    main()