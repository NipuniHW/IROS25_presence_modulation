import math
import pdb
import cv2
import mediapipe as mp
import numpy as np
from time import time, sleep
from collections import deque
# from connection import Connection
import threading
import queue
import pandas as pd
import sys
import os
# import vision_definitions
from mdp_formulation import low_gaze_config
from pepper import Pepper
import random
# from testing import load_q_table, update_lights, update_movements, update_volume, set_all_leds, get_gaze_bin

class AttentionDetector:
    def __init__(self, 
                 attention_threshold=0.5,  # Time in seconds needed to confirm attention
                 pitch_threshold=15,       # Maximum pitch angle for attention
                 yaw_threshold=20,         # Maximum yaw angle for attention
                 history_size=10):         # Number of frames to keep for smoothing
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize parameters
        self.attention_threshold = attention_threshold
        self.pitch_threshold = pitch_threshold
        self.yaw_threshold = yaw_threshold
        self.attention_start_time = None
        self.attention_state = False
        
        # Initialize angle history for smoothing
        self.angle_history = deque(maxlen=history_size)
        
        # Define the 3D face model coordinates
        self.face_3d = np.array([
            [285, 528, 200],  # Nose tip
            [285, 371, 152],  # Chin
            [197, 574, 128],  # Left eye corner
            [173, 425, 108],  # Left mouth corner
            [360, 574, 128],  # Right eye corner
            [391, 425, 108]   # Right mouth corner
        ], dtype=np.float64)
        
        # Metrics for gaze quality and time
        self.gaze_quality = 0.0  # Percentage of frames with attention detected
        self.gaze_time = 0.0     # Total time of sustained attention in seconds
        self.frames_with_attention = 0
        self.total_frames = 0
        
    def rotation_matrix_to_angles(self, rotation_matrix):
        """Convert rotation matrix to Euler angles (pitch, yaw, roll)"""
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], 
                        math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.array([pitch, yaw, roll]) * 180.0 / math.pi
    
    def smooth_angles(self, angles):
        """Apply smoothing to angles using a moving average"""
        self.angle_history.append(angles)
        return np.mean(self.angle_history, axis=0)
    
    def is_looking_at_robot(self, pitch, yaw):
        """Determine if the person is looking at the robot based on angles"""
        return abs(pitch) < self.pitch_threshold and abs(yaw) < self.yaw_threshold
    
    def process_frame(self, frame):
        """Process a single frame and return attention state and visualization"""
        h, w, _ = frame.shape
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # Initialize return values
        attention_detected = False
        sustained_attention = False
        angles = None
        face_found = False
        
        if results.multi_face_landmarks:
            face_found = True
            face_2d = []
            for face_landmarks in results.multi_face_landmarks:
                # Get 2D coordinates for key landmarks
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [1, 9, 57, 130, 287, 359]:  # Key facial landmarks
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_2d.append([x, y])
                
                face_2d = np.array(face_2d, dtype=np.float64)
                
                # Camera matrix
                focal_length = 1 * w
                cam_matrix = np.array([
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1]
                ])
                
                # Distortion matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                
                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(
                    self.face_3d, face_2d, cam_matrix, dist_matrix
                )
                
                # Get rotation matrix
                rot_matrix, _ = cv2.Rodrigues(rot_vec)
                
                # Get angles
                angles = self.rotation_matrix_to_angles(rot_matrix)
                
                # Apply smoothing
                smoothed_angles = self.smooth_angles(angles)
                pitch, yaw, roll = smoothed_angles
                
                # Check if looking at robot
                attention_detected = self.is_looking_at_robot(pitch, yaw)
                
                # Track sustained attention
                current_time = time()
                if attention_detected:
                    if self.attention_start_time is None:
                        self.attention_start_time = current_time
                    elif (current_time - self.attention_start_time) >= self.attention_threshold:
                        sustained_attention = True
                else:
                    self.attention_start_time = None
                
                # Visualization
                color = (0, 255, 0) if sustained_attention else (
                    (0, 165, 255) if attention_detected else (0, 0, 255)
                )
                
                # Add text overlays
                cv2.putText(frame, f'Pitch: {int(pitch)}', (20, 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f'Yaw: {int(yaw)}', (20, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw attention status
                status = "Sustained Attention" if sustained_attention else (
                    "Attention Detected" if attention_detected else "No Attention"
                )
                cv2.putText(frame, status, (20, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw nose direction
                nose_2d = face_2d[0]
                nose_3d = self.face_3d[0]
                nose_3d_projection, _ = cv2.projectPoints(
                    nose_3d.reshape(1, 1, 3), rot_vec, trans_vec, cam_matrix, dist_matrix
                )
                
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + yaw), int(nose_2d[1] - pitch))
                
                cv2.line(frame, p1, p2, color, 2)
        
                        
        return frame, attention_detected, sustained_attention, angles, face_found,

class AttentionCalibrator:
    def __init__(self, 
                 calibration_time=10.0,    # Time in seconds needed for calibration
                 samples_needed=300,        # Number of samples to collect
                 angle_tolerance=15.0):     # Tolerance for angle variation during calibration
        
        self.calibration_time = calibration_time
        self.samples_needed = samples_needed
        self.angle_tolerance = angle_tolerance
        
        # Storage for calibration samples
        self.pitch_samples = []
        self.yaw_samples = []
        
        # Calibration state
        self.calibration_start_time = None
        self.is_calibrated = False
        self.baseline_pitch = None
        self.baseline_yaw = None
        self.pitch_threshold = None
        self.yaw_threshold = None
        
    def start_calibration(self):
        """Start the calibration process"""
        self.calibration_start_time = time()
        self.pitch_samples = []
        self.yaw_samples = []
        self.is_calibrated = False
        print("Starting calibration... Please look directly at the robot.")
        
    def process_calibration_frame(self, pitch, yaw):
        """Process a frame during calibration"""
        if self.calibration_start_time is None:
            return False, "Calibration not started"
        
        current_time = time()
        elapsed_time = current_time - self.calibration_start_time
        
        # Add samples
        self.pitch_samples.append(pitch)
        self.yaw_samples.append(yaw)
        
        # Check if we have enough samples
        if len(self.pitch_samples) >= self.samples_needed:
            # Calculate baseline angles and thresholds
            self.baseline_pitch = np.mean(self.pitch_samples)
            self.baseline_yaw = np.mean(self.yaw_samples)
            
            # Calculate standard deviations
            pitch_std = np.std(self.pitch_samples)
            yaw_std = np.std(self.yaw_samples)
            
            # Set thresholds based on standard deviation and minimum tolerance
            self.pitch_threshold = max(2 * pitch_std, self.angle_tolerance)
            self.yaw_threshold = max(2 * yaw_std, self.angle_tolerance)
            
            self.is_calibrated = True
            return True, "Calibration complete"
        
        # Still calibrating
        return False, f"Calibrating... {len(self.pitch_samples)}/{self.samples_needed} samples"

class CalibratedAttentionDetector(AttentionDetector):
    def __init__(self, calibrator, attention_threshold=0.5, history_size=10):
        super().__init__(
            attention_threshold=attention_threshold,
            pitch_threshold=None,  # Will be set by calibrator
            yaw_threshold=None,    # Will be set by calibrator
            history_size=history_size
        )
        
        # Store calibrator
        self.calibrator = calibrator
        
        # Set thresholds from calibrator
        if calibrator.is_calibrated:
            self.pitch_threshold = calibrator.pitch_threshold
            self.yaw_threshold = calibrator.yaw_threshold
            self.baseline_pitch = calibrator.baseline_pitch
            self.baseline_yaw = calibrator.baseline_yaw
    
    def is_looking_at_robot(self, pitch, yaw):
        """Override the original method to use calibrated values"""
        if not self.calibrator.is_calibrated:
            return False
            
        # Calculate angle differences from baseline
        pitch_diff = abs(pitch - self.calibrator.baseline_pitch)
        yaw_diff = abs(yaw - self.calibrator.baseline_yaw)
        
        return pitch_diff < self.calibrator.pitch_threshold and yaw_diff < self.calibrator.yaw_threshold

def calculate_attention_metrics(attention_window, interval_duration=3.0):
   
    if not attention_window:
        return {
            'gaze_time': 0.0,
            'attention_ratio': 0.0,
            'gaze_entropy': 0.0,
            'frames_in_interval': 0,
            'robot_looks': 0,
            'non_robot_looks': 0
        }        

    # Get current time and filter window to only include last interval_duration seconds
    current_time = attention_window[-1][0]  # Latest timestamp
    filtered_window = [(t, a) for t, a in attention_window 
                      if current_time - t <= interval_duration]
    
    # Count frames
    frames_in_interval = len(filtered_window)
    robot_looks = sum(1 for _, attention in filtered_window if attention)
    non_robot_looks = frames_in_interval - robot_looks
    
    # Calculate attention ratio
    attention_ratio = robot_looks / frames_in_interval if frames_in_interval > 0 else 0.0
    
    # Calculate stationary gaze entropy
    gaze_entropy = 0.0
    if frames_in_interval > 0:
        p_robot = robot_looks / frames_in_interval
        p_non_robot = non_robot_looks / frames_in_interval
        
        # Calculate entropy using Shannon formula
        if p_robot > 0:
            gaze_entropy -= p_robot * math.log2(p_robot)
        if p_non_robot > 0:
            gaze_entropy -= p_non_robot * math.log2(p_non_robot)
    
    # Calculate continuous gaze time for robot
    continuous_gaze_time = 0.0
    continuous_gaze_time = sum(
        timestamp - start_time
        for start_time, timestamp in zip(
            [t for t, a in filtered_window if a],
            [t for t, a in filtered_window[1:] if a]
        )
    ) if filtered_window and filtered_window[0][1] else 0.0

    return {
        'gaze_time': continuous_gaze_time,
        'attention_ratio': attention_ratio,
        'gaze_entropy': gaze_entropy,
        'frames_in_interval': frames_in_interval,
        'robot_looks': robot_looks,
        'non_robot_looks': non_robot_looks
    }
    
def calibration_main():
    global LeRobot
    """Run the calibration process"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 10)
    detector = AttentionDetector()
    calibrator = AttentionCalibrator()
    attention = AttentionCalibrator(detector)
    
    # Start calibration
    calibrator.start_calibration()
    
    # while cap.isOpened():
    #     success, frame = cap.read()
    #     if not success:
    #         break
    
    execute_action(0, 0, 7)
    
    while True:
        # Get the latest frame from Pepper’s camera
        image_container = LeRobot.video_proxy.getImageRemote(LeRobot.subscriber_id)
        if not image_container:
            continue  # Skip if no image is received
        
        width, height = image_container[0], image_container[1]
        frame = np.frombuffer(image_container[6], dtype=np.uint8).reshape((height, width, 3))
        
        # Process frame using existing detector
        frame, attention, sustained, angles, face_found = detector.process_frame(frame)
        
        if face_found and angles is not None:
            pitch, yaw, _ = angles
            is_complete, message = calibrator.process_calibration_frame(pitch, yaw)
            
            # Display calibration status
            cv2.putText(frame, message, (20, 110), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            if is_complete:
                print(f"Calibration complete!")
                print(f"Baseline Pitch: {calibrator.baseline_pitch:.2f}")
                print(f"Baseline Yaw: {calibrator.baseline_yaw:.2f}")
                print(f"Pitch Threshold: {calibrator.pitch_threshold:.2f}")
                print(f"Yaw Threshold: {calibrator.yaw_threshold:.2f}")
                break
        
        cv2.imshow('Calibration', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return calibrator

def calculate_gaze_score(output_queue, metrics, interval_duration=3.0):
    # Extract values from the metrics dictionary
    continuous_gaze_time = metrics['gaze_time']
    attention_ratio = metrics['attention_ratio']
    gaze_entropy = metrics['gaze_entropy']
    
    # Normalize the continuous gaze time
    normalized_gaze_time = min(continuous_gaze_time / interval_duration, 1.0)
    
    # Normalize the gaze entropy (lower entropy is better for focused attention)
    normalized_entropy = 1.0 - min(gaze_entropy, 1.0)  # Invert to reward focus
    
    # Define weights for each metric
    weight_gaze_time = 0.5  # Highest weight for continuous gaze
    weight_attention_ratio = 0.3  # Moderate weight for overall attention ratio
    weight_entropy = 0.2  # Lowest weight for entropy
    
    # Apply non-linear scaling to emphasize sustained gaze and high attention ratios
    normalized_gaze_time = normalized_gaze_time ** 1.5
    attention_ratio = attention_ratio ** 1.2
    
    # Calculate raw gaze score (weighted sum of metrics)
    raw_score = (
        weight_gaze_time * normalized_gaze_time +
        weight_attention_ratio * attention_ratio +
        weight_entropy * normalized_entropy
    )
    
    # Scale raw score to 0–100 and clamp it
    gaze_score = min(max(raw_score * 100, 0), 100)
    output_queue.put(gaze_score)

def main():
    # global gaze_score
    global LeRobot
    
    # Start by setting all behaviors to default values
    execute_action(0, 0, 7)
    
    # First run calibration
    print("Starting calibration process...")
    calibrator = calibration_main()
    
    if not calibrator.is_calibrated:
        print("Calibration failed or was interrupted.")
        return
    
    # Initialize camera and detector with calibration
    print("\nStarting attention detection with calibrated values...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 10)
    detector = CalibratedAttentionDetector(calibrator)
    
    # Create an output queue to store results from threads
    output_queue = queue.Queue()
    
    # Initialize attention window
    attention_window = []
    
    attention = AttentionDetector()
    
    while True:
        # Get the latest frame from Pepper’s camera
        image_container = LeRobot.video_proxy.getImageRemote(LeRobot.subscriber_id)
        if not image_container:
            continue  # Skip if no image is received
        
        width, height = image_container[0], image_container[1]
        frame = np.frombuffer(image_container[6], dtype=np.uint8).reshape((height, width, 3))
    
        # Process frame
        frame, attention, sustained, angles, face_found = detector.process_frame(frame)
        
        # Update attention window
        current_time = time()
        attention_window.append((current_time, attention))
    
        # Remove old entries from attention window (older than 3 seconds)
        attention_window = [(t, a) for t, a in attention_window if t > current_time - 3]
        
        # Calculate metrics
        metrics = calculate_attention_metrics(attention_window)
        
        # Start threads
        gaze_thread_obj = threading.Thread(target=calculate_gaze_score, args=(output_queue, metrics, 3.0), daemon=True)
        gaze_thread_obj.start()
        gaze_thread_obj.join()  # Wait for thread to finish before moving forward

         # # Get gaze score from the output queue
        if not output_queue.empty():
            gaze_score = output_queue.get()
            # print(f"Gaze: {gaze_score}")
            yield gaze_score

        # Display the frame
        if face_found:
            h, w, _ = frame.shape
            # Add calibration values
            cv2.putText(frame, f'Baseline Pitch: {calibrator.baseline_pitch:.1f}', 
                      (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.putText(frame, f'Baseline Yaw: {calibrator.baseline_yaw:.1f}', 
                      (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # Add metrics
            cv2.putText(frame, f'Attention Ratio: {metrics["attention_ratio"]:.2f}', 
                      (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.putText(frame, f'Gaze Entropy: {metrics["gaze_entropy"]:.2f}', 
                      (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.putText(frame, f'Frames in Window: {metrics["frames_in_interval"]}', 
                      (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        cv2.imshow('Calibrated HRI Attention Detection', frame)
        
        # Break loop on 'ESC'
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
    # cap.release()
    LeRobot.video_proxy.unsubscribe(LeRobot.subscriber_id)
    cv2.destroyAllWindows()
    # print(f"Gaze: {gaze_score}")

def load_q_table(file_path):
    q_table = pd.read_csv(file_path)
    q_table.index = q_table.index.astype(int)
    q_table.set_index(q_table.columns[0], inplace=True)
    q_table.index.name = "State"
    return q_table

# Main testing loop
def test_q_learning(q_table_path, duration_minutes=2):
    q_table = load_q_table(q_table_path)
    gaze_generator = main()   
    light, movement, volume = 0, 0, 0  # Default values 

    def process_q_learning(gaze_score):
        state = get_gaze_bin(gaze_score)
        print(f"Current state: {state}")
        action = choose_action(state, q_table)
        print(f"Chosen action: {action}")
        nonlocal light, movement, volume
        light, movement, volume = update_behavior(action, light, movement, volume)

    start_time = time()
    interval = 2  # Interval in seconds
    end_time = start_time + duration_minutes * 60  # Calculate end time
    gaze_score_average_vector = []
    
    while time() < end_time:
        gaze_score = next(gaze_generator)
        gaze_score_average_vector.append(gaze_score)
        print("added the number ", gaze_score) 
        
        current_time = time()
        
        if current_time - start_time >= interval:
            gaze_score_average = sum(gaze_score_average_vector) / len(gaze_score_average_vector)
            process_q_learning(gaze_score_average)
            gaze_score_average_vector = []
            print("########average gaze score ", gaze_score_average)
            print("########updated the behavior")
            # process_q_learning(gaze_score)
            start_time = current_time  # Reset the timer

        # process_q_learning(gaze_score)
        
        # Delay the loop by 180ms
        sleep(0.18)

    print("Test completed")
'''
def test_q_learning(q_table_path, iterations=1000):
    q_table = load_q_table(q_table_path)
    gaze_generator = main()   
    light, movement, volume = 0, 0, 0  # Default values 
    frame_skip = 6  # Process every 6th frame
    frame_count = 0

    def process_q_learning(gaze_score):
        state = get_gaze_bin(gaze_score)
        print(f"Current state: {state}")
        action = choose_action(state, q_table)
        print(f"Chosen action: {action}")
        nonlocal light, movement, volume
        light, movement, volume = update_behavior(action, light, movement, volume)

    start_time = time()
    interval = 2  # Interval in seconds
    
    for _ in range(iterations):  # Test for 100 steps
        gaze_score = next(gaze_generator)
        
        current_time = time()
        
        if current_time - start_time >= interval:
            process_q_learning(gaze_score)
            start_time = current_time  # Reset the timer

        frame_count += 1
        #print(f"Current gaze score: {gaze_score}")
        
        process_q_learning(gaze_score)
        
        # if frame_count % frame_skip == 0:
            # process_q_learning(gaze_score)
            # q_learning_thread = threading.Thread(target=process_q_learning, args=(gaze_score,))
            # q_learning_thread.start()
            # q_learning_thread.join()  # Wait for the thread to finish

        frame_count += 1

    print("Test completed")
    '''


def get_gaze_bin(gaze_score):
    if gaze_score < 0.0 or gaze_score > 100.0:
        raise ValueError("Raw gaze score must be between 0.0 and 100.0")

    if gaze_score <= 30.0:
        return int((gaze_score / 30.0) * 3)  # Scale 0-30 to 0-3
    elif gaze_score <= 60.0:
        return int(4 + ((gaze_score - 31.0) / 29.0) * 2)  # Scale 31-60 to 4-6
    else:
        return int(7 + ((gaze_score - 61.0) / 39.0) * 3)  # Scale 61-100 to 7-10

# Function to choose an action based on the current state
def choose_action(state, q_table):
    if state not in q_table.index:
        raise ValueError(f"State {state} is not in the Q-table. Available states: {q_table.index.tolist()}")

    action_values = q_table.loc[state]
    max_q_value = action_values.max()
    best_actions = action_values[action_values == max_q_value].index.tolist()

    chosen_action = np.random.choice(best_actions)  # Randomly select among best actions if tie

    # Ensure the chosen action is split into a list of three components
    return chosen_action.split(", ")  # Split by ", " to separate the values

def update_behavior(action, light, movement, volume):
    l_action, m_action, v_action = action
    
    if l_action == "Increase L":
        light = min(10, light + 1)
    elif l_action == "Decrease L":
        light = max(0, light - 1)
        
    if m_action == "Increase M":
        movement = min(5, movement + 1)
    elif m_action == "Decrease M":
        movement = max(0, movement - 1)
            
    if v_action == "Increase V":
        volume = min(10, volume + 1)
    elif v_action == "Decrease V":
        volume = max(0, volume - 1)
            
    # Keep Same
    if l_action == "Keep L":
        light = light
    elif m_action == "Keep M":
        movement = movement
    elif v_action == "Keep V":
        volume = volume
    
    print(f"Light: {light}, Movement: {movement}, Volume: {volume}")
    
    # Perform the action
    execute_action(light, movement, volume)
    return light, movement, volume

# Function to execute an action (this is an example, modify as needed)
def execute_action(light, movement, volume):
    update_lights(light)
    update_movements(movement)
    update_volume(volume)   
    
# To update volume
def update_volume(volume):    
    global LeRobot
    volume_n = round(max(0, volume/10), 1)
    # print(f"Volume_n: {volume, volume_n}")
    LeRobot.tts.setVolume(volume_n)
    
    # List of random greetings or catchphrases
    greetings = [
        "Hello there!",
        "How's it going?",
        "Nice to see you!",
        "What's up?",
        "Greetings!",
        "Hey, how are you?",
        "Good day!",
        "Hi there!",
        "Howdy!",
        "Welcome!",
        "beep boop beep",
        "I am here!",
        "Hello, human!",
        "beep beep beep" # just for Damith
    ]    
    # Randomly pick a greeting
    random_greeting = random.choice(greetings)
    LeRobot.tts.say(random_greeting)
    
# To update movements
def update_movements(movement):
    global LeRobot
    # LeRobot.behavior_mng_service.stopAllBehaviors()
    LeRobot.behavior_mng_service.startBehavior("attention_actions_2/" + str(movement)) 
 
# To update lights
def update_lights(light):
    global LeRobot
    if light == 0:
        light_n = 0.1
    else:
        light_n = round(max(0, light/10), 1)
    set_all_leds(LeRobot.leds, light_n)    
    
def set_all_leds(leds, light_n):
    for led in low_gaze_config.led_actuators:
        leds.setIntensity(led, light_n)
        
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Q-Learning Testing')
    # parser.add_argument('--q_table', type=str, required=True, help='Path to the Q-table CSV file')
    # args = parser.parse_args()
    global LeRobot
    LeRobot = Pepper()
    try:
        LeRobot.connect("pepper.local", 9559)
        if not LeRobot.is_connected:
            sys.exit(1)
            
        # test_q_learning(args.q_table)
        q_table = "/home/nipuni/Documents/IROS25_presence_modulation/Finals/table_high_ql.csv"
        test_q_learning(q_table)
        del LeRobot 
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Cleaning up...")
        del LeRobot   
        cv2.destroyAllWindows()
        sys.exit(0)