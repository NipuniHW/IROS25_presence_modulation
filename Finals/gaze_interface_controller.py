from copy import deepcopy
import math
import cv2
import mediapipe as mp
import numpy as np
from time import time, sleep
from collections import deque
from threading import Thread, Lock

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
                
        return frame, attention_detected, sustained_attention, angles, face_found

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
    """
    Calculate attention metrics for a given time window of attention data.
    
    Args:
        attention_window (list): List of tuples (timestamp, attention_state)
        interval_duration (float): Duration of analysis interval in seconds
        
    Returns:
        dict: Dictionary containing attention metrics:
            - attention_ratio: Ratio of frames with attention detected
            - gaze_entropy: Shannon entropy of gaze distribution
            - frames_in_interval: Number of frames in analyzed interval
            - robot_looks: Number of frames looking at robot
            - non_robot_looks: Number of frames not looking at robot
    """
    if not attention_window:
        return {
            'attention_ratio': 0.0,
            'gaze_entropy': 0.0,
            'frames_in_interval': 0,
            'robot_looks': 0,
            'non_robot_looks': 0,
            'gaze_score': 0.0
        }
    
    # Get current time and filter window to only include last interval_duration seconds
    current_time = attention_window[-1][0]  # Latest timestamp
    filtered_window = [(t, a) for t, a in attention_window 
                      if current_time - t <= interval_duration]
    # print("The filtered window is ", filtered_window)
    # print("the size of the filtered window is ", len(filtered_window))
    
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
    
    ### Cacluations for gaze score on pepper

    # Compute gaze score using the new formula
    # # High gc Low entrophy: gaze score high 
    # # High gc High entropy: gaze score low
    # # low gc high entrophy: gaze score low
    # # low gc low entrophy: gaze score low

    if gaze_entropy == 1.0 or (robot_looks > 30 and 1.0 > gaze_entropy > 0.7):
        gaze_score = 100 * attention_ratio
    else:
        gaze_score = 100 * (attention_ratio * (1 - gaze_entropy))
    
    gaze_score = max(0, min(100, gaze_score))  # Ensure score is within 0-100
    
    ### End additional calculations
    
    return {
        'attention_ratio': attention_ratio,
        'gaze_entropy': gaze_entropy,
        'frames_in_interval': frames_in_interval,
        'robot_looks': robot_looks,
        'non_robot_looks': non_robot_looks,
        'gaze_score': gaze_score
    }

class GazeInterfaceController:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = AttentionDetector()
        self.calibrator = AttentionCalibrator()
        self.is_in_attention_detection_mode = False
        self.attention_window_lock = Lock()
        self.gaze_score = 0.0
        self.gaze_score_lock = Lock()
        self.attention_thread = Thread(target=self.attention_detection_loop)
        self.robot_looks_lock = Lock()
        self.robot_looks = 0
        self.gaze_entropy_lock = Lock()
        self.gaze_entropy = 0.0
        
    def get_gaze_score(self):
        self.gaze_score_lock.acquire()
        score = self.gaze_score
        self.gaze_score_lock.release()
        return score
    
    def get_robot_looks(self):
        self.robot_looks_lock.acquire()
        score = self.robot_looks
        self.robot_looks_lock.release()
        return score
    
    def get_gaze_entropy(self):
        self.gaze_entropy_lock.acquire()
        score = self.gaze_entropy
        self.gaze_entropy_lock.release()
        return score
    
    def get_visualisation_frame(self):
        self.gaze_score_lock.acquire()
        frame = self.visualisation_frame.copy()
        self.gaze_score_lock.release()
        return frame
    
    def kill(self):
        self.is_in_attention_detection_mode = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        sleep(0.5)
        self.attention_thread.join()
        
        
    def calibration_exe(self):        
        # Start calibration
        self.calibrator.start_calibration()
        is_complete = False
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
                
            # Process frame using existing detector
            frame, attention, sustained, angles, face_found = self.detector.process_frame(frame)
            
            if face_found and angles is not None:
                pitch, yaw, _ = angles
                is_complete, message = self.calibrator.process_calibration_frame(pitch, yaw)
                
                # Display calibration status
                cv2.putText(frame, message, (20, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                if is_complete:
                    print(f"Calibration complete!")
                    print(f"Baseline Pitch: {self.calibrator.baseline_pitch:.2f}")
                    print(f"Baseline Yaw: {self.calibrator.baseline_yaw:.2f}")
                    print(f"Pitch Threshold: {self.calibrator.pitch_threshold:.2f}")
                    print(f"Yaw Threshold: {self.calibrator.yaw_threshold:.2f}")
                    break
            
            self.gaze_score_lock.acquire()
            self.visualisation_frame = frame
            self.gaze_score_lock.release()
            
            cv2.imshow('Calibration', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        if not is_complete:
            print("Calibration interrupted or failed.")
            raise ValueError("Calibration failed")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
    
        
    def start_detecting_attention(self):
        print("\nStarting attention detection with calibrated values...")
        self.is_in_attention_detection_mode = True
        # Start the attention detection loop in a separate thread
        self.attention_thread.start()
        
        
    def attention_detection_loop(self):
        self.cap = cv2.VideoCapture(0)
        
        # Initialize camera and detector with calibration
        self.detector = CalibratedAttentionDetector(self.calibrator)
        
        self.is_in_attention_detection_mode = True
        
        
        self.attention_window = []
        
        while self.cap.isOpened() and self.is_in_attention_detection_mode:
            success, frame = self.cap.read()
            if not success:
                print("Failed to read frame. Stopping attention detection.")
                break
                
            # print("Processing frame")
            # Process frame
            frame, attention, sustained, angles, face_found = self.detector.process_frame(frame)
            
            # Update attention window
            current_time = time()
            self.attention_window.append((current_time, attention))
            
            # Calculate metrics
            metrics = calculate_attention_metrics(self.attention_window)
            
            self.gaze_score_lock.acquire()
            self.gaze_score = metrics["gaze_score"]
            self.gaze_score_lock.release()
            
            self.robot_looks_lock.acquire()
            self.robot_looks = metrics["robot_looks"]
            self.robot_looks_lock.release()
            
            self.gaze_entropy_lock.acquire()
            self.gaze_entropy = metrics["gaze_entropy"]
            self.gaze_entropy_lock.release()
            
            # Add metrics and calibration values to display
            if face_found:
                h, w, _ = frame.shape
                # Add calibration values
                cv2.putText(frame, f'Baseline Pitch: {self.calibrator.baseline_pitch:.1f}', 
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(frame, f'Baseline Yaw: {self.calibrator.baseline_yaw:.1f}', 
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                # Add metrics
                cv2.putText(frame, f'Attention Ratio: {metrics["attention_ratio"]:.2f}', 
                        (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(frame, f'Gaze Entropy: {metrics["gaze_entropy"]:.2f}', 
                        (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(frame, f'Frames in Window: {metrics["frames_in_interval"]}', 
                        (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                self.gaze_score_lock.acquire()
                self.visualisation_frame = frame
                self.gaze_score_lock.release()
                
            # Display the frame
            # cv2.imshow('Calibrated HRI Attention Detection', frame)
            
            # Break loop on 'ESC'
            # if cv2.waitKey(5) & 0xFF == 27:
            #     break
        
        # cap.release()
        # cv2.destroyAllWindows()
        
if __name__=="__main__":
    controller = GazeInterfaceController()
    controller.calibration_exe()
    controller.start_detecting_attention()
    
    start_time = time()
    duration = 5 * 60  # 3 minutes in seconds
    interval = 3  # Interval in seconds
    next_print_time = start_time + interval
    try:
        
        while time() - start_time < duration:            
            # Print the gaze score every 3 seconds
            current_time = time()
            if current_time >= next_print_time:
                print(f"####### Gaze Score: {controller.get_gaze_score()}")
                print(f"Robot looks: {controller.get_robot_looks()}")
                print(f"Gaze entropy: {controller.get_gaze_entropy()}")
                next_print_time = current_time + interval
            
            frame = controller.get_visualisation_frame()
            if frame is not None:
                f = deepcopy(frame)
                # print("the type of frame is ", type(f))
                cv2.imshow('Calibrated HRI Attention Detection', f)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            else:
                print("Frame is None")
            sleep(0.05)
        
        cv2.destroyAllWindows()
        controller.kill()
        exit(0)
    except KeyboardInterrupt:
        controller.kill()
        exit(0)
    
    print("Attention detection completed.")
    
    
    