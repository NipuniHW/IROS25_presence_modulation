import math
import cv2
import mediapipe as mp
import numpy as np
from time import time
from collections import deque

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
    
    return {
        'attention_ratio': attention_ratio,
        'gaze_entropy': gaze_entropy,
        'frames_in_interval': frames_in_interval,
        'robot_looks': robot_looks,
        'non_robot_looks': non_robot_looks
    }

def calibration_main():
    """Run the calibration process"""
    cap = cv2.VideoCapture(0)
    detector = AttentionDetector()
    calibrator = AttentionCalibrator()
    
    # Start calibration
    calibrator.start_calibration()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
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

def main():
    # First run calibration
    print("Starting calibration process...")
    calibrator = calibration_main()
    
    if not calibrator.is_calibrated:
        print("Calibration failed or was interrupted.")
        return
        
    # Initialize camera and detector with calibration
    print("\nStarting attention detection with calibrated values...")
    cap = cv2.VideoCapture(0)
    detector = CalibratedAttentionDetector(calibrator)
    
    # Initialize attention window
    attention_window = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Process frame
        frame, attention, sustained, angles, face_found = detector.process_frame(frame)
        
        # Update attention window
        current_time = time()
        attention_window.append((current_time, attention))
        
        # Calculate metrics
        metrics = calculate_attention_metrics(attention_window)
        
        # Add metrics and calibration values to display
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
        
        # Display the frame
        cv2.imshow('Calibrated HRI Attention Detection', frame)
        
        # Break loop on 'ESC'
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()