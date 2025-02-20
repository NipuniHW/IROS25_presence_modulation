import sys
import cv2
import time
import argparse
from pepper import Pepper
from gaze_controller import *
from experiment_functions import *

def test_q_learning(q_table_path, duration_minutes, L1, M1, V1):
    pepper = Pepper()
    # pepper.connect("pepper.local", 9559)
    pepper.connect("localhost", 36989)
    
    try:
        if not pepper.is_connected:
            sys.exit(1)
            del pepper 
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Cleaning up...")
        del pepper   
        cv2.destroyAllWindows()
        sys.exit(0)
    q_table = load_q_table(q_table_path)
    
     # Change the camera ID to 2 if using external usb webcam, 0 if using the laptop webcam
    controller = GazeInterfaceController(camera_id=2)
    sleep(1)
    # Ask the user to press enter to start a calibration
    print('Press Enter to start the calibration')
    input()
    controller.calibration_exe()
    controller.start_detecting_attention()

    # Ask the user to press enter to start the training
    sleep(1)
    print('Press Enter to start the testing')
    input()
    
    # Start the training
    current_time = time()
    
    light, movement, volume = L1, M1, V1 # Default values 
    online_episodes_duration_minutes = duration_minutes*60
    
    start_time_inner_loop = time()
    
    while time() - current_time < online_episodes_duration_minutes:
        frame = controller.get_visualisation_frame()
        if frame is not None:
            f = deepcopy(frame)
            # print("the type of frame is ", type(f))
            cv2.imshow('Calibrated HRI Attention Detection', f)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            
        if time() - start_time_inner_loop >= 5:
            start_time_inner_loop = time()
            # Get the current gaze score
            gaze_score = controller.get_gaze_score()
            state = int(round(gaze_score/20))
            print(f"Gaze score: {gaze_score} -- giving state: {state}")
            action = choose_action(state, q_table)
            print(f"Chosen action: {action}")
            # nonlocal light, movement, volume
            light, movement, volume = pepper.update_behavior(action, light, movement, volume)
            print("Updated the behavior\n")

        # Delay the loop by 180ms
        sleep(0.18)

    print("Test completed")

if __name__ == "__main__":                   
    parser = argparse.ArgumentParser(description='Q-Learning Configuration')
    parser.add_argument('--q_table', type=str, required=True, help='Path to the Q-table CSV file')
    parser.add_argument('--duration', type=int, required=True, help='Required testing duration')
    parser.add_argument('--L', type=int, required=True, help='Initial L')
    parser.add_argument('--M', type=int, required=True, help='Initial M')
    parser.add_argument('--V', type=int, required=True, help='Initial V')
    args = parser.parse_args()
    
    test_q_learning(args.q_table, args.duration, args.L, args.M, args.V)
        
   