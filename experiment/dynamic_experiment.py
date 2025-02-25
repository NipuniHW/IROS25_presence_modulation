import os
import sys
import cv2
from time import time, sleep
import argparse
from pepper import Pepper
from gaze_controller import *
from experiment_functions import *
from mdp_formulation import high_gaze_config_6, low_gaze_config_6

def test_q_learning(q_table_low_path, q_table_high_path, testing_runname):
    pepper = Pepper()
    pepper.connect("pepper.local", 9559)
    # pepper.connect("localhost", 38975)
    
    try:
        if not pepper.is_connected:
            sys.exit(1)
            del pepper 
        else:
            q_table_low = load_q_table_from_csv(q_table_low_path)
            q_table_high = load_q_table_from_csv(q_table_high_path)
            
            # Change the camera ID to 2 if using external usb webcam, 0 if using the laptop webcam
            controller = GazeInterfaceController(camera_id=2)
            sleep(1)
            # Ask the user to press enter to start a calibration
            print('Press Enter to start the calibration')
            input()
            controller.calibration_exe()
            controller.start_detecting_attention()

            # Ask the user to press enter to test the calibration
            sleep(1)
            print('Press Enter to test the calibration')
            input()
            
            curr_time = time()
            
            while time() - curr_time < 10:
                frame = controller.get_visualisation_frame()
                if frame is not None:
                        f = deepcopy(frame)
                        # print("the type of frame is ", type(f))
                        cv2.imshow('Testing Calibrating', f)
                        if cv2.waitKey(5) & 0xFF == 27:
                            cv2.destroyAllWindows()
            cv2.destroyAllWindows()    
            
            # Ask the user to press enter to start the training
            sleep(1)
            print('Press Enter to start the testing')
            input()
            
            # Start the training
            current_time = time()
            
            light, movement, volume = 0,0,0 # Default values 
            
            start_time_inner_loop = time()
            switch_time = 90  # 90 seconds
            testing_duration_minutes = 271
            current_time = time()
            time_step_count = 0
            save_dictionary = {}

            while time() - current_time < testing_duration_minutes:
                frame = controller.get_visualisation_frame()
                if frame is not None:
                    f = deepcopy(frame)
                    cv2.imshow('Calibrated HRI Attention Detection', f)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                if time() - start_time_inner_loop >= 3:
                    start_time_inner_loop = time()
                    # Get the current gaze score
                    gaze_score = controller.get_gaze_score()
                    state = int(round(gaze_score / 20))
                    if gaze_score > 0 and state == 0:
                        state = 1
                        gaze_score = 15.5678
                    print(f"Gaze score: {gaze_score} -- giving state: {state}")

                    # Determine which Q-table to use based on the elapsed time
                    elapsed_time = time() - current_time
                    if (elapsed_time // switch_time) % 2 == 0:
                        action = choose_action(state, q_table_low)
                        print(f"Chosen action from q_table_low: {action}")
                        q_table_name = 'Low'
                    else:
                        action = choose_action(state, q_table_high)
                        print(f"Chosen action from q_table_high: {action}")
                        q_table_name = 'High'

                    # Update the behavior
                    light, movement, volume = pepper.update_behavior(action, light, movement, volume, state)
                    
                    save_dictionary['state_subject_1_timestep_'+str(time_step_count)] = state
                    save_dictionary['nextstate_subject_1_timestep_'+str(time_step_count)] = gaze_score
                    save_dictionary['action_subject_1_timestep_'+str(time_step_count)] = action
                    save_dictionary['q_table_'+str(time_step_count)] = q_table_name

                    time_step_count+=1
                    
            save_trajectory_ep_to_yaml_3(testing_runname, save_dictionary)
            
    
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Cleaning up...")
        del pepper   
        cv2.destroyAllWindows()
        sys.exit(0)
                    
    del pepper 
    controller.kill_attention_thread()
    print("Test completed\n")

if __name__ == "__main__":                   

    q_table_low = "/home/nipuni/Documents/IROS25_presence_modulation/Finals/trained_q_tables/q_table_low.csv"
    q_table_high =  "/home/nipuni/Documents/IROS25_presence_modulation/Finals/trained_q_tables/q_table_high.csv"
    
    parser = argparse.ArgumentParser(description='Q-Learning Configuration')
    parser.add_argument('--testing_runname', type=str, required=True, help='CSV file name to save the testing data')
    args = parser.parse_args()
       
    if not os.path.exists(args.testing_runname):
        print('we are executing a new initial testing session')
        os.makedirs(args.testing_runname)
        print('made directory:' + args.testing_runname + ' for testing initial data')
        
        
    while True:

        user_input = input('Would you like to start/ continue testing for another subject? (Y/N): ')
        if user_input.lower() == 'y' or user_input.lower() == 'Y':
            test_q_learning(q_table_low, q_table_high, args.testing_runname)
        else:
            print('Your input was not Y/y. Exiting testing')
            break
    
        
   