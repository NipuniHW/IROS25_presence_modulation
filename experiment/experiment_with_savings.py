import os
import sys
import cv2
from time import time, sleep
import argparse
from pepper import Pepper
from gaze_controller import *
from experiment_functions import *
from mdp_formulation import high_gaze_config_6, low_gaze_config_6

def test_q_learning(q_table_path, duration_minutes, L1, M1, V1, testing_runname, subject_count):
    pepper = Pepper()
    pepper.connect('pepper.local', '9559')
    # pepper.connect("localhost", 38975)
    
    try:
        if not pepper.is_connected:
            sys.exit(1)
            del pepper 
        else:
            q_table = load_q_table_from_csv(q_table_path)
            
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
            
            # Ask the user to press enter to start the experiment
            sleep(1)
            print('Press Enter to start the experiment')
            input()
            
            # Start the training
            current_time = time()
            
            light, movement, volume = L1, M1, V1 # Default values 
            testing_duration_minutes = duration_minutes*60
            
            start_time_inner_loop = time()
            time_step_count = 0
            save_dictionary = {}
            
            while time() - current_time < testing_duration_minutes:
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
                    if gaze_score > 0 and state == 0:
                        state = 1
                        gaze_score = 15.5678
                    print(f"Gaze score: {gaze_score} -- giving state: {state}")
                    action = choose_action(state, q_table)
                    print(f"Chosen action: {action}")
                    # nonlocal light, movement, volume
                    light, movement, volume = pepper.update_behavior(action, light, movement, volume, state)
                    # print("Updated the behavior")
                
                    #Wait for 3s
                    sleep(3)
                    # Get the current gaze score
                    gaze_score_ = controller.get_gaze_score()
                    ## CHANGED
                    next_state = int(round(gaze_score_/20))
                    
                    if gaze_score_ > 0 and next_state == 0:
                        next_state = 1
                        gaze_score_ = 15.5678
                    # Get the reward
                    reward = config.reward_function(state, action, next_state, config.gaze_threshold)
                    
                    save_dictionary['previousstate_subject_' + str(subject_count)+'_timestep_'+str(time_step_count)] = state
                    save_dictionary['nextstate_subject_' + str(subject_count)+'_timestep_'+str(time_step_count)] = next_state
                    save_dictionary['action_subject_' + str(subject_count)+'_timestep_'+str(time_step_count)] = action
                    save_dictionary['reward_subject_' + str(subject_count)+'_timestep_'+str(time_step_count)] = reward
                    
                    time_step_count+=1
                    
                    print(f"Next gaze score: {gaze_score_} -- giving state:{next_state}\n")
              
            save_trajectory_ep_to_yaml(testing_runname, subject_count-1, save_dictionary)
            
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Cleaning up...")
        del pepper   
        cv2.destroyAllWindows()
        sys.exit(0)
        
    del pepper 
    controller.kill_attention_thread()
    print("Experiment completed\n")

if __name__ == "__main__":                   
    parser = argparse.ArgumentParser(description='Q-Learning Configuration')
    parser.add_argument('--q_table', type=str, required=True, help='Path to the Q-table CSV file')
    parser.add_argument('--duration', type=int, required=True, help='Required testing duration')
    parser.add_argument('--config', type=str, 
                                    choices=['low_gaze_config_6',
                                             'high_gaze_config_6'
                                             ], 
                                             required=True, 
                                             help='Choose the configuration')
    parser.add_argument('--L', type=int, required=True, help='Initial L')
    parser.add_argument('--M', type=int, required=True, help='Initial M')
    parser.add_argument('--V', type=int, required=True, help='Initial V')
    parser.add_argument('--testing_runname', type=str, required=True, help='CSV file name to save the testing data')
    args = parser.parse_args()
    
    if args.config == 'low_gaze_config_6':
        config = low_gaze_config_6
    elif args.config == 'high_gaze_config_6':
        config = high_gaze_config_6
        
    subject_count = 0
    
    if not os.path.exists(args.testing_runname):
        print('we are executing a new testing session')
        os.makedirs(args.testing_runname)
        print('made directory:' + args.testing_runname + ' for testing data')
        
    while True:
        subject_count += 1
        user_input = input('\nWould you like to start/ continue testing for another subject? (Y/N): ')
        if user_input.lower() == 'y' or user_input.lower() == 'Y':
            test_q_learning(args.q_table, args.duration, args.L, args.M, args.V,args.testing_runname, subject_count)
        else:
            print('Your input was not Y/y. Exiting testing')
            break
    
        
   