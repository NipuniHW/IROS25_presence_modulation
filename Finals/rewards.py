'''
if 0 <= gaze_score <= 30:
        reward = 50
    if 31 <= gaze_score <= 60:
        reward = -10
    if 61 <= gaze_score <= 100:
        reward = -50
    return reward

'''
# s , a, threshold -> r
'''
gaze will be 0.0 - 100.0, threshold will be 0.0 - 30.0
'''

def low_gaze_reward_LVM(state, action_vector, gaze_threshold=[0, 3]):
    gaze = state[0]
    distance_to_goal_state = gaze - (sum(gaze_threshold) / 2)
    action_sum_gaze_alter = sum(action_vector)*5 # = -3 - 3
    distance_to_goal_after_action = distance_to_goal_state + (action_sum_gaze_alter)
    new_gaze = gaze + distance_to_goal_after_action

    #if currently within threshold before action
    if gaze_threshold[0] <= gaze <= gaze_threshold[1]:
        #  The action extimator says we'll keep the agent within the threshold
        if gaze_threshold[0] <= new_gaze <= gaze_threshold[1]:
            desired_gaze = sum(gaze_threshold) / 2
            return 5 + (desired_gaze - abs(desired_gaze - new_gaze))*5
            # return abs(distance_to_goal_state + action_sum_gaze_alter)
        else:
            return -1
    else:
        return -abs(distance_to_goal_after_action)

def low_gaze_reward(gaze, action_vector, gaze_threshold=[0, 3]):
    distance_to_goal_state = gaze - (sum(gaze_threshold) / 2)
    action_sum_gaze_alter = sum(action_vector)*0.5 # = -3 - 3
    distance_to_goal_after_action = distance_to_goal_state + (action_sum_gaze_alter)
    new_gaze = gaze + distance_to_goal_after_action

    #if currently within threshold before action
    if gaze_threshold[0] <= gaze <= gaze_threshold[1]:
        #  The action extimator says we'll keep the agent within the threshold
        if gaze_threshold[0] <= new_gaze <= gaze_threshold[1]:
            desired_gaze = sum(gaze_threshold) / 2
            return 0.5 + (desired_gaze - abs(desired_gaze - new_gaze))*0.5
            # return abs(distance_to_goal_state + action_sum_gaze_alter)
        else:
            return -1
    else:
        return -abs(distance_to_goal_after_action)
    
def medium_gaze_reward(gaze, action_vector, gaze_threshold=[4, 6]):
    distance_to_goal_state = gaze - (sum(gaze_threshold) / 2)
    action_sum_gaze_alter = sum(action_vector)*0.5 # = -3 - 3
    distance_to_goal_after_action = distance_to_goal_state + (action_sum_gaze_alter)
    new_gaze = gaze + distance_to_goal_after_action

    #if currently within threshold before action
    if gaze_threshold[0] <= gaze <= gaze_threshold[1]:
        #  The action extimator says we'll keep the agent within the threshold
        if gaze_threshold[0] <= new_gaze <= gaze_threshold[1]:
            desired_gaze = sum(gaze_threshold) / 2
            return 0.5 + (desired_gaze - abs(desired_gaze - new_gaze))*0.5
            # return abs(distance_to_goal_state + action_sum_gaze_alter)
        else:
            return -1
    else:
        return -abs(distance_to_goal_after_action)
    
#TODO::Test
def high_gaze_reward(previous_gaze, action_vector, next_gaze, gaze_threshold=[7,10]):
    # if gaze is not an integer between 0 and 10, raise an error
    if previous_gaze not in range(11) or next_gaze not in range(11) or not all(isinstance(i, int) for i in [previous_gaze, next_gaze]):
        raise ValueError(f"Gaze values must be integers between 0 and 10. Got {previous_gaze} and {next_gaze}")
    # if the action vector is not a list of 3 integers, raise an error
    # if not all(isinstance(i, str) for i in action_vector) or len(action_vector) != 3:
    #     raise ValueError(f"Action vector must be a list of 3 integers. Got {action_vector}")
    
    previous_distance_to_goal_state = abs(previous_gaze - (sum(gaze_threshold) / 2))
    next_distance_to_goal_state = abs(next_gaze - (sum(gaze_threshold) / 2))
    
    difference = previous_distance_to_goal_state - next_distance_to_goal_state
    # if the current state is not within the threshold
    if not gaze_threshold[0] <= next_gaze <= gaze_threshold[1]:    
        if previous_distance_to_goal_state < next_distance_to_goal_state:
            # This is bad because the agent is moving away from the goal in the next step
            return -(abs(difference))
        elif previous_distance_to_goal_state > next_distance_to_goal_state:
            # This is good because the agent is moving towards the goal in the next step
            return abs(difference)
        else:
            # They're equal which is also bad
            return -2
    else:
        return 5
        
    
# distance_to_goal_state = gaze - (sum(gaze_threshold) / 2)
# action_sum_gaze_alter = sum(action_vector)*5 # = -3 - 3
# distance_to_goal_after_action = distance_to_goal_state + (action_sum_gaze_alter)
# new_gaze = gaze + distance_to_goal_after_action

# #if currently within threshold before action
# if gaze_threshold[0] <= gaze <= gaze_threshold[1]:
#     #  The action extimator says we'll keep the agent within the threshold
#     if gaze_threshold[0] <= new_gaze <= gaze_threshold[1]:
#         desired_gaze = sum(gaze_threshold) / 2
#         return 5 + (desired_gaze - abs(desired_gaze - new_gaze))*5
#         # return abs(distance_to_goal_state + action_sum_gaze_alter)
#     else:
#         return -1
# else:
#     return -abs(distance_to_goal_after_action)
    
# write main function to test the rewards
if __name__=="__main__":
    testnum = 0
    #test case 1
    previous_gaze = 2
    action_vector = [1, 1, 1]
    next_gaze = 8
    expected_reward = 5
    reward = high_gaze_reward(previous_gaze, action_vector, next_gaze)
    result = reward == expected_reward
    assert reward == expected_reward, f"Test case {testnum} failed: {reward} != {expected_reward}"
    print(f"Test case {testnum} result: {result} with reward: {reward} and expected reward: {expected_reward}")
    testnum += 1
    
    #test case 2
    previous_gaze = 2
    action_vector = [1, 1, 1]
    next_gaze = 4
    expected_reward = 2
    reward = high_gaze_reward(previous_gaze, action_vector, next_gaze)
    result = reward == expected_reward
    assert reward == expected_reward, f"Test case {testnum} failed: {reward} != {expected_reward}"
    print(f"Test case {testnum} result: {result} with reward: {reward} and expected reward: {expected_reward}")
    testnum += 1
    
    #test case 3
    previous_gaze = 8
    action_vector = [1, 1, 1]
    next_gaze = 4
    expected_reward = -4
    reward = high_gaze_reward(previous_gaze, action_vector, next_gaze)
    result = reward == expected_reward
    assert reward == expected_reward, f"Test case {testnum} failed: {reward} != {expected_reward}"
    print(f"Test case {testnum} result: {result} with reward: {reward} and expected reward: {expected_reward}")
    testnum += 1
    
    #test case 4
    previous_gaze = 10
    action_vector = [1, 1, 1]
    next_gaze = 0
    expected_reward = -7
    reward = high_gaze_reward(previous_gaze, action_vector, next_gaze)
    result = reward == expected_reward
    assert reward == expected_reward, f"Test case {testnum} failed: {reward} != {expected_reward}"
    print(f"Test case {testnum} result: {result} with reward: {reward} and expected reward: {expected_reward}")
    testnum += 1
    
    # #test case 1
    # gaze = 15
    # action_vector = [1, 1, 1]
    # expected_reward = 5
    # reward = low_gaze_reward(gaze, action_vector)
    # result = reward == expected_reward
    # assert reward == expected_reward, f"Test case {testnum} failed: {reward} != {expected_reward}"
    # print(f"Test case {testnum} result: {result} with reward: {reward} and expected reward: {expected_reward}")
    # testnum += 1

    # #test case 2
    # gaze = 90
    # action_vector = [-1, -1, -1]
    # expected_reward = -60
    # reward = low_gaze_reward(gaze, action_vector)
    # result = reward == expected_reward
    # assert reward == expected_reward, f"Test case {testnum} failed: {reward} != {expected_reward}"
    # print(f"Test case {testnum} result: {result} with reward: {reward} and expected reward: {expected_reward}")
    # testnum += 1

    # #test case 3
    # gaze = 15
    # action_vector = [-1, -1, -1]
    # expected_reward = 5
    # reward = low_gaze_reward(gaze, action_vector)
    # result = reward == expected_reward
    # assert reward == expected_reward, f"Test case {testnum} failed: {reward} != {expected_reward}"
    # print(f"Test case {testnum} result: {result} with reward: {reward} and expected reward: {expected_reward}")
    # testnum += 1

    #  #test case 4
    # gaze = 0
    # action_vector = [-1, -1, -1]
    # expected_reward = -1
    # reward = low_gaze_reward(gaze, action_vector)
    # result = reward == expected_reward
    # assert reward == expected_reward, f"Test case {testnum} failed: {reward} != {expected_reward}"
    # print(f"Test case {testnum} result: {result} with reward: {reward} and expected reward: {expected_reward}")
    # testnum += 1

    #  #test case 5
    # gaze = 100
    # action_vector = [1, 1, 1]
    # expected_reward = -100
    # reward = low_gaze_reward(gaze, action_vector)
    # result = reward == expected_reward
    # assert reward == expected_reward, f"Test case {testnum} failed: {reward} != {expected_reward}"
    # print(f"Test case {testnum} result: {result} with reward: {reward} and expected reward: {expected_reward}")
    # testnum += 1

