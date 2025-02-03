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
def low_gaze_reward(gaze, action_vector, gaze_threshold=[0.0, 30.0]):
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
    
def high_gaze_reward(gaze, gaze_threshold):
    if gaze > gaze_threshold:
        return 1
    else:
        return 0
    
def medium_gaze_reward(gaze, gaze_threshold):
    if gaze < gaze_threshold - 0.1 or gaze > gaze_threshold + 0.1:
        return 1
    else:
        return 0
    
# write main function to test the rewards
if __name__=="__main__":
    testnum = 0
    #test case 1
    gaze = 15
    action_vector = [1, 1, 1]
    expected_reward = 5
    reward = low_gaze_reward(gaze, action_vector)
    result = reward == expected_reward
    assert reward == expected_reward, f"Test case {testnum} failed: {reward} != {expected_reward}"
    print(f"Test case {testnum} result: {result} with reward: {reward} and expected reward: {expected_reward}")
    testnum += 1

    #test case 2
    gaze = 90
    action_vector = [-1, -1, -1]
    expected_reward = -60
    reward = low_gaze_reward(gaze, action_vector)
    result = reward == expected_reward
    assert reward == expected_reward, f"Test case {testnum} failed: {reward} != {expected_reward}"
    print(f"Test case {testnum} result: {result} with reward: {reward} and expected reward: {expected_reward}")
    testnum += 1

    #test case 3
    gaze = 15
    action_vector = [-1, -1, -1]
    expected_reward = 5
    reward = low_gaze_reward(gaze, action_vector)
    result = reward == expected_reward
    assert reward == expected_reward, f"Test case {testnum} failed: {reward} != {expected_reward}"
    print(f"Test case {testnum} result: {result} with reward: {reward} and expected reward: {expected_reward}")
    testnum += 1

     #test case 4
    gaze = 0
    action_vector = [-1, -1, -1]
    expected_reward = -1
    reward = low_gaze_reward(gaze, action_vector)
    result = reward == expected_reward
    assert reward == expected_reward, f"Test case {testnum} failed: {reward} != {expected_reward}"
    print(f"Test case {testnum} result: {result} with reward: {reward} and expected reward: {expected_reward}")
    testnum += 1

     #test case 5
    gaze = 100
    action_vector = [1, 1, 1]
    expected_reward = -100
    reward = low_gaze_reward(gaze, action_vector)
    result = reward == expected_reward
    assert reward == expected_reward, f"Test case {testnum} failed: {reward} != {expected_reward}"
    print(f"Test case {testnum} result: {result} with reward: {reward} and expected reward: {expected_reward}")
    testnum += 1

