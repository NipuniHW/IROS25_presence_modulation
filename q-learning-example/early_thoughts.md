# Q-Learning Example

## Overview

- This folder holds an example notebook with Q-learning for learning navigation in an example gridworld, from this [YouTube link](https://www.youtube.com/watch?v=iKdlKYG78j4).
- The video gives all the major information required.
- With four actions against an 11-sized square grid, the Q-table ends up being 484 cells with 121 possible states against four possible actions.
- The reward given is:
  - -100 in certain terminal states
  - -1 for every valid action
  - 100 for reaching the proposed goal state
- This is just an example Q-learning application.

The rest of this document just has my notes and thoughts about the current approach after discussion.

## MDP Formulation for Affective Computing Gaze Problem

- At present moment, there are 27 possible actions for the robot altering its lights, volume, and movement, all 10 possible actions between 1-10, to take depending on gaze and context data.
- As a result, the Q-table will be the size of 27 * (all possible states).

Possible state configurations discussed include:
- Context [disengaged, social, alarm] combined with gaze_score ([10, 20, ... 90, 100], len10). As a result, the Q-table in this formulation would be 810 (3 * 10 * 27).
- Context [disengaged, social, alarm] combined with gaze_score ([10, 20, ... 90, 100], len10) alongside the current movement, light, and volume (all of len 10) values. As a result, a Q-table of this size would be 810,000? (3 context * 10 gaze score * 10 volume * 10 move * 10 light * 27 actions).
  - Concerns with this formulation is that it is a large Q-table value.
  - Need to check with literature how effective Q-learning is at a table of this scale.
  - Regarding this point, it is quite possible that it will be more effective for a Q-learning formulation and mitigations could be taken to minimize the table's size (i.e., reducing the dimensionality of current move, light, and vol) to a discrete range of 5 in the state, thereby the Q-table would end up being 101,250. Which while still large feels far more manageable than a value near 1 million.

- Understanding the size of the Q-table and how it will impact our learning approach is an important step to refining the effectiveness of the affective learning.
- Experimentation with the MDP formulations should reveal insights.

## Proposed Reward Structure

- Reward Function broad description follows the following function
```python
contexts = ["Disengaged", "Social", "Alarmed"]
expected_ranges = {
    "Disengaged": (0, 30),
    "Social": (31, 60),
    "Alarmed": (61, 100)
}

def get_reward(context, gaze_score):
    # will get the ranges for the desired context
    expected_min, expected_max = expected_ranges[context]

    # Gets the center of the range
    expected_center = (expected_min + expected_max) / 2

    # Gets the range of the current expect value
    expected_range_width = expected_max - expected_min

    # in this context, I lack the the understanding of gaze score and how it related to the expected center, in reading this value, I would expect to a gaze score to ideally be a value between 0 and 100, and where it matches to the 'expected centre, i.e. if it's 15 in disengaged, that's the ideal score, 45 for social etc.' I'll need elaboration here 
    reward = 1 - (abs(gaze_score - expected_center) / (expected_range_width / 2)) ** 2

    return reward
```

- The first thought that strikes me is that we can probably simplify the reward function, i.e. if the gaze score is not in the range, simply give -1, if its roughly in the gaze range but not ideal, give 4, and if its in the ideal range for the situation give 15.
- I'll need to check this point in discussions.

## Request to examine gaze.py and context_identification.py

- Request to examine the way the gaze_score and context is correct or not.
- So I took a look through the code, truthfully without visualisation or test cases it's hard to determine the cause, will discuss at next meeting.