# Q-Learning Example

## Overview

This folder holds an example notebook with q-learning for learning navigation in an example gridworld, from this [youtube link](https://www.youtube.com/watch?v=iKdlKYG78j4). The video gives all the major information required etc. With four actions agains an 11 sized square grid, the q-table ends up being 484 cells with 121 possible states against four possible actions. The reward given is -100 in certain terminal states, -1 for every valid action, and 100 for reaching the proposed goal state. This is just an example q learning application.

The rest of this document just has my notes and thoughts about the current approach after discussion.

<!-- ## Table of Contents -->

<!-- 1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Components](#components)
5. [Diagrams](#diagrams)
6. [Contributing](#contributing)
7. [License](#license) -->

## MDP formulation for affective computing gaze problem

At present moment, there are 27 possible actions for the robot altering it's lights, volume and movement, all 10 possible actions between 1-10, to take depending of gaze and context data. As a result, the q-table will be size of 27*(all possible states).

Possible state configurations discussed include:
    - Context [disengaged, social, alarm] combined with gaze_score ([10,20, ... 90, 100], len10). As a result, the q-table in this formulation would be 810 (3*10*27)
    - Context [disengaged, social, alarm] combined with gaze_score ([10,20, ... 90, 100], len10) alongside the current movement, light and volume (all of len 10) values. As a result, a q-table of this size would be 810,000? (3 context * 10 gaze score * 10 volume * 10 move * 10 light * 27 actions). Concerns with this formulation is that it is a large q-table value. Need to check with literature how effective q-learning is at a table of this scale. Regarding this point it is quite possible that it will be more effective for a q-learning formulation and migitgations could be taken to minimize the tables size (i.e. reducing the dimensionality of current move, light, and vol) to a discrete range of 5 in the state, there by the q-table would end up being 101,250. Which while still large feels far more managable than a value near 1 million.

Understanding the size of the q-table and how it will impact our learning approach is an important step to refining the effectiveness of the affective learning. Experimentation with the mdp formulations should reveal insights.

## Proposed reward structure
    Notes from  my end incoming... still studying the reward function right now.
## 