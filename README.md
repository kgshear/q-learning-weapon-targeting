# q-learning-weapon-targeting
Approximate Q-Learning algorithm designed for the WTA (weapon target assignment) problem. Created for an AI/ML Competition held by NSWCDD Dahlgren. 

# The Problem

A battle simulation is generated where some amount of missiles is targeting some amount of stationary ships. These ships have a limited quantity of weapons to defend themselves. Weapons also have different speeds, meaning that a slow weapon might not be able to reach a missile before it sucessfully hits a ship. In addition, some ships are  high-value assets, meaning that they are worth more points. 

# The Solution

Based on the current state (represented by features), picks an availible action. Calculates reward based on the result of that action. This reward is used to update the weight of important features, which allows it to pick better actions in the future. Essentially, this code picks the best missile to target with an availible weapon at every time step. 

An action is considered availible if
1. The target is an enemy missile
2. The weapon is able to reach the missile in time 
3. The missile has not already been targeted by another weapon
4. The weapon can be used (has not already been fired and is not empty)

The features used to represent the state is a descending-order counter containing the ratio of (ship value / enemy_time_to_hit). This counter contains the ratio of every missile. Ship value is determined by whether or not the ship is a high value asset. A greater ship value and a lower time to hit will result in a higher number. 

#### Privacy Notice:
JCORE simulations and data were provided by the NSWCDD organizers; as such, it cannot be shared, except at the allowance of those organizers. Due to this, the data that this code relies on will not be shared. 
    

