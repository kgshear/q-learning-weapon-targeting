# ships: [0 , 0 , 0 , 0, 0]
# missile: [0 . . . 0]
# numTargeting [ 1 2 3 4 5]
# missileTarget [ 2  4 0 0 0 0 0 0 0 0 ]
# actions [ [0 , 0], [0, 0] ... ]
# shipHealth [ 2 3 1 4 5 ]
# nearTarget [ 0 0 0 0 0 0 0 0 0 0 0  ] 0 or 1 depending if missile is 40 s from target
import numpy as np
import os

import UtilityFunctions
from PlannerProto_pb2 import StatePb, ShipActionPb, AssetPb, TrackPb
from UtilityFunctions import get_threat_target, parse_msg_to_ships_weapons_tracks, get_time_to_impact
from collections import Counter

class Extractor():
    def __init__(self, state: StatePb):
        self.state = state
        self.ships, self.weapons, self.enemy = UtilityFunctions.parse_msg_to_ships_weapons_tracks(state)
    def getFeatures(self, state, action):
        pass

class ArrayExtractor(Extractor):
    #more detailed features describing the game state
    def __init__(self, state: StatePb):
        super().__init__(state)

    def getThreatArray(self, action):
        # returns an array of enemy missiles, sorted by closest
            # does not include missile that was targeted in "action"
        actionID = 100
        if action != None:
            actionID = action.split(",")[1]
        threatArray = [0]*30
        count = 0
        if (len(self.ships) != 0):
            for threat in self.enemy:
                if threat.ThreatRelationship == "Hostile" and threat.TrackId != actionID:
                    target = get_threat_target(threat, self.ships)
                    if target.isHVU:
                        value = 2
                    else:
                        value = 1
                    hitTime = float(get_time_to_impact(threat, target))
                    if hitTime > 0.0:
                        ratio = value / (hitTime + 0.001)
                        threatArray[count] = ratio
                    count += 1
        threatArray.sort(reverse=True)
        count = 1
        enemyArray = Counter()
        for item in threatArray:
            enemyArray["ClosestEnemy_"+ str(count)] = item
            count += 1
        return enemyArray



    def getFeatures(self, state, action):
        enemyArray = self.getThreatArray(action)
        return enemyArray
