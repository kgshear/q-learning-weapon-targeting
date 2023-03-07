from collections import Counter
from random import randrange
from UtilityFunctions import time_to_hit_enemy_with_friendly, get_time_to_impact, parse_msg_to_ships_weapons_tracks
from PlannerProto_pb2 import StatePb, ShipActionPb, WeaponPb, TrackPb, OutputPb
from Agents.NavyAgent import OrderProtectedNavyAgent
from FeatureExtractor import ArrayExtractor
import numpy as np
import pickle



class QLearner(OrderProtectedNavyAgent):

    def __init__(self, epsilon=0.05, alpha=0.1, discount=.8):
        super().__init__()
        self.runs_so_far = 0
        self.accum_train_rewards = 0.0
        self.accum_test_rewards = 0.0
        self.num_training = 450 # change this (depending on how many runs you want it to train for)
        self.epsilon = epsilon      #exploration rate
        self.alpha = alpha          #learning rate
        self.discount = discount
        self.last_state = None
        self.last_actions = []
        self.run_rewards = 0.0
        self.curState = None
        self.weights = Counter()
        self.generate_weights()
        self.legal_actions = None
        self.last_features = Counter()                                       # constantly updated
        self.actions = []
        self.run_targets = []
        self.run_weapons = []
        self.change_in_score = 0
        self.elapsed_score = 0


    def _update(self, msg: StatePb):
        if self.curState != None:
            self.last_state = self.curState
        self.change_in_score = msg.score - self.elapsed_score
        self.elapsed_score = msg.score
        self.curState = msg
        if self.is_in_training():
            self.calc_reward(self.curState)
        self.ships, self.weapons, self.tracks = parse_msg_to_ships_weapons_tracks(msg)
        self.enemy = [missile for missile in self.tracks if missile.ThreatRelationship == "Hostile"]
        self.legal_actions = self.get_legal_actions(self.weapons, self.enemy)


    def _calculate_feature_vector(self):
        self.featExtract = ArrayExtractor(self.curState)
        ## insert auto-encoder here

    def get_legal_actions(self, weapons: list[WeaponPb], tracks: list[TrackPb]):
        # returns actions that can be taken at a given state
        # action cannot be taken if: (1) track is not hostile, (2) weapon cannot reach missile in time,
            # (3) missile has already been targeted, (4) weapon has already been fired
        legal_actions = []
        weapon_index = 0
        for weapon in weapons:
            if weapon[1].WeaponState == "Ready" and weapon[1].Quantity > 0 and ((weapon[0], weapon[1].SystemName) not in self.run_weapons):
                for enemy in tracks:
                    friendly = [ship for ship in self.ships if ship.AssetName == weapon[0]]
                    if (enemy.ThreatRelationship  == "Hostile") and \
                        (time_to_hit_enemy_with_friendly(friendly[0], enemy, weapon[1].SystemName) < get_time_to_impact(enemy, friendly[0])) \
                        and (enemy.TrackId not in self.run_targets):

                        legal_actions.append(weapon[0] +','+ str(enemy.TrackId) + ','+ weapon[1].SystemName)
            # check if ship weapon works, if so add to legal actions
            weapon_index += 1
        return legal_actions

    def get_q(self, state, action):
        # calculates dot product of weights and features
        total = 0
        x = self.weights
        y = self.featExtract.getFeatures(state, action)
        for key in x:
            total += x[key] * y[key]
        return total

    def compute_value_from_q_values(self, state):
        #returns highest q value of all actions

        if len(self.legal_actions) == 0:
            return 0.0
        # returns action with highest q value
        max_action = max([self.get_q(state, action)
                          for action in self.legal_actions])

        return max_action

    def generate_weights(self):

        self.weights = Counter()
        count = 1
        while count != 31:
            self.weights["ClosestEnemy_" + str(count)] = 0
            count += 1
        return self.weights


    def compute_action_from_q_values(self, state):
        # returns action with highest q value
        if len(self.legal_actions) == 0:
            return None
        max_q = float('-inf')
        best_action = None
        meanq = []
        for action in self.legal_actions:
            q = self.get_q(state, action)
            if (q > max_q):
                meanq.append(q)
                max_q = q
                best_action = action
        meanq = np.mean(meanq)
        if (max_q < meanq): # if q is lower than threshold, return nothing (prevents us from making useless actions)
            return None
        return best_action

    def get_action(self, state):
        # Pick Action
        self.legal_actions = self.get_legal_actions(self.weapons, self.enemy)

        if len(self.legal_actions) == 0:
            return None
        if np.random.binomial(1, self.epsilon):
            action = self.legal_actions[randrange(0, len(self.legal_actions))]
        else:
            action = self.compute_action_from_q_values(state)

        actionId = action.split(',')
        self.run_targets.append(actionId[1])
        self.run_weapons.append((actionId[0], actionId[2]))
        return action

    def _select_action(self):
        # selects the best action and appends it to OutputPb
        actionnum = 5 # max actions availible at a step
        while actionnum != 0:
            action = self.get_action(self.curState)
            self.actions.append(action)
            actionnum -= 1
        output = OutputPb()
        count = 0
        ship_action: ShipActionPb = ShipActionPb()
        for action in self.actions:
            if action is not None:
                print(self.runs_so_far, count, action)
                actionIndex = action.split(',')
                ship_action.AssetName = actionIndex[0]
                ship_action.TargetId = int(actionIndex[1])
                ship_action.weapon = actionIndex[2]
                self.fired.append(int(actionIndex[1]))
                output.actions.append(ship_action)
                count += 1
        if (len(self.actions) == 0):
            output.actions.append(ship_action)
        return output

    def updateQ(self, state, action, next_state, reward):
        # updates the weight of the state
        diff = (reward + self.discount * self.get_value(next_state)) - self.get_q(state, action)
        for i in self.last_features:
            self.weights[i] += self.alpha * diff * self.last_features[i]


    def observe_change(self, state, actions, next_state, delta_reward):
        self.run_rewards += delta_reward
        for action in actions:
            self.updateQ(state, action, next_state, delta_reward)

    def start_run(self):
        self.run_rewards = 0.0
        self.curState = None
        self.fired = []
        self.elapsed_score = 0
        self.last_actions = self.actions
        self.actions = []
        self.run_weapons = []
        self.run_targets = []

    def stop_run(self):
        if self.runs_so_far < self.num_training:
            self.accum_train_rewards += self.run_rewards
        else:
            self.accum_test_rewards += self.run_rewards
        self.runs_so_far += 1
        if self.runs_so_far >= self.num_training:
            # Take off the training wheels
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning
        if self.curState != None:
            self.final(self.curState)
            self.last_features = self.featExtract.getFeatures(self.curState, None)

    def is_in_training(self):
        return False

    def is_in_testing(self):
        return not self.is_in_training()

    def calc_reward(self, state):
        if self.last_state is not None:
            reward = self.change_in_score
            self.observe_change(self.last_state, self.last_actions, state, reward)

    def get_value(self, state):
        return self.compute_value_from_q_values(state)

    def load_weights(self):
        with open("Agents/QData/q_weights", 'rb') as handle:
            self.weights = pickle.load(handle)
        print("Loaded weights")

    def save_weights(self):
        #saving weights and features
        with open("Agents/QData/q_weights", 'wb') as handle:
            pickle.dump(self.weights, handle)
        print("Saving weights")

    def new_run(self):
        if (self.curState == None):
            self.load_weights()
        self.stop_run()
        self.start_run()

    def final(self, state):
        NUM_EPS_UPDATE = 10  #change depending on how often you want status updates

        if 'last_window_accum_rewards' not in self.__dict__:
            self.last_window_accum_rewards = 0.0
        self.last_window_accum_rewards += (state.score)
        if self.runs_so_far % NUM_EPS_UPDATE == 0:
            print('Reinforcement Learning Status:')
            print(self.weights)


            window_avg = self.last_window_accum_rewards / float(NUM_EPS_UPDATE)
            print('\tAverage Rewards for last %d episodes: %.2f' % (
                NUM_EPS_UPDATE, window_avg))
            self.last_window_accum_rewards = 0.0
            if self.runs_so_far <= self.num_training:
                train_avg = self.accum_train_rewards / float(self.runs_so_far)
                print('\tCompleted %d out of %d training episodes' % (
                    self.runs_so_far, self.num_training))
                print('\tAverage Rewards over all training: %.2f' % (
                    train_avg))
                print("=======Feature Weights=======")
                for i in self.featExtract.getFeatures(self.last_state, None):
                    print("{0} : {1}".format(i, self.weights[i]))

            else:
                test_avg = float(self.accum_test_rewards) / \
                           (self.runs_so_far - self.num_training)
                print('\tCompleted %d test episodes' % (self.runs_so_far - self.num_training))
                print('\tAverage Rewards over testing: %.2f' % test_avg)

            if self.runs_so_far == self.num_training:
                self.save_weights()


