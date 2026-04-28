# basic 1d treadmill environment to test context-dependent reinforcement learning problems 
# models each state as a unique one-hot encoding
import numpy as np
        
# state-encoding is one_hot: [normal cell?, *which cue position, *which reward position]
# cue position and reward position are both one hot encodings (1 at "cue number" index for each array)

class Cue_Treadmill():
    def __init__(self, length, cue_position, cue_length, reward_position, reward_length, reward_positions, cue_number, total_cues, reward_values=[0, 1, -1]):
        '''
        length: length of treadmill
        cue_position: index of cue position start
        cue_length: length of cue location
        reward_position: index of reward position
        reward_length: length of reward location (ending on actual rewarded cell)
        reward_positions: list of positions of all rewards
        cue_number: some numerical representation of the cue (index < total_cues)
        total_cues: total number of possible cues
        reward_values: reward for [ignoring, licking correctly, licking correctly]
        '''

        assert cue_position < reward_position
        assert reward_position < length

        self.current_state = 0
        self.length = length
        self.reward_position = reward_position
        self.cue_position = cue_position
        self.cue_number = cue_number
        self.total_cues = total_cues
        self.reward_values = reward_values
        self.one_hot_states = np.eye(self.length)

        self.actions = {
            0: 'ignore', 
            1: 'lick',
        }

        self.state_representations = np.zeros((self.length, 1+2*total_cues))
        
        # first run of blanks
        self.state_representations[:, 0] = 1

        # run of cues
        self.state_representations[cue_position:cue_position+cue_length, 0] = 0
        self.state_representations[cue_position:cue_position+cue_length, 1+cue_number] = 1


        # run of rewards (only rewarded on last location)
        for i, pos in enumerate(reward_positions):
            self.state_representations[pos-reward_length+1:pos+1, 0] = 0
            self.state_representations[pos-reward_length+1:pos+1, 1+total_cues+i] = 1

        self.state_representations[-1, 0] = 0

    def get_action_list(self):
        return [0, 1]

    def get_next_state(self, action):
        # return new_state and whether or not to terminate the run

        new_state = self.current_state + 1
        if new_state == self.length:
            return 0, True
        
        return new_state, False
        
    def move_state(self, new_state):
        self.current_state = new_state

    def _get_reward(self, action):
        if action == 0:
            # if self.current_state == self.reward_position: # ignoring at the reward is the same now as licking incorrectly
            #     return self.reward_values[2]
            return self.reward_values[0]
        elif self.current_state == self.reward_position:
            return self.reward_values[1]
        return self.reward_values[2]

    def take_action(self, action):
        # take action, receive reward (returned) and move to next state if possible (else terminate)
        # rewards are as follows:
        #   ignore -> 0
        #   wrongly lick -> -0.1
        #   properly lick -> +1

        reward = self._get_reward(action)
        observation = self.get_observation(self.current_state)
        
        next_state, terminated = self.get_next_state(action)

        return observation, reward, next_state, terminated

    def get_observation(self, state):
        return self.state_representations[state]
    
    def __str__(self):
        out_string = "length: {}, cue position: {}, cue value: {}, reward position: {}".format(self.length, self.cue_position, self.cue_number, self.reward_position)
        tread_rep = np.argmax(self.state_representations, 1).astype(object)
        tread_rep[self.reward_position] = str(tread_rep[self.reward_position]) + '*'
        return out_string + '\n' + str(tread_rep)


if __name__ == '__main__':

    length = 23
    cue_position = 8
    cue_length = 3
    reward_position = 15
    reward_length = 2
    reward_positions = [15, 19]
    cue_numbers = list(range(len(reward_positions)))
    total_cues = 2
    reward_values = [0, 1, -0.1]


    treadmill = Cue_Treadmill(length, cue_position, cue_length, reward_positions[0], reward_length, reward_positions, 0, total_cues)
    print(treadmill)

    treadmill = Cue_Treadmill(length, cue_position, cue_length, reward_positions[1], reward_length, reward_positions, 1, total_cues)
    print(treadmill)