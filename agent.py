import torch
from game import board, player, QuoridorGame
from collections import deque
import numpy as np
import model

MAX_MEMORY = 10000000
BATCH_SIZE = 100
LR = 0.001

class agent():
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #popleft()
        self.model = model.Linear_QNet(289,140,500)
        self.trainer = None #TO DO

    def get_state(self,game):
        state = np.concatenate(game.board.squares,game.board.walls_vert,game.board.walls_hor,game.board.vertices, axis=None)
        return state
    
    def get_action(self, state, game):
        self.epsilon = 1000 - self.num_games
        final_move = np.zeros(140)
        if np.random.randint(0,200) < self.epsilon:
            move_nums = np.random.shuffle(np.arange(140)-1)
            while True:
                i = 0
                movetype, direction, locations = game.get_action_vars(move_nums[i])
                if game.is_legal(movetype, direction, locations):
                    return 
                i += 1
        else:
            pass

    def read_state_dict(self, state_dict):
        self.model = model.Linear_QNet().load_state_dict(state_dict)

    def remember(self,state, action, reward, next_state, game_over):
        self.memory.append([state,action,reward,next_state,game_over])

    def train_long_memory(self):
        pass

def train():
    env = board()
    p1 = player(1, 'blue_pawn.png')
    p2 = player(2, 'white_pawn.png')
    game = QuoridorGame(p1,p2,env)
    agent_1 = agent()
    agent_2 = agent()

    while True:

        #get old state
        state_old = agent_1.get_state(game)

        #get move
        final_move = agent_1.get_action(state_old)

        #perform move and get interim state
        turn_reward, game_over = game.play_step(final_move)
        state_interim = agent.get_state(game)

        if game_over:
            game.reset()
            agent.train_long_memory()

        #opponent agent plays move, get the final state for the 'turn'
        opp_move = agent_2.get_action(state_interim)
        opp_reward, game_over = game.play_step(opp_move)
        state_new = agent.get_state(game)

        #calc total turn reward
        reward = opp_reward + turn_reward

        #remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            game.reset()
            agent.train_long_memory()

        return 0

#prediction_values = np.argsort
#while True:
#   i = 0
#   if move_vector[prediction_values[0]] == 1
#       action = prediction_values[0]
#       break
#   else:
#       i += 1

if __name__ == '__main__':
    train()