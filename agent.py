import torch
from game import board, player, QuoridorGame
from collections import deque
import numpy as np
import model
import random

MAX_MEMORY = 10000000
BATCH_SIZE = 1000
LR = 0.001

class agent():
    def __init__(self, player):
        self.number_of_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0.9 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #popleft()
        self.model = model.Linear_QNet(290,140,500)
        self.trainer = model.QTrainer(self.model, self.gamma, LR)
        self.player = player
        self.wins = 0

    def get_state(self, game):
        #build game state
        state = np.concatenate([game.board.squares,game.board.walls_vert,game.board.walls_hor,game.board.vertices,self.player], axis=None)
        return state
    
    def get_action(self, state, game):
        
        self.epsilon = 200000 - self.number_of_games
        final_move = np.zeros(140)
            
        #epsilon greedy exploration vs exploitation
        if np.random.randint(0,180000) < self.epsilon:
            move_nums = np.arange(140)
            np.random.shuffle(move_nums)
            for i in range(len(move_nums)):
                movetype, direction, locations = game.get_action_vars(move_nums[i])
                if game.is_legal(movetype, direction, locations):
                    final_move[move_nums[i]] = 1
                    return final_move, movetype, direction, locations
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            prediction_sorted = np.argsort(prediction.detach().numpy())
            for i in range(len(prediction_sorted)):
                movetype, direction, locations = game.get_action_vars(prediction_sorted[i])
                if game.is_legal(movetype, direction, locations):
                    final_move[prediction_sorted[i]] = 1
                    return final_move, movetype, direction, locations

                
    def read_state_dict(self, state_dict):
        self.model = self.model.load_state_dict(state_dict)

    def remember(self,state, action, reward, next_state, game_over):
        self.memory.append([state,action,reward,next_state,game_over])


    def train_long_memory(self):
        #create mini sample
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states,actions,rewards,next_states,game_overs = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,game_overs)

def train():
    #set up game environment and agents
    env = board()
    p1 = player(1, 'blue_pawn.png')
    p2 = player(2, 'white_pawn.png')
    game = QuoridorGame(p1,p2,env)
    agent_1 = agent(1)
    agent_2 = agent(2)

    while True:
        #reset reward
        reward = 0

        #if it is the first move, and the learning agent is p2, do a move 
        if agent_1.player == 2 and game.turn == 0:
            move, mtype, dir, loc = agent_2.get_action(agent_2.get_state(game),game)
            game.play_step(mtype, dir, loc)

        #get old state
        state_old = agent_1.get_state(game)

        #get move
        final_move, movetype, direction, locations = agent_1.get_action(state_old, game)

        #perform move and get interim state
        game_over, p1_reward, p2_reward = game.play_step(movetype, direction, locations)
        rewards = [p1_reward, p2_reward]
        reward += rewards[agent_1.player - 1]

        state_new = agent_1.get_state(game)

        #remember
        agent_1.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            game.reset()
            agent_1.train_long_memory()

            agent_1.player = np.random.randint(1,3)
            agent_2.player = (agent_1.player % 2) + 1
            agent_1.number_of_games += 1
            agent_2.number_of_games += 1

            if reward == 10:
                agent_1.wins += 1

            #print summary stats
            win_percent = agent_1.wins/agent_1.number_of_games    
            print(f'game: {agent_1.number_of_games}, wins: {agent_1.wins} win percent: {win_percent}')
        
        else:

            #opponent agent plays move, get the final state for the 'turn'
            opp_move, movetype, direction, locations = agent_2.get_action(state_new, game)
            game_over, p1_reward, p2_reward= game.play_step(movetype, direction, locations)


            if game_over:
                reward = -10
                agent_1.memory.pop()
                agent_1.remember(state_old, final_move, reward, state_new, game_over)
                game.reset()
                agent_1.train_long_memory()

                agent_1.player = np.random.randint(1,3)
                agent_2.player = (agent_1.player % 2) + 1
                agent_1.number_of_games += 1
                agent_2.number_of_games += 1

                win_percent = agent_1.wins/agent_1.number_of_games   
                print(f'game: {agent_1.number_of_games}, wins: {agent_1.wins} win percent: {win_percent}')
                



#also TO DO: Save state dicts, update the opponent's model every so often so
#the opponent also improves during training

if __name__ == '__main__':
    train()