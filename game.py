import numpy as np
import pygame
import torch

pygame.init()


class board():
    def __init__(self):
        self.reset_board()

    def reset_board(self):
        # numpy arrays for the squares, and the spaces between for walls
        # 1 = occupied, 0 = unoccupied
        # for squares, 1 = occupied by p1, 2 = occupied by p2, 0 = unoccupied
        self.squares = np.zeros((9,9))
        self.walls_vert = np.zeros((9,8))
        self.walls_hor = np.zeros((8,9))
        self.vertices = np.zeros((8,8))


class player(pygame.sprite.Sprite):
    def __init__(self, player_num, image):
        super().__init__()
        self.image = pygame.image.load(image)
        self.player = player_num
        if self.player == 1:
            self.start_pos = [0,4]
        else:
            self.start_pos = [8,4]
        self.reset_player()

    def reset_player(self):
        self.walls = 10
        self.position = self.start_pos


class QuoridorGame():
    def __init__(self, p1, p2, board):
        self.w = 800
        self.h = 800
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Quoridor')
        self.icon = pygame.image.load('white_pawn.png')
        pygame.display.set_icon(self.icon)
        self.display.fill((40,40,40))
        
        
        #init board and players
        self.board = board
        self.p1 = p1
        self.p2 = p2
        self.players = [self.p1, self.p2]

        self.reset()


    def reset(self):
        #init game state
        self.board.reset_board()
        self.p1.reset_player()
        self.p2.reset_player()
        self.turn = 0
        self.turn_player = self.players[self.turn]

    def place_wall(self, wall_direction, locations):
    #put down a wall on the board
        if wall_direction == 'v':
            self.board.walls_vert[locations[0]] = 1
            self.board.walls_vert[locations[1]] = 1
        elif wall_direction == 'h':
            self.board.walls_hor[locations[0]] = 1
            self.board.walls_hor[locations[1]] = 1
        self.board.vertices[locations[0]] = 1
        self.turn_player.walls -= 1

    def move_piece(self,direction):
        #move turn player's piece
        self.squares[self.turn_player.position] = 0
        self.turn_player.position += direction #NEEDS IMRPOVEMENT
        self.squares[self.turn_player.position] = self.turn_player + 1

    def play_step(self,action):
        #check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        #update game board
        self.render()

        #set it to be the other players turn
        self.turn += 1
        self.turn = self.turn % 2
        self.turn_player = self.players[self.turn]

    def render(self):
        #load current game state into pygame display
        for i in np.linspace(80,680,9): #board steps of 75 pixels
            for j in np.linspace(80,680,9):
                square = pygame.Rect(i,j,40,40)
                pygame.draw.rect(self.display,(255,255,255),square)
        self.display.blit(self.p1.image, (384,84))
        self.display.blit(self.p2.image, (384,684))
        pygame.display.update()

    def legal_moves(self):
        #returns a move_vector representing the action space, move_vector_n = 1 
        #indicates that the move represented by move_vector_n is legal
        move_vector = np.zeros(140)
        n=0
        #vertical wall places
        for i in range(8):
            for j in range(8):
                #checks if the wall spaces are free
                free_check = self.board.walls_vert[i,j] == 0 and self.board.walls_vert[i+1,j] == 0
                #checks if the wall would cross a horizontal wall
                cross_check = self.board.vertices[i,j] == 0
                #checks if the wall space is protected due to the blocking constraint
                '''NEED TO CHECK IF THE WALLS BLOCKS A PLAYER OFF'''
                is_not_protected = True         
                if free_check and cross_check and is_not_protected:
                    move_vector[n] = 1
                    n += 1                      
        #horizontal wall places
        for i in range(8):
            for j in range(8):
                #checks if the wall spaces are free
                free_check = self.board.walls_hor[i,j] == 0 and self.board.walls_hor[i,j+1] == 0
                #checks if the wall would cross a horizontal wall
                cross_check = self.board.vertices[i,j] == 0
                #checks if the wall space is protected due to the blocking constraint
                '''NEED TO CHECK IF THE WALLS BLOCKS A PLAYER OFF'''
                is_not_protected = True  
                if free_check and cross_check and is_not_protected:
                    move_vector[n] = 1
                    n += 1
        #normal piece move - check not either at edge or blocked by a wall
        pos = self.turn_player.position
        if pos[0] != 0 and self.board.walls_hor[pos] == 0: #backwards move
            move_vector[n] = 1
            n += 1
        if pos[0] != 9 and self.board.walls_hor[(pos[0],pos[1]-1)] == 0: #forwards move
            move_vector[n] = 1
            n += 1
        if self.turn_player.position[1] != 0: #left move
            move_vector[n] = 1
            n += 1
        if self.turn_player.position[1] != 9: #right move
            move_vector[n] = 1
            n += 1
        #double piece move
        #diagonal piece move

        return move_vector



if __name__ == '__main__':

    #create board and players
    env = board()
    p1 = player(1,'white_pawn.png')
    p2 = player(2,'blue_pawn.png')
    game = QuoridorGame( p1, p2, env)
    
    while True:
        game.play_step()





