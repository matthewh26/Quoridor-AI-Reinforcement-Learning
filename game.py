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
            self.start_pos = (0,4)
        else:
            self.start_pos = (8,4)
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
        self.clock = pygame.time.Clock()
        self.display.fill((40,40,40))
        
        
        #init board and players
        self.board = board
        self.p1 = p1
        self.p2 = p2
        self.players = [self.p1, self.p2]

        self.reset()

       
    def get_player_positions(self):
        pass

    def reset(self):
        #init game state
        self.board.reset_board()
        self.p1.reset_player()
        self.p2.reset_player()
        self.turn_player = 0

    def place_wall(self, wall_direction, locations):
    #put down a wall on the board
        if wall_direction == 'v':
            self.board.walls_vert[locations[0]] = 1
            self.board.walls_vert[locations[1]] = 1
        elif wall_direction == 'h':
            self.board.walls_hor[locations[0]] = 1
            self.board.walls_hor[locations[1]] = 1
        self.board.vertices[locations[0]] = 1
        self.players[self.turn_player].walls -= 1

    def move_piece(self,direction):
        #move turn player's piece
        self.players[self.turn_player].position += direction

    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            self.render()

    def render(self):
        for i in np.linspace(80,680,9): #board steps of 75 pixels
            for j in np.linspace(80,680,9):
                square = pygame.Rect(i,j,40,40)
                pygame.draw.rect(self.display,(255,255,255),square)
        self.display.blit(self.p1.image, (384,84))
        self.display.blit(self.p2.image, (384,684))
        pygame.display.update()

    def legal_moves(self):
        pass



if __name__ == '__main__':

    #create board and players
    env = board()
    p1 = player(1,'white_pawn.png')
    p2 = player(2,'blue_pawn.png')
    game = QuoridorGame( p1, p2, env)
    
    while True:
        game.play_step()





