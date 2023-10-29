import numpy as np
import pygame

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

        self.vert_wall_positions = []
        self.hor_wall_positions = []
        self.vertex_positions = []


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
        self.set_x_y()

    def reset_player(self):
        self.walls = 10
        self.position = self.start_pos
        self.reward = 0

    def set_x_y(self):
        self.x = (self.position[1]*75) + 84
        self.y = (self.position[0]*75) + 84


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
        self.board.squares[self.p1.position] = 1
        self.board.squares[self.p2.position] = 2
        self.turn = 0
        self.turn_player = self.players[self.turn]

    def place_wall(self, wall_direction, locations):
        #put down a wall on the board
        if wall_direction == 'v':
            self.board.walls_vert[locations[0]] = 1
            self.board.walls_vert[locations[1]] = 1
            self.board.vert_wall_positions.append(locations[0])
            self.board.vert_wall_positions.append(locations[1])
        elif wall_direction == 'h':
            self.board.walls_hor[locations[0]] = 1
            self.board.walls_hor[locations[1]] = 1
            self.board.hor_wall_positions.append(locations[0])
            self.board.hor_wall_positions.append(locations[1])
        self.board.vertices[locations[0]] = 1
        self.board.vertex_positions.append(locations[0])
        self.turn_player.walls -= 1
    
    def remove_wall(self, wall_direction, locations):
        #pick a wall back up from the board
        if wall_direction == 'v':
            self.board.walls_vert[locations[0]] = 0
            self.board.walls_vert[locations[1]] = 0
            self.board.vert_wall_positions.remove(locations[0])
            self.board.vert_wall_positions.remove(locations[1])
        elif wall_direction == 'h':
            self.board.walls_hor[locations[0]] = 0
            self.board.walls_hor[locations[1]] = 0
            self.board.hor_wall_positions.remove(locations[0])
            self.board.hor_wall_positions.remove(locations[1])
        self.board.vertices[locations[0]] = 0
        self.board.vertex_positions.remove(locations[0])
        self.turn_player.walls += 1


    def move_piece(self,direction):
        #move turn player's piece
        self.board.squares[self.turn_player.position] = 0
        self.turn_player.position += direction 
        self.turn_player.set_x_y()
        self.board.squares[self.turn_player.position] = self.turn_player.player


    def play_step(self,movetype,direction=None,locations=None):
        #check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        #make move
        if 'wall' in movetype:
            wall_direction = movetype[0]
            self.place_wall(wall_direction,locations)
        else:
            self.move_piece(direction)

        #update game board
        self.render()

        #check if game over
        game_over = False
        if self.p1.position[0] == 8 or self.p2.position[0] == 0:
            game_over = True
            winner = self.turn_player.player
            self.turn_player.reward = 10
            self.players[(self.turn + 1) % 2].reward = -10
            return game_over, self.p1.reward, self.p2.reward

        #set it to be the other players turn
        self.turn += 1
        self.turn_player = self.players[self.turn % 2]

        return self.p1.reward, self.p2.reward

    def render(self):
        #load current game state into pygame display
        for i in np.linspace(80,680,9): #board steps of 75 pixels
            for j in np.linspace(80,680,9):
                square = pygame.Rect(i,j,40,40)
                pygame.draw.rect(self.display,(255,255,255),square)
        self.display.blit(self.p1.image, (self.p1.x,self.p1.y))
        self.display.blit(self.p2.image, (self.p2.x,self.p2.y))
        #draw the walls
        for wall in self.board.hor_wall_positions:
            wall_x = (wall[1]*75) + 70
            wall_y = (wall[0]*75) + 130
            wall_piece = pygame.Rect(wall_x,wall_y,60,15)
            pygame.draw.rect(self.display,(181, 101, 29), wall_piece)
        for wall in self.board.vert_wall_positions:
            wall_x = (wall[1]*75) + 130
            wall_y = (wall[1]*75) + 70
            wall_piece =  pygame.Rect(wall_x,wall_y,12.5,60)
            pygame.draw.rect(self.display,(181, 101, 29), wall_piece)
        #draw vertex blockers
        for vertex in self.board.vertex_positions:
            vertex_x = (vertex[1]*75) + 130
            vertex_y = (vertex[0]*75) + 130
            vertex_piece =  pygame.Rect(vertex_x,vertex_y,15,15)
            vertex_piece = pygame.draw.rect(self.display,(181,101,29),vertex_piece)
        
        pygame.display.update()
 


    
def is_legal(self,movetype,direction=None,locations=None):
    #checks if the chosen move is a legal move
    if locations is not None:
        i = locations[0]
        j = locations[1]
    if 'wall' in movetype:
        wall_direction = movetype[0]
        if self.turn_player.walls == 0:
            return False
    pos = self.turn_player.position
    
    #wall moves
    if movetype == 'vertical wall':
        #checks if the wall spaces are free
        free_check = self.board.walls_vert[i,j] == 0 and self.board.walls_vert[i+1,j] == 0
        #checks if the wall would cross a horizontal wall
        cross_check = self.board.vertices[i,j] == 0
        if free_check and cross_check:
            pass
        else:
            return False
        #checks if the wall space is protected due to the blocking constraint
        not_protected_check = not self.is_protected('v',((i,j),(i+1,j)))     
        if not_protected_check:
            return True
        else:
            return False
    elif movetype == 'horizontal wall':
        #checks if the wall spaces are free
        free_check = self.board.walls_hor[i,j] == 0 and self.board.walls_hor[i,j+1] == 0
        #checks if the wall would cross a horizontal wall
        cross_check = self.board.vertices[i,j] == 0
        if free_check and cross_check:
            pass
        else:
            return False
        #checks if the wall space is protected due to the blocking constraint
        not_protected_check = not self.is_protected('h',((i,j),(i,j+1)))
        if not_protected_check:
            return True
        else:
            return False
    #normal piece move - check not either at edge or blocked by a wall or player piece
    elif movetype == 'regular move':
        if direction == (1,0): #downwards move
            if pos[0] != 8 and self.board.walls_hor[pos] == 0 and self.board.squares[pos[0]+1,pos[1]] == 0:
                return True
            else:
                return False
        elif direction == (-1,0): #upwards move
            if pos[0] != 0 and self.board.walls_hor[(pos[0],pos[1]-1)] == 0 and self.board.squares[pos[0]-1,pos[1]] == 0:
                return True
            else:
                return False
        elif direction == (0,1): #right move
            if self.turn_player.position[1] != 8 and self.board.walls_vert[pos] == 0 and self.board.squares[pos[0], pos[1]+1] == 0:
                return True
            else: 
                return False
        else: #left move
            if self.turn_player.position[1] != 0 and self.board.walls_vert[pos[0],pos[1]-1] == 0 and self.board.squares[pos[0], pos[1]-1] == 0:
                return True
            else:
                return False
    #double piece move - check at least two spaces from edge and is blocked by player piece and not blocked by walls 1 or 2 spaces away
    elif movetype == 'straight jump':                      
        return False
    #diagonal piece move - check not at least 2 spaces from edge, and blocked by player with wall piece behind
    #NB: FOR EACH DIAGONAL MOVE, THERE ARE TWO WAYS TO ACHIEVE - (eg. diag down left blocked by player down or to the left)
    elif movetype == 'diagonal jump':
        return False

def is_protected(self,wall_direction,locations):
    #check whether a wall location is protected by the blocking constraint
    protected = False
    self.place_wall(wall_direction,locations)
    for player in self.players:
        if find_route(player):
            pass
        else:
            protected = True
    self.remove_wall(wall_direction, locations)
    return protected


def set_grid(player):
    #sets up a grid to use for the search algorithm
    grid = np.zeros((9,9))
    if player.player == 1:
        grid[8,:] = 2
    else:
        grid[0,:] = 2
    return grid
    
def find_route(player):
    #maze search to find the quickest route for the player to the finish
    #will return None if there is no possible route to the finish
    grid = set_grid(player)
    route = search(player.position[0],player.position[1],grid)
    return route

def search(self,i,j,grid):
    #searches for a possible route from a given x and y to the goal side of a grid
    if grid[(i,j)] == 2: #2 = reached the other side
            return True
    elif grid[(i,j)] == 3: #3 = already visited square
        return False
    
    grid[(i,j)] = 3 #set square to visited
    
    if (((i > 0) and (self.board.hor_walls[(i-1,j)]==0) and (self.search(i-1,j))) or 
        ((j < 2) and (self.board.vert_walls[i,j]==0) and self.search(i,j+1)) or 
        ((i < 2) and (self.board.hor_walls[i,j]==0) and self.search(i+1,j)) or 
        ((j > 0) and (self.board.vert_walls[(i,j-1)]==0) and self.search(i,j+1))):
        return True
    
    return False




'''
if __name__ == '__main__':

    env = board()
    p1 = player(1, 'blue_pawn.png')
    p2 = player(2, 'white_pawn.png')
    game = QuoridorGame(p1, p2, env)
    game.board.hor_wall_positions.append((0,3))
    game.board.hor_wall_positions.append((0,4))
    game.board.vertex_positions.append((0,3))

    while True:
        game.play_step('wall')
'''




