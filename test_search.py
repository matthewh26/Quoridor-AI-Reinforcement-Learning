import numpy as np

#create test state

class board():
    def __init__(self):

        self.squares = np.array([[2, 2, 2],
                                 [0, 0, 0],
                                 [0, 0, 0]])

        self.vert_walls = np.array([[0, 0],
                                    [1, 0],
                                    [0, 1]])

        self.hor_walls = np.array([[1, 0, 0],
                                   [1, 1, 1]])

    def search(self, i, j):

        if self.squares[(i,j)] == 2:
            return True
        elif self.squares[(i,j)] == 3:
            return False
        
        self.squares[(i,j)] = 3
        
        if (((i > 0) and (self.hor_walls[(i-1,j)]==0) and (self.search(i-1,j))) or 
            ((j < 2) and (self.vert_walls[i,j]==0) and self.search(i,j+1)) or 
            ((i < 2) and (self.hor_walls[i,j]==0) and self.search(i+1,j)) or 
            ((j > 0) and (self.vert_walls[(i,j-1)]==0) and self.search(i,j+1))):
            return True
        
        return False
    

my_board = board()
my_board.search(i=2,j=0)

        
        


