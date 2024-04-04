import pygame
import math 
from queue import PriorityQueue

WIDTH = 800

# pygame.display allows us to control / modify the display window and screen
# set_mode initializes our display using specific arguments | In this case, since WIDTH is equal to 800, our display is an 800x800 square
WIN = pygame.display.set_mode((WIDTH, WIDTH))

#This sets a caption for our display
pygame.display.set_caption('A-STAR')

#Assigns RGB values of specific colors to corresponding color variables

RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)
WHITE = (255,255,255)
BLACK = (0,0,0)
PURPLE = (127,0,255)
ORANGE = (255,128,0)
GREY = (160,160,160)
TURQUOISE = (0,153,153)

# each node will need to hold differing values
# where it is, width, neighbors of the node, color of the node G, H, and F values, much more.
# so we use the class below to keep track.

class Node: 
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width # Keeps track of the starting x position of a specific node
        self.y = col * width # Keeps track of the starting y position of a specific node
        self.color = WHITE # Sets our initial grid to all white
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows
    
    # Returns the positional values of a node.
    def get_pos(self):
        return self.row, self.col
    
    #Checks if a node is in the closed set.
    def is_closed(self):
        return self.color == RED #RED is equal to a considered path which was defined a closed node.
    
    #Checks if a node is in the open set.
    def is_open(self):
        return self.color == GREEN #GREEN is equal to a considered path which was defined as an open node.
    
    #Checks if a node is defined as a barrier.
    def is_barrier(self):
        return self.color == BLACK #BLACK is equal to a barrier.
    
    #Checks if a node is the starting node
    def is_start(self):
        return self.color == ORANGE #ORANGE is equal to the starting node.
    
    #Checks if a node is the ending node
    def is_end(self):
        return self.color == TURQUOISE #PURPLE is equal to the ending node.
    
    #Allows for a node to be reset back to white.
    def reset(self):
        self.color = WHITE #WHITE is equal to a default node.

    def make_start(self):
        self.color = ORANGE

    #Turns a node to RED (closed set)
    def make_closed(self):
            self.color = RED

    #Turns a node to GREEN (open set)
    def make_open(self):
        self.color = GREEN

    #Turns a node to BLACK (barrier)
    def make_barrier(self):
        self.color = BLACK
    
    #Turns a node to TURQUOISE (end)
    def make_end(self):
        self.color = TURQUOISE
    
    #Turns a node to PURPLE (path)
    def make_path(self):
        self.color = PURPLE
    
    #Allows us to draw on our grid.
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width)) #pygame.draw.rect() allows us to draw a rectangle (or a square)
        #self.x and self.y identifies the starting point at which a node is drawn.
        #self.width and self.width represent the width and height for our node. Given that our node is a square, we can use self.width for height, as width and height will be equal.
    
    #Allows us to identify the neighbors of a specific node and add it to our neighbors list
    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): #Checks to see if we can move down a row, if there isn't a barrier.
            self.neighbors.append(grid[self.row + 1][self.col]) #Appends the next row if possible to neighbors list

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): #Checks to see if we can move up a row, if there isn't a barrier.
            self.neighbors.append(grid[self.row - 1][self.col]) #Appends the next row if possible to neighbors list
       
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): #Checks to see if we can move right, if there isn't a barrier.
            self.neighbors.append(grid[self.row][self.col + 1]) #Appends the next col if possible to neighbors list
       
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): #Checks to see if we can move left, if there isn't a barrier.
            self.neighbors.append(grid[self.row][self.col - 1]) #Appends the next col if possible to neighbors list


    # __lt__ stands for 'less than'. We compare the node 'self' to the 'other' node. If the value of 'self' is less than 'other', then it returns false.
    def __lt__(self, other):
        return False
    
#Calculates the Manhattan Distance -- the shortest distance between two points expressed in only vertical / horizontal paths
def h(p1, p2):
    x1, y1 = p1 # Position of node 1
    x2, y2 = p2 # Position of node 2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, current, draw):
    while current in came_from: 
        current = came_from[current]
        current.make_path()
        draw()

def algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue() #Allows us to identify elements with a higher priority and then dequeue them for use in a specific purpose
    open_set.put((0, count, start)) # Appends items | 0 represents the f-score, count is the tie breaker if we have the same f score based on what node was added first, start is the current node itself
    came_from = {} #Keeps track of where each node came from. To keep track of the current path to identify the optimal path.
    g_score = {node: float("inf")for row in grid for node in row} # Each node is set to have a g_score of infinity, only for the start
    g_score[start]= 0 # Starting node g score is set to 0. Due to the fact that we're at the start node and to reach it we need 0 steps
    f_score = {node: float("inf")for row in grid for node in row} # Each node is set to have a f_score of infinity, only for the start 
    f_score[start]= h(start.get_pos(), end.get_pos()) 

    open_set_hash = {start}

    while not open_set.empty(): # Allows for us to quit and stop running the algorithm. 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2] #Gets the current node we're looking at based on the Open set
        open_set_hash.remove(current) #Removes a node from the Open Set

        if current == end: #if we found the shortest path (checks current node)
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

 #If we found a better way to reach a neighbor, update the path based on the best node
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash: #If the best found neighbor isn't in the open hash,
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()
        
        draw()

        if current != start:
            current.make_closed()

    return False

#Defines the function to make a grid based on intended rows and width.
def make_grid(rows, width):
    grid = [] #creates an empty list which will hold our individual nodes
    gap = width // rows # Determines the width of each individual node 

    
    '''The following creates a 2D list which has the structure of the following:
    [[]
    []
    []
    []]
    Each list within parent list holds it's own node'''
    for i in range(rows):
         grid.append([]) # Appends a new row per value (i) of 'rows'
         for j in range(rows):
             node = Node(i, j, gap, rows) # Makes use of the Node class to define "row, col, width, total_rows" per value (j) of 'rows'
             grid[i].append(node) # For every row i, a new node is appended to the end of our grid. [Appending a new list inside the 'grid' list]
    
    return grid

# Draws how we draw the black lines of our grid
def draw_grid(win, rows, width):
    gap = width // rows #Determines width of each individual node.
    for i in range(rows): 
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap)) # Draws our horizontal line based on integer i, denoting each individual row in range rows.
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width)) # Draw our vertical lines based on integer j, denogint each individualrow in range rows.

def draw(win, grid, rows, width):
    win.fill(WHITE) #resets entire screen with 1 color white for each frame in order to allow for the redrawing of every predetermiend color.

    # Draws / Fills in all the spots with the color white
    for row in grid: 
            for node in row:
                node.draw(win)
    
    draw_grid(win, rows, width) #draws our grid lines once more by calling the previous draw_grid() function.
    pygame.display.update() 
    '''Takes what we drew and updates it onto the display prior pygame functions
        would "draw" our nodes and grid lines and set it to an off-screen buffer. 
        This function actually updates our display based on our drawings to make 
        them visible'''
    
def get_clicked_pos(pos, rows, width): # Uses the position of our pointer, the row, and the width to identify which node was clicked
    gap = width // rows
    y, x = pos # Finds the position of our pointer in values y and x

    row = y // gap # Finds the row of our pointer by taking the y value and dividing it by the width (well height) of each node
    col = x // gap # Finds the row of our pointer by taking the x value and dividing it by the width of each node

    return row, col #returns the row and column of the pointer.


# The Main Loop (func), enscaptulating how our program works
def main(win, width):
    ROWS = 50 #Defines the # of rows we want to make
    grid = make_grid(ROWS, width) # Uses the make_grid() function, taking in our # of ROWS and width to actually make the 2D array which'll hold our nodes which make up our grid

    start = None #Set to none as default. That'll be dependent on the user
    end = None #Set to none as default. That'll be dependent on the user

    run = True #Defines if the main loop is running.
    started = False #Defaults to false as the algorithm doesn't automatically start.

    while run: #When run is True, the following runs
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: #Checks if we close the game
                run = False #Stops / ends the main loop from running
            
            if started: #Checks if the algorithm started
                continue #The user won't be able to do anything besides stop the game once the algorithm is running.

            if pygame.mouse.get_pressed()[0]: #Checks if our click button (LEFT Click) was pressed.
                pos = pygame.mouse.get_pos() #Gets the (x, y) coordinate of the click
                row, col = get_clicked_pos(pos, ROWS, width) #Gets the row and column of the click, indicating which node was clicked
                node = grid[row][col] #Our row and column is indexed and assigned to a node
                if not start and node != end: #If no node is set as the start | we also can't select an end node as start
                    start = node
                    start.make_start() #Our next clicked node will be set as the start.
                
                elif not end and node != start: #If no node is set as the end | w also can't select a start node as end.
                    end = node
                    end.make_end() #Our next clicked node will be set as the end

                elif node != end and node != start: #After we select start and end
                    node.make_barrier() #The rest of the nodes that are clicked will be set as barriers (BLACK)

            elif pygame.mouse.get_pressed()[2]: #Checks if our RIGHT click was pressed
                #We get the position of our mouse click.
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos (pos, ROWS, width)
                node = grid[row][col]
                node.reset() # Resets our node to WHITE on the right clock, indicating a default node
                if node == start: # Allows us to reset our start node to WHITE on the right click
                    start = None
                elif node == end:  # Allows us to reset our end node to WHITE on the right click
                    end = None
            
            if event.type == pygame.KEYDOWN: #Identifies if we press a key on our keyboard
                if event.key == pygame.K_SPACE and start and end: #If our space key is pressed and there is a start + end update the neighbors of x (starting?) node
                    for row in grid: #in each row in our grid
                        for node in row: #in each node in our row
                            node.update_neighbors(grid) #updates our neighbors list based on the node
                    
                    algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit() #closes our window

main(WIN, WIDTH)
 