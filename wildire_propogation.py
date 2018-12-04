# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:22:42 2018
0 = firebreak, 1 = fire, 2 = tree
@author: _mgreen13_, _DaveLanday_
"""
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import matplotlib.colors as colors
import sys

class Cell():

    # constructor for cell

    def __init__(self,x,y,z,state):
        self.x = x
        self.y = y
        self.position = x,y
        self.z = z
        self.state = state
        self.visited = False
        self.risk = 0
        self.fireT = 0
        self.partition = -1
        self.dz1 = None
        self.dz2 = None
        self.dz3 = None
        self.dz4 = None
        self.nStates = 2
        self.p = []
        
    def getNTFire(self,landscape):
        """Get fire times of all neighbors
        """
        neighbor_fire_times = []
        i,j = self.getPosition()
        for n in self.getN(landscape):
            neighbor_fire_times.append(landscape[n].fireT)
        return(neighbor_fire_times)
    

    def getNState(self,landscape):
        i = self.x
        j = self.y

        try:
            n1 = landscape[i-1,j].getState()
        except:
            IndexError
        try:
            n2 = landscape[i,j+1].getState()
        except:
            IndexError
        try:
            n3 = landscape[i+1,j].getState()
        except:
            IndexError
        try:
            n4 = landscape[i,j-1].getState()
        except:
            IndexError

        # Build case for each border area, upper/lower left corner, upper/lower right corner
        # All Four borders
        # Upper Left Corner (No n1 or n4)
        if i == 0 and j == 0:
            return(n2,n3)
        # Upper right corner(no n1 or n2)
        elif i==0 and j==len(landscape)-1:
            return(n3,n4)
        # Lower left corner(no n3,n4)
        elif i == len(landscape)-1 and j == 0:
           return(n1,n2)
        # Lower right corner(no n2 or n3)
        elif i == (len(landscape)-1) and j == (len(landscape)-1):
            return(n1,n4)
        # On top of matrix
        elif i ==0:
            return(n2,n3,n4)
        # Bottom of matrix
        elif i == len(landscape)-1:
            return(n1,n2,n4)
        # Right side of matrix
        elif j == len(landscape)-1:
            return(n1,n3,n4)
        # Left Side of matrix
        elif j == 0:
            return(n1,n2,n3)
        else:
            return(n1,n2,n3,n4)

    def getN(self,landscape):
        i,j = self.getPosition()
        # TRY EXCEPT BLOCK TO ATTEMPT TO ASSIGN NEIGHBOR LOCATIONS
        try:
            n1 = landscape[i-1,j].getPosition()
        except:
            IndexError
        try:
            n2 = landscape[i,j+1].getPosition()
        except:
            IndexError
        try:
            n3 = landscape[i+1,j].getPosition()
        except:
            IndexError
        try:
            n4 = landscape[i,j-1].getPosition()
        except:
            IndexError
            # Build case for each border area, upper/lower left corner, upper/lower right corner
        # All Four borders

        # Upper Left Corner (No n1 or n4)
        if i == 0 and j == 0:
            return(n2,n3)
        # Upper right corner(no n1 or n2)
        elif i==0 and j==len(landscape)-1:
            return(n3,n4)
        # Lower left corner(no n3,n4)
        elif i == len(landscape)-1 and j == 0:
           return(n1,n2)
        # Lower right corner(no n2 or n3)
        elif i == (len(landscape)-1) and j == (len(landscape)-1):
            return(n1,n4)
        # On top of matrix
        elif i ==0:
            return(n2,n3,n4)
        # Bottom of matrix
        elif i == len(landscape)-1:
            return(n1,n2,n4)
        # Right side of matrix
        elif j == len(landscape)-1:
            return(n1,n3,n4)
        # Left Side of matrix
        elif j == 0:
            return(n1,n2,n3)
        else:
            return(n1,n2,n3,n4)

    # getter for state of cell
    def getState(self):
        return self.state

    #setter for state of cell
    def setState(self,state):
        self.state = state

    # Get position of cell in matrix
    def getPosition(self):
        return(self.x,self.y)

    # Get height of cell
    def getZ(self):
        return(self.z)


    # Set dz values between site and neighbouring nodes
    def setDz(self,landscape):
        #INITIALIZD DELZ AS NONE
        self.dz2 = None
        self.dz4 = None
        self.dz1 = None
        self.dz3 = None

        # Exception for higher borders of grid
        try:
            self.dz1 = landscape[i,j].getZ() - landscape[i+1,j].getZ()
            self.dz3 = landscape[i,j].getZ() - landscape[i,j+1].getZ()
        except:
            IndexError
        # Exception for lower borders of grid
        if i!= 0:
            self.dz2 = landscape[i,j].getZ() - landscape[i-1,j].getZ()
        if j!= 0:
            self.dz4 = landscape[i,j].getZ() - landscape[i,j-1].getZ()

    def getDz(self):
        return(self.dz1,self.dz2,self.dz3,self.dz4)

    def getDzSum(self,landscape):
        """
        Get sum of height differences from neighbor of current fire cell
        """
        nbs = self.getN(landscape)
        zs = []
        for n in nbs:
            if landscape[n].state == 1:
                zs.append(landscape[n].getZ()-self.z)
        sumDz= np.sum(zs)
        return(sumDz)
        
        
class Agent():
    
    def __init__ (self):
        self.position = None
        self.partition = None
        self.partition_mates = None
        
    def set_partition_mates(self,landscape):
        """
        Get list of all cells in the same partition as agent
        """
        partition_mates = []

        for i in range(len(landscape)):
            for j in range(len(landscape)):
                if self.partition == landscape[i,j].partition:
                    partition_mates.append((i,j))
        self.partition_mates = partition_mates
                    
        
    
    
def place_agent(landscape,agents):
    """
    Given the landscape and the number of agents,
    place the agent in a partition of the landscape
    """
    
    for num,agent in enumerate(agents):
        # Assign partition to agent
        agent.partition = num+1
        partition_fire_sites = []
        for i in range(len(landscape)):
            for j in range(len(landscape)):
                # IF SITE IS ON FIRE AND IN PARTITION, APPEND TO PART_FIRE_SITES
                if landscape[i,j].partition == num+1 and landscape[i,j].state == 1:
                    print(1)
                    partition_fire_sites.append((i,j))
        agent_position = max_risk_pos(landscape,partition_fire_sites)
        # ReASSIGN POSITION
        agent.position = agent_position
        # BUILD FIRE BREAK AT AGENTS POSITION
        landscape[agent.position].state = 3
        
def update_agent(landscape,agents):
    """
    Update the position of the agent based to block off the neighbors of the cells with the highest risk of catching fire
    """
    for agent in agents:
        agent.set_partition_mates(landscape)
        agent_neighbor_set = set(landscape[agent.position].getN(landscape))
        cells_in_partition = set(agent.partition_mates)
        
        neighbors_in_partition = agent_neighbor_set.intersection(cells_in_partition)
        
        n_fire = []
        for n in neighbors_in_partition:
            if landscape[n].state == 2:
                n_fire.append(n)
                
#        for cell in agent.partition_mates:
#            if landscape[cell].state == 1:
#                n_fire.extend(list(set(landscape[cell].getN(landscape)).intersection(set(agent.partition_mates))))
#        fires_part_neighors = list(neighbors_in_partition.intersection(set(n_fire)))
#        
#        if len(fires_part_neighors) == 0:
#            pass
#        else:
        try:
            agent.position = max_risk_pos(landscape,n_fire)
        except:
            pass

        landscape[agent.position].state = 3
        landscape[agent.position].risk = 0
            
          
def partition_grid(landscape, num_agents):
    """
    PARTITION LANDSCAPE INTO EITHER 2 OR 4 PARTITIONS PENDING OF THE NUMBER OF AGENTS
    USED TO CONTROL THE FIRE.
    """
    
    # DRAW BOX AROUND CELLS THAT ARE ON FIRE
    fire_i = []
    fire_j = []
    for i in landscape:
        for j in i:
            if j.state ==1:
                fire_i.append(j.position[0])
                fire_j.append(j.position[1])
    max_i = max(fire_i)
    max_j = max(fire_j)
    min_i = min(fire_i)
    min_j = min(fire_j)

    centroid_i = int((max_i-min_i)/2 + min_i)
    centroid_j = int((max_j-min_j)/2 + min_j)

    if num_agents <4:
        for i in landscape:
            for j in i:
                if j.position[0] < centroid_i:
                    j.partition = 1
                else:
                    j.partition = 2
    # FOR FIRES WITH MORE THAN 4 AGENTS
    else:
        for i in landscape:
            for j in i:
                # upper left corner
                if j.position[0] < centroid_i and j.position[1] < centroid_j:
                    j.partition =1
                # upper right corner
                elif j.position[0] <centroid_i and j.position[1] >= centroid_j:
                    j.partition = 2
                # bottom left
                elif j.position[0] >= centroid_i and j.position[1] < centroid_j:
                    j.partition = 3
                # bottom right
                elif j.position[0] >= centroid_i and j.position[1] >= centroid_j:
                    j.partition = 4
    return(centroid_i,centroid_j)
        
def getStates(landscape):
    """ RETURN ARRAY WITH THE STATE OF EVERY SITE IN LANDSCAPE"""
    state_map = np.zeros([len(landscape), len(landscape)])
    for i in landscape:
        for j in i:
            state_map[j.position] = j.state
    return(state_map)


def check_contained(landscape,threshold):
    """
    FIRE CONTAINMENT TEST: visit all fire sights, if they have 
    neighboring  trees, the fire is not contained.
    
    Fire cells that will expire 
    INPUT: landscape at time t
    OUTPUT: boolean
    """
    contained = False
    nieghborState = []
    for i in range(len(landscape)):
        for j in range(len(landscape)):
            if landscape[i,j].getState() == 1:
                # check to see if fire cell has a neighbor that is a tree
                nieghborState.extend(landscape[i,j].getNState(landscape))
    if 2 not in nieghborState:
        contained = True
    return(contained)

    
def max_risk_pos(landscape, potential_fire_sites):
    """
        MAX_RISK_POS: calculates the riskiest site for the agent to move to
    """
    #store a list of risks:
    risks = []

    #get the risk values for the potential fire sites:
    for site in potential_fire_sites:
        risks.append(landscape[site].risk)

    #get the coordinate for the most risky site:
    riskiest = potential_fire_sites[np.argmax(risks)]

    #return the riskiest site:
    return(riskiest)
    
def get_fire_sites(landscape):
    fire_sites = []
    for i in range(len(landscape)):
        for j in range(len(landscape)):
            if landscape[i,j].getState() == 1:
                fire_sites.append((i,j))
    return(fire_sites)
                


def update_p_fire(landscape,gamma,zMax):
    """
    UPDATE RISK OF EVERY CELL IN THE LANDSCAPE 
    """
    for i in landscape:
        for j in i:
            # ONLY UPDATE IF CELL IS A TREE
            if j.state == 2:
                # GET STATES OF BORDERS SITES NEIGHBORS
                nStates = j.getNState(landscape)
                # GET SUM OF DELTA Z 
                dzSum = j.getDzSum(landscape)
                nF = Counter(nStates)
                nF = nF[1]
                nS = len(nStates)
                # ASSIGN RISK
                if dzSum == 0:
                    dzSum =1
                # TODO FIX THIS!!!!! PROBABILITY FUNCTION
                j.risk = gamma + (1-gamma)*(dzSum)/(4*(nS*zMax))
            # IF CELL IS ALREADY ON FIRE, RISK IS ZERO
            else:
                j.risk = 0
                
def fire_init(landscape,gamma, zMax,maxN,contained,threshold,init_time):
    """
    INITIALIZE FIRE BY RUNNING T TIMESTEPS OF THE FIRE PROPOGATION
     """
    stateMaps = []
    fired = []
    # START FIRE AT RANDOM SITE
    i = int(len(landscape)/2)
    j = int(len(landscape)/2)
    # SET STATE OF CELL TO FIRE
    landscape[i,j].setState(1)
    # ADD TO LIST OF FIRED CELLS
    fired.append((i,j))
    t = 0
    # BEGIN FIRE PROPOGATION
    while t < init_time:
        border = []
        # CREATE FIRE BORDER BY VISTING FIRE CELLS THAT ARE NEIGHBORS WITH TREES
        for site in fired:
            # LOOP OVER LIST OF NEIGHBORS OF FIRE CELLS
            for idxN,neighbor in enumerate(landscape[site].getN(landscape)):
                # IF CELL HAS A NEIGHBOR THAT IS A TREE, ADD TREE CELL TO BORDER
                if landscape[neighbor].state == 2:
                    border.append(neighbor)
                    # TURN OLD FIRES INTO ASH/FIREBREAKS
            if landscape[site].fireT == threshold:
                landscape[site].setState(0)
                # KEEP TRACK OF TIME THAT FIRE HAS BEEN BURNING AT A SITE
            landscape[site].fireT += 1
        # CONSIDER ALL BORDER SITES FOR POTENTIAL FIRE SPREADING
        for site in border:
            # DETERMINE PROBABILITY OF FIRE SPREAD
            probFire = landscape[site].risk
            # SET FIRE DEPENDING ON LIKELYHOOD
            if probFire > np.random.rand():
                landscape[site].setState(1)
                fired.append(site)
        t = t+1
        # UPDATE RISK VALUES FOR ALL CELLS IN LANDSCAPE
        update_p_fire(landscape,gamma,zMax)
        stateMaps.append(getStates(landscape))
    return(stateMaps)              
    
def fire_prop(landscape,gamma, zMax,maxN,contained,threshold,num_agents,statemaps):
    """
    SEMI SYNCHRONOUS UPDATE.
  INPUTS
       1) Landscape matrix: matrix of forest cell objects
       2) Gamma: probability space partition
       3) zMax: Maximum height of cell in forest
       4) contained: global boolan variable indicating the state of the fire
       5) threshold:
       4) maxN: maximum number of neighrbors a cell is allowed to have

   OUTPUTS
       1) Updated landscape matrix
       """
    stateMaps = []
    fired = []
    # ADD TO LIST OF FIRED CELLS
    fire_sites = get_fire_sites(landscape)
    fired.extend(fire_sites)
    t = 0
    # BEGIN FIRE PROPOGATION
    while not contained:
        border = []
        # CREATE FIRE BORDER BY VISTING FIRE CELLS THAT ARE NEIGHBORS WITH TREES
        for site in fired:
            # LOOP OVER LIST OF NEIGHBORS OF FIRE CELLS
            for idxN,neighbor in enumerate(landscape[site].getN(landscape)):
                # IF CELL HAS A NEIGHBOR THAT IS A TREE, ADD TREE CELL TO BORDER
                if landscape[neighbor].state == 2:
                    border.append(neighbor)
                    # TURN OLD FIRES INTO ASH/FIREBREAKS
            if landscape[site].fireT == threshold:
                landscape[site].setState(0)
                # KEEP TRACK OF TIME THAT FIRE HAS BEEN BURNING AT A SITE
            landscape[site].fireT += 1
        # CONSIDER ALL BORDER SITES FOR POTENTIAL FIRE SPREADING
        for site in border:
            # DETERMINE PROBABILITY OF FIRE SPREAD
            probFire = landscape[site].risk
            # SET FIRE DEPENDING ON LIKELYHOOD
            if probFire > np.random.rand():
                landscape[site].setState(1)
                fired.append(site)
            # PLACE AGENTS ONLY ONCE
            if t == 0:
                agents = []
                for g in range(num_agents):
                    A = Agent()
                    agents.append(A)
                place_agent(landscape,agents)
            if t != 0:
                update_agent(landscape,agents)
                update_p_fire(landscape,gamma,zMax)
                update_agent(landscape,agents)
                update_agent(landscape,agents)


            t = t+1
        # UPDATE RISK VALUES FOR ALL CELLS IN LANDSCAPE
        update_p_fire(landscape,gamma,zMax)
        stateMaps.append(getStates(landscape))
        contained = check_contained(landscape,threshold)
    return(stateMaps)

bowlSmall = np.load("50x50_bowl_zmax_10.npy")
# initialize contained
contained = False

#zVals= np.random.randint(1,10,[N,N])
zVals = bowlSmall
N = len(zVals)
landscape = np.ndarray([N,N],dtype = Cell)
for i,ik in enumerate(zVals):
    for j,jk in enumerate(ik):
        z = zVals[i,j]
        a = Cell(i,j,z,2)
        landscape[i,j] = a

# SET HEIGHTS OF CELLS
for i in list(range(len(landscape))):
            for j in list(range(len(landscape))):
                landscape[i][j].setDz(landscape)
#TEST FIRE_PROP
#initialize fire cluster
stateMaps = fire_init(landscape,.5,10,4,False,5,4)
# IF CELL HAS A NEIGHBOR THAT IS A TREE, ADD TREE CELL TO BORDER
partition_grid(landscape,4)
#propogate fire
state_maps = fire_prop(landscape,.5,10,4,False,10,4,stateMaps)







part_map = np.zeros([40,40])
for i in range(len(landscape)):
    for j in range(len(landscape)):
        part_map[i,j] = landscape[i,j].partition

fig, ax = plt.subplots(figsize=(15, 10));
cmap = colors.ListedColormap(['white', 'red', 'green','blue'])
img = ax.imshow(state_maps[65], interpolation = 'nearest', cmap = cmap)
plt.contour(zVals, colors = "b")
plt.show()


fig, ax = plt.subplots(figsize=(15, 10));
cmap = ListedColormap(['r', 'green'])
cax = ax.matshow(stateMaps[-1],cmap=cmap)

plt.contour(zVals, colors = "b")
plt.show()


#for i,frame in enumerate(state_maps):
#    fig, ax = plt.subplots(figsize=(15, 10))
# 
#    cmap = ListedColormap(['w', 'r', 'green','b'])
#    cax = ax.matshow(frame,cmap=cmap)
#    plt.contour(zVals, colors = "b")
#    figname = "{}.png".format(i)
#    plt.savefig(figname)
#    plt.close(fig)
