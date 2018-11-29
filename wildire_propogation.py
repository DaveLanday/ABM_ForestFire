# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:22:42 2018
0 = firebreak, 1 = fire, 2 = tree
@author: mgreen13
"""
import numpy as np
from collections import Counter
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from scipy import signal
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.ioff()

class Cell():
    
    # constructor for cell
    
    def __init__(self,x,y,z,state):
        self.x = x
        self.y = y
        self.position = x,y
        self.z = z
        self.state = state
        self.visited = False 
        
        self.dz1 = None
        self.dz2 = None
        self.dz3 = None
        self.dz4 = None
        self.nStates = 2
        self.p = []
    
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
        nbs = self.getN(landscape)
        zs = []
        for n in nbs:
            if landscape[n].state == 1:
                zs.append(self.z - landscape[n].getZ())
        avgDz= np.sum(zs)
        return(avgDz)


  
def stateMat(landscape):
    
    """ 
    Retrieve matrix of states from landscape
    """   
    mat = np.zeros([len(landscape),len(landscape)])
    for i in range(len(landscape)):
        for j in range(len(landscape)):
            mat[i,j] = landscape[i,j].state
    return(mat)
                
def zMat(landscape):
    mat = np.zeros([len(landscape),len(landscape)])
    for i in range(len(landscape)):
        for j in range(len(landscape)):
            mat[i,j] = landscape[i,j].z
    return(mat)
                
    
def  growTree(p,landscape):
    """GROW TREE AT I,J WITH PROBABILITY P""" 

    for i in range(len(landscape)):
        for j in range(len(landscape)):
            if landscape[i,j].getState() == 0:
                if np.random.rand(1) < p:
                    landscape[i,j].setState(2)
    return(landscape)
                    
def check_contained(landscape):
    for i in range(len(landscape)):
        for j in range(len(landscape)):
            if landscape[i,j].getState() == 2:
                if 1 in landscape[i,j].getNState(landscape):
                    contained = False
                else:
                    contained = True
                
    
    
def fire_prop(land,gamma, zMax,maxN,contained,threshold):
                    
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
    unvisited = []
    fired = []
    starting_z = []
    
    # START FIRE AT RANDOM SITE
    i = np.random.randint(0,len(land))
    j = np.random.randint(0,len(land))
    # SET STATE OF CELL TO FIRE
    landscape[i,j].setState(1)  
    starting_z.append(landscape[i,j].z)
    unvisited.extend(land[i,j].getN(land))
    # ADD TO LIST OF FIRED CELLS
    fired.append((i,j))
        
    # BEGIN FIRE PROPOGATION
    while not contained:
        border = []
        # CREATE FIRE BORDER BY VISTING FIRE CELLS THAT ARE NEIGHBORS WITH TREES 
        for site in fired:
            # LOOP OVER LIST OF NEIGHBORS OF FIRE CELLS
            for idxN,neighborState in enumerate(landscape[site].getNState(landscape)):
                # IF CELL HAS A NEIGHBOR THAT IS A TREE, ADD FIRE CELL TO BORDER
                if neighborState == 2:
                    border.append(landscape[site].getN(landscape)[idxN])
                # IN THIS CASE, THE FIRE IS SURROUNDED BY FIRE BREAKS 
                else:
                    contained = True
                    # TURN OLD FIRES INTO ASH/FIREBREAKS
                if landscape[site].fireT == threshold:
                    landscape[site].setState(0)
                # KEEP TRACK OF TIME THAT FIRE HAS BEEN BURNING AT A SITE
                landscape[site].fireT += 1
        for site in border:
            if pFire(site) > np.random.rand:
                site.setState(1)
            
                
                    
                    
                
                    
                    
        for indn,n in enumerate(unvisited):
            nStates = land[n].getNState(land)
            dzSum = land[n].getDzSum(land)
            nF = Counter(nStates)
            nF = nF[1]
            nS = len(nStates)
            for nNeigh in land[n].getN(land):
                if land[nNeigh].getState() == 2 and nNeigh != (i,j):
                    unvisited.append(nNeigh)
            pFire = gamma + (1-gamma)*(dzSum*nF)/(nS*zMax)
            
            land[0,0].p.append(pFire)
            if np.random.rand(1)<pFire:
                land[n].setState(1)
                unvisited.pop(indn)
                fired.append(n)
                mapS = stateMat(land)
                stateMaps.append(mapS)
                
    for fire in fired:
        land[fire].setState(0)
    stateMaps.append(stateMat(land))
    
    return(land,stateMaps,fired,starting_z)
    
bowl = np.load("150x150_bowl_z_10.npy")
hill = np.load("150x150_slant_z_10.npy")
hillsmall = np.load("50x50_slant_zmax_25.npy")
bowlSmall = np.load("50x50_bowl_zmax_10.npy")

# initialize contained
contained = False






#zVals= np.random.randint(1,10,[N,N])
zVals = hillsmall
N = len(zVals)
landscape = np.ndarray([N,N],dtype = Cell)
for i,ik in enumerate(zVals):
    for j,jk in enumerate(ik):
        z = zVals[i,j]
        a = Cell(i,j,z,0)
        landscape[i,j] = a
        
# SET HEIGHTS OF CELLS
for i in list(range(len(landscape))):
            for j in list(range(len(landscape))):
                landscape[i][j].setDz(landscape)

statemaps = []
firedMaps = []
startZList = []
# Start with Dense Forest
landscape = growTree(1,landscape)

for i in range(100):
    landscape,statemap,fired,startZ= lightStrike(landscape,.7,10,4)
    statemaps.append(statemap)
    firedMaps.append(fired)
    startZList.append(startZ)
    
    
    
    
    
