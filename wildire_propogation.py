# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:06:26 2018

@author: mgreen13
"""

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
import argparse
import sys
import os

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
        self.maxDz = 0
        self.spatial_risk = None

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
        for i in range(len(landscape)):
            for j in range(len(landscape)):
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
        zs.extend([1])
        self.dzMax = np.max(zs)
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




def place_agent(landscape,agents,gamma,zMax):
    """
    Given the landscape and the number of agents,
    place the agent in a partition of the landscape. The agent must be the riskeist nieghbor of the riskist cell
    """
    for num,agent in enumerate(agents):
        # Assign partition to agent
        agent.partition = num+1
        agent.set_partition_mates(landscape)
        partition_fire_sites = []
        for i in range(len(landscape)):
            for j in range(len(landscape)):
                # IF SITE IS ON FIRE AND IN PARTITION, APPEND TO PART_FIRE_SITES
                if landscape[i,j].partition == num+1 and landscape[i,j].state == 1:
                    partition_fire_sites.append((i,j))

        # GET LIST OF NEIGHBORS OF THE FIRE CELLS
        nieghbor_position = []
        for site in partition_fire_sites:
            nieghbor_position.extend(landscape[site].getN(landscape))

        # GET LIST OF CELLS IN PARTITION
        cells_in_partition = agent.partition_mates
        nieghbor_partition_pos = list(set(nieghbor_position).intersection(cells_in_partition))

        # Get site with maximum risk out of the potential sites
        risk_nieghbor = max_risk_pos(landscape,nieghbor_partition_pos,True)
        landscape[risk_nieghbor].state = 1
        update_p_fire(landscape,gamma,zMax)


        # ReASSIGN POSITION
        agent_positions = list(set(landscape[risk_nieghbor].getN(landscape)).intersection(cells_in_partition))
        safe_agent_positions = []
        for site in agent_positions:
            for n_site in landscape[site].getN(landscape):
                if 1 not in landscape[n_site].getNState(landscape):
                    safe_agent_positions.append(site)

        # REMOVE ANY POSITIONS THAT ARE NEIGHBORS OF FIRE
        position = max_risk_pos(landscape,safe_agent_positions,True)
        agent.position = position
        landscape[position].state = 3
        # peel back predicted fire and associated risks
        landscape[risk_nieghbor].state = 2
        update_p_fire(landscape,gamma,zMax)


def update_agent(landscape,agents):
    """
    Update the position of the agent based to block off the neighbors of the cells with the highest risk of catching fire.
    Agents must move to sites with high risk that are not on fire
    """
    for agent in agents:
        agent.set_partition_mates(landscape)
        agent_neighbor_set = list(landscape[agent.position].getN(landscape))
        #cells_in_partition = set(agent.partition_mates)
        #neighbors_in_partition = agent_neighbor_set.intersection(cells_in_partition)
        possible = []
        for n in agent_neighbor_set:
            if landscape[n].state == 2:
                possible.append(n)


        not_fire_neighbors = []
        for site in possible:
            if 1 not in landscape[site].getNState(landscape):
                not_fire_neighbors.append(site)

        next_not_fire_neighbors = []
        for site in not_fire_neighbors:
            if 1 not in landscape[site].getNState(landscape):
                next_not_fire_neighbors.append(site)

#        for cell in agent.partition_mates:
#            if landscape[cell].state == 1:
#                n_fire.extend(list(set(landscape[cell].getN(landscape)).intersection(set(agent.partition_mates))))
#        fires_part_neighors = list(neighbors_in_partition.intersection(set(n_fire)))
#
#        if len(fires_part_neighors) == 0:
#            pass
#        else:

        next_pos = agent.position
        try:
            next_pos = max_risk_pos(landscape,next_not_fire_neighbors,True)
        except:
            pass

        agent.position = next_pos
        landscape[agent.position].state = 3
        landscape[agent.position].risk = 0

def update_dz(landscape):
    for i in landscape:
        for j in i:
            nbs = j.getN(landscape)
            dzs = []
            for n in nbs:
                dzs.append(landscape[n].getZ()-j.z)
            j.dzMax = np.max(dzs)
    return(max(dzs))


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

def max_risk_pos(landscape, potential_fire_sites,place):
    """
        MAX_RISK_POS: calculates the riskiest site for the agent to move to
    """
    #store a list of risks:
    risks = []
    spatial_risks = []
    potential_fire_sites = list(potential_fire_sites)
    #get the risk values for the potential fire sites:
    for site in potential_fire_sites:
        risks.append(landscape[site].risk)
        spatial_risks.append(landscape[site].spatial_risk)

    #get the coordinate for the most risky site:
    if place == True:
        riskiest = potential_fire_sites[np.argmax(spatial_risks)]
    else:
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

def spatial_risk_mat(landscape):
    spat_r = np.zeros([len(landscape),len(landscape)])
    for i in range(len(landscape)):
        for j in range(len(landscape)):
            spat_r[i,j] = landscape[(i,j)].spatial_risk
    return(spat_r)


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
                    dzSum = 1
                # TODO FIX THIS!!!!! PROBABILITY FUNCTION
                if j.maxDz == 0:
                    j.maxDz=1
                j.risk = gamma + ((1-gamma)*(dzSum))/(nS*2*j.maxDz)#j.maxDz)
            # IF CELL IS ALREADY ON FIRE, RISK IS ZERO
            else:
                j.risk = 0
def update_spatial_risk(landscape):
    fires = get_fire_sites(landscape)
    for i in landscape:
        for j in i:
            x_dists_from_fire = abs(np.array(fires)[:,0] - j.position[0])
            y_dists_from_fire = abs(np.array(fires)[:,1] - j.position[1])
            dist_from_fire = np.sqrt(x_dists_from_fire**2 + y_dists_from_fire**2)
            nStates = j.getNState(landscape)
            if min(dist_from_fire) == 0 or j.state == 0 or j.state == 3 and 1 not in nStates and 0 not in nStates and 2 not in nStates:
                j.spatial_risk = 0
            else:
                j.spatial_risk = max(1/dist_from_fire)

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
        update_spatial_risk(landscape)
        stateMaps.append(getStates(landscape))
    return(stateMaps)

def fire_prop(landscape,gamma, zMax,maxN,contained,threshold,num_agents,statemaps, num_blocks):
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
    risk_maps = []

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
            place_agent(landscape,agents,gamma,zMax)
        if t != 0:
            update_agent(landscape,agents)
            update_p_fire(landscape,gamma,zMax)
            update_spatial_risk(landscape)

        t = t+1
        # UPDATE RISK VALUES FOR ALL CELLS IN LANDSCAPE
        for i in range(num_blocks):
            update_p_fire(landscape,gamma,zMax)
            stateMaps.append(getStates(landscape))
            risk_maps.append(spatial_risk_mat(landscape))

        contained = check_contained(landscape,threshold)
    return(stateMaps,risk_maps)
def save_maps(maps, f_name, outDir):
    """
        SAVE_MAPS: saves the state maps as individual .npy files so that the vacc
                   can easily create analyses and plots  at each time-step

                 ARGS:
                      Maps: the statemaps array compiled by make_sim().
                            Type: ndarray

                      f_name: name of the file to save. Type: str

                      outDir: path to the ouput directory where files should b
                              saved. Type: str

              RETURN:
                    NONE, saves the maps as individual .npy files
    """

    #make the directory:
    try:
        os.mkdir(outDir)
    except:
        pass

    #save the file:
    np.save(outDir+'/'+f_name, maps)

def parser():
    """
    ##########################__WILDFIRE_PROPAGATION__##########################
    An agent based model to simulate strategies to combat the spread of wildfire.
    """

    #create the argument parser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    #add some positional arguments:

    #file to the .npy file representing topography:
    parser.add_argument('-i',
                        '--inFile',
                        help='Path to input file (.npy 2D matrix)',
                        required=True,
                        type=str)

    #choose a gamma:
    parser.add_argument('-g',
                        '--gamma',
                        help='Probability that a site catches fire, given its neighbors are on the same plane (dz = 0)',
                        type=float,
                        default=0.5)

    #choose number of agents:
    parser.add_argument('-a',
                        '--numAgents',
                        help='Number of agents to place on the map',
                        type=int,
                        default=4)

    #choose threshold:
    parser.add_argument('-t',
                        '--threshold',
                        help='amount of time a site can remain on fire before extinguishing',
                        type=int,
                        default=4)

    #number of statemaps:
    parser.add_argument('-s',
                        '--initTime',
                        help='amount of time allowed for the fire to initialize',
                        type=int,
                        default=5)

    #number of blocks placed per timestep:
    parser.add_argument('-b',
                        '--blocks',
                        help='number of blocks the agent is allowed to place',
                        required=True,
                        type=int)

    #where do you want to save output?
    parser.add_argument('-o',
                        '--outDir',
                        help='Path to output directory (where to save landscape statemap and spatial risk maps at each timestep)',
                        required=True,
                        type=str)

    parser.add_argument('-n',
                        '--simNumber',
                        help='Indicates the trial number for the simulation you are conducting, i.e: the number of runs of the simulation that you have conducted',
                        required=True,
                        type=int)

    parser.add_argument('-y',
                        '--Yield',
                        help='txt file to return yield',
                        required=True,
                        type=str)

    parser.add_argument('-r',
                        '--save',
                        help='Indicates whether to save each statemap/riskmap from the simulation',
                        nargs='?',
                        const=0,
                        type=int,
                        default=0)


    return parser.parse_args()

def make_sim(infile, outDir, simNumber, blocks, y_ield, **kwargs):
    """
        MAKE_SIM: Runs an agent based model, whereby agents are tasked at
                  containing the spread of a wildfire over a topographical
                  surface, while maximizing the density of the forest
                  (i.e. num_trees / total_area).

                ARGS:
                    infile: path to a matrix representing the topography of the
                    model space. Type: str

                    outDir: a path to the desired output directory. Type: str

                    simNumber: simulation currently in progress. Type: int

                    blocks: number of blocks an agent is allowed to place

                    y_ield: path to txt file where yield quantities are saved.
                            Type: str

            RETURNS:
                    NONE, saves sitemaps and spatial risk maps at each timestep
    """

    #get commandline arguments:
    gamma      = kwargs.get('gamma', 0.1)
    num_agents = kwargs.get('numAgents', 4)
    threshold  = kwargs.get('threshold', 4)
    init_time  = kwargs.get('initTime', 5)
    save       = kwargs.get('save', False)
    #load in the height map:
    height_map = np.load(infile)
    z_max = np.max(height_map)

    #the initial state of the fire is not contained:
    contained = False

    #create a landscape from the height_maps:
    L = len(height_map)
    landscape = np.ndarray([L,L], dtype=Cell)

    #give cells relevamt information, plant trees at all sites:
    for i,ik in enumerate(height_map):
        for j,jk in enumerate(ik):
            z = height_map[i,j]
            a = Cell(i,j,z,2)
            landscape[i,j] = a

    #set delta-zs of all cells
    for i in list(range(len(landscape))):
        for j in list(range(len(landscape))):
            landscape[i][j].setDz(landscape)

    # set dzMax
    update_dz(landscape)

    #initialize fire cluster: landscape,gamma, zMax,maxN,contained,threshold,init_time
    stateMaps = fire_init(landscape,
                          gamma,
                          z_max,
                          4,
                          contained,
                          threshold,
                          init_time)
    # IF CELL HAS A NEIGHBOR THAT IS A TREE, ADD TREE CELL TO BORDER
    partition_grid(landscape,4)
    #propogate fire: landscape, gamma, zMax,maxN,contained,threshold,num_agents,statemaps
    state_maps, risk_mats = fire_prop(landscape,
                                     gamma,
                                     z_max,
                                     4,
                                     contained,
                                     threshold,
                                     num_agents,
                                     stateMaps,
                                     blocks)

    file = infile.split('/')[-1]

    flatten_map = state_maps[-1].flatten()
    map_quant = Counter(flatten_map)
    y = map_quant[2] / len(flatten_map)

    #save yield to yield txt file:
    with open(file[:-4]+'_'+'gamma_'+str(gamma)+'_'+y_ield, 'a+') as f:
        f.write(str(y)+'\n')

    if save:

        #save the files: maps, f_name, outDir
        for i, maps in enumerate(state_maps):
            save_maps(maps,
                      str(i)+'_'+file[:-4]+'_state_map',
                      outDir+'/'+str(simNumber)+'_'+file[:-4])

        for i, risks in enumerate(risk_mats):
            save_maps(risks,
                      str(i)+'_'+file[:-4]+'_risk_mat',
                      outDir+'/'+str(simNumber)+'_'+file[:-4])

def main():
    """
        MAIN: takes commandline arguments and runs a simulation of the model
              based on input parameters ( see parser() ).

            ARGS:
                NONE, uses commandline input from user

        RETURNS:
                NONE, saves the statemaps and the spatial risk maps to the
                specified output dir.
    """

    #get the arguments passed into the commandline prompt:
    args = parser()

    #run the simulation:
    make_sim(infile=args.inFile,
             outDir=args.outDir,
             gamma=args.gamma,
             numAgents=args.numAgents,
             threshold=args.threshold,
             initTime=args.initTime,
             blocks=args.blocks,
             simNumber=args.simNumber,
             save=args.save,
             y_ield=args.Yield)

if __name__ == '__main__':
    main()
#
# bowlSmall = np.load("150x150_bowl_z_10.npy")
# # initialize contained
# contained = False

# #zVals= np.random.randint(1,10,[N,N])
# zVals = bowlSmall
# N = len(zVals)
# landscape = np.ndarray([N,N],dtype = Cell)
# for i,ik in enumerate(zVals):
#     for j,jk in enumerate(ik):
#         z = zVals[i,j]
#         a = Cell(i,j,z,2)
#         landscape[i,j] = a
#
# # SET HEIGHTS OF CELLS
# for i in list(range(len(landscape))):
#             for j in list(range(len(landscape))):
#                 landscape[i][j].setDz(landscape)
# #TEST FIRE_PROP
# #initialize fire cluster
# stateMaps = fire_init(landscape,.5,10,4,False,5,4)
# # IF CELL HAS A NEIGHBOR THAT IS A TREE, ADD TREE CELL TO BORDER
# partition_grid(landscape,4)
# #propogate fire
# state_maps,risk_mats = fire_prop(landscape,.1,10,4,False,10,4,stateMaps)
# #
# #risk_space_map = np.zeros([len(landscape),len(landscape)])
# #part_map = np.zeros([len(landscape),len(landscape)])
# #for i in range(len(landscape)):
# #    for j in range(len(landscape)):
# #        part_map[i,j] = landscape[i,j].state
# #        risk_space_map[i,j] = landscape[i,j].spatial_risk
#
#
# #
# #fig, ax = plt.subplots(figsize=(15, 10));
# #img = ax.imshow(risk_mats[42], interpolation = 'nearest')
# #plt.contour(zVals, colors = "b")
# #plt.show()
#
# a = os.getcwd()
# os.chdir('gif1')
# for i,frame in enumerate(state_maps):
#     fig, ax = plt.subplots(figsize=(15, 10))
#     cmap = colors.ListedColormap(['white', 'red', 'green','blue'])
#     cax = ax.matshow(frame,cmap = cmap)
#     plt.contour(zVals, colors = "b")
#     figname = "{}.png".format(i)
#     plt.savefig(figname)
#     plt.close(fig)
#
#
# for i,frame in enumerate(risk_mats):
#     fig, ax = plt.subplots(figsize=(15, 10))
#     cax = ax.matshow(frame)
#     plt.contour(zVals, colors = "b")
#     figname = "risk{}.png".format(i)
#     plt.savefig(figname)
#     plt.close(fig)
# os.chdir(a)
