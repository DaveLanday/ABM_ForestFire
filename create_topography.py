#!/usr/bin/env python
# -*-coding: utf-8 -*-


#create_topography.py
#David W. Landay
#LAST UPDATED: 12-03-2018

import sys
import numpy as np
from scipy import interpolate
from time import time
from skimage.measure import perimeter, label
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def fade(t):
    """
        FADE: fade function to generate perlin noise_type

            ARGS:
                t: value
                mode:
    """

    return 6*t**5 - 15*t**4 + 10* t**3
def grad(site, neighbors):
    pass

def lerp(a,b,X):
    return a + X * (b - a)
class Landscape(object):

    def __init__(self, L=20):
        #square dimension of the landscape lattice:
        self.L = L

        #array  representing the lattice(fill it with trees):
        self.land = np.full((self.L, self.L), 2, dtype=int)

        #array representing the topography:
        self.top = np.zeros((self.L, self.L), dtype=float)

        #dictionary describing neighbors for each cell:
        self.neighbors = {}

        #ordered list of all indices:
        self.indices = np.array(list(np.ndindex(self.land.shape)))

        #indices of the sites on fire:
        self.on_fire = []

    def get_neighbors(self):
        """
            GET_NEIGHBORS: creates an dictionary of adjacency lists for each site on
                          the lattice:

                        ARGS:
                            NONE

                     RETURNS:
                            NONE initializes the dictionary of adjacency lists
        """
        #create a reference list to check if we have exhausted the row:
        row_len = [i for i in range(self.L)]

        for idx, site in enumerate(self.indices):

            #every site must have an adjacency list (set of 4 neighbors):
            adj = []

            #check neighbor directly above first:
            if site[0] - 1 in row_len:
                adj.append((site[0] - 1, site[1]))

            #next, check neighbor directly to the right:
            if site[1] + 1 in row_len:
                adj.append((site[0], site[1] + 1))

            #next, check neighbor directly below:
            if site[0] + 1 in row_len:
                adj.append((site[0] + 1, site[1]))

            #finally, check neighbor directly to the left:
            if site[1] - 1 in row_len:
                adj.append((site[0], site[1] - 1))

            #update the dictionary of adjacency lists:
            self.neighbors[tuple(site)] = adj
        return

    def initialize_fire(self, amt_spread):
        """
            INITIALIZE_FIRE: Randomly construct a connected component of sites
                             that have the value of 1. Do this by selecting a
                             coordinate on the lattice at random, then following
                             random edges of its neighbors to ignite (turn 1).
                             Continue following excess edges for the input
                             amount of spreading. Nearest neighbors to the
                             selected coordinate are always burned. This way,
                             a centroid is defined and the fire is not biased
                             in one direction.

                            ARGS:
                                amt_spreading: the number of edges to follow
                                               out from the point of outbreak.
                                               Type: int

                        RETURNS:
                                NONE, assigns the selected coordinates in
                                self.land to be in state 1.

        """

        #select a site to ignite:
        selection = np.random.choice(list(range(self.L**2)), p=None, size=1)

        #ignite site at coordinate associated with the seleciton (index):
        coord = (self.indices[selection][0][0], self.indices[selection][0][1])

        self.land[coord] = 1

        #propagate fire using random selections from the excess degree of coord:
        neighbors = self.neighbors[coord]

        #randomly ignite 2 of the nearest neighbors:
        init_ignite = np.random.choice(range(len(neighbors)), size=2)
        for n in neighbors:
            self.land[n] = 1

        #now, follow the desired number of excess edges and spread the fire:
        t = 0
        while t <= amt_spread:
            self.get_sites_on_fire()
            seen = set((i[0],i[1]) for i in self.on_fire)

            for num_edges in self.on_fire:
                temp = []
                for neigh in self.neighbors[num_edges[0], num_edges[1]]:
                    #generate a random number:
                    if neigh not in seen:
                        temp.append(neigh)

                #randomly choose a neighbor to ignite:
                try:
                    n2choosefrom = [i for i in range(len(temp))]

                    rn = np.random.choice(n2choosefrom)
                    seen.update([temp[rn]])
                except:
                    pass

            for i in seen:
                self.land[i] = 1
            t+=1
        return

    def get_sites_on_fire(self):
        """
            GET_SITES_ON_FIRE: gets the indices of sites that are assigned the
                               value 1 (on fire).

                               ARGS:
                                    None

                            RETURNS:
                                    an array of indices
        """
        #get the row and column indices for each of the sites on fire:
        i, j = np.where(self.land == 1)

        #zip them into a series of tuples:
        coords = list(zip(i,j))

        #update on_fire:
        self.on_fire = np.array(coords)

    def perlin_terrain(self, f, noise, scale=100):
        """
            PERLIN_TERRAIN: Generates ranodom 2D height maps that mimic smoothed
                            terrain.

                        ARGS:
                            f: a scale factor and the height displacement
                              (phase displacement shift/ translation). Type: int

                            noise: the noise function that defines the deformation
                                   of a point across the 2D space.
                                   Acceptible Input:
                                            'Sinosoid': generates a sinosoidal
                                                      surface centered at h+1.
                                                      (see: https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html)

                                            'Perlin': generates perlin noise
                                                      over the 2D space to
                                                      represent a height map.
                                                      requires a scale input,
                                                      which determines the
                                                      amount of smoothing
                                                      (i.e interpolation between
                                                      sites on the lattice)
                                   Type: str
                    RETURNS:
                            2D nparray describing a height map.
        """
        #NOTES:
        #octaves are the number of waves to combine
        #frequency is the number of oscillations per time step
        #amplitude is the height of any given point on the waveform
        #get row,col:

        if noise == 'Saddle':

            #initialize a meshgrid:
            X,Y = np.mgrid[-1:1:self.L+0j, -1:1:self.L+0j]

            #calculate heights at each site (centered about Z= f + 1):
            Z = ((f*(X + Y) * np.exp(-f*(X**2 + Y**2)) + 1) *f + f) + 1
            self.top = Z

        if noise == 'Perlin':

            #choose different intervals to pick samples from:
            low_high = np.array([(0,1), (0.25, 0.75), (0.33, 0.67), (0.375, 0.625)])

            #at each site in the grid, initialize some noise
            for j in range(self.L):
                for i in range(self.L):
                    x = fade(np.random.uniform(low=0,high=1))*f
                    y = fade(np.random.uniform(low=0,high=1))*f

                    self.top[i,j] = x + y

            #how much zooming in or out we wish to do (smoothing as a function of time):
            t=0
            while  t <= scale:
                for site in self.indices:

                    #choose the interval to sample from:
                    lh = np.random.choice([0,1,2,3])

                    #take a sample from the interval:
                    rn = np.random.uniform(low=low_high[lh][0],
                                           high=low_high[lh][1],
                                           size=len(self.neighbors[site[0], site[1]]))

                    #store the gradient interpolations:
                    lerps = np.zeros(len(self.neighbors[site[0], site[1]]))

                    # interpolate gradient given heights of sites neighbors:
                    for idx, n in enumerate(self.neighbors[site[0], site[1]]):
                        lerps[idx] = lerp(self.top[n], self.top[site[0], site[1]], rn[idx])

                    #average over all gradients and save as a height:
                    self.top[site[0], site[1]] = np.average(lerps)

                #increment the time step:
                t+=1

    def display_topography(self):
        X, Y = np.mgrid[:self.L, :self.L]

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1, projection='3d')
        surf = ax.plot_surface(X,Y,self.top,cmap='Greens')
        plt.show()

#________________________________PLOTTING_______________________________________
a = Landscape(L=50)
a.get_neighbors()
a.perlin_terrain(f=2, noise='Perlin', scale=20)
a.display_topography()
