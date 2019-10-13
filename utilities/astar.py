#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""astar: A* pathfinding algorithm implemented using heaps.

More information on A*: http://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html#the-a-star-algorithm
Code implemented by: Christian Careaga (christian.careaga7@gmail.com)
"""

import sys  # To close the window when the game is over
from os import environ, path  # To center the game window the best possible
import numpy as np
from heapq import *

__author__ = "Victor Neves"
__license__ = "MIT"
__maintainer__ = "Victor Neves"
__email__ = "victorneves478@gmail.com"
__status__ = "Production"


def heuristic(a, b):
    """Euclidian distance of a and b.

    The heuristic calculates the Euclidian distance using the expression:
    h(n)²​​ = (n.x−goal.x)²​ + (n.y−goal.y)²

    Return
    -----------
    heuristic: int
        The euclidian distance between a and b.
    ​​"""
    heuristic = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

    return heuristic


def astar(array, start, goal):
    """A* algorithm for pathfinding.

    It searches for paths excluding diagonal movements. The function is composed
    by two components, gscore and fscore, as seem below.

    f(n) = g(n) + h(n)
    """
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))

    while oheap:
        current = heappop(oheap)[1]

        if current == goal:
            data = []

            while current in came_from:
                data.append(current)
                current = came_from[current]

            return data

        close_set.add(current)

        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)

            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [
                i[1] for i in oheap
            ]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))

    return False


def resource_path(relative_path):
    """Function to return absolute paths. Used while creating .exe file."""
    if hasattr(sys, "_MEIPASS"):
        return path.join(sys._MEIPASS, relative_path)

    return path.join(path.dirname(path.realpath(__file__)), relative_path)


def load_map(self, path):
    """Load map file to play. """
    map_path = self.resource_path(path)

    with open(map_path) as map_file:
        self.map = np.loadtxt(map_file).transpose()
