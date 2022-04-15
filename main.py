import random
from math import floor

from tabulate import tabulate

def generate_walls(m, wm):
    good_walls = False
    walls = []
    while not good_walls:
        good_walls = True
        walls = [0, m - 1]  # border walls
        for i in range(wm):
            r = random.randint(1, m - 2)
            if r not in walls:
                walls.append(r)
        walls.sort()
        for i in range(1, len(walls)):
            if len(walls) < wm + 2:
                good_walls = False
                break
            if walls[i] - walls[i - 1] < 5:
                good_walls = False
                break
    return walls

def build_walls(g, m, n ,horizontal_walls, vertical_walls):
    for i in horizontal_walls:
        for j in range(n):
            g[i][j] = '*'
    for i in vertical_walls:
        for j in range(m):
            g[j][i] = '*'
    return g

def add_doors(g, m, n ,horizontal_walls, vertical_walls):
    doors=[]
    for i in range(1, len(horizontal_walls)-1):
        for j in range(1, len(vertical_walls)):
            r=random.randint(vertical_walls[j-1]+1, vertical_walls[j]-1)
            doors.append((horizontal_walls[i], r))
            g[horizontal_walls[i]][r]=" "
    for i in range(1, len(vertical_walls) - 1):
        for j in range(1, len(horizontal_walls)):
            r = random.randint(horizontal_walls[j - 1] + 1, horizontal_walls[j] - 1)
            doors.append((r, vertical_walls[i]))
            g[r][vertical_walls[i]] = " "
    if len(horizontal_walls)<len(vertical_walls):
        min_rooms=len(horizontal_walls)-1
    else:
        min_rooms=len(vertical_walls)-1
    r=random.randint(0, min_rooms-1) #number of doors to randomly remove
    print(r)
    for count in range(r):
        i, j = random.choice(doors)
        g[i][j]="*"
        doors.remove((i, j))
    return g

def create_grid(m, n, wm, wn):
    g = [[" "] * n for i in range(m)] #creates a m*n table
    horizontal_walls=generate_walls(m,wm) #returns a list of indexes for the horizontal walls
    vertical_walls=generate_walls(n,wn)
    g=build_walls(g, m, n, horizontal_walls, vertical_walls)
    g=add_doors(g, m, n, horizontal_walls, vertical_walls)
    return (g, horizontal_walls, vertical_walls)

def distance(i1, j1, i2, j2):
    return abs(i1-i2) + abs(j1-j2)

def add_agents(g, m, n):
    corners=[(0, 0), (0, n-1), (m-1, 0), (m-1, n-1)]
    agent=(0, 0)
    agent_in_corner=False
    while not agent_in_corner:
        i=random.randint(1, m-2)
        j=random.randint(1, n-2)
        if g[i][j]==" ":
            for (i1, j1) in corners:
                if distance(i, j, i1, j1) < 6:
                    agent_in_corner=True
                    agent=(i, j)
                    g[i][j]="|"
                    break
    goal_faraway=False
    goal=(0, 0)
    while not goal_faraway:
        i = random.randint(1, m - 2)
        j = random.randint(1, n - 2)
        max_side = max(m-1, n-1)
        if g[i][j]==" " and distance(i, j, agent[0], agent[1]) > max_side:
            goal_faraway=True
            goal=(i, j)
            g[i][j]="O" #big letter o, not zero
    enemy_agents=[]
    num_enemies=floor(m*n/16)
    for count in range(num_enemies):
        i = random.randint(1, m - 2)
        j = random.randint(1, n - 2)
        while g[i][j]!=" " or distance(i, j, agent[0], agent[1]) <= 3: #enemies can't be on the goal, fix later?
            i = random.randint(1, m - 2)
            j = random.randint(1, n - 2)
        enemy_agents.append((i, j))
        g[i][j]="1"
    return (g, agent, goal, enemy_agents)


def draw(g):
    print(tabulate(g))

m=17
n=24
wm=2
wn=2
g, h, v = create_grid(m, n, wm, wn)
print(h, v)
g, agent, goal, enemies = add_agents(g, m, n)
print(agent, goal)
draw(g)
