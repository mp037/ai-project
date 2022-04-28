import random
from math import floor
from enemy import Enemy, Agent
from tabulate import tabulate
import sys
from io import StringIO
import copy


class Game() :
    def __init__(self):
        m=17
        n=24
        wm=2
        wn=2
        self.vis_range = 5
        self.g_raw, h, v = self.create_grid(m, n, wm, wn)
        self.g, self.agent, self.goal, self.enemies = self.add_agents(self.g_raw, m, n)
        self.g = self.redraw_map()
        self.draw()
    
    def generate_walls(self, m, wm):
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
    
    def build_walls(self, g, m, n ,horizontal_walls, vertical_walls):
        for i in horizontal_walls:
            for j in range(n):
                g[i][j] = '*'
        for i in vertical_walls:
            for j in range(m):
                g[j][i] = '*'
        return g
    
    def add_doors(self, g, m, n ,horizontal_walls, vertical_walls):
        doors = []
        for i in range(1, len(horizontal_walls)-1):
            for j in range(1, len(vertical_walls)):
                r = random.randint(vertical_walls[j-1]+1, vertical_walls[j]-1)
                doors.append((horizontal_walls[i], r))
                g[horizontal_walls[i]][r] = " "
        for i in range(1, len(vertical_walls) - 1):
            for j in range(1, len(horizontal_walls)):
                r = random.randint(horizontal_walls[j - 1] + 1, horizontal_walls[j] - 1)
                doors.append((r, vertical_walls[i]))
                g[r][vertical_walls[i]] = " "
        if len(horizontal_walls) < len(vertical_walls):
            min_rooms = len(horizontal_walls)-1
        else:
            min_rooms = len(vertical_walls)-1
        r = random.randint(0, min_rooms-1) #number of doors to randomly remove
        print(r)
        for count in range(r):
            i, j = random.choice(doors)
            g[i][j] = "*"
            doors.remove((i, j))
        return g
    
    def create_grid(self, m, n, wm, wn):
        g = [[" "] * n for i in range(m)] #creates a m*n table
        horizontal_walls = self.generate_walls(m,wm) #returns a list of indexes for the horizontal walls
        vertical_walls = self.generate_walls(n,wn)
        g = self.build_walls(g, m, n, horizontal_walls, vertical_walls)
        g = self.add_doors(g, m, n, horizontal_walls, vertical_walls)
        return (g, horizontal_walls, vertical_walls)
    
    def distance(self, i1, j1, i2, j2):
        return abs(i1-i2) + abs(j1-j2)
    
    def add_agents(self, g, m, n):
        corners = [(0, 0), (0, n-1), (m-1, 0), (m-1, n-1)]
        agent_pos = (0, 0)
        agent_in_corner = False
        while not agent_in_corner:
            i = random.randint(1, m-2)
            j = random.randint(1, n-2)
            if g[i][j] == " " :
                for (i1, j1) in corners:
                    if self.distance(i, j, i1, j1) < 6:
                        agent_in_corner = True
                        agent = Agent((i, j))
                        #g[i][j] = "|"
                        break
        goal_faraway = False
        goal = (0, 0)
        while not goal_faraway:
            i = random.randint(1, m - 2)
            j = random.randint(1, n - 2)
            max_side = max(m-1, n-1)
            if g[i][j] == " " and self.distance(i, j, agent.get_pos()[0], agent.get_pos()[1]) > max_side:
                goal_faraway = True
                goal = (i, j)
                #g[i][j] = "O" #big letter o, not zero
        enemy_agents = []
        num_enemies = floor(m*n/16)
        for count in range(num_enemies):
            i = random.randint(1, m - 2)
            j = random.randint(1, n - 2)
            while g[i][j] != " " or self.distance(i, j, agent.get_pos()[0], agent.get_pos()[1]) <= 3: #enemies can't be on the goal, fix later?
                i = random.randint(1, m - 2)
                j = random.randint(1, n - 2)
            enemy_agents.append(Enemy((i, j), self.vis_range))
            #g[i][j] = "1"
        return (g, agent, goal, enemy_agents)
    
    def redraw_map(self) :
        g = copy.deepcopy(self.g_raw)
        agent_pos = self.agent.get_pos()
        
        g[agent_pos[0]][agent_pos[1]] = '|'
        g[self.goal[0]][self.goal[1]] = 'O'
        
        for enemy in self.enemies :
            if enemy.is_alive() :
                enemy_pos = enemy.get_pos()
                g[enemy_pos[0]][enemy_pos[1]] = enemy.get_symbol()  
            
        return g
    
    def do_action(self, in_key) :
        direction = (0, 0)
        if in_key == 'w' :
            direction = (-1, 0)
        elif in_key == 's' :
            direction = (1, 0)
        elif in_key == 'a' :
            direction = (0, -1)
        elif in_key == 'd' :
            direction = (0, 1)
        else :
            print('something went wrong')
            return False
    
        self.agent.move(direction, self.g)
        agent_pos = self.agent.get_pos()
        
        if agent_pos == self.goal :
            print('You won!')
            return False
        
        for enemy in self.enemies :
            if enemy.is_alive() :
                enemy_pos = enemy.get_pos()
                if agent_pos == enemy_pos :
                    enemy.kill()
                    continue
                
                enemy.move_towards(self.agent.get_pos(), self.g)
                
                if agent_pos == enemy.get_pos() :
                    self.agent.kill()
                    print("You lose.")
        
        self.g = self.redraw_map()
        self.draw()
        
        return self.agent.is_alive()
        
    
    def draw(self):
        print(tabulate(self.g))


game = Game()

run = True
allowed = ['w', 'a', 's', 'd']
while(run) :
    in_key = 'b'
    while in_key not in allowed :
        in_key = input()
        if in_key == 'stop' :
            run = False
            break
        
    if run :
        run = game.do_action(in_key)

print('Done.')


