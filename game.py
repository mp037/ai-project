import random
from math import floor
from enemy import Enemy1, Enemy2, Enemy3, Agent
from tabulate import tabulate
import sys
from io import StringIO
import copy
import numpy as np
import copy

class Game() :
    def __init__(self, visualize=True, m=10, n=10, wm=0, wn=0, num_enemies=3):
        self.visualize = visualize
        self.num_enemies = num_enemies
        self.vis_range = 5
        self.g_raw, h, v = self.create_grid(m, n, wm, wn)
        self.g, self.agent, self.goal, self.enemies = self.add_agents(self.g_raw, m, n)
        self.g_visited = [[0]*len(self.g[0])]*len(self.g)
        
        self.og_agent = copy.deepcopy(self.agent)
        self.og_goal = copy.deepcopy(self.goal)
        self.og_enemies = copy.deepcopy(self.enemies)
        
        
        self.g = self.redraw_map()
        self.action_space = [0, 1, 2, 3]
        
        print('Ready.')
        if self.visualize or True :
            self.draw()
            
            
    def reset(self) :
        self.agent = copy.deepcopy(self.og_agent)
        self.goal = copy.deepcopy(self.og_goal)
        self.enemies = copy.deepcopy(self.og_enemies)
        
        return self.get_state()
    
    def distance_coord(self, pos1, pos2):
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])  
    
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
        if self.visualize :
            print('Walls generated.')
            
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
        if self.visualize :
            print('Grid created.')
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
        if self.visualize :
            print('Agent added.')
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
        num_enemies = self.num_enemies
        enemies_of_a_type = num_enemies/3
        for count in range(num_enemies):
            i = random.randint(1, m - 2)
            j = random.randint(1, n - 2)
            same_pos = False
            for enemy in enemy_agents:
                if (i, j) == enemy.get_pos():
                    same_pos = True
                    break
            while g[i][j] != " " or self.distance(i, j, agent.get_pos()[0], agent.get_pos()[1]) <= 3 or (i, j) == goal or same_pos: #enemies can't be on the goal, fix later?
                i = random.randint(1, m - 2)
                j = random.randint(1, n - 2)
                same_pos = False
                for enemy in enemy_agents:
                    if (i, j) == enemy.get_pos():
                        same_pos = True
                        break
            
            enemy_agents.append(Enemy1((i, j), self.vis_range))
            """
            if count < enemies_of_a_type:
                enemy_agents.append(Enemy1((i, j), self.vis_range))
            elif count < 2*enemies_of_a_type:
                enemy_agents.append(Enemy2((i, j), self.vis_range))
            else:
                enemy_agents.append(Enemy3((i, j), self.vis_range))
            """
            #g[i][j] = "1"
        if self.visualize :
            print('Enemies added.')
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
    
    def get_state(self) :
        state = []
        agent_pos = self.agent.get_pos()
        
        state.append(agent_pos[0])
        state.append(agent_pos[1])
        #state.append(self.goal[0])
        #state.append(self.goal[1])
        
        for enemy in self.enemies :
            enemy_pos = enemy.get_pos()
            state.append(enemy_pos[0])
            state.append(enemy_pos[1])
    
        return np.array(state)
    
    def do_action(self, in_key) :
        done = False
        
        direction = (0, 0)
        if in_key == 'w' or in_key == 0:
            direction = (-1, 0)
        elif in_key == 'a' or in_key == 1:
            direction = (0, -1)
        elif in_key == 's' or in_key == 2:
            direction = (1, 0)
        elif in_key == 'd' or in_key == 3 :
            direction = (0, 1)
        else :
            print('something went wrong')
            return self.get_state(), -100, True
        
        #old_dist = self.distance_coord(self.goal, self.agent.get_pos())
        self.agent.move(direction, self.g, self.enemies)
        agent_pos = self.agent.get_pos()
        """
        nu_dist = self.distance_coord(self.goal, self.agent.get_pos())
        
        if old_dist - nu_dist > 0 :
            reward = 1
        else:
            reward = -1
        """
        reward = (-1)*(self.g_visited[agent_pos[0]][agent_pos[1]]**2)
        self.g_visited[agent_pos[0]][agent_pos[1]] += 1
        
        if agent_pos == self.goal :
            if self.visualize:
                print('You won!')
            return self.get_state(), 100, True
        
        for enemy in self.enemies :
            if enemy.is_alive() and enemy.get_lives() == 0:
                #reward = 10
                if self.visualize:
                    print('Mama! Just killed a man ' + str(enemy.symbol))
                enemy.kill()

            if enemy.is_alive() :
                enemy_pos = enemy.get_pos()
                if agent_pos == enemy_pos :
                    enemy.kill()
                    continue
                
                enemy.move_towards(self.agent.get_pos(), self.g, self.enemies)
                
                if agent_pos == enemy.get_pos() :
                    self.agent.kill()
                    if self.visualize :
                        print("You lose.")
                    reward = -100 #- self.agent.distance_to(self.goal) * 10
                    done = True
        
        self.g = self.redraw_map()
        
        if self.visualize :
            self.draw()
        
        return self.get_state(), reward, done
        
    
    def draw(self):
        print(tabulate(self.g))

if __name__ == '__main__':
    game = Game(visualize=True)
    
    done = False
    allowed = ['w', 'a', 's', 'd']
    while not done :
        in_key = 'b'
        while in_key not in allowed :
            sys.stdout.flush()
            in_key = input()
            in_key = in_key.lower()
            if in_key == 'stop' :
                done = True
                break
        if not done :
            _, _, done = game.do_action(in_key)
    
    print('Done.')
    
    
