# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:04:47 2022

@author: Kert PC
"""

class Enemy() :
    def __init__(self, pos=(0,0), vis_range=5, moves=1, symbol='1'):
        self.position = pos
        self.moves = moves
        self.vis_range = vis_range
        self.alive = True
        self.symbol = symbol
        
    def get_pos(self):
        return self.position
    
    def get_symbol(self) :
        return self.symbol
    
    def kill(self):
        self.alive = False
    
    def is_alive(self) :
        return self.alive
    
    def distance(self, pos1, pos2):
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])    
    
    def is_close_enough(self, pos) :
        return self.vis_range >= self.distance(self.position, pos)
        
    def move_towards(self, pos, g) :
        if self.is_close_enough(pos) :
            moves_remain = self.moves
            
            while moves_remain > 0 :
                moves_remain -= 1
                x_dist = pos[0] - self.position[0]
                y_dist = pos[1] - self.position[1] 
                max_dist = (x_dist // abs(x_dist), 0) if abs(x_dist) > abs(y_dist) else (0, y_dist // abs(y_dist))
                
                nu_pos = (self.position[0] + max_dist[0], self.position[1] + max_dist[1])
                if g[nu_pos[0]][nu_pos[1]] == ' ' or g[nu_pos[0]][nu_pos[1]] == '|':
                    self.position = nu_pos
                
                
class Agent() :
    def __init__(self, pos=(0,0)):
        self.position = pos
        self.alive = True
        
    def kill(self):
        self.alive = False
    
    def is_alive(self) :
        return self.alive

    def get_pos(self):
        return self.position

    def move(self, direction, g) :
        nu_pos = (self.position[0] + direction[0], self.position[1] + direction[1])
        if g[nu_pos[0]][nu_pos[1]] != '*':
            self.position = nu_pos