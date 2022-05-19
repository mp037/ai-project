# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:04:47 2022

@author: Kert PC
"""
class Enemy() :
    def get_pos(self):
        return self.position
    
    def get_symbol(self) :
        return self.symbol

    def get_lives(self) :
        return self.lives

    def remove_life(self):
        self.lives -= 1
    
    def kill(self):
        self.alive = False
        self.position = (-1, -1)  # enemies can now move to spaces where previous ones have been killed
    
    def is_alive(self) :
        return self.alive
    
    def distance(self, pos1, pos2):
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])    
    
    def is_close_enough(self, pos) :
        return self.vis_range >= self.distance(self.position, pos)

    def reduce_turns(self):
        self.move_every_turn -= 1

    def reset_turns(self):
        self.move_every_turn = self.moves_per_turn
        
    def can_move_to(self, nu_pos, g) :
        return (g[nu_pos[0]][nu_pos[1]] == ' '
                or g[nu_pos[0]][nu_pos[1]] == '|')
        
    def move_towards(self, pos, g, enemies):
        if self.is_close_enough(pos):
    
            self.reduce_turns()
            if self.move_every_turn == 0:
                self.reset_turns()
    
                moves_remain = self.moves
    
                while moves_remain > 0:
                    moves_remain -= 1
                    x_dist = pos[0] - self.position[0]
                    y_dist = pos[1] - self.position[1]
    
                    if abs(x_dist) > abs(y_dist):
                        if abs(y_dist) > 0:
                            moves = [(0, y_dist // abs(y_dist)), (x_dist // abs(x_dist), 0)]
                        else:
                            moves = [(x_dist // abs(x_dist), 0)]
                    else:
                        if abs(x_dist) > 0:
                            moves = [(x_dist // abs(x_dist), 0), (0, y_dist // abs(y_dist))]
                        else:
                            moves = [(0, y_dist // abs(y_dist))]
    
    
                    for (x, y) in moves:
                        nu_pos = (self.position[0] + x, self.position[1] + y)
                        if self.can_move_to(nu_pos, g) :
                            same_pos = False
                            for enemy in enemies:
                                if enemy.get_pos() == nu_pos:
                                    same_pos = True
                                    break
                            if same_pos:
                                break
                            self.position = nu_pos
                            break


class Enemy1(Enemy) :
    def __init__(self, pos=(0,0), vis_range=5, moves=1, symbol='1', lives=1, moves_per_turn=1):
        self.position = pos
        self.moves = moves
        self.vis_range = vis_range
        self.alive = True
        self.symbol = symbol
        self.lives = lives
        self.move_every_turn = moves_per_turn
        self.moves_per_turn = moves_per_turn


class Enemy2(Enemy):
    def __init__(self, pos=(0, 0), vis_range=5, moves=1, symbol='2', lives=2, moves_per_turn=2):
        self.position = pos
        self.moves = moves
        self.vis_range = vis_range
        self.alive = True
        self.symbol = symbol
        self.lives = lives
        self.move_every_turn = moves_per_turn
        self.moves_per_turn = moves_per_turn


class Enemy3(Enemy):
    def __init__(self, pos=(0, 0), vis_range=5, moves=1, symbol='3', lives=1, moves_per_turn=1):
        self.position = pos
        self.moves = moves
        self.vis_range = vis_range
        self.alive = True
        self.symbol = symbol
        self.lives = lives
        self.move_every_turn = moves_per_turn
        self.moves_per_turn = moves_per_turn

    def can_move_to(self, nu_pos, g) :
        return (g[nu_pos[0]][nu_pos[1]] == ' '
            or g[nu_pos[0]][nu_pos[1]] == '|'
            or g[nu_pos[0]][nu_pos[1]] == '*')

                
class Agent() :
    def __init__(self, pos=(0,0)):
        self.position = pos
        self.alive = True
        
    def kill(self):
        self.alive = False
        
    def distance_to(self, pos):
        return abs(self.position[0]-pos[0]) + abs(self.position[1]-pos[1])  
    
    def is_alive(self) :
        return self.alive

    def get_pos(self):
        return self.position

    def move(self, direction, g, enemies) :
        nu_pos = (self.position[0] + direction[0], self.position[1] + direction[1])
        if g[nu_pos[0]][nu_pos[1]] != '*':
            if g[nu_pos[0]][nu_pos[1]] == ' ' or g[nu_pos[0]][nu_pos[1]] == 'O':
                self.position = nu_pos
            else:
                for enemy in enemies:
                    if enemy.get_pos() == nu_pos:
                        enemy.remove_life()
                        
            return 1
        else :
            return 0