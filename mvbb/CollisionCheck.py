#!/usr/bin/env python
from klampt.model import collide

def CheckCollision(world,robot,obj):
    collision = collide.WorldCollider(world)
    R_O = collision.robotObjectCollisions(robot,obj)
    li = []
    for i in R_O:
        li.append(R_O)
    R_w = collision.robotTerrainCollisions(robot)
    li2 = []
    for j in R_w:
        li2.append(R_w)
    if(len(li)>0 or len(li2)>0):
        return True
    else:
        return False