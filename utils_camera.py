import pkg_resources
pkg_resources.require("klampt>=0.7.0")
from klampt import *
from klampt import vis 
from klampt.vis.glrobotprogram import *
from klampt.math import *
from klampt.model import collide
from klampt.io import resource
from klampt.sim import *
from moving_base_control import *
import numpy as np
import csv
from IPython import embed
from klampt.math import se3,so3
import math



def Read_Poses(nome,vector_set):

    obj_dataset = '3DCNN/NNSet/Pose/pose/%s.csv'%(nome)
    with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
        file_reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
        for row in file_reader:
            T = row[9:12]
            pp = row[:9]
            vector_set.append( np.array(se3.homogeneous( (pp,T)) ) ) 
            # break

def Write_image(camera,dataset):
    '''Write the dataset'''
    # embed()
    import csv
    f = open(dataset, 'w')
    # embed()
    for i in camera:
        f.write(','.join([str(i)]))
        f.write('\n')
    f.close()


def Find_min_angle(pose,axis):
    # embed()
    # x = pose[:3,0]
    # y = pose[:3,1]
    # z = pose[:3,2]
    # pose_temp = se3.homogeneous(pose)
    if isinstance(pose, np.ndarray):
        x = [pose[0][0],pose[1][0],pose[2][0] ]
        y = [pose[0][1],pose[1][1],pose[2][1] ]
        z = [pose[0][2],pose[1][2],pose[2][2] ]
    else:
        x = pose[0]
        y = pose[1]
        z = pose[2]



    # embed()
    Dx = math.acos(np.dot(axis,x) / np.dot(np.sqrt(np.dot(x,x)),np.sqrt(np.dot(axis,axis))) )
    Dy = math.acos(np.dot(axis,y) / np.dot(np.sqrt(np.dot(y,y)),np.sqrt(np.dot(axis,axis))) )
    Dz = math.acos(np.dot(axis,z) / np.dot(np.sqrt(np.dot(z,z)),np.sqrt(np.dot(axis,axis))) )
    return [Dx,Dy,Dz]



def Find_long_side_and_axis(bmin,bmax):
    side_x = math.sqrt( pow( bmin[0] - bmax[0] , 2) )
    side_y = math.sqrt( pow( bmin[1] -  bmax[1] , 2) )
    side_z = math.sqrt( pow( bmin[2] - bmax[2], 2) )

    figure = []
    ori = None

    figure.append(side_x) 
    figure.append(side_y)
    figure.append(side_z)


    maxi = -1000 #assign max a value to avoid garbage

    for k in range(0,len(figure)):
        if (maxi <= figure[k]):
            maxi = figure[k]
            ori = k

    if ori == None:
        assert "Problem with long side"
    #ori=0 axis x
    #ori=1 axis y
    #ori=2 axis z
    # axis = []
    # index = None
    # if ori == 0:
    #     axis = [1,0,0]
    #     index = 0
    # elif ori ==1:
    #     axis = [0,1,0]
    #     index = 1
    # elif ori ==2:
    #     # axis = [0,0,1]
    #     index = 2

    print 'index',ori
    return ori








def  FromCamera2rgb(camera_measure):
    abgr = (np.array(camera_measure)[0:len(camera_measure)/2]).reshape(256,256).astype(np.uint32)
    
    rgb = np.zeros((256,256,3),dtype=np.uint8)
    rgb[:,:,0] =                np.bitwise_and(abgr,0x000f)
    rgb[:,:,1] = np.right_shift(np.bitwise_and(abgr,0x00f0), 8)
    rgb[:,:,2] = np.right_shift(np.bitwise_and(abgr,0x0f00), 16)
    # embed()
    return abgr,rgb


def perpendicular_vector(v):
    """ Finds an arbitrary perpendicular vector to *v*. """

    a, b = random.random(), random.random()
    if not iszero(v.z):
        x, y, z = v.x, v.y, v.z
    elif not iszero(v.y):
        x, y, z = v.x, v.z, v.y
    elif not iszero(v.x):
        x, y, z = v.y, v.z, v.x
    else:
        raise ValueError('zero-vector')

    c = (- x * a - y * b) / z

    if not iszero(v.z):
        return Vector(a, b, c)
    elif not iszero(v.y):
        return Vector(a, c, b)
    elif not iszero(v.x):
        return Vector(b, c, a)




def Find_long_side(bbox):
    '''Find longest boxes axis'''
    #distance between vertex 0 and 7

    side_x = math.sqrt( pow( bbox.Isobox[0,0] - bbox.Isobox[1, 0] , 2) )
    side_y = math.sqrt( pow( bbox.Isobox[0,1] -  bbox.Isobox[1,1] , 2) )
    side_z = math.sqrt( pow( bbox.Isobox[0,2] - bbox.Isobox[1,2], 2) )

    figure = []
    ori = None

    figure.append(side_x) 
    figure.append(side_y)
    figure.append(side_z)


    maxi = -1000 #assign max a value to avoid garbage

    for k in range(0,len(figure)):
        if (maxi <= figure[k]):
            maxi = figure[k]
            ori = k

    if ori == None:
        assert "Problem with long side"
    #ori=0 axis x
    #ori=1 axis y
    #ori=2 axis z
    # axis = []
    index = None
    if ori == 0:
        axis = [bbox.T[0,0],bbox.T[1,0],bbox.T[2,0]]
        index = 0
    elif ori ==1:
        axis = [bbox.T[0,1],bbox.T[1,1],bbox.T[2,1]]
        index = 1
    elif ori ==2:
        axis = [bbox.T[0,2],bbox.T[1,2],bbox.T[2,2]]
        index = 2


    return axis,index

def Compute_box(obj):
    if isinstance(obj, np.ndarray):
        vertices = obj
        n_vertices = vertices.shape[0]
        box = pydany_bb.Box(n_vertices)

        box.SetPoints(vertices)
    else:
        tm = obj.geometry().getTriangleMesh()
        n_vertices = tm.vertices.size() / 3
        box = pydany_bb.Box(n_vertices)

        for i in range(n_vertices):
            box.SetPoint(i, tm.vertices[3 * i], tm.vertices[3 * i + 1], tm.vertices[3 * i + 2])

    I = np.eye(4)
    print "doing PCA"
    box.doPCA(I)
    print box.T
    print "computing Bounding Box"
    bbox = pydany_bb.ComputeBoundingBox(box)

    long_side,index  = Find_long_side(bbox)

    angle = math.acos(np.dot([0,0,1],long_side)) / np.dot(np.sqrt(np.dot(long_side,long_side)),np.sqrt(np.dot([0,0,1],[0,0,1]))) 

    if angle <= math.radians(10):
        standing = True
    else:
        standing = False
    # embed()
    return long_side,index,standing,box.T