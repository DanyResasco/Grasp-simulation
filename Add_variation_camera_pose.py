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
import random
from utils_camera import FromCamera2rgb, Find_axis_rotation,Find_min_angle
from IPython import embed
import PyKDL as kdl
import pydany_bb

from create_mvbb import MVBBVisualizer, compute_poses, skip_decimate_or_return

def Add_variation(o_T_p_r_original,o_T_p_r):


    theta_deg = random.randint(-10,10)
    if theta_deg == 0.0:
        theta_deg = random.randint(-10,10)
    theta = math.radians(theta_deg)

    # embed()
    axis_x = o_T_p_r_original[0][0:3]
    axis_y = o_T_p_r_original[0][3:6]
    axis_z = o_T_p_r_original[0][6:9]
    ##Variation on x
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)


    #  #variazione su y
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)
    
    #  #variazione su z
    R = np.array(se3.homogeneous((so3.from_axis_angle(([0,0,1],theta)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)


    # #variazione su x e muovo su z
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)


    # #variazione su y e muovo su z
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)


    # #variazione su y e muovo su z
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_z,theta)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    # o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)

    # # muovo su - z
    o_T_p_r_temp = o_T_p_r_original
    move_z = random.randrange(-5,5)
    o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)

    # #muovo su + z
    o_T_p_r_temp = o_T_p_r_original
    move_z = random.randrange(-5,5)
    o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - move_z*0.011
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)

    #  #variazione su y e muovo su z
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_y,theta)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)

    #  #variazione su x e muovo su y
    move_z = random.randrange(-5,5)
    R = np.array(se3.homogeneous((so3.from_axis_angle((axis_x,theta)),[0,0,0] )))
    o_T_p_r_temp = se3.from_homogeneous(( np.dot(R,np.array(se3.homogeneous(o_T_p_r_original)))))
    o_T_p_r_temp[1][0:3] = o_T_p_r_original[1][0:3]
    o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - move_z*0.01
    # o_T_p_r_temp[1][1] = o_T_p_r_original[1][1] - 0.7
    o_T_p_r.append(o_T_p_r_temp)




def create_camera_frame(z):
    """
    create_camera_frame returns a rotation matrix with z parallel to z
    """
    _R = np.eye(3)

    _z = np.array(z).reshape(3)
    _z = _z / np.linalg.norm(_z)
    _z_world = np.array([0,0,1])
    if np.abs(_z.dot(_z_world)) > (1 - 1e-9):  # collinearity check
        if _z.dot(_z_world) < 0.:
            _R = np.array([[1, 0, 0],
                           [0,-1, 0],
                           [0, 0,-1]])
    else:
        _x_cand = np.cross(_z_world, _z)
        _y = np.cross(_x_cand, -_z)
        _y = _y / np.linalg.norm(_y)
        _x = -np.cross(_z, _y)
        # _R = np.vstack((_x, _y, _z)).transpose()
        _R = np.vstack((_z, _y, -_x)).transpose()
    return _R



def Find_long_side(bbox):
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
    embed()
    return long_side,index,standing,box.T



def Make_camera_poses(o_T_p,obj):
    o_T_p_r = []


    print "len: ", len(o_T_p)

    long_side,index, standing, T_box = Compute_box(obj)

    R_o,t = obj.getTransform()
    bmin,bmax = obj.geometry().getBB()
    centerX = 0.5 * ( bmax[0] - bmin[0] ) + t[0]
    centerY = 0.5 * ( bmax[1] - bmin[1] ) + t[1]
    centerZ = 0.5 * ( bmax[2] - bmin[2] ) + t[2]
    
    # if index == 0:
    #     if standing == True: # x up
    #         T_o = [centerX -0.7, centerY , centerZ ]
    #     else:
    #         T_o = [centerX -0.7, centerY, centerZ ]
    # elif index ==1:
    #     if standing == True: # x up
    #         T_o = [centerX, centerY + 0.7, centerZ]
    #     else:
    T_o = [centerX  , centerY  , centerZ]

    # embed()
    for k in range(0,len(o_T_p)):
        if k == 0:
            # R_np = create_camera_frame(long_side)
            # R_temp = list(R_np.reshape(-1))
            o_T_p_r.append((se3.from_homogeneous(T_box)[0],T_o))
            
        else:
            pass
            # o_T_p_r.append((R_temp,T_o))
            # tx = se3.from_homogeneous( o_T_p[k])[1][0] 
            # ty = se3.from_homogeneous( o_T_p[k])[1][1] 
            # tz = se3.from_homogeneous( o_T_p[k])[1][2] 
            
            # z = (T_o - np.array([tx,ty,tz])) / np.linalg.norm(T_o - np.array([tx,ty,tz]))
            # x = np.cross(z,np.array([0,0,-1])) / np.linalg.norm(np.cross(z,np.array([0,0,-1])) )
            # y = -np.cross(z,x)

           


            # R_np = create_camera_frame(z)
            # R_temp = list(R_np.reshape(-1))
            # Rz = so3.from_axis_angle(([R_np[0,1],R_np[1,1],R_np[2,1]] ,math.radians(180)))
            # R = np.dot(np.array(se3.homogeneous((R_np.transpose().reshape(-1),[0,0,0]))), np.array( se3.homogeneous(( Rz,[0,0,0])) ))  

            # embed()
            # z =  (np.array(t) - np.array(se3.from_homogeneous( o_T_p[k]))[1]) / np.linalg.norm( np.array(t) - np.array(se3.from_homogeneous( o_T_p[k]))[1])
            # x = [-1,0,0]
            # y  = np.cross(z,x) / np.linalg.norm(np.cross(z,x))
            # m = [x[0],y[0],z[0],x[1],y[1],z[1],x[2],y[2],z[2]]
            # m = [x[0],y[0],z[0],x[1],y[1],z[1],x[2],y[2],z[2]]

            # embed()

            # temp = m, [tx,ty,tz]

            # embed()


            # o_T_p_r.append( temp)

        Add_variation(o_T_p_r[k],o_T_p_r)
        break


    return o_T_p_r



# w_T_p is a numpy array

# r = w_T_p[0:2,3]
# r_des = 0.5         # desired distance from object
# z_des = 1.0
# w_T_p[0:3,3] *= r_des / np.linalg.norm(r)
# w_T_p[2,3] = z_des

# p_2 = w_T_p[0:3,3]
# z = -p_2 # that is, p_1 - p_2 with p_1 = [0,0,0]
# z *= 1. / np.linalg.norm(z)
# e l'origine della camera sara p_2