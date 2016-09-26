  

def PoseVariation(pose,longSide):

    I = np.eye(4)

    %rotation
    import math
    
    degree = 10*math.pi/180;   %angle in rad
    R = np.ones(4)
    
    rot_x = np.array([[1, 0, 0],[0, cos(degree), -sin(degree)],[ 0, sin(degree), cos(degree)]])
    R[0:3,0:3] = rot_x

    dim = longSide/5 

    listVarPose = []
    I[0:3,3] = np.array([[0,0,0]])
    listVarPose.append(pose * I)

    I[0:3,3] = np.array([[dim,0,0]])
    listVarPose.append(pose * I)

    I[0:3,3] = np.array([[2*dim,0,0]])
    listVarPose.append(pose * I)

    I[0:3,3] = np.array([[-dim,0,0]])
    listVarPose.append(pose * I)

    I[0:3,3] = np.array([[-2*dim,0,0]])
    listVarPose.append(pose * I)

    I[0:3,3] = np.array([[3*dim,0,0]])
    listVarPose.append(pose * I)


    #Add rotation R_variation
    listVarPose.append(pose *R* np.array([[0,0,0]]))
    listVarPose.append(pose *R* np.array([[dim,0,0]]))
    listVarPose.append(pose *R* np.array([[2*dim,0,0]]))
    listVarPose.append(pose *R* np.array([[-dim,0,0]]))
    listVarPose.append(pose *R* np.array([[-2*dim,0,0]]))
    listVarPose.append(pose *R* np.array([[3*dim,0,0]]))

    return listVarPose



