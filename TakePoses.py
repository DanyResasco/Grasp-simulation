#!/isr/bin/env python
from klampt import *
from klampt.math import so3, se3
import numpy as np


def SimulationPoses(Posedesired,Th_w,Tow_temp):
	#Th_w end-effector in word frame
	#Th_o object in end-effector frame
	#Tw_o pbject in word frame

	Ryaw = so3.rotation([0,0,1],Th_w[3])
	Rpitch = so3.rotation([0,1,0],Th_w[4])
	Rroll = so3.rotation([1,0,0],Th_w[5])
	R =  so3.mul(so3.mul(Ryaw,Rpitch) ,Rroll)
	To_h = np.array([[R[0],R[1],R[2],Th_w[0]], [R[3],R[4],R[5],Th_w[1]], [R[6], R[7], R[8], Th_w[2]]])	# T end-effector in object frame

	To_w = np.array([[Tow_temp[0][0],Tow_temp[0][1],Tow_temp[0][2],Tow_temp[1][0]], 
		[Tow_temp[0][3],Tow_temp[0][4],Tow_temp[0][5],Tow_temp[1][1]], [Tow_temp[0][6],Tow_temp[0][7],Tow_temp[0][8],Tow_temp[1][2]]])
	
	TDes_m_o = se3.inv(se3.from_homogeneous(Posedesired))	#T object in end-effector frame
	Tw_o	= se3.inv(se3.from_homogeneous(To_w))	
	Tact_m_o = se3.inv(se3.mul(Tw_o,se3.from_homogeneous(To_h)))	#T object in end-effector frame
	posedict = {}
	posedict['desired'] = [TDes_m_o]
	posedict['actual'] = [Tact_m_o]
	return posedict
