
from __future__ import division

import random, time, threading
import numpy as np
from math import pi
from multiprocessing import Pool

if __name__ == '__main__':
	
	workspace_x_min = -1.3
	workspace_x_max = 1.3
	workspace_y_min = -1.3
	workspace_y_max = 1.3
	workspace_z_min = 0
	workspace_z_max = 1.7

	grid_res = 0.05


class BitmapEnvironment:
	def __init__(self):
		self.total_num_configs = 4726
		self.robot_occ_maps = np.load('bitmaps/robot_occ.npz')
		self.configs = self.robot_occ_maps['configs']
		self.robot_occ_maps = self.robot_occ_maps['occ_data']
		self.obs_occs = np.load('bitmaps/cylinder_occs.npz')
		fsv = np.load('bitmaps/fsv.npz')
		self.full_swept_volume = fsv['fsv']
		self.fsv_sum = fsv['sum']
		

	def gen_data(self, num_frames, with_robot=False, num_obstacles=3, save_name=None):
		'''
		generate completely independent frames
		'''
		X = np.zeros((num_frames, 52*52*34+1), dtype='float32')
		y = np.zeros(num_frames, dtype='float32')
		assert with_robot==False, 'With Robot is not Implemented'
		for i in xrange(num_frames):
			while True: # generating an environment that collides some of the configuration
				env = np.zeros((52,52,34))
				for j in xrange(num_obstacles):
					pos_x, pos_y = [random.uniform(workspace_x_min, workspace_x_max), random.uniform(workspace_y_min, workspace_y_max)]
					x_idx = int((pos_x - workspace_x_min)/grid_res)
					y_idx = int((pos_y - workspace_y_min)/grid_res)
					cylinder_idx = random.choice(range(1,9))
					cylinder_map = self.obs_occs[str(cylinder_idx)]
					cylinder_map[:,0] += x_idx
					cylinder_map[:,1] += y_idx
					valid_idxs = reduce(np.logical_and, [cylinder_map[:,0]>=0, cylinder_map[:,1]>=0, cylinder_map[:,0]<52, cylinder_map[:,1]<52])
					cylinder_map = cylinder_map[valid_idxs]
					env[cylinder_map[:,0], cylinder_map[:,1], cylinder_map[:,2]] = 1
				if np.logical_or(env, self.full_swept_volume).sum()==env.sum() + self.fsv_sum: # no collision with any part of the path
					continue
				found_robot_config = False
				for _ in xrange(10):
					config_idx = int(random.random()*self.total_num_configs)
					robot_occ = self.robot_occ_maps[config_idx]
					if np.logical_xor(env, robot_occ).sum()==env.sum() + robot_occ.sum(): # no collision with current config
						found_robot_config = True
						break
				if found_robot_config:
					break
			X[i,1:] = env.flatten()
			X[i,0] = config_idx
			c_idx = config_idx + 1
			config_count = 0
			while True:
				if c_idx==self.total_num_configs:
					c_idx = 0
				cur_occ = self.robot_occ_maps[c_idx]
				if np.logical_xor(env, cur_occ).sum()!=env.sum() + cur_occ.sum(): # found collision
					break
				c_idx += 1
				config_count += 1
			y[i] = config_count
		if save_name is not None:
			np.savez_compressed(save_name, X=X, y=y)
		else:
			return X, y

def gen_data_top_level(idx):
	BitmapEnvironment().gen_data(50, save_name='data/data_%i.npz'%idx)

if __name__ == '__main__':
	i = 0
	while True:
		gen_data_top_level(i)