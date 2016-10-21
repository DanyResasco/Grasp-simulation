#!/usr/bin/env python

import csv
from klampt.math import se3
import numpy as np

class MVBBLoader(object):
    def __init__(self, new_method = False, suffix = ''):
        self.new_method = new_method
        self.filename = 'db/database.csv' if not new_method else 'db/database_populated.csv'
        self.filename_scored = 'db/database_scored%s.csv'%suffix if not new_method else 'db/database_populated_scored%s.csv'%suffix
        self.db = {}
        self.db_scored = {}
        self._load_mvbbs()
        self._load_mvbbs_scored()

    def _load_mvbbs(self):
        try:
            f = open(self.filename)
            self.db = {}
            reader = csv.reader(f)
            for row in reader:
                object_name = row.pop(0).strip()
                R = [float(v) for i, v in enumerate(row) if i in range(9)]
                t = [float(v) for i, v in enumerate(row) if i in range(9,12)]
                T = (R,t)
                if object_name not in self.db:
                    self.db[object_name] = []
                self.db[object_name].append(np.array(se3.homogeneous(T)))

        except:
            print "Error loading file", self.filename

    def _load_mvbbs_scored(self):
        try:
            f = open(self.filename_scored)
            reader = csv.reader(f)
            self.db_scored = {}
            for row in reader:
                object_name = row.pop(0).strip()
                R = [float(v) for i, v in enumerate(row) if i in range(9)]
                t = [float(v) for i, v in enumerate(row) if i in range(9,12)]
                T = (R,t)
                grasped = True if row[12] == 'True' else False
                kindness = None
                try:
                    if row[13] != 'x':
                        kindness = float(row[13])
                except:
                    pass

                if object_name not in self.db_scored:
                    self.db_scored[object_name] = []
                obj_pose_score = {'T': np.array(se3.homogeneous(T)),
                                  'grasped': grasped,
                                  'kindness': kindness}
                self.db_scored[object_name].append(obj_pose_score)
        except:
            print "Error loading file", self.filename_scored

    def save_score(self, object_name, pose, grasped, kindness = None):
        if not self.has_score(object_name, pose):
            if kindness is not None and self.has_score(object_name, pose, True):
                for i,p in enumerate(self.db_scored[object_name]):
                    if np.all(pose == p['T']):
                        self.db_scored[object_name][i] = {'T': pose,
                                                          'grasped': grasped,
                                                          'kindness': kindness}
                f = open(self.filename_scored, 'w')
                for pose_obj_name in self.db_scored:
                    values = [pose_obj_name]
                    pose = se3.from_homogeneous(self.db_scored[pose_obj_name]['T'])
                    values += pose[0]
                    values += pose[1]
                    grasped = self.db_scored[pose_obj_name]['grasped']
                    kindness = self.db_scored[pose_obj_name]['kindness']
                    values.append(grasped)
                    values.append(kindness if kindness is not None else 'x')
                    f.write(','.join([str(v) for v in values]))
                    f.write('\n')
            else:
                f = open(self.filename_scored, 'a')
                values = [object_name]
                if isinstance(pose, np.ndarray):
                    pose = se3.from_homogeneous(pose)
                values += pose[0]
                values += pose[1]
                values.append(grasped)
                values.append(kindness if kindness is not None else 'x')
                f.write(','.join([str(v) for v in values]))
                f.write('\n')
            f.close()
        self._load_mvbbs_scored()

    def has_score(self, object_name, pose, need_kindness = False):
        self._load_mvbbs_scored()
        if object_name in self.db_scored:
            if not isinstance(pose, np.ndarray):
                pose = np.array(se3.homogeneous(pose))
            poses = [p['T'] for p in self.db_scored[object_name] if not need_kindness or p['kindness'] is not None]
            for p in poses:
                if np.all((pose - p) < 1e-12):
                    return True
        return False

    def get_poses(self, object_name):
        if self.db == {}:
            self._load_mvbbs()
        if object_name in self.db:
            return self.db[object_name]
        return []

    def get_successful_poses(self, object_name):
        scored_poses = self.get_scored_poses(object_name)
        successful_poses = []
        for sp in scored_poses:
            if sp['grasped']:
                successful_poses.append(sp)
        return successful_poses

    def get_scored_poses(self, object_name, only_successful = True):
        if self.db_scored == {}:
            self._load_mvbbs_scored()
        if object_name in self.db_scored:
            return [ pose['T'] for pose in self.db_scored[object_name] if pose['grasped'] or only_successful == False]
        return []

    def get_all_scored_poses(self, only_successful = True):
        if self.db_scored == {}:
            self._load_mvbbs_scored()
        obj_poses = {}
        for object_name in self.db_scored:
            poses = [ pose['T'] for pose in self.db_scored[object_name] if pose['grasped'] or only_successful == False]
            if len(poses) > 0:
                obj_poses[object_name] = poses
        return obj_poses