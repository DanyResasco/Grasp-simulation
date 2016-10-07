#!/usr/bin/env python

import csv
from klampt.math import se3
import numpy as np

class MVBBLoader(object):
    def __init__(self, new_method = False):
        self.new_method = new_method
        self.filename = 'db/database.csv' if not new_method else 'db/database_populated.csv'
        self.db = {}

    def _load_mvbbs(self):
        try:
            f = open(self.filename)
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

    def get_poses(self, object_name):
        if self.db == {}:
            self._load_mvbbs()
        if object_name in self.db:
            return self.db[object_name]
        return []