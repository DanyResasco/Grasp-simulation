import numpy as np
from klampt.math import se3
'''Write two output dataset:
    1) is composed by: obj_name, n_simulation_total, n_simulation_succesfull, Percentage
    2) is composed by: obj_name, each pose succesfull founded
    In input require the csv dataset composed by: obj_name, poses, grasp status (true or false) and kindness'''


def str2bool(st):
    '''From string to bool'''
    try:
        return ['false', 'true'].index(st.lower())
    except (ValueError, AttributeError):
        raise ValueError('no Valid Conversion Possible')

def Draw_Grasph(kindness):
    import matplotlib.pyplot as plt
    plt.plot(kindness)
    plt.ylabel('kindness')
    plt.xlabel('poses')
    plt.show()

class DataAnalysis():
    def __init__(self):
        self.n_simulation_total = 0
        self.n_simulation_obj = 1
        self.grasp_succesful = 0
        self.Mesh_data = []
        self.first_time = 0
        self.grasp_status = ''
        self.kindness = []
        self.poses = []
        self.count = 0

    def Write_Poses(self,dataset,parameter):
        '''Write the dataset'''
        import csv
        f = open(dataset, 'w')
        for i in range(0,len(self.poses)):
            f.write(','.join([str(v) for v in self.poses[i]]))
            f.write('\n')
        f.close()


    def Write_Results(self,dataset,parameter):
        '''Write the dataset'''
        import csv
        with open(dataset, 'wb')  as csvfilereader:
            writer = csv.writer(csvfilereader, delimiter=',')
            for i in parameter:
                writer.writerow([i])
            csvfilereader.close()


    def Save_parameter(self):
        '''Save the parameters in the vector and if the grasp is true in csv file for neural networks set'''

        self.Mesh_data.append({'Name':self.Nome_finito,'n_simulation_obj':self.n_simulation_obj ,
                'grasp_succesful': self.grasp_succesful,'Percentage': (self.grasp_succesful / float(self.n_simulation_obj))*100})

        if self.grasp_succesful > 0:
            data_set = '3DCNN/NNSet/Pose/%s.csv'%self.Nome_finito
            self.Write_Poses(data_set,self.poses)
        # del self.poses[:]
            if (self.grasp_succesful / float(self.n_simulation_obj))*100 > 50.0:
                data_set = '3DCNN/NNSet/Percentage/%s.csv'%self.Nome_finito
            else:
                data_set = '3DCNN/NNSet/Nsuccessfull/%s.csv'%self.Nome_finito
            self.Write_Poses(data_set,self.poses)
            del self.poses[:]
        else:
            # print "No grasp_succesful in ",self.Nome_finito
            self.count +=1


    def Set_Name(self, obj_name):
        if self.first_time is 0:
            self.Nome_finito = obj_name
            self.first_time = 1
        else:
            if self.Nome_finito == obj_name:
                self.n_simulation_obj += 1
            else:
                self.Save_parameter()
                self.Nome_finito = obj_name
                self.n_simulation_obj = 1
                self.grasp_succesful = 0
                del self.kindness[:]

    def Set_Nsimulation(self):
        self.n_simulation_total +=1

    def Get_Kindness(self, value):
        self.kindness.append(value)

    def Get_Grasp_Status(self,status,poses):
        boolean = str2bool(status)
        if boolean is 1:
            self.grasp_succesful += 1
            self.Get_Pose(poses)

    def Get_Pose(self,pose):
        pose_temp = [float(v) for v in pose]
        self.poses.append(pose_temp)


import csv
import sys
obj_dataset = sys.argv[1] #input file with obj and poses
res_dataset = sys.argv[2] #output results_dataset
with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
    file_reader = csv.reader(csvfile, delimiter=',')
    analysis = DataAnalysis()
    for row in file_reader:
        analysis.Set_Name(row[0])
        analysis.Get_Grasp_Status(row[13], row[1:13])
        analysis.Get_Kindness(row[13])
        # Draw_Grasph(analysis.kindness)
    analysis.Save_parameter()
    analysis.Write_Results(res_dataset,analysis.Mesh_data)
    analysis.Write_Poses('3DCNN/NNSet/Pose/%s.csv'% analysis.Nome_finito,analysis.poses)
    if (analysis.grasp_succesful / float(analysis.n_simulation_obj))*100 > 50.0:
        data_set = '3DCNN/NNSet/Percentage/%s.csv'%analysis.Nome_finito
    else:
        data_set = '3DCNN/NNSet/Nsuccessfull/%s.csv'%analysis.Nome_finito
    analysis.Write_Poses(data_set,analysis.poses)
    print "Number of object with zeros poses ", analysis.count