import numpy as np

def str2bool(st):
    '''From string to bool'''
    try:
        return ['false', 'true'].index(st.lower())
    except (ValueError, AttributeError):
        raise ValueError('no Valid Conversion Possible')

def Draw_Grasph(kindness):
    print "disegno"
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
        # self.parameter = ''

    def Write_Results(self,dataset,parameter):
        '''Write the dataset'''
        import csv
        with open(dataset, 'wb')  as csvfilereader:
            writer = csv.writer(csvfilereader, delimiter=',')
            for i in parameter:
                writer.writerow([i])
            csvfilereader.close()

    def Save_parameter(self):
        '''Save the parameters in the vector and in csv file for neural networks set'''
        self.Mesh_data.append({'Name':self.Nome_finito,'n_simulation_obj':self.n_simulation_obj ,
                'grasp_succesful': self.grasp_succesful,'Percentage': (self.grasp_succesful / float(self.n_simulation_obj))*100})

        if self.grasp_succesful > 0:
            data_set = 'CNN/NNSet/%s.csv'%self.Nome_finito
            self.Write_Results(data_set,self.poses)
        del self.poses[:]

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
        self.poses.append(pose)


import csv
import sys
obj_dataset = sys.argv[1] #object_dataset
res_dataset = sys.argv[2] #results_dataset
with open(obj_dataset, 'rb') as csvfile: #open the file in read mode
    file_reader = csv.reader(csvfile, delimiter=',')
    analysis = DataAnalysis()
    for row in file_reader:
        analysis.Set_Name(row[0])
        analysis.Get_Grasp_Status(row[13], row[1:12])
        analysis.Get_Kindness(row[14])
    # Draw_Grasph(self.kindness)
    analysis.Save_parameter()
    analysis.Write_Results(res_dataset,analysis.Mesh_data,)
    analysis.Write_Results('CNN/NNSet/%s.csv'% analysis.Nome_finito,analysis.poses)