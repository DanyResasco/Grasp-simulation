import numpy as np

# obj_dataset = 'db/database_scorereflex.csv'
obj_dataset = 'db/testdataanalysis.csv'

def str2bool(st):
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
        # self.index = 0
        # self.nome = ''
        self.grasp_succesful = 0
        self.Mesh_data = []
        self.first_time = 0
        # self.grasp_status = ''
        self.kindness = ''
        self.parameter = ''
    
    def Set_Name(self):
        if self.first_time is 0:
            self.Nome_finito = self.parameter
            self.parameter = ''
            self.first_time = 1
        else:
            if self.Nome_finito == self.parameter:
                self.n_simulation_obj += 1
                self.parameter = ''
            else:
                self.Mesh_data.append({'Name':self.Nome_finito,'n_simulation_obj':self.n_simulation_obj ,'grasp_succesful': self.grasp_succesful})
                print self.Mesh_data
                self.Nome_finito = self.parameter
                self.parameter = ''
                self.n_simulation_obj = 1
                analysis.grasp_succesful = 0

    def Set_Nsimulation(self):
        self.n_simulation_total +=1

    def Get_Kindness(self):
        # Draw_Grasph(self.parameter)
        self.kindness = self.parameter
        self.parameter = ''
        return True

    def Get_Grasp_Status(self):
        # if valuate is 1:
            print self.parameter
            boolean = str2bool(self.parameter)
            if boolean is 1:
                self.grasp_succesful += 1
                self.parameter = ''
            else:
                self.parameter = ''
            return True

import csv

with open(obj_dataset, 'rb') as csvfile:
    file_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    analysis = DataAnalysis()
    count_comma = 0
    active_grasp_evaluate = 0
    valuate = 0
    stop = False

    for row in file_reader:
        analysis.Set_Nsimulation()
        for i in range(0,len(row[0])):
            if row[0][i] is',':
                count_comma +=1
                if count_comma is 1:
                    analysis.Set_Name()
                elif count_comma <= 13:
                    analysis.parameter = ''
                if count_comma is 14: #finito di leggere grasp status e inizio con kindness
                    analysis.Get_Grasp_Status()
            else:
                analysis.parameter += row[0][i]
                if count_comma is 14:
                    analysis.Get_Kindness()
                    count_comma = 0
                    break






    # for row in file_reader:
    #     analysis.Set_Nsimulation()
    #     for i in range(0,len(row[0])):
    #         if row[0][i] is',':
    #             count_comma +=1
    #             # print "n_comma: ", count_comma
    #             if count_comma is 1:
    #                 analysis.Set_Name()
    #             if count_comma is 13: # inizio a leggere grasp status
    #                 active_grasp_evaluate = 1
    #             if count_comma is 14: #finito di leggere grasp status e inizio con kindness
    #                 valuate = 1

    #         else:
    #             if count_comma < 1:
    #                 analysis.nome += row[0][i]
    #             if active_grasp_evaluate is 1:
    #                 print "entro if"
    #                 stop = analysis.Get_Grasp_Status()
    #                 if stop:
    #                     count_comma = 0
    #                     active_grasp_evaluate = 0
    #                     print "qui"
    #                     stop = False
    #                     valuate = 0
    #                     break