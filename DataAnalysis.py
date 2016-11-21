import numpy as np
# from mvbb.dataAnalysislog import DanyDataAnalysis
obj_dataset = 'db/testdataanalysis.csv'
# obj_dataset = 'db/database_scoredreflex_DanyFinito.csv'

def str2bool(st):
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
        self.parameter = ''
        # self.datares = DanyDataAnalysis(suffix='results')
    def Save_Results(self):
        import csv
        with open('db/dataAnalysislog.csv', 'wb') as csvfilereader:
            writer = csv.writer(csvfilereader, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in self.Mesh_data:
                writer.writerow([i])
            csvfilereader.close()



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
                self.Mesh_data.append({'Name':self.Nome_finito,'n_simulation_obj':self.n_simulation_obj ,
                'grasp_succesful': self.grasp_succesful,'Percentage': (self.grasp_succesful / float(self.n_simulation_obj))*100})
                self.Nome_finito = self.parameter
                self.parameter = ''
                self.n_simulation_obj = 1
                analysis.grasp_succesful = 0

    def Set_Nsimulation(self):
        self.n_simulation_total +=1

    def Get_Kindness(self, end_file):
        self.kindness.append(self.parameter)
        if end_file:
                # Draw_Grasph(self.kindness)
                self.Mesh_data.append({'Name':self.Nome_finito,'n_simulation_obj':self.n_simulation_obj ,
                    'grasp_succesful': self.grasp_succesful,'Percentage': (self.grasp_succesful / float(self.n_simulation_obj))*100})
                self.Save_Results()
        self.parameter = ''
        return True

    def Get_Grasp_Status(self):
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
    N_row_total = sum(1 for i in file_reader)
    csvfile.seek(0)
    N_row_count = 0
    count_par = 0
    for row in file_reader:
        N_row_count +=1
        analysis.Set_Nsimulation()
        n_par = len(range(0,len(row[0])))
        for i in range(0,len(row[0])):
            count_par += 1
            if row[0][i] is',':
                count_comma +=1
                if count_comma is 1:
                    analysis.Set_Name()
                elif count_comma <= 13:
                    analysis.parameter = ''
                if count_comma is 14: #finito di leggere grasp status e inizio con kindness
                    analysis.Get_Grasp_Status( )
            else:
                analysis.parameter += row[0][i]
                if count_comma is 14 and count_par is n_par:
                    analysis.Get_Kindness(True if N_row_count == N_row_total else False)
                    count_comma = 0
                    count_par = 0
                    break