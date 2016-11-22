import numpy as np

res_dataset = 'db/Results.csv'

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

    def Save_Results(self):
        import csv
        print "save"
        with open(res_dataset, 'wb') as csvfilereader:
            print "salvo"
            writer = csv.writer(csvfilereader, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in self.Mesh_data:
                writer.writerow([i])
            csvfilereader.close()

    def Set_Name(self, obj_name):
        if self.first_time is 0:
            self.Nome_finito = obj_name
            self.first_time = 1
        else:
            if self.Nome_finito == obj_name:
                self.n_simulation_obj += 1
            else:
                self.Mesh_data.append({'Name':self.Nome_finito,'n_simulation_obj':self.n_simulation_obj ,
                'grasp_succesful': self.grasp_succesful,'Percentage': (self.grasp_succesful / float(self.n_simulation_obj))*100})
                self.Nome_finito = obj_name
                self.n_simulation_obj = 1
                analysis.grasp_succesful = 0

    def Set_Nsimulation(self):
        self.n_simulation_total +=1

    def Get_Kindness(self, value, end_file):
        self.kindness.append(value)
        if end_file:
                print "dentro end_file"
                # Draw_Grasph(self.kindness)
                self.Mesh_data.append({'Name':self.Nome_finito,'n_simulation_obj':self.n_simulation_obj ,
                    'grasp_succesful': self.grasp_succesful,'Percentage': (self.grasp_succesful / float(self.n_simulation_obj))*100})
                self.Save_Results()
        return True

    def Get_Grasp_Status(self,status):
        boolean = str2bool(status)
        if boolean is 1:
            self.grasp_succesful += 1


import csv
import sys
obj_dataset = sys.argv[1]
with open(obj_dataset, 'rb') as csvfile:
    file_reader = csv.reader(csvfile, delimiter=',')
    analysis = DataAnalysis()
    count_comma = 0
    N_row_total = sum(1 for i in file_reader)
    N_row_count = 0
    print N_row_total
    csvfile.seek(0)
    for row in file_reader:
        N_row_count += 1
        analysis.Set_Name(row[0])
        analysis.Get_Grasp_Status(row[13])
        analysis.Get_Kindness(row[14],True if N_row_count == N_row_total else False)