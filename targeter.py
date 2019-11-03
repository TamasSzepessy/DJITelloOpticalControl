import csv
import numpy as np

class TargetDefine():
    def __init__(self):
        with open('marker_nav.csv', 'rt', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            self.marker_nav = list(reader)

    def changeTarget(self, ID):
        selected = 'Origin'
        for i in self.marker_nav:
            if i[0] == str(ID):
                selected = i[1]
                break

        print(selected + " marker")

        switcher={
                'Origin':                np.array([[0., 0., 0.8, 0.]]),
                'Right sideways':        np.array([[0., 0., 0.8, -40.]]),
                'Left sideways':         np.array([[0., 0., 0.8, 40.]]),
                'Rotate right corner 1': np.array([[0., 0., 0.8, 5.]]),
                'Rotate right corner 2': np.array([[0., 0., 0.8, -10.]]),
                'Rotate right corner 3': np.array([[0., 0., 0.8, -20.]]),
                'Rotate left corner 1':  np.array([[0., 0., 0.8, -5.]]),
                'Rotate left corner 2':  np.array([[0., 0., 0.8, 10.]]),
                'Rotate left corner 3':  np.array([[0., 0., 0.8, 20.]]),
                'End':                   np.array([[0., 0., 0.8, 0.]])
             }
        return switcher.get(selected, "Invalid marker type")

# target = TargetDefine()
# print(target.changeTarget(50))
