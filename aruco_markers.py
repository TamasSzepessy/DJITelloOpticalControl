import cv2 as cv
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def make_markers():
    "Creates 4X4 ArUco Markers"
    # Stolen code from the ArUco tutorial
    aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_100)

    fig = plt.figure()
    nx = 2
    ny = 1
    for i in range(1, nx*ny+1):
        ax = fig.add_subplot(ny,nx, i)
        img = aruco.drawMarker(aruco_dict,i+2, 600)
        plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
        #cv.imwrite("markers/img_"+str(i)+".jpg", img)
        ax.axis("off")
    plt.savefig("markers/ArUco_Markers.pdf")
    plt.show()

make_markers()