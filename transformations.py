import numpy as np
import math
import cv2
import cam_class
from scipy.spatial.transform import Rotation

def TransformBetweenMarkers(tvec_m, tvec_n, rvec_m, rvec_n):
    tvec_m = np.transpose(tvec_m) # tvec of 'm' marker
    tvec_n = np.transpose(tvec_n) # tvec of 'n' marker
    dtvec = tvec_m - tvec_n # vector from 'm' to 'n' marker in the camera's coordinate system
    
    # get the markers' rotation matrices respectively
    R_m = cv2.Rodrigues(rvec_m)[0]
    R_n = cv2.Rodrigues(rvec_n)[0]

    tvec_mm = np.matmul(-R_m.T, tvec_m) # camera pose in 'm' marker's coordinate system
    tvec_nn = np.matmul(-R_n.T, tvec_n) # camera pose in 'n' marker's coordinate system
    # translational difference between markers in 'm' marker's system,
    # basically the origin of 'n'
    dtvec_m = np.matmul(-R_m.T, dtvec)

    # this gets me the same as tvec_mm,
    # but this only works, if 'm' marker is seen
    # tvec_nm1 = dtvec_m + np.matmul(-R_m.T, tvec_n)

    ## identical
    # something with the rvec difference must give the transformation(???)
    rvec_mm = np.transpose(np.matmul(-R_m.T, np.transpose(rvec_m)))
    rvec_nm = np.transpose(np.matmul(-R_m.T, np.transpose(rvec_n)))
    print(rvec_mm)
    drvec_mm = rvec_mm - rvec_nm
    dR_m = cv2.Rodrigues(drvec_mm)[0]
    # I want to transform tvec_nn with a single matrix,
    # so it would be interpreted in 'm' marker's system
    tvec_nm = dtvec_m + np.matmul(dR_m, tvec_nn)
    ##

    ## identical
    drvec = rvec_m - rvec_n
    drvec_m = np.transpose(np.matmul(R_m.T, np.transpose(drvec)))
    dR_m = cv2.Rodrigues(drvec_m)[0]
    tvec_nm2= dtvec_m + np.matmul(dR_m.T, tvec_nn)
    ##

    print("tvec_mm:")
    print(tvec_mm)
    print("tvec_nm:")
    print(tvec_nm)
    print("tvec_nm2:")
    print(tvec_nm2)
    # objective: tvec_mm == tvec_nm
    

def RotateXYZ(pitch, roll, yaw, x, y, z):
        rotx=np.array([[1, 0, 0],[0, math.cos(pitch), -math.sin(pitch)],[0, math.sin(pitch), math.cos(pitch)]])
        roty=np.array([[math.cos(roll), 0, math.sin(roll)],[0, 1, 0],[-math.sin(roll), 0, math.cos(roll)]])
        rotz=np.array([[math.cos(pitch), -math.sin(pitch), 0],[math.sin(pitch), math.cos(pitch), 0],[0, 0, 1]])
        rot=np.matmul(roty,rotz)
        rot=np.matmul(rotx,rot)
        temp = np.matmul(np.column_stack((x,y,z)),rot)
        xr, yr, zr = temp[:,0], temp[:,1], temp[:,2]

        return xr, yr, zr

def compensateTranslation(rvec, tvec, inv):
    R, _ = cv2.Rodrigues(rvec)
    if inv:
        R = R.T
        tvec=np.transpose(np.matmul(-R, np.transpose(tvec)))
    else:
        tvec=np.matmul(tvec,-R)
    return tvec

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6 
 
# Calculates rotation matrix to euler angles
def rotationVectorToEulerAngles(rvec):
    R, _ = cv2.Rodrigues(rvec)
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])