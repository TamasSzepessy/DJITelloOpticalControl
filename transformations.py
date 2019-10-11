import numpy as np
import math
import cv2

def RotateXYZ(self, pitch, roll, yaw, x, y, z):
        rotx=np.array([[1, 0, 0],[0, math.cos(pitch), -math.sin(pitch)],[0, math.sin(pitch), math.cos(pitch)]])
        roty=np.array([[math.cos(roll), 0, math.sin(roll)],[0, 1, 0],[-math.sin(roll), 0, math.cos(roll)]])
        rotz=np.array([[math.cos(pitch), -math.sin(pitch), 0],[math.sin(pitch), math.cos(pitch), 0],[0, 0, 1]])
        rot=np.matmul(roty,rotz)
        rot=np.matmul(rotx,rot)
        temp = np.matmul(np.column_stack((x,y,z)),rot)
        xr, yr, zr = temp[:,0], temp[:,1], temp[:,2]

        return xr, yr, zr

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