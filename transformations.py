import numpy as np
import math
import cv2
#import cam_class
from scipy.spatial.transform import Rotation

# Calculate transformation matrices and average them
def getTransformations(tvec_m, tvec_n, rvec_m, rvec_n, tvec_orig_m, tvec_orig_n, rvec_orig_m,
                       rvec_orig_n, dRot_m, dRot_n, allow_use, ALLOW_LIMIT, tvec_max_n, tvec_min_n):
    if allow_use < ALLOW_LIMIT:
        tvec_m = np.transpose(tvec_m) # tvec of 'm' marker
        tvec_n = np.transpose(tvec_n) # tvec of 'n' marker
        tvec_orig_m = np.transpose(tvec_orig_m) # origin of 'm' in global coordinates
        tvec_orig_n = np.transpose(tvec_orig_n) # origin to be of 'n' in global coordinates
        dtvec = tvec_m - tvec_n # vector from 'm' to 'n' marker in the camera's coordinate system
        # get the markers' rotation matrices respectively
        R_m = cv2.Rodrigues(rvec_m)[0]
        R_n = cv2.Rodrigues(rvec_n)[0]

        tvec_temp = tvec_orig_m + dRot_m.dot(-R_m.T.dot(dtvec))
        if np.linalg.norm(tvec_temp) > np.linalg.norm(tvec_max_n):
            tvec_max_n = tvec_temp
        if np.linalg.norm(tvec_temp) < np.linalg.norm(tvec_min_n):
            tvec_min_n = tvec_temp

        tvec_orig_n = tvec_orig_n + tvec_temp

        dRot_n = dRot_n + dRot_m.dot(R_m.T.dot(R_n))

        # rotations
        rvec_m = np.transpose(rvec_m)
        rvec_n = np.transpose(rvec_n)
        rvec_orig_m = np.transpose(rvec_orig_m)
        rvec_orig_n = np.transpose(rvec_orig_n)
        drvec = rvec_m - rvec_n
        rvec_orig_n = rvec_orig_n + rvec_orig_m + dRot_m.dot(-R_m.T.dot(drvec))

        allow_use += 1
    
    if allow_use == ALLOW_LIMIT:
        tvec_mm = tvec_orig_m + dRot_m.dot(-R_m.T.dot(tvec_m)) # camera pose in 'm' marker's coordinate system
        tvec_nn = -R_n.T.dot(tvec_n) # camera pose in 'n' marker's coordinate system
        tvec_orig_n = tvec_orig_n - tvec_max_n - tvec_min_n
        tvec_orig_n = tvec_orig_n/(ALLOW_LIMIT-2) # the origin of 'n' in global
        dRot_n = dRot_n/ALLOW_LIMIT # rotation matrix from 'n' to global
        tvec_nm = tvec_orig_n + dRot_n.dot(tvec_nn) # camera pose from 'n' to global

        rvec_orig_n = rvec_orig_n/ALLOW_LIMIT
        print("rvec_orig_n:")
        print(rotationVectorToEulerAngles(rvec_orig_n)*180/math.pi)
        print("tvec_mm:")
        print(tvec_mm)
        print("tvec_nm:")
        print(tvec_nm)
    
    tvec_orig_n = np.transpose(tvec_orig_n)
    rvec_orig_n = np.transpose(rvec_orig_n)

    return tvec_orig_n, dRot_n, rvec_orig_n, allow_use, tvec_max_n, tvec_min_n

# Calculate position data from stored values and current values
def calculatePos(tvec, rvec, tvec_orig, dRot):
    tvec = np.transpose(tvec)
    tvec_orig = np.transpose(tvec_orig)
    R = cv2.Rodrigues(rvec)[0]
    tvec = -R.T.dot(tvec)
    tvec = tvec_orig + dRot.dot(tvec)
    tvec = np.transpose(tvec)
    return tvec

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