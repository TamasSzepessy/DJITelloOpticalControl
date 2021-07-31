import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation

# Calculate transformation matrices and average them
def getTransformations(n_id, tvec_m, tvec_n, rvec_m, rvec_n, tvec_orig_m, tvec_orig_n, rvec_orig_m,
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

        # for filtering out min and max values
        tvec_temp = tvec_orig_m + dRot_m.dot(-R_m.T.dot(dtvec))
        if np.linalg.norm(tvec_temp) > np.linalg.norm(tvec_max_n):
            tvec_max_n = tvec_temp
        if np.linalg.norm(tvec_temp) < np.linalg.norm(tvec_min_n):
            tvec_min_n = tvec_temp

        tvec_orig_n = tvec_orig_n + tvec_temp

        dRot_n = dRot_n + dRot_m.dot(R_m.T.dot(R_n))

        # rotations
        # rvec_m = np.transpose(rvec_m)
        # rvec_n = np.transpose(rvec_n)
        # rvec_orig_m = np.transpose(rvec_orig_m)
        # rvec_orig_n = np.transpose(rvec_orig_n)
        # drvec = rvec_m - rvec_n
        # rvec_orig_n = rvec_orig_n + rvec_orig_m - dRot_m.dot(-R_m.T.dot(drvec))

        allow_use += 1
    
    if allow_use == ALLOW_LIMIT:
        #tvec_mm = tvec_orig_m + dRot_m.dot(-R_m.T.dot(tvec_m)) # camera pose in 'm' marker's coordinate system
        #tvec_nn = -R_n.T.dot(tvec_n) # camera pose in 'n' marker's coordinate system
        tvec_orig_n = tvec_orig_n - tvec_max_n - tvec_min_n
        tvec_orig_n = tvec_orig_n/(ALLOW_LIMIT-2) # the origin of 'n' in global
        dRot_n = dRot_n/ALLOW_LIMIT # rotation matrix from 'n' to global
        #tvec_nm = tvec_orig_n + dRot_n.dot(tvec_nn) # camera pose from 'n' to global

        rvec_orig_n = rotationMatrixToRotationVector(-dRot_n.T) # the orientation of 'n'
        # print("rvec_orig_n:")
        # print(rotationVectorToEulerAngles(rvec_orig_n)*180/math.pi)
        # print("tvec_mm:")
        # print(tvec_mm)
        # print("tvec_nm:")
        # print(tvec_nm)
        print("Marker "+str(n_id)+" transformations calculated")
    
    # transose for row vector
    tvec_orig_n = np.transpose(tvec_orig_n)
    #rvec_orig_n = np.transpose(rvec_orig_n)

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

# Get position in marker's coordinate system
def TranslationInMarker(rvec, tvec):
    tvec = np.transpose(tvec)
    R = cv2.Rodrigues(rvec)[0]
    tvec = -R.T.dot(tvec)
    tvec = np.transpose(tvec)
    return tvec

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6 
 
# Calculates rotation vector to euler angles
def rotationVectorToEulerAngles(rvec):
    r = Rotation.from_rotvec(rvec)
    return r.as_euler('xyz')

# Calculates rotation matrix to rotation vector
def rotationMatrixToRotationVector(dR):
    r = Rotation.from_dcm(dR)
    return r.as_rotvec()

# r = Rotation.from_rotvec(np.array([0.4472136,-0.4472136,-0.77459667]))
# r = Rotation.from_dcm(np.array([[0,0.5,-math.sqrt(3)/2],[-1,0,0],[0,math.sqrt(3)/2,0.5]]))
# print(r.as_euler('zyz')*180/np.pi)