from .helpers import FACIAL_LANDMARKS_IDXS
from .helpers import shape_to_np
import cv2 as cv
import numpy as np

class FaceAligner:
    def __init__(self, predictor, desiredLeftEye = (0.35, 0.35), desiredFaceWidth = 256, desiredFaceHeight = None)
        #defining all the variables
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth

        #To get the output image in square shape
        if desiredFaceHeight is None:
            self.desiredFaceHeight = desiredFaceWidth
    def align(self, image, gray, rect):
        #Converting to landmarks into a numpy array
        shape = self.predictor(gray, rect)
        shape = self.shape_to_np(shape)

        #Getting the left eye and right eye points
        (lstart, lend) = FACIAL_LANDMARKS_IDXS('left_eye')
        (rstart, rend) = FACIAL_LANDMARKS_IDXS('right_eye')
        lpoints = shape[lstart:lend]
        rpoints = shape[rstart:rend]

        #Find the center of each eye
        lcenter = lpoints.mean(axis = 0).astype('int')
        rcenter = rpoints.mean(axis = 0).astype('int')

        #Find the angle between both the centroids
        dx = lcenter[0] - rcenter[0]
        dy = lcenter[1] - rcenter[1]
        angle = np.degrees(np.arctan2(dy, dx)) - 180

        desiredRightEyeX = 1.0 - desiredLeftEye[0]
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        #Find the center of both the eyes
        eyesCenter = ((lcenter[0] + rcenter[0]) // 2, (lcenter[1] + rcenter[1]) // 2)

        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output
