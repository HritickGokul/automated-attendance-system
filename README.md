## Automated Attendance System
Real-time multiple Face Detection and Recognition using OpenCV and Python. 

I trained the model to recognise my family members using a pipeline of multiple steps. The first step is face detection, which is done using haar cascade and frontal face detector. The second step is to find the landmarks ( using shape_predictor_68_face_landmarks) in the face and align them to get more accurate results. And the final step is to find 128 unique measurements in each face and use it to classify the face using a classifier (Support Vector Machine classifier)

I am currently working on this project and trying to improve the accuracy so that the model can be used for future projects.

Tools Used
-Python, OpenCV, dlib
-Scikit-learn, NumPy, pandas
-Atom environment
