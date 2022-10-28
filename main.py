import cv2
from cvzone.PoseModule import PoseDetector 

'''cvzone is a cv package  that makes its easy to run Image processing and AI functions. 
 At the core it uses OpenCV and Mediapipe libraries.
 Google's ML Kit Pose Detection to detect the pose of a subject's body in real time 
 from a continuous video or static image'''

cap = cv2.VideoCapture("Video.mp4") #it is a method of cv2 to capture the visdeo and save it in cap variable

detector = PoseDetector() 
posList = []    #it will store the coordinates of the moving subject
while True:
    success,img = cap.read() 
    ''' cap.read() returns is a boolean (True/False) and image content. 
    If you remove success, the img variable takes that boolean and image data as a tuple. 
    This is why you get an error.'''
    img = detector.findPose(img) #it will find poses of the subject and stoores them in img
    
    lmList,bboxInfo = detector.findPosition(img)
    
    if bboxInfo:
        lmString = '' #land marks list
        for lm in lmList:   
            #print(lm)
            # lm --> [num,x,y,z]
            # opencv uses starting point top left
            # unity uses starting point from bottom left
            lmString += f'{lm[1]},{img.shape[0]-lm[2]},{lm[3]},'
            
        posList.append(lmString)
    
    print(len(posList))
    
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    # press to save animation coordinates in a file
    if key == ord('s'):
        with open("AnimationFile.txt",'w') as f:
            f.writelines(["%s\n"% item for item in posList])