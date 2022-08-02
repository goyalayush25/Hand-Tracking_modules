import cv2
import numpy as np

# Found using colorDetection
myColor=[[93,71,107,161,255,255]] # We can add more colors here if we want to
myColorValues=[[255,0,127]]  #  for purple
myPoints=[] # [x,y,colorId]

def findColor(img,myColor,myColorValues):
    newPoints=[]
    count=0
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    for color in myColor:
        lower = np.array(myColor[0][0:3])
        upper = np.array(myColor[0][3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x,y=getContours(mask)
        cv2.circle(imgResult,(x,y),5,myColorValues[count],cv2.FILLED)
        if x!=0 and y!=0:
            newPoints.append([x,y,count])
        count+=1
        # cv2.imshow("img",mask)
    return newPoints

def getContours(img):
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h=0,0,0,0
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>500:
            # cv2.drawContours(imgResult,cnt,-1,(0,0,255),2)
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h=cv2.boundingRect(approx)
    return x+w//2,y

def draw(myPoints,myColorValues):
    for point in myPoints:
        cv2.circle(imgResult,(point[0],point[1]),10,myColorValues[point[2]],cv2.FILLED)

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)  # For Brightness
while True:
    frame, img = cap.read()
    img=cv2.flip(img,1)
    imgResult=img.copy()
    newPoints = findColor(img,myColor,myColorValues)
    if len(newPoints)!=0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints)!=0:
        draw(myPoints,myColorValues)
    cv2.imshow("Image", imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
