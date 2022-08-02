import cv2
import HandTrackingModule as htm


cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionConfidence=0.8)
colorR = (255,0,0)
# cx,cy,w,h = 40,40,120,120

class DragRect():
    def __init__(self,posCenter,size = [200,200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx,cy=self.posCenter
        w,h=self.size
        # If index finger tip is in rectangle region then change pos of box
        if cx - w // 2 < cursor[1] < cx + h // 2 and cy - h // 2 < cursor[2] < cy + h // 2:
            # colorR = (0, 255, 0)
            self.posCenter = cursor[1], cursor[2]

rectList=[]
for x in range(2):
    rectList.append(DragRect([x*250+150,150]))

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        l,_,_ = detector.findDistance(lmList[8], lmList[12], img, draw = False)
        # print(l)
        if l < 35:
            cursor = lmList[8]  # index finger tip landmark
            # Call update here
            for rect in rectList:
                rect.update(cursor)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(img, (cx-w//2,cy-h//2), (cx+w//2,cy+h//2), colorR,cv2.FILLED)
    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
