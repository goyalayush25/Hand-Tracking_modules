import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self,mode=False, maxHands = 2,modelComplex = 1, detectionConfidence = 0.5, trackConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplex
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLandmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks((img),handLandmark, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo = 0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)  # since lm.x and lm.y are in decimals so to obtain pixels they are multiplied with w and h
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                # if id == 4:
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,255),cv2.FILLED)

        return lmList

    def findDistance(self,p1, p2, img=None, draw=True):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1[1],p1[2]
        x2, y2 = p2[1],p2[2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            if draw:
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
            return length,info, img
        else:
            return length, info



def main():
    # To know fps
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == "__main__":
    main()