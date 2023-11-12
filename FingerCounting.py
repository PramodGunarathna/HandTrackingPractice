import cv2
import time
import os
import HandTrackingModule as HTM


wCam,hCam = 640,480

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)


mylist = os.listdir("FingerImages")
Audiolist = os.listdir("AudioList")
#print(mylist)
overlayList = []
pTime = 0
for imPath in mylist:
    image = cv2.imread(f"FingerImages/{imPath}")

    overlayList.append(image)

detector = HTM.handDetector(detectionCon=0.75)
tipId = [4,8,12,16,20]

while True:
    success,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    #print(lmList)

    if len(lmList) !=0 :

        finger = []

        if lmList[tipId[0]][1] < lmList[tipId[0]-1][1]:
            finger.append(1)
        else:
            finger.append(0)

        for id in range(1,5):

            if lmList[tipId[id]][2] < lmList[tipId[id]-2][2]:   # picture eke usa  maninne uda idan maximum height eka thiyenne yatama
                #print("Index Finger Open")
                finger.append(1)
            else:
                finger.append(0)
        #print(finger)
        totalFingers = finger.count(1)
        #print(totalFingers)
        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]



        cv2.rectangle(img,(20,200),(175,425),(200,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)





    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    #cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


    cv2.imshow("image",img)
    cv2.waitKey(1)