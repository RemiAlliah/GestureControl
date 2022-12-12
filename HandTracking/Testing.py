import HandTrackingModule as htm
import cv2
import mediapipe as mp
import time

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
tracker = htm.handTracker()
tracker.draw_hand = True
tracker.draw_position = True
while True:
    isSuccess, img = cap.read()
    img  = tracker.detectHands(img)
    lmList = tracker.detectPosition(img)
    if len(lmList) != 0:
        print(lmList[4])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    text = "Testing FPS:" + str(int(fps))
    cv2.putText(img, text, (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
    cv2.imshow("Testing Imape", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        quit()