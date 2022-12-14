import HandTrackingModule as htm
import cv2
import mediapipe as mp
import time

prev_time = 0
curr_time = 0
cap = cv2.VideoCapture(0)
tracker = htm.handTracker()
tracker.draw_hand = True
tracker.draw_position = True

key = None

print("Press 'q' to quit!")
while key != ord('q'):
    success, image = cap.read()
    image  = tracker.detectHands(image)
    landmarkList = tracker.detectPosition(image)
    if len(landmarkList) != 0:
        print(landmarkList[4])
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    text = "Testing FPS:" + str(int(fps))
    cv2.putText(image, text, (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
    cv2.imshow("Testing Imape", image)
    key = cv2.waitKey(1)

print("Camera is turned off.")

