import cv2
import mediapipe as mp
import time

class handTracker():
    
    def __init__(self, mode=False, draw_hand=True, draw_position=True, numHands=2, detectLimit=0.6, trackLimit=0.6):
        self.mode = mode
        self.draw_hand = draw_hand
        self.draw_position = draw_position
        self.numHands = numHands
        self.detectLimit = detectLimit
        self.trackLimit = trackLimit
        

        # Initialize mediapipe to detect hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.numHands, self.detectLimit, self.trackLimit)
        self.mp_draw = mp.solutions.drawing_utils

    def detectHands(self, image):

        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert the image from BGR 2 RGB because RGB is suitble for mediapipe
        self.hand_results = self.hands.process(image_RGB)

        if self.hand_results.multi_hand_landmarks and self.draw_hand:
            for landmark in self.hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, landmark, self.mp_hands.HAND_CONNECTIONS) # Draw the landmark on the hand and connect them
        return image
        
    
    def detectPosition(self, image, handNo=0):
        landmarkList = []

        if self.hand_results.multi_hand_landmarks:
            myHand = self.hand_results.multi_hand_landmarks[handNo]

            for i, landm in enumerate(myHand.landmark):
                    h, w, _ = image.shape
                    cent_x, cent_y = int(landm.x * w), int(landm.y * h)
                    # print(i, cent_x, cent_y)
                    landmarkList.append([i,cent_x,cent_y])
                    if self.draw_position:
                        cv2.circle(image, (cent_x, cent_y), 10, (255,0,0), cv2.FILLED)

        return landmarkList

def main():
    prev_time = 0
    curr_time = 0
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
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

        text = "FPS:" + str(int(fps))
        cv2.putText(image, text, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

        cv2.imshow("Image", image)
        key = cv2.waitKey(1)
    print("Camera is turned off.")


if __name__ == "__main__":
    main()