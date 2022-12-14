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

        self.hand_landmarks = self.hand_results.multi_hand_landmarks
        if self.hand_landmarks and self.draw_hand:
            for i in range(len(self.hand_landmarks)):
                # Draw the landmark on the hand and connect them
                self.mp_draw.draw_landmarks(image, self.hand_landmarks[i], self.mp_hands.HAND_CONNECTIONS)

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
                        cv2.circle(image, (cent_x, cent_y), 8, (120,120,0), cv2.FILLED)

        return landmarkList
