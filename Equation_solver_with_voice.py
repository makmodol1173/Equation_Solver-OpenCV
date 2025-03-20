import cv2
import mediapipe as mp
import numpy as np
import sympy as sp
import time
import pyttsx3

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


# Function to count extended fingers
def count_fingers(landmarks):
    tips = [8, 12, 16, 20]  # Index, middle, ring, pinky fingertips
    count = 0

    # Thumb detection
    if landmarks.landmark[4].x < landmarks.landmark[3].x:
        count += 1

    # Other fingers
    for tip in tips:
        if landmarks.landmark[tip].y < landmarks.landmark[tip - 2].y:
            count += 1

    return count


# OpenCV Video Capture
cap = cv2.VideoCapture(0)

equation = ""
last_number_time = time.time()
last_operator_time = time.time()


# Function to check time elapsed
def has_time_elapsed(last_time, seconds):
    return time.time() - last_time > seconds


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    detected_number = None
    detected_operator = None
    pinky_up_detected = False
    hand_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            avg_x = sum(x_coords) / len(x_coords)
            hand_list.append((avg_x, hand_landmarks))

        hand_list.sort(key=lambda x: x[0])

        if len(hand_list) >= 1 and has_time_elapsed(last_number_time, 3):
            detected_number = count_fingers(hand_list[-1][1])  # Rightmost hand (numbers)
            last_number_time = time.time()
            speak(f"Number {detected_number}")

        if len(hand_list) >= 2 and has_time_elapsed(last_operator_time, 3):
            fingers_count = count_fingers(hand_list[0][1])  # Leftmost hand (operators)
            if fingers_count == 1:
                detected_operator = '+'
            elif fingers_count == 2:
                detected_operator = '-'
            elif fingers_count == 3:
                detected_operator = '*'
            elif fingers_count == 4:
                detected_operator = '/'
            elif fingers_count == 5:
                detected_operator = '('
            elif fingers_count == 0:
                detected_operator = ')'
            last_operator_time = time.time()
            speak(f"Operator {detected_operator}")

        # Detect pinky up gesture (for result calculation)
        for _, hand_landmarks in hand_list:
            if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:  # Pinky extended
                pinky_up_detected = True
                break

    if detected_number is not None:
        equation += str(detected_number)

    if detected_operator is not None:
        equation += detected_operator

    if pinky_up_detected:
        try:
            result = sp.sympify(equation)
            speak(f"Result is {result}")
        except:
            result = "?"
        equation = str(result)
    else:
        result = "?"

    cv2.putText(frame, f'Equation: {equation}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f'Result: {result}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Finger Gesture Calculator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()