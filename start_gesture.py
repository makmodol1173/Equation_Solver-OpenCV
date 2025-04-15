import cv2
import mediapipe as mp
import numpy as np


def detect_thumbs_up(hand_landmarks):
    mp_hands = mp.solutions.hands
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    return thumb_tip.y < thumb_ip.y < index_mcp.y  # Thumb is raised above the index MCP


def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    thumbs_up_count = 0  # Count thumbs up hands detected
    display_text = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            thumbs_detected = 0
            for hand_landmarks in result.multi_hand_landmarks:
                if detect_thumbs_up(hand_landmarks):
                    thumbs_detected += 1
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if thumbs_detected >= 2:  # Both hands showing thumbs up
                thumbs_up_count += 1
                if thumbs_up_count >= 2:
                    display_text = "Start"
                    thumbs_up_count = 0  # Reset after detecting thumbs up

        if display_text:
            cv2.putText(frame, display_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
