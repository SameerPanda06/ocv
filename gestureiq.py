import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# =========================
# SETTINGS
# =========================
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

CAM_W, CAM_H = 640, 480
SMOOTHING = 0.25

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# =========================
# CAMERA SETUP
# =========================
cap = cv2.VideoCapture(0)
cap.set(3, CAM_W)
cap.set(4, CAM_H)

screen_w, screen_h = pyautogui.size()

# =========================
# CURSOR VARIABLES
# =========================
smooth_x, smooth_y = 0, 0
click_cooldown = 0

# FPS
pTime = 0

# =========================
# KALMAN FILTER
# =========================
kalman = cv2.KalmanFilter(4, 2)

kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], np.float32)

kalman.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)

kalman.processNoiseCov = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0.03, 0],
    [0, 0, 0, 0.03]
], np.float32)

# =========================
# HELPER FUNCTIONS
# =========================
def get_dist(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def all_pinched(lm):
    tips = [8, 12, 16, 20]

    cx = np.mean([lm[i].x for i in tips])
    cy = np.mean([lm[i].y for i in tips])

    thumb_near = get_dist(lm[4], lm[8]) < 0.08

    dists = [
        np.sqrt((lm[i].x - cx) ** 2 + (lm[i].y - cy) ** 2)
        for i in tips
    ]

    return thumb_near and all(d < 0.08 for d in dists), cx, cy

# =========================
# MAIN LOOP
# =========================
while True:

    success, frame = cap.read()

    if not success:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand detection
    result = hands.process(rgb)

    gesture = "IDLE"

    # =========================
    # HAND FOUND
    # =========================
    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            lm = hand_landmarks.landmark

            # Finger states
            index_up = finger_up(lm, 8, 6)
            middle_up = finger_up(lm, 12, 10)
            ring_up = finger_up(lm, 16, 14)
            pinky_up = finger_up(lm, 20, 18)

            # Pinch detection
            pinched, cx, cy = all_pinched(lm)

            # =========================
            # DOUBLE CLICK
            # =========================
            if pinky_up and not index_up and not middle_up and not ring_up:

                gesture = "DOUBLE CLICK"

                if click_cooldown == 0:
                    pyautogui.doubleClick()
                    click_cooldown = 30

            # =========================
            # SCREENSHOT
            # =========================
            elif ring_up and not index_up and not middle_up and not pinky_up:

                gesture = "SCREENSHOT"

                if click_cooldown == 0:
                    pyautogui.hotkey('win', 'shift', 's')
                    click_cooldown = 40

            # =========================
            # LEFT CLICK
            # =========================
            elif index_up and not middle_up and not ring_up and not pinky_up:

                gesture = "LEFT CLICK"

                if click_cooldown == 0:
                    pyautogui.click()
                    click_cooldown = 25

            # =========================
            # RIGHT CLICK
            # =========================
            elif middle_up and not index_up and not ring_up and not pinky_up:

                gesture = "RIGHT CLICK"

                if click_cooldown == 0:
                    pyautogui.rightClick()
                    click_cooldown = 25

            # =========================
            # CURSOR MOVEMENT
            # =========================
            elif pinched:

                gesture = "TRACKING ACTIVE"

                # Convert camera coords → screen coords
                tx = np.interp(cx, [0.1, 0.9], [0, screen_w])
                ty = np.interp(cy, [0.1, 0.9], [0, screen_h])

                # =========================
                # KALMAN FILTER PREDICTION
                # =========================
                measurement = np.array([[np.float32(tx)],
                                        [np.float32(ty)]])

                kalman.correct(measurement)

                prediction = kalman.predict()

                pred_x = prediction[0][0]
                pred_y = prediction[1][0]

                # Smooth movement
                smooth_x = smooth_x + SMOOTHING * (pred_x - smooth_x)
                smooth_y = smooth_y + SMOOTHING * (pred_y - smooth_y)

                # Move cursor
                pyautogui.moveTo(int(smooth_x), int(smooth_y))

                # Draw tracking point
                dot_x = int(cx * CAM_W)
                dot_y = int(cy * CAM_H)

                cv2.circle(frame, (dot_x, dot_y), 12, (0, 255, 0), -1)

    # =========================
    # COOLDOWN
    # =========================
    if click_cooldown > 0:
        click_cooldown -= 1

    # =========================
    # FPS CALCULATION
    # =========================
    cTime = time.time()

    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0

    pTime = cTime

    # =========================
    # UI COLORS
    # =========================
    color_map = {
        "TRACKING ACTIVE": (0, 255, 0),
        "LEFT CLICK": (0, 0, 255),
        "RIGHT CLICK": (255, 0, 0),
        "DOUBLE CLICK": (255, 0, 255),
        "SCREENSHOT": (0, 165, 255),
        "IDLE": (180, 180, 180)
    }

    # =========================
    # TITLE UI
    # =========================
    cv2.putText(
        frame,
        f"GestureIQ | {gesture}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color_map.get(gesture, (255, 255, 255)),
        2
    )

    # FPS Display
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Tracking status
    status = "TRACKING STABLE" if gesture == "TRACKING ACTIVE" else "WAITING"

    cv2.putText(
        frame,
        status,
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # =========================
    # SHOW WINDOW
    # =========================
    cv2.imshow("GestureIQ", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()