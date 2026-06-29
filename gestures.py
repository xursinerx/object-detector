# ABOUTME: Defines recognized hand gestures as finger-state patterns.
# ABOUTME: Provides landmark analysis to determine which fingers are extended.

gestures = {
    (True, True, True, True, True): "open hand",
    (False, False, False, False, False): "fist",
    (False, True, False, False, False): "point",
    (False, True, True, False, False): "peace",
    (True, True, False, False, False): "gun",
}

def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    fingers = []
    for i, (t, p) in enumerate(zip(tips, pips)):
        if i == 0:
            fingers.append(hand_landmarks.landmark[t].x < hand_landmarks.landmark[p].x)
        else:
            fingers.append(hand_landmarks.landmark[t].y < hand_landmarks.landmark[p].y)
    return fingers