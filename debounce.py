# ABOUTME: Debounces gesture detections to prevent rapid-fire triggering.
# ABOUTME: Requires a gesture to be held for N frames before firing, then enforces a cooldown.

import time

class Debouncer:
    def __init__(self, cooldown, threshold):
        self.last_gest = None
        self.frame_count = 0
        self.last_trig = 0
        self.cooldown = cooldown
        self.threshold = threshold
    
    def update(self, gesture):
        if gesture in ("No hands", "Unknown"):
            return None
        elif gesture != self.last_gest:
            self.frame_count = 0
            self.last_gest = gesture
        elif gesture == self.last_gest:
            self.frame_count += 1
            if self.frame_count == self.threshold and time.time() - self.last_trig > self.cooldown:
                self.last_trig = time.time()
                self.frame_count = 0
                return gesture
        else:
            return None