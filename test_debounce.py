from debounce import Debouncer
import time

bounce = Debouncer(1, 3)
result = None
for i in range(4):
    result = bounce.update("open hand")
print(f"Test 1 - should trigger 'open hand': {result}")

bounce = Debouncer(cooldown=1, threshold=3)
bounce.update("open hand")
bounce.update("open hand")
bounce.update("fist")
result = bounce.update("fist")
print(f"Test 2 - Should be None: {result}")

bounce = Debouncer(cooldown=1, threshold=3)
for i in range(4):
    result = bounce.update("open hand")
for i in range(4):
    result = bounce.update("fist")
print(f"Test 3 - should not trigger 'fist': {result}")

bounce = Debouncer(cooldown=0.5, threshold=3)
for i in range(4):
    bounce.update("open hand")
time.sleep(0.6)
result = None
for i in range(4):
    result = bounce.update("fist")
print(f"Test 4 - should trigger 'fist': {result}")

bounce = Debouncer(cooldown=1, threshold=3)
for i in range(4):
    result = bounce.update("Unknown")
for i in range(4):
    result = bounce.update("fist")
print(f"Test 5 - should trigger 'fist': {result}")