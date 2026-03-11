"""
keep_awake.py
Moves the mouse cursor by 1 pixel every 60 seconds to prevent Windows sleep.
Run this in a terminal and leave it running overnight.
Kill it manually with Ctrl+C when done.
"""
import time
import ctypes

print("Keep-awake script running. Press Ctrl+C to stop.")
i = 0
while True:
    # Simulate tiny mouse movement using Windows API (no external libraries needed)
    ctypes.windll.user32.mouse_event(0x0001, 1, 0, 0, 0)   # move +1
    ctypes.windll.user32.mouse_event(0x0001, -1, 0, 0, 0)  # move back
    i += 1
    print(f"  tick {i} — {time.strftime('%H:%M:%S')}")
    time.sleep(60)