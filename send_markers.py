#!/usr/bin/env python3
import sys, termios, tty, time
from pylsl import StreamInfo, StreamOutlet

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

info = StreamInfo('ExperimentMarkers', 'Markers', 1, 0, 'string', 'marker123')
outlet = StreamOutlet(info)

print("Marker-Stream gestartet (ohne Root).")
print("Tasten:  b=Baseline  r=Recovery  1=Stress1  2=Stress2  3=Stress3  q=Quit")

while True:
    ch = getch()
    if ch == 'b':
        outlet.push_sample(["Baseline"]); print("Marker: Baseline")
    elif ch == 'r':
        outlet.push_sample(["Recovery"]); print("Marker: Recovery")
    elif ch == '1':
        outlet.push_sample(["Stress1"]); print("Marker: Stress1")
    elif ch == '2':
        outlet.push_sample(["Stress2"]); print("Marker: Stress2")
    elif ch == '3':
        outlet.push_sample(["Stress3"]); print("Marker: Stress3")
    elif ch in ('q', '\x03', '\x1b'):   # q, Ctrl-C, ESC
        print("Beende Marker-Stream."); break
