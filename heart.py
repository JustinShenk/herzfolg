#!/usr/bin/env python
from __future__ import print_function

import cv2
import numpy as np
import os
import sys

looping = True

CASCADE_PATH = 'face.xml'
cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Initialize camera
cv2.namedWindow('CamFace')
cam = cv2.VideoCapture(0)
print("Camera initialized")

opacity = 0.2
size = (480, 320)
screenwidth, screenheight = size
cam.set(3, screenwidth)
cam.set(4, screenheight)

tickcount = 0

_, im = cam.read()


def detect(im):
    global cascade

    frame_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50),
        flags=0
    )
    if len(rects) == 0:
        return []
    return rects if len(rects) == 1 else rects[0]


def color(face, im, overlay):
    if not len(face):
        return
    for x, y, w, h in face:
        if tickcount % 10 > 5:
            phase = tickcount % 10
            intensity = 3 - abs(phase - 7)  # { 2, 3, 2, 1 }
            cv2.ellipse(overlay, (x + w // 2, y + h // 2),
                        (int(w / 2.5), int(h / 2.2)), 0, 0, 360, (0, 40, 211), -1)
            cv2.addWeighted(overlay, opacity * 0.3 * intensity,
                            im, 1 - opacity * 0.3 * intensity, 0, im)


while looping:
    tickcount += 1
    print(tickcount)
    _, im = cam.read()
    face = detect(im)
    overlay = im.copy()
    color(face, im, overlay)

    cv2.imshow('CamFace', im)

    keypress = cv2.waitKey(5) & 0xFF
    if keypress != 255:
        print(keypress)
        if keypress == 32:  # Spacebar
            pass
        elif keypress == 113 or 27:  # 'q' pressed to quit
            print("Escape key entered")
            looping = False

# When everything is done, release the capture
cam.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
