# MIDI
from __future__ import print_function
import sys
import rtmidi
from mido import Message

midiout = rtmidi.MidiOut()
available_ports = midiout.get_ports()
print(available_ports)
midiout.open_port(1)

from datetime import datetime, date
import numpy as np
import cv2
import time, random
import os

cap = cv2.VideoCapture('D:\OBSRR\Video.mov')

areamin = 200  # spalvos detektinimo plotas  
areamax = 1000  # spalvos detektinimo plotas

# -----------------------------------------------------------
interested = ['car', 'truck', 'bus']
# -----------------------------------------------------------

classNames = []
classFile = 'Resources/coco.names'
with open(classFile) as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'Resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'Resources/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

filename = 'video.avi'
frames_per_second = 24.0
res = '720p'


# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

    # Standard Video Dimensions Sizes


STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height


# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))

while True:
    success, img = cap.read()
    thresh = 0.6
    classIds, confs, bbox = net.detect(img, confThreshold=thresh)
    if interested is None:
        interested = classNames

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            label = classNames[classId - 1]
            if label in interested:
                cv2.rectangle(img, box, color=(0, 200, 0), thickness=1)
                cv2.putText(img, classNames[classId - 1].lower(), (int(box[0]), int(box[1] - 20)), 0, 0.70, (0, 200, 0),
                            1)
                cv2.putText(img, str(round(confidence * 100, 2)) + '%', (int(box[0]), int(box[1] - 5)), 0, 0.40,
                            (0, 200, 0), 1)

                notes = [48, 52, 55]
                if label == "car":
                    freq = random.choice(notes)
                    note_on = [0x90, freq, 112]
                    note_off = [0x80, freq, 0]
                    midiout.send_message(note_on)
                    time.sleep(0.001)
                    midiout.send_message(note_off)

    imh = cv2.imshow('Output', img)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        del midiout
        break

cap.release()
out.release()
cv2.destroyAllWindows()
