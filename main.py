import cv2
import numpy as np
import matplotlib.pyplot as plt

PLOT_BUFFER = []
MAX_BUFFER_SIZE = 180

prev_frame = None

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()

    light = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HLS))[1]

    light = cv2.GaussianBlur(light, (49, 49), 0)
    mask = cv2.threshold(light, 250, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

    PLOT_BUFFER.append(np.sum(mask))
    if len(PLOT_BUFFER) == MAX_BUFFER_SIZE:
        PLOT_BUFFER.pop(0)
    thresh = np.mean(PLOT_BUFFER)

    plt.clf()
    plt.plot(PLOT_BUFFER, color='b')
    plt.axhline(thresh, color='r')
    plt.pause(0.01)

    cv2.imshow('Mask', mask)
    cv2.imshow('Result', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()


