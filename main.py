import os
import os.path as path
import time
import random

from threading import Thread
from PIL import ImageStat, Image
import cv2
import numpy as np
import statistics
import d3dshot

from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio

soundsDir = "./sounds/"
threshold = 24  # tune to your liking; 24 is a pretty good value
debounceCooldown = 2
debug = True  # show computer vision

capture = d3dshot.create(capture_output="numpy")
previousState = False
onCooldown = False

def trigger_event():
    random.seed(time.time())  # make sound choice less predictable
    fileName = path.join(soundsDir + random.choice(os.listdir(soundsDir)))
    _play_with_simpleaudio(AudioSegment.from_file(fileName))


def crop_from_center(image, dim):
    width, height = image.shape[1], image.shape[0]

    # process crop width and height for max available dimension
    crop_width = min(dim[0], image.shape[1])
    crop_height = min(dim[1], image.shape[0])
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    return image[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]


def resize_window_keep_aspect(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def putText(img, text, org=(10, 48), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1):
    cv2.putText(
        img=img,
        text=text,
        org=org,
        fontFace=fontFace,
        fontScale=fontScale,
        color=(0, 0, 0),
        thickness=thickness*2
    )
    cv2.putText(
        img=img,
        text=text,
        org=org,
        fontFace=fontFace,
        fontScale=fontScale,
        color=color,
        thickness=thickness
    )


while(True):
    np_frame = capture.screenshot()
    frame_height, frame_width, frame_channel = np_frame.shape

    np_frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)
    # make feed/debug window half the size of the screen
    np_frame = resize_window_keep_aspect(np_frame, height=int(frame_height/2))
    np_frame = crop_from_center(np_frame, (int(frame_width/3), int(frame_height/3)))

    pil_frame = Image.fromarray(np.uint8(np_frame)).convert("RGB")
    pil_frame = Image.fromarray(np_frame.astype("uint8"), "RGB")

    stat = round(
        statistics.mean(
            [x[0] for x in ImageStat.Stat(pil_frame).extrema]
        ), 4)
    state = (stat > threshold)

    if not onCooldown and state and previousState != state:
        def cooldown():
            global onCooldown
            onCooldown = True
            time.sleep(debounceCooldown)
            onCooldown = False

        Thread(target=cooldown).start()
        Thread(target=trigger_event).start()

    if debug:
        # extrema

        putText(
            img=np_frame,
            text=f"EXTREMA AVG: {stat}",
            org=(10, 48),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.75,
            color=(255, 255, 255),
            thickness=1,
        )

        # is flashed
        putText(
            img=np_frame,
            text=f"FLASHED: {'YES' if state else 'NO'}",
            org=(10, 72),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.75,
            color=(255, 255, 255),
            thickness=1,
        )

        # threshold
        putText(
            img=np_frame,
            text=f"USING THRESHOLD: {threshold}",
            org=(10, 88),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
        )

        cv2.imshow("frame", np_frame)

    previousState = state

    if cv2.waitKey(1) & 0xFF == ord("q"):  # quit
        break

cv2.destroyAllWindows()
