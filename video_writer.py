import cv2
import numpy as np
import time
import queue
import threading

class WriteVideo():
    def __init__(self, frame_queue, FPS, endVideo_event):
        self.frame_queue = frame_queue
        self.frame_list = []
        self.wroteBefore = False
        self.FPS = FPS
        self.endVideo_event = endVideo_event

    def writer(self):
        while True:
            if not self.frame_queue.empty() and not self.wroteBefore:
                frame = self.frame_queue.get()
                h, w = frame.shape[:2]
                timestr = time.strftime("%Y%m%d_%H%M%S")
                out = cv2.VideoWriter("videos/"+timestr+".avi",cv2.VideoWriter_fourcc('M','J','P','G'), self.FPS, (w,h))
                self.wroteBefore = True
            elif not self.frame_queue.empty() and self.wroteBefore:
                frame = self.frame_queue.get()
                out.write(frame)
            elif self.frame_queue.empty() and self.endVideo_event.is_set():
                out.release()
                print("Video saved")
                self.wroteBefore = False
                self.endVideo_event.clear()