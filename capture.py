"""
CamCapture Class file
"""

#From Stephen Kirby

import cv2
import threading, queue
from datetime import datetime
from dateutil.tz import tzlocal


def get_axis_rtsp(username: str, password: str, ip: str) -> str:
    """ Creates RTSP URL for Axis Camera. """
    return f"rtsp://{username}:{password}@{ip}/axis-media/media.amp"


class CamCapture:
    # Ulrich is a hero of our time:
    # https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
    
    def __init__(self, path: str, verbose: bool = True):
        self.path = path
        self.cap_count = 0
        self.verbose = verbose
        self.start()
        return
        
    def start(self):
        self.cap = cv2.VideoCapture(self.path)
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.running = True
        self.thread.start()
        return
    
    def isOpened(self) -> bool:
        return self.cap.isOpened()
    
    def pv(self, *args) -> None:
        """ Verbose printing. """
        if self.verbose:
            print(*args)

    def _reader(self):
        while True:
            # self.pv("[CAMCAPTURE INFO] Grabbing Frame.")
            self.cap_count += 1
            if self.cap_count == 600:
                self.cap_count = 0
            try:
                ret, frame = self.cap.read()
            except cv2.error as error:
                print("[CV2 Error in CamCapture]: {}".format(error))
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait() # ignores unprocessed frame  
                except queue.Empty:
                    pass
            if not self.running:
                self.pv("[CAMCAPTURE INFO] Capture no longer running!")
                return
            self.q.put(frame)
            # self.pv("[CAMCAPTURE INFO] Successfully Queued Frame.")
        return

    def read(self):
        cap_time = datetime.now(tzlocal())
        while True:
            try: 
                next_image = self.q.get(timeout=15)
                break
            except Exception:
                print("Queue timed out, restarting thread.")
                self.restart_capture()
        return cap_time, next_image

    def restart_capture(self): 
        """ An attempt to restart the video capture upon failure. """
        self.close()
        self.start()
    
    def close(self):
        self.running = False
        self.thread.join()
        self.cap.release()
        return None