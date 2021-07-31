from djitellopy import Tello
import cv2
import pygame
from pygame.locals import *
import numpy as np
import time
import queue
import threading
from cam_class import Camera
from timeit import default_timer as timer
from video_writer import WriteVideo

# Speed of the drone
S = 60
# Speed for autonomous navigation
S_prog = 15
# Frames per second of the pygame window display
FPS = 25

class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations
            - W and S: Up and down.
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.width = 640
        self.height = 480
        self.screen = pygame.display.set_mode([self.width, self.height])

        # create queue for data communications
        self.data_queue=queue.Queue()

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello(self.data_queue)

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        
        # Variables for drone's states
        self.battery = 0
        self.angles = [0., 0., 0., 0.]

        # Direction queue for navigation
        self.dir_queue=queue.Queue()
        self.dir_queue.queue.clear()

        # Bool variables for setting functions
        self.send_rc_control = False
        self.calibrate = False
        self.getPoints = False
        self.resetPoints = False
        self.save = False
        self.getOrigin = False

        # Creating video queue
        self.video_queue = queue.Queue()
        self.video_queue.queue.clear()
        self.END_event = threading.Event()
        self.END_event.clear()
        self.videoWrite = WriteVideo(self.video_queue, FPS, self.END_event)
        # Run video writer in the background
        thread_vid = threading.Thread(target=self.videoWrite.writer)
        thread_vid.daemon = True
        thread_vid.start()

        # Data collection event
        self.getCoords_event = threading.Event()
        self.getCoords_event.clear()
        # Navigate between markers
        self.navigate_event = threading.Event()
        self.navigate_event.clear()

        # Camera class
        self.cam = Camera(S_prog, self.dir_queue, 'calibration_files/camcalib.npz',
                          self.getCoords_event, self.navigate_event, self.END_event)

        # Create update timer
        pygame.time.set_timer(USEREVENT + 1, 50)

    def run(self):

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        frame_read = self.tello.get_frame_read()
        directions = np.zeros(4)

        should_stop = False
        while not should_stop:
            img=cv2.resize(frame_read.frame, (960,720))

            # Read from drone state queue
            if not self.data_queue.empty():
                pitch, roll, yaw, tof, bat = self.data_queue.get()
                self.data_queue.queue.clear()
                self.battery = bat
                self.angles_tof = [pitch, roll, yaw, tof]
                #print([pitch, roll, yaw, tof])

            # Calibrate drone camera
            if self.calibrate:
                img = self.cam.calibrator(img)

            # Detect ArUco markers
            img = self.cam.aruco(img, self.getPoints, self.resetPoints, self.angles_tof)

            # Reset measurements
            if self.resetPoints:
                self.resetPoints=False

            for event in pygame.event.get():
                if event.type == USEREVENT + 1:
                    self.update(directions)
                elif event.type == QUIT:
                    should_stop = True
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                frame_read.stop()
                break

            # Save image on 'M' press
            if self.save:
                timestr = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite("images/"+timestr+".jpg", img)
                self.save = False

            # Navigation started, points and video capture
            if self.getCoords_event.is_set():
                self.video_queue.put(np.copy(img))

            # Write battery percentage
            img = self.cam.writeBattery(img, self.battery)

            img=cv2.resize(img, (640,480))
            
            # Resize pyGame window
            if (img.shape[1] != self.width) or (img.shape[0] != self.height):
                self.width = img.shape[1]
                self.height = img.shape[0]
                self.screen=pygame.display.set_mode((self.width, self.height))
            
            self.screen.fill([0, 0, 0])
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw counter clockwise velocity
            self.yaw_velocity = S

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False
        elif key == pygame.K_k: # camera calibration
            if self.calibrate:
                self.calibrate = False
            else:
                self.calibrate = True
        elif key == pygame.K_c: # get aruco marker points
            if self.getPoints:
                self.getPoints=False
            else:
                self.getPoints = True
                self.resetPoints = True
        elif key == pygame.K_m:  # save image
            self.save = True
        elif key == pygame.K_o:  # start navigation
            if self.navigate_event.is_set():
                self.navigate_event.clear()
            else:
                self.navigate_event.set()
        elif key == pygame.K_x:  # end video
            self.END_event.set()
            self.getPoints = False

    def update(self, dirs):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            if self.navigate_event.is_set() and not self.dir_queue.empty():
                # Auto navigation, read directions queue
                x, y, z, yaw = self.dir_queue.get()
                self.tello.send_rc_control(int(x), int(y), int(z), int(yaw))
            else:
                # Clear directions queue to avoid storing old data
                self.dir_queue.queue.clear() 
                # Control tello manually
                self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                           self.yaw_velocity)

def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
