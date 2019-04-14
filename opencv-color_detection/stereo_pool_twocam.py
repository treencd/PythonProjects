# Imports
import cv2
import time
import threading
import imutils
import argparse
import sys
from collections import deque
import numpy as np

# working_on_the_Pi = False

# if working_on_the_Pi:
#     try:
#         from gpiozero import LED
#         from picamera.array import PiRGBArray
#         from picamera import PiCamera
#     except ImportError as imp:
#         print("IMPORTANT  :   ARE YOU WORKING THE RASPBERRY PI ?:  ", imp)
#         sys.stdout.flush()
#     else:
#         GREEN = LED(5)

# Camera settings go here
imageWidth = 1008
imageHeight = 256
frameRate = 30
processingThreads = 1

# Shared values
global running_1
running_1 = True

global cap_1
global frameLock_1
frameLock_1 = threading.Lock()
global processorPool_1
global running_2
running_2 = True

global cap_2
global frameLock_2
frameLock_2 = threading.Lock()
# global processorPool_2
# global processorPool_1
global processorPool_2

focalsize = 3.04e-03
pixelsize = 1.12e-06
baseline = 0.012#0.737

# Setup the camera
cap_1 = cv2.VideoCapture(0)
cap_1.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth);
cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeight);
cap_1.set(cv2.CAP_PROP_FPS, frameRate);
if not cap_1.isOpened():
    cap_1.open()

cap_2 = cv2.VideoCapture(2)
cap_2.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth);
cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeight);
cap_2.set(cv2.CAP_PROP_FPS, frameRate);
if not cap_2.isOpened():
    cap_2.open()

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=8,
                help="max buffer size")
args = vars(ap.parse_args())

pts = deque(maxlen=args["buffer"])

# LOCATION OPTIONS ARE:
# - hall
# - gym
# - capstone

COLOR = "RED"

if COLOR == "PINK":
    hsvLower = (160, 60, 60)
    hsvUpper = (170, 255, 255)
if COLOR == "RED":
    hsvLower = (0, 180, 180)  # currently set for red
    hsvUpper = (10, 255, 255)

class data_object:
    def __init__(self, *args):
        self.args = args

# def gpio_blinker(led_color, loop_count):
#     if working_on_the_Pi:
#         if loop_count % 2 == 0:
#             led_color.on()
#         else:
#             led_color.off()

def loop_counter(loop_number):
    loop_number += 1
    if loop_number >= 10:
        loop_number = 1
    return loop_number

class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        if not (self.isEmpty()):
            return self.items[len(self.items) - 1]
        else:
            return None

    def size(self):
        return len(self.items)



# Image processing thread, self-starting
class ImageCapture_1(threading.Thread):
    def __init__(self):
        super(ImageCapture_1, self).__init__()
        self.start()

    # Stream delegation loop
    def run(self):
        # This method runs in a separate thread
        global running_1
        global cap_1
        global processorPool_1
        global frameLock_1
        while running_1:
            # Grab the oldest unused processor thread
            with frameLock_1:
                if processorPool_1:
                    processor = processorPool_1.pop()
                else:
                    processor = None
            if processor:
                # Grab the next frame and send it to the processor
                success, frame = cap_1.read()
                if success:
                    processor.nextFrame = frame
                    processor.event.set()
                else:
                    print('Capture stream lost...')
                    running_1 = False
            else:
                # When the pool is starved we wait a while to allow a processor to finish
                time.sleep(0.01)
        print('Capture thread terminated')




# Processing for each image goes here
### TODO ###

class ImageProcessor_1(threading.Thread):
    def __init__(self, name, output_data, stereo_data, show_cam, autoRun=True):
        super(ImageProcessor_1, self).__init__()
        self.event = threading.Event()
        self.output_data = output_data
        self.stereo_data = stereo_data
        self.show = show_cam
        self.eventWait = (2.0 * processingThreads) / frameRate
        self.name = str(name)
        print('Processor_1 thread %s started with idle time of %.2fs' % (self.name, self.eventWait))
        self.start()

    def run(self):
        # This method runs in a separate thread
        global running_1
        global frameLock_1
        global processorPool_1
        while running_1:
            # Wait for an image to be written to the stream
            self.event.wait(self.eventWait)
            if self.event.isSet():
                if not running_1:
                    break
                try:
                    self.ProcessImage(self.nextFrame, self.show)
                finally:
                    # Reset the event
                    self.nextFrame = None
                    self.event.clear()
                    # Return ourselves to the pool at the back
                    with frameLock_1:
                        processorPool_1.insert(0, self)

        print('Processor thread %s terminated' % (self.name))

    def ProcessImage(self, image, show):
        frame = image
        show_cam = show
        # handle the frame from VideoCapture or VideoStream
        frame = frame[1] if args.get("video", False) else frame

        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, hsvLower, hsvUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                self.stereo_data.left_x = x
                self.output_data.push(self.stereo_data)


        # update the points queue
        pts.appendleft(center)
        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # show the frame to our screen
        if show_cam.is_set():
            try:
                cv2.imshow("Frame", mask)
                key = cv2.waitKey(1) & 0xFF
            except:
                pass
# Image capture thread, self-starting

class ImageCapture_2(threading.Thread):
    def __init__(self):
        super(ImageCapture_2, self).__init__()
        self.start()

    # Stream delegation loop
    def run(self):
        # This method runs in a separate thread
        global running_2
        global cap_2
        global processorPool_2
        global frameLock_2
        while running_2:
            # Grab the oldest unused processor thread
            with frameLock_2:
                if processorPool_2:
                    processor = processorPool_2.pop()
                else:
                    processor = None
            if processor:
                # Grab the next frame and send it to the processor
                success, frame = cap_2.read()
                if success:
                    processor.nextFrame = frame
                    processor.event.set()
                else:
                    print('Capture stream lost...')
                    running_2 = False
            else:
                # When the pool is starved we wait a while to allow a processor to finish
                time.sleep(0.01)
        print('Capture thread terminated')

class ImageProcessor_2(threading.Thread):
    def __init__(self, name, output_data, stereo_data, show_camera, autoRun=True):
        super(ImageProcessor_2, self).__init__()
        self.event = threading.Event()
        self.output_data = output_data
        self.stereo_data = stereo_data
        self.show = show_camera
        self.eventWait = (2.0 * processingThreads) / frameRate
        self.name = str(name)
        print('Processor thread %s started with idle time of %.2fs' % (self.name, self.eventWait))
        self.start()

    def run(self):
        # This method runs in a separate thread
        global running_2
        global frameLock_2
        global processorPool_2
        while running_2:
            # Wait for an image to be written to the stream
            self.event.wait(self.eventWait)
            if self.event.isSet():
                if not running_2:
                    break
                try:
                    self.ProcessImage(self.nextFrame, self.show)
                finally:
                    # Reset the event
                    self.nextFrame = None
                    self.event.clear()
                    # Return ourselves to the pool at the back
                    with frameLock_2:
                        processorPool_2.insert(0, self)

        print('Processor_2 thread %s terminated' % (self.name))

    def ProcessImage(self, image, show):
        frame = image
        show_cam = show
        # handle the frame from VideoCapture or VideoStream
        frame = frame[1] if args.get("video", False) else frame

        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, hsvLower, hsvUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                self.stereo_data.right_x = x

                self.output_data.push(self.stereo_data)


        # update the points queue
        pts.appendleft(center)
        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # show the frame to our screen
        if show_cam.is_set():
            try:
                cv2.imshow("Frame", mask)
                key = cv2.waitKey(1) & 0xFF
            except:
                pass

def Stereoscopics(stereo_data, kill_event, show_camera, pause_event, data_lock):
    stereo_info = stereo_data
    # GREEN = led_color
    # working_on_the_Pi = pi_no_pi
    get_lock = data_lock
    kill = kill_event
    pause = pause_event
    show = show_camera

    # Define Objects:
    output_data = Stack()
    stereo_data = data_object
    frame_counter = 0
    global processorPool_1
    global processorPool_2
    global running_1
    global running_2

    # Create some threads for processing and frame grabbing
    processorPool_1 = [ImageProcessor_1(i + 1, output_data, stereo_data, show) for i in range(processingThreads)]
    processorPool_2 = [ImageProcessor_2(a + 1, output_data, stereo_data, show) for a in range(processingThreads)]
    allProcessors_1 = processorPool_1[:]
    allProcessors_2 = processorPool_2[:]
    allProcessors = allProcessors_1 + allProcessors_2
    captureThread_1 = ImageCapture_1()
    captureThread_2 = ImageCapture_2()


    # Main loop, basically waits until you press CTRL+C
    # The captureThread_1 gets the frames and passes them to an unused processing thread

    try:
        print('Press CTRL+C to quit')
        #start_time = time.time()
        while running_1 and running_2 and not kill.is_set():
            # if working_on_the_Pi:
            #     gpio_blinker(GREEN, stereo_loop_count)
            # stereo_loop_count = loop_counter(stereo_loop_count)
            try:
                data = output_data.pop()
                try:
                    left_x = data.left_x
                    left = True
                except IndexError:
                    left = False
                    pass
                except AttributeError:
                    left = False
                    pass

                try:
                    right_x = data.right_x
                    right = True
                except IndexError:
                    right=False
                    pass
                except AttributeError:
                    right = False
                    pass
            except IndexError:
                pass
            except AttributeError:
                pass
            else:
                #frame_counter += 1
                #end_time = time.time()
                #time_since_beg = end_time - start_time
                #print(frame_counter / time_since_beg )#,"    ", left_x)
                # if left:
                #     print("LEFT:  ", left_x)
                # if right:
                #     print("RIGHT:  ", right_x)
                if left and right:
                    disparity = abs(left_x - right_x)
                    try:
                        distance = round((focalsize * baseline) / (disparity * pixelsize), 2)
                    except ZeroDivisionError as z:
                        print(z)
                    # print("[STEREO:]  Distance:  ",distance)
                    data = str(right_x), ",", str(left_x), ",", str(distance)
                    stereo_info.put(data)



            #time.sleep(1)
    except KeyboardInterrupt:
        print('\nUser shutdown')
    except:
        e = sys.exc_info()
        print()
        print(e)
        print('\nUnexpected error, shutting down!')

    # Cleanup all processing threads
    running_1 = False
    running_2 = False

    while allProcessors:
        # Get the next running thread
        with frameLock_1:
            processor = allProcessors_1.pop()
        # Send an event and wait until it finishes
        processor.event.set()
        processor.join()

        with frameLock_2:
            processor = allProcessors_1.pop()
        # Send an event and wait until it finishes
        processor.event.set()
        processor.join()

    # Cleanup the capture thread
    captureThread_1.join()
    captureThread_2.join()


    # Cleanup the camera object
    cap_1.release()
    cap_2.release()


if __name__ == "__main__":
    import multiprocessing as mp
    from threading import Thread
    from guizero import App, Text, PushButton, TextBox

    class run_app():
        def __init__(self):

            self.stereo_data = mp.Queue()
            self.kill_event = mp.Event()
            self.show_camera = mp.Event()
            self.pause_event = mp.Event()
            self.data_lock = mp.Lock()
            self.StereoProcess = mp.Process(target=Stereoscopics, args=[self.stereo_data, self.kill_event, self.show_camera, self.pause_event, self.data_lock])

            # SHOW THE CAMERA?:
            self.show_camera.set()

            # working_on_the_Pi = True
            # if working_on_the_Pi:
            #     GREEN = LED(16)

        def run_stereo(self):
            self.StereoProcess.daemon = True
            self.StereoProcess.start()

            while True:
                data = self.stereo_data.get()
                tempData = "".join(data)
                tempData = tempData.strip("<")
                tempData = tempData.strip(">")
                tempData = tempData.split(",")
                RightXcoord = int(float(tempData[0]))
                LeftXcoord = int(float(tempData[1]))
                stereoDist = float(tempData[2])
                self.distBox.value = stereoDist
                print("RightXcoord:  ", RightXcoord, "  LeftXcoord:  ", LeftXcoord, "  stereoDist:  ", stereoDist)

        def start(self,distance):
            self.distBox = distance
            self.startThread = Thread(target=self.run_stereo,args=[])
            if not self.startThread.isAlive():
                self.startThread.start()

        def stop(self):
            self.kill_event.set()

            # self.StereoProcess.terminate()
            self.startThread.join()
            sys.exit()

    app = App(title="Stereoscopics", layout="auto", height=320, width=480, bg="#424242", visible=True)

    Text(app, "S.T.A.R.S. Stereoscopics", size=20, font="Calibri Bold", color="red")

    distance = TextBox(app)
    distance.text_size = 128
    distance.text_color = "white"
    distance.font = "Calibri Bold"

    start = PushButton(app, command=run_app().start, args=[distance], text="START",
                         width=20)

    stop = PushButton(app, command=run_app().stop, args=[], text="STOP",
                       width=20)

    app.display()


