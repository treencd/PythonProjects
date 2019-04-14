# Imports
import cv2
import time
from threading import Event, Thread, Lock
import imutils
import argparse
import sys
from collections import deque
import numpy as np
import socket
import io

# HOST = '169.254.116.12'
# PORT = 5025
# BUFFER_SIZE = 24
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def Stereoscopics(stereo_data, led_color, kill_event, ):
    Old_val = 1
    focalsize = 3.04e-03
    pixelsize = 1.12e-06
    baseline = 0.737
    stereo_data = stereo_data
    GREEN = led_color
    kill_event = kill_event

    # Camera settings go here
    imageWidth = 1008
    imageHeight = 256
    frameRate = 10
    processingThreads = 4

    # Shared values
    global running
    global cap
    global frameLock
    global processorPool
    running = True
    frameLock = Lock()

    COLOR = "RED"

    # Setup the camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeight);
    cap.set(cv2.CAP_PROP_FPS, frameRate);
    if not cap.isOpened():
        cap.open()

    if COLOR == "PINK":
        hsvLower = (160, 80, 100)
        hsvUpper = (170, 255, 255)
    if COLOR == "RED":
        hsvLower1 = (160, 100, 100)  # currently set for red
        hsvUpper1 = (179, 255, 255)
        hsvLower2 = (0, 100, 100)  # currently set for red
        hsvUpper2 = (10, 255, 255)

    # pts = deque(maxlen=args["buffer"])

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

    output_data = Stack()

    def gpio_blinker(led_color, loop_count):
        if loop_count % 2 == 0:
            led_color.on()
        else:
            led_color.off()

    def loop_counter(loop_number):
        loop_number += 1
        if loop_number >= 10:
            loop_number = 1
        return loop_number

    # Image processing thread, self-starting
    class ImageProcessor(Thread):
        def __init__(self, name, output_data, autoRun=True):
            super(ImageProcessor, self).__init__()
            self.event = Event()
            self.output_data = output_data
            self.eventWait = (2.0 * processingThreads) / frameRate
            self.name = str(name)
            print('Processor thread %s started with idle time of %.2fs' % (self.name, self.eventWait))
            self.start()

        def run(self):
            # This method runs in a separate thread
            global running
            global frameLock
            global processorPool
            while running:
                # Wait for an image to be written to the stream
                self.event.wait(self.eventWait)
                if self.event.isSet():
                    if not running:
                        break
                    try:
                        self.ProcessImage(self.nextFrame)
                    finally:
                        # Reset the event
                        self.nextFrame = None
                        self.event.clear()
                        # Return ourselves to the pool at the back
                        with frameLock:
                            processorPool.insert(0, self)

            print('Processor thread %s terminated' % (self.name))

        def ProcessImage(self, image):
            frame = image

            frame = imutils.resize(frame, width=imageWidth, height=imageHeight)
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            # blurred = cv2.medianBlur(frame, 13)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            if COLOR == "RED":
                mask1 = cv2.inRange(hsv, hsvLower1, hsvUpper1)
                mask2 = cv2.inRange(hsv, hsvLower2, hsvUpper2)
                mask = mask1+ mask2

            else:
                mask = cv2.inRange(hsv, hsvLower, hsvUpper)

            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None

            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                centroid = (round((M["m10"] / M["m00"]), 3), round((M["m01"] / M["m00"]), 3))
                centroid = (centroid[0] * 3280 / imageWidth, centroid[1] * 2464 / imageHeight)

                self.output_data.push(int(centroid[0]))

                print(self.name, " ",centroid[0])
                ((x, y), radius) = cv2.minEnclosingCircle(c)

                if radius > 0.5:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

            cv2.imshow("Frame", mask)  # mask
            key = cv2.waitKey(1) & 0xFF



    # Image capture thread, self-starting
    class ImageCapture(Thread):
        def __init__(self):
            super(ImageCapture, self).__init__()
            self.start()

        # Stream delegation loop
        def run(self):
            # This method runs in a separate thread
            global running
            global cap
            global processorPool
            global frameLock
            while running:
                # Grab the oldest unused processor thread
                with frameLock:
                    if processorPool:
                        processor = processorPool.pop()
                    else:
                        processor = None
                if processor:
                    # Grab the next frame and send it to the processor
                    success, frame = cap.read()
                    if success:
                        processor.nextFrame = frame
                        processor.event.set()
                    else:
                        print('Capture stream lost...')
                        running = False
                else:
                    # When the pool is starved we wait a while to allow a processor to finish
                    time.sleep(0.01)
            print('Capture thread terminated')

    # def receive_data():
    #     try:
    #         data = clientPort.recv(BUFFER_SIZE)
    #         client_data = data.decode()
    #         #            print(client_data)
    #         tempData = "".join(client_data)
    #         tempData = tempData.split("<")
    #         try:
    #             right_x = float(tempData[1])
    #             #                print(right_x)
    #             Old_val = right_x
    #             return right_x
    #         except ValueError:
    #             right_x = Old_val
    #             pass
    #     except socket.error:
    #         print("[Stereo] : missed client data")
    #         while not kill_event.is_set():  # and not shutdown_event.is_set() and not kill_event.is_set():
    #             try:
    #                 data = clientPort.recv(BUFFER_SIZE)
    #                 client_data = data.decode()
    #                 #            print(client_data)
    #                 tempData = "".join(client_data)
    #                 tempData = tempData.split("<")
    #                 try:
    #                     right_x = float(tempData[1])
    #                     return right_x
    #                 except ValueError:
    #                     right_x = Old_val
    #                     pass
    #             except socket.error:
    #                 print("[Stereo] : Lost the client connection")
    #                 time.sleep(0.5)
    #                 continue

    # connected = False
    # while not connected and not kill_event.is_set():  # and not shutdown_event.is_set() and not kill_event.is_set():
    #     try:
    #         s.bind((HOST, PORT))
    #         s.listen(1)
    #         clientPort, addr = s.accept()
    #         connected = True
    #         print("[Stereo] : Client connected")
    #     #            return clientPort
    #     except socket.error as se:
    #         print('[Stereo] :       No Client', se)
    #         sys.stdout.flush()
    #         time.sleep(3)
    #         continue
    #     except Exception as e:
    #         print('[Stereo] :       No Client' + str(e))
    #         sys.stdout.flush()
    #         time.sleep(3)
    #         continue

    # Create some threads for processing and frame grabbing
    processorPool = [ImageProcessor(i + 1, output_data, ) for i in range(processingThreads)]
    allProcessors = processorPool[:]
    captureThread = ImageCapture()

    # Main loop ___________________________________________________________
    # The captureThread gets the frames and passes them to an unused processing thread
    try:
        start_time = time.time()
        stereo_loop_count = 1
        first = True
        while running:
            try:
                # gpio_blinker(GREEN, stereo_loop_count)

                # stereo_loop_count = loop_counter(stereo_loop_count)
                time_since_last = time.time()

                while (time.time() - time_since_last) <= 0.1:
                    time.sleep(0.01)
                #                start_time = time.time()
                left_x = output_data.pop()
                # right_x = receive_data()
                # print("Left:  ", left_x)#, "   Right:  ", right_x)
                # disparity = abs(left_x - float(right_x))
                # if disparity == 0:
                #     disparity = 1
                # distance = round((focalsize * baseline) / (disparity * pixelsize), 2)
                #                print("[STEREO:]  Distance:  ", distance)
                data = str(left_x)
                #                if data_flag.is_set():
                stereo_data.put(data)
            #                    data_flag.clear()

            except IndexError as i:
                pass
    except KeyboardInterrupt:
        print('\nUser shutdown')
    except:
        e = sys.exc_info()
        print()
        print(e)
        print('\nUnexpected error, shutting down!')
    # Cleanup all processing threads
    running = False
    while allProcessors:
        # Get the next running thread
        with frameLock:
            processor = allProcessors.pop()
        # Send an event and wait until it finishes
        processor.event.set()
        processor.join()

    # Cleanup the capture thread
    captureThread.join()

    # Cleanup the camera object
    cap.release()


if __name__ == "__main__":
    import multiprocessing as mp
    # from gpiozero import LED

    stereo_data = mp.Queue()
    kill_event = mp.Event()
    # GREEN = LED(16)
    GREEN = "GREEN"
    led_color = GREEN

    Stereo = mp.Process(target=Stereoscopics, args=[stereo_data, led_color, kill_event])
    Stereo.daemon = True
    Stereo.start()
    #    print("starting loop")
    while True:
        #        print("Getting Data")
        data = stereo_data.get()
        # tempData = "".join(data)
        # tempData = tempData.strip("<")
        # tempData = tempData.strip(">")
        # tempData = tempData.split(",")
        # RightX = int(float(tempData[0]))
        # LeftX = int(float(tempData[1]))
        # stereoDist = float(tempData[2])

        # print("LeftX:  ", data) # , "  RightX:  ", RightX)
