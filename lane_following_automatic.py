## task_lane_following_auto.py
from pal.products.qcar import QCar, QCarCameras
from pal.utilities.math import *
from hal.utilities.image_processing import ImageProcessing

import time
import numpy as np
import cv2
import math

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Timing and Camera Parameters
sampleRate = 30
dt = 0.033
imageWidth, imageHeight = 820, 410

myCam = QCarCameras(
    frameWidth=imageWidth,
    frameHeight=imageHeight,
    frameRate=sampleRate,
    enableFront=True,
)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Hardware Initialization
myCar = QCar()

# Filter initialization
steeringFilter = Filter().low_pass_first_order_variable(25, 0.033)
next(steeringFilter)

# Configuration
baseThrottle = 0.1  # Base speed (0.0 to 1.0). Start low for safety!

## Main Loop
try:
    print("Starting Automatic Lane Following... Press Ctrl+C to stop.")
    while True:
        start = time.time()
        myCam.readAll()

        # 1. Processing: Crop -> HSV -> Binary Threshold
        # Cropping the image to focus on the road immediately in front
        croppedRGB = myCam.csiFront.imageData[262:352, 0:410]
        hsvBuf = cv2.cvtColor(croppedRGB, cv2.COLOR_BGR2HSV)

        binaryImage = ImageProcessing.binary_thresholding(
            frame=hsvBuf,
            lowerBounds=np.array([10, 0, 0]),
            upperBounds=np.array([45, 255, 255]),
        )

        # 2. Lane Math
        slope, intercept = ImageProcessing.find_slope_intercept_from_binary(
            binary=binaryImage
        )

        # 3. Automatic Control Logic
        # Check if we have valid lane data. If slope is NaN, we lost the line.
        if slope is None or math.isnan(slope) or math.isnan(intercept):
            # LINE LOST: Stop the car, but keep the loop running
            throttle = 0
            steering = 0
            # Optional: Print warning so you know why it stopped
            # print("Line Lost - Stopping")
        else:
            # LINE FOUND: Calculate steering and throttle

            # Steering Calculation (Original Formula)
            rawSteering = 0.7 * (slope - 0.3419) + (1 / 150) * (intercept + 5)

            # Apply Filter and Clip
            steering = steeringFilter.send((np.clip(rawSteering, -0.5, 0.5), dt))

            # Throttle Calculation
            # Reduce speed slightly when turning (cos(steering) < 1)
            throttle = baseThrottle * np.cos(steering)

        # 4. Output to Car
        myCar.read_write_std(throttle, steering, np.array([0, 0, 0, 0, 0, 0, 1, 1]))

        # 5. Display
        cv2.imshow("Binary View", binaryImage)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        dt = time.time() - start

except KeyboardInterrupt:
    print("User interrupted!")

finally:
    # Ensure car stops when script ends
    myCar.read_write_std(0, 0, np.array([0, 0, 0, 0, 0, 0, 1, 1]))
    myCar.terminate()
    cv2.destroyAllWindows()
