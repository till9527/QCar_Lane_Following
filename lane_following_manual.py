## task_lane_following.py
from pal.products.qcar import QCar, QCarCameras
from pal.utilities.math import *
from pal.utilities.gamepad import LogitechF710
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
gpad = LogitechF710(deviceID=1)
steeringFilter = Filter().low_pass_first_order_variable(25, 0.033)
next(steeringFilter)


def control_from_gamepad(LB, RT, leftLateral, A):
    if LB == 1:
        # Scale inputs for safety: Throttle is 10%, Steering is 50%
        direction = 1 if A == 1 else -1
        throttle_axis = 0.2 * direction * (RT - 0.5)
        steering_axis = leftLateral * 0.5
    else:
        throttle_axis, steering_axis = 0, 0
    return np.array([throttle_axis, steering_axis])


## Main Loop
try:
    while True:
        start = time.time()
        myCam.readAll()

        # 1. Processing: Crop -> HSV -> Blur -> Threshold
        croppedRGB = myCam.csiFront.imageData[262:352, 0:410]
        hsvBuf = cv2.cvtColor(croppedRGB, cv2.COLOR_BGR2HSV)
        # hsvBuf = cv2.GaussianBlur(hsvBuf, (5, 5), 0)  # Smooths out the "jutting"

        binaryImage = ImageProcessing.binary_thresholding(
            frame=hsvBuf,
            lowerBounds=np.array([10, 0, 0]),
            upperBounds=np.array([45, 255, 255]),
        )

        # 2. Lane Math
        slope, intercept = ImageProcessing.find_slope_intercept_from_binary(
            binary=binaryImage
        )

        # 3. Steering Calculation
        rawSteering = 0.7 * (slope - 0.3419) + (1 / 150) * (intercept + 5)
        steering = steeringFilter.send((np.clip(rawSteering, -0.5, 0.5), dt))

        # 4. Controller (FIXED: Changed leftLateralAxis to leftHorizontal)
        gpad.read()
        QCarCommand = control_from_gamepad(
            gpad.buttonLeft, gpad.trigger, gpad.leftJoystickX, gpad.buttonA
        )

        # Automatic mode (Hold X)
        if gpad.buttonX == 1:
            QCarCommand[1] = 0 if math.isnan(steering) else steering
            QCarCommand[0] = QCarCommand[0] * np.cos(steering)

        # 5. Output
        myCar.read_write_std(
            QCarCommand[0], QCarCommand[1], np.array([0, 0, 0, 0, 0, 0, 1, 1])
        )

        # Display - Comment these out if running via basic SSH
        cv2.imshow("Binary View", binaryImage)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        dt = time.time() - start

except KeyboardInterrupt:
    print("User interrupted!")

finally:
    myCar.terminate()
    cv2.destroyAllWindows()
