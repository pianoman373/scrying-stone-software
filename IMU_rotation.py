import time
import board
import adafruit_bno055
import numpy as np


i2c = board.I2C()
sensor = adafruit_bno055.BNO055_I2C(i2c)

last_val = 0xFFFF

def rotationMatrix():
    eAngles = sensor.euler
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(eAngles[0]), -1*np.sin(eAngles[0])],
                   [0, np.sin(eAngles[0]), np.cos(eAngles[0])]])
    Ry = np.array([[np.cos(eAngles[1]), 0, np.sin(eAngles[1])],
                   [0, 1, 0],
                   [-1*np.sin(eAngles[1]), 0, np.cos(eAngles[1])]])
    Rz = np.array([[np.cos(eAngles[2]), -1*np.sin(eAngles[2]), 0],
                   [np.sin(eAngles[2]), np.cos(eAngles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

if __name__ == "__main__":
    while True:
    	print("Euler angles: {}".format(sensor.euler))
    	print("Rotation matrix: {}".format(rotationMatrix()))
    	time.sleep(1)
