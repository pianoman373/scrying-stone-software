import time
import board
import adafruit_bno055
import numpy as np


i2c = board.I2C()
sensor = adafruit_bno055.BNO055_I2C(i2c)

last_val = 0xFFFF
prevR = np.eye(3)

def rotationMatrix():
    global prevR
    eAngles = sensor.euler
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(eAngles[0]), -1*np.sin(eAngles[0])],
                   [0, np.sin(eAngles[0]), np.cos(eAngles[0])]])
    Ry = -1*np.array([[np.cos(eAngles[2]), 0, np.sin(eAngles[2])],
                   [0, 1, 0],
                   [-1*np.sin(eAngles[2]), 0, np.cos(eAngles[2])]])
    Rz = np.array([[np.cos(eAngles[1]), -1*np.sin(eAngles[1]), 0],
                   [np.sin(eAngles[1]), np.cos(eAngles[1]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    R_t = np.transpose(R)
    Rdelta = np.dot(R_t, prevR)
    prevR = R
    return Rdelta

if __name__ == "__main__":
    while True:
    	print("Euler angles: {}".format(sensor.euler))
    	print("Rotation matrix: \n\r{}".format(rotationMatrix()))
    	time.sleep(1)
