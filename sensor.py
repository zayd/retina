"""
Optimize sensor placements (instead of reconstruction based method) for natural images

July 2013
"""

import numpy as np
from numpy import dot, tile

def grid(size=64, stride=32):
  return [(x*stride,x*stride) for x in range(size/stride)]

def update(sensors, epic):


def gaze(sensors, epic):
  num_sensors = sensors.shape[1]
  time = epic.shape[2]

  observations = np.zeros((num_sensors*2,time))

  # First observation only first sensor
  observations[:num_sensors,0] = dot(sensors.T, tile(epic[0], (1,num_sensors)))

  for t in range(1,time):
    observations[:num_sensors,t] = dot(sensors.T, tile(epic[t], (1, num_sensors)))

    sensors = update(sensors, epic[t-1], epic[t])

    observations[num_sensors:,t] = dot(sensors.T, tile(epic[t], (1, num_sensors)))
