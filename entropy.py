"""
Experiment to test correlation of output of a center-surround filter
with entropy of pixels of  image patch
"""
import numpy as np
from numpy import dot, mean, std, cov, arange, log, exp
import matplotlib.pyplot as plt

def extract_patches(image_dir='./images/vanhateran/', size=256, num_patches=800, padding=4):
  import random
  import os
  import array

  def preprocess(filename, database='vanhateran',log=False):
    # Preprocess Van Hateran Natural Images
    R = 1024
    C = 1536
    extra = (C-R)/2

    with open(filename, 'rb') as handle:
      s = handle.read()
    arr = array.array('H', s)
    arr.byteswap()
    img = np.array(arr, dtype='uint16').reshape(R, C)

    img = img[:,extra-1:C-extra-1] # crop to make square
    img = img.reshape(R*R,1)
    return img.reshape(R,R)

  side = np.sqrt(size)
  num_images = 20

  filenames = os.listdir(image_dir)
  filenames = random.sample(filenames, num_images)

  I = np.zeros((size,num_patches))

  for j,image_file in enumerate(filenames):
    image = preprocess(image_dir + image_file)
    image_size = image.shape[0]
    for i in range(num_patches/num_images):
      r = padding + np.ceil((image_size-side-2*padding) * random.uniform(0,1))
      c = padding + np.ceil((image_size-side-2*padding) * random.uniform(0,1))
      patch = np.reshape(image[r:r+side, c:c+side], (size))
      # Zero-mean & Unit-variance
      patch = patch - mean(patch) / std(patch)
      I[:,j*num_patches/num_images+i] = patch
  return I

def main():
  patches=1000
  I = extract_patches(size=256, num_patches=patches)
  print "Center Surround: "
  cs = center_surround(on_radius=6, off_radius=8, center=(4,4) plot=True)
  correlation(I, cs)

  print "Summation: "
  cs = center_surround(on_radius=8, plot=True)
  correlation(I, cs)

  print "Sum of 3 Shifted Center Surrounds: "
  cs = center_surround(on_radius=8, off_radius=6)

def correlation(I, cs):
  patches = I.shape[1]
  filter_out = np.dot(I.T,cs)
  entropy_out = np.var(I, axis=0)
  entropy_out = 0.5*np.log(2*np.pi*np.exp(1)*entropy_out)

  print "Filter out std dev: " + str(np.std(filter_out))
  print "Entropy std dev: " + str(np.std(entropy_out))

  entropy_out = entropy_out.reshape(1,patches)
  filter_out = filter_out.reshape(1,patches)
  X = np.concatenate((entropy_out, filter_out), axis=0)
  correlation = np.cov(X)/(np.std(entropy_out)*np.std(filter_out))

  print "Correlation" + str(correlation)

def center_surround(size=256, on_radius=4, off_radius=6, center=(8,8) plot=False):
  side = np.sqrt(size).astype(int)
  A = np.zeros((side, side))

  for i in range(-side/2,side/2):
    for j in range(-side/2,side/2):
      if np.sqrt(i**2 + j**2) < on_radius:
        A[i+side/2,j+side/2] = 1
      if np.sqrt(i**2 + j**2) >= on_radius and np.sqrt(i**2 + j**2) < off_radius:
        A[i+side/2,j+side/2] = -1

  if plot:
    plt.imshow(A, interpolation='nearest')
    plt.show()

  return A.reshape((size))





