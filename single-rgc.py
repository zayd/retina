import numpy as np
import retina
import os
import random
import matplotlib.pyplot as plt

"""
Finds optimal size of single RGC from log-transformed natural images
and a Gaussian noise model of retinal cone cells
"""
img_dir = './images/vanhateran/'
filenames = os.listdir(img_dir)
num_iterations=50

noise_std = 5
noise_mean = 0

def extract_patches(image, size=4096, num_patches=50):
  BUFF = 4
  im_size = image.shape[0]
  side = np.sqrt(size)

  I = np.zeros((side,side,num_patches))
  for i in range(num_patches):
    r = BUFF + np.ceil((im_size-side-2*BUFF) * random.uniform(0,1))
    c = BUFF + np.ceil((im_size-side-2*BUFF) * random.uniform(0,1))
    I[:,:,i] = image[r:r+side, c:c+side]

  return I

radius = 16
entropy = np.zeros(radius)

trials = 4
for t in range(trials):
  for r in range(1,radius):
    I = extract_patches(retina.preprocess(img_dir + str(random.choice(filenames)), log=True))
    rgc = retina.Receptor(x=0, y=0, type='circle', radius=r, noise=(noise_std,noise_mean))
    weights = rgc.weights()
    y = np.zeros(I.shape[2])
    for i in range(num_iterations):
      y[i] = rgc.sample_pixels(radius, radius, I[:,:,i])

    # print weights
    # print (noise_std**2)*np.sum(weights**2)
    entropy[r] = 0.5*np.log((np.std(y)**2)/((noise_std**2)*np.sum(weights**2)))
    print "Receptor radius: " + str(r) + " Entropy: " + str(entropy[r])

  plt.plot(range(1,radius),entropy[1:])

plt.show()



