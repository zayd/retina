import numpy as np
from numpy import dot, arange, ones, exp
import matplotlib.pyplot as plt
import scipy.io

# import theano
# import theano.tensor as T

num_iterations = 5000

def extract_patches(image_dir='./images/vanhateran/', size=256, num_patches=800, padding=4):
  import random
  import retina
  import os

  side = np.sqrt(size)
  num_images = 20

  filenames = os.listdir(image_dir)
  filenames = random.sample(filenames, num_images)

  I = np.zeros((size,num_patches))

  for j,image_file in enumerate(filenames):
    image = retina.preprocess(image_dir + image_file)
    image_size = image.shape[0]
    for i in range(num_patches/num_images):
      r = padding + np.ceil((image_size-side-2*padding) * random.uniform(0,1))
      c = padding + np.ceil((image_size-side-2*padding) * random.uniform(0,1))
      patch = np.reshape(image[r:r+side, c:c+side], (size))
      patch = patch / float(np.max(patch))
      patch = patch - np.mean(patch)
      I[:,j*num_patches/num_images+i] = patch
  return I

def display(W, M, side):
  plt.close()
  print "Iteration " + str(t)
  image = np.ones((side*np.sqrt(M)+np.sqrt(M),side*np.sqrt(M)+np.sqrt(M)))
  for i in range(np.sqrt(M).astype(int)):
    for j in range(np.sqrt(M).astype(int)):
      image[i*side+i:i*side+side+i,j*side+j:j*side+side+j] = np.reshape(W[:,i*np.sqrt(M)+j],(side,side))
  plt.imshow(image, cmap='jet', interpolation="nearest")
  plt.show(block=False)

def weight(theta, size):
  w = ones((size,size))

  for i in range(size):
    for j in range(size):
      w[i,j] = exp(-((i-size/2)**2 + (j-size/2)**2)/theta)
  plt.close()
  plt.imshow(w)
  plt.show(block=False)
  w = 1 + (1 - w)
  return w.reshape((size*size,1))

# tI = T.dmatrix('tI')
# tW = T.dmatrix('tW')
# tG = T.dmatrix('tG')

#cost = T.sum((tI - t.dot(tW, s))**2) + tk
batch_size = 100
M = 256
N = 36
k = 0.025
W = 0.1*np.random.randn(N,M)
W = np.dot(W.T,np.diag(1/np.sqrt(np.sum(W.T*W.T, axis = 0)))).T
G = 0.1*np.random.randn(M,N)
G = np.dot(G,np.diag(1/np.sqrt(np.sum(G*G, axis = 0))))

eta = 0.005

# Use data.mat
#IMAGES = scipy.io.loadmat('./images/patches.mat')
#IMAGES = IMAGES['IMAGES']
#num_images = IMAGES.shape[1]

# Use Raw Van Hateran files
num_images = batch_size*500
IMAGES = extract_patches('./images/vanhateran/', size=M, num_patches=num_images)
w = weight(theta=24,size=16)

for t in range(num_iterations+1):

    I = IMAGES[:,np.random.permutation(arange(0,num_images))[:batch_size]]
    y = dot(W, I)
    #noisy_y = y + np.random.normal(0, 0.05, y.shape)
    # e = I - dot(W.T,y)
    e = I - dot(G,y)
    e = e * np.tile(w,(1,batch_size))
    #dW = dot(y, e.T).T
    dW = dot(y, e.T)
    dG = dot(y, e.T).T
    dW = dW - k * np.sign(W)
    # dG = dG - k * np.sign(G)
    # print dW
    W = W + eta * dW
    G = G + eta * dG
    W = dot(W.T,np.diag(1/np.sqrt(np.sum(W.T*W.T, axis = 0)))).T
    G = np.dot(G,np.diag(1/np.sqrt(np.sum(G*G, axis = 0))))

    if np.mod(t,200) == 0:
      display(W.T, N, np.sqrt(M))



