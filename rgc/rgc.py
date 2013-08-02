# - H(X|R) = -E[1/2 * ln(2*pi*det(C_(X|R)^i))]
import scipy.io
import numpy as np
from numpy import dot, mean, std, cov, arange, log, exp
import random
import theano
import theano.sandbox.linalg as tl
import theano.tensor as T

import matplotlib.pyplot as plt
from matplotlib import cm

class RetinalGanglionCells(object):
  def __init__(self, N=256, neurons=100, sigma_nx=0.4, sigma_nr=2, penalty=0.1, non_lin=None):
    self.N = N
    self.neurons = neurons
    self.sigma_nx = sigma_nx
    self.sigma_nr = sigma_nr
    self.C_nx = sigma_nx * np.eye(N)
    self.C_nr = sigma_nr * np.eye(neurons)
    self.non_lin = non_lin
    self.W = np.random.randn(self.N, self.neurons)
    self.W = dot(self.W,np.diag(1/np.sqrt(np.sum(self.W * self.W, axis = 0))))
    self.spikes = np.zeros((neurons))
    self.penalty = penalty*np.ones((self.neurons))
    self.eta = 3

    # Theano Variables
    tW = T.dmatrix("tW")
    tG = T.dmatrix("tG")
    tC_x = T.dmatrix("tC_x")
    tC_nx = T.dmatrix("tC_nx")
    tC_nr = T.dmatrix("tC_nr")
    tlambda = T.dvector("tlambda")
    tx = T.dvector("tx")
    tn_x = T.dvector("tn_x")

    # W*G
    tWtG = T.dot(tW,tG)
    # C_x|r
    tC_xr = tl.ops.matrix_inverse(tl.ops.matrix_inverse(tC_x)
      + T.dot(T.dot(tWtG,tl.ops.matrix_inverse(T.dot(T.dot(tWtG.T, tC_nx), tWtG) + tC_nr)),
      tWtG.T))
    # -H(X|R)
    H_xr = -0.5*T.log(2*np.pi*np.exp(1)*tl.ops.det(tl.ops.psd(tC_xr)))
    # Do not include tn_r and f0 as they do not affect gradient
    E = H_xr - T.dot(tlambda.T, T.dot(tG, T.dot(tW.T, tx + tn_x)))
    # Energy gradient w.r.t
    gE_W = T.grad(E, tW)
    gE_G = T.grad(E, tG)


    self.ce = theano.function([tW,tG,tC_x,tC_nx,tC_nr],H_xr)
    self.gce_w = theano.function([tW,tG,tC_x,tC_nx,tC_nr,tlambda,tx,tn_x],gE_W)
    self.gce_g = theano.function([tW,tG,tC_x,tC_nx,tC_nr,tlambda,tx,tn_x],gE_G)

  def r(self, x):
    # Computes neuron's response r to input image x
    return dot(self.G(x),dot(dot(W.T,(x + self.__sample_noise(self.sigma_nx)))) +
      self.__sample_noise(self.sigma_nr)) + f_0

  def G(self, x, noise):
    # Computes first order approximation to input vector x. Individual x's along columns
    return np.diag(self.non_lin.derivative(self.W,x + noise))

  def update_W(self, X, C_x):
    # Computes grad H(X|R) w.r.t. W and updates W over sample of images X
    # X: Sample of images 2-D array. Images by column
    # C_x: Covariance matrix of input data
    (N, M) = np.shape(X) # Dimensions, Images in batch

    update_W = 0
    update_G = 0
    for m in range(M):
      nx = self.sigma_nx*np.random.randn(self.N)
      update_W = update_W + (1.0/M)*self.gce_w(self.W, self.G(X[:,m], nx), C_x,
        self.C_nx, self.C_nr, self.penalty, X[:,m], nx)
      update_G = update_G + (1.0/M)*self.gce_g(self.W, self.G(X[:,m], nx), C_x,
        self.C_nx, self.C_nr, self.penalty, X[:,m], nx)

      self.spikes = self.spikes + self.non_lin.evaluate(self.W, X[:,m] + nx)

    self.W = self.W + update_W * self.eta/float(M)#- 0.001 * np.sign(self.W)
    norms = np.apply_along_axis(np.linalg.norm, 0, self.W)
    self.W = self.W / np.tile(norms,(self.W.shape[0],1))

    self.non_lin.update(update_G * self.eta/float(M))

  def conditional_entropy(self, X, C_x):
    (N, M) = np.shape(X)
    H_xr = 0
    for m in range(M):
      nx = self.sigma_nx*np.random.randn(self.N)
      #print self.ce(self.W, self.G(X[:,m]), C_x, self.C_nx, self.C_nr)
      H_xr = H_xr + (1.0/M)*self.ce(self.W, self.G(X[:,m], nx), C_x, self.C_nx, self.C_nr)
    return H_xr

class NonLinearity(object):
  """
  Takes input vector x and returns output y. Stores parameters as vectors, with
  seperate instances for each neuron.

  Exponential:
  y = log(exp(a*x + b)) """

  def __init__(self, type='exponential', neurons=100, a=None, b=None):
    self.type = type
    self.neurons = neurons
    self.a = np.ones(self.neurons) if a == None else a
    self.b = 1*np.ones(self.neurons) if b == None else b

  def evaluate(self, W, X):
    # Matrix W. Matrix X (Image patches as columns). Not vectorized yet.
    if self.type == 'rectifying':
      r = dot(W.T,X)
      r[r <= self.b] = 0
      return (r - self.b) * self.a

    if self.type =='exponential':
      # output = exp(a*(w^T*x) + b) + c. not vectorized
        return log(exp(self.a * (dot(W.T, X) - self.b)) + 1)
        # return np.exp(self.a * dot(W.T, X) + self.b) + self.c
      # Returns vector with output of neuron to each image patch

  def derivative(self, W, X):
    # Matrix W. Matrix X (Image patches as columns). Not vectorized yet.
    # derivative with respect to y = w^T*x
    if self.type == 'rectifying':
      # if x <= b: return 0 if x > b: return a * x. not vectorized
      g = dot(W.T,X)
      g[g <= self.b] = 0
      g[g > self.b] = 1
      return g * self.a

    elif self.type == 'exponential':
      # not vectorized.
      g = dot(W.T,X)
      return (self.a*exp(self.a*g))/(exp(self.a*self.b) + exp(self.a * g))
      # return self.a * np.exp(self.a * dot(W.T, X) + self.b)
    # Returns vector with output of of neuron's derivative to each image patch

  def update(self, dG):
    if self.type == 'exponential':
      self.a = self.a + np.diag(dG)

  def plot(self, x):
    """Returns matrix of values of non-linearity over range of x for each neuron"""
    if self.type == 'exponential':
      return log(exp(self.a * (np.tile(x,(self.neurons,1)) - self.b)) + 1)

def extract_patches(image_dir='../images/vanhateran/', size=256, num_patches=800, padding=4):
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

def display(t, W, neurons, side, firing_rate=None, non_lin=None):
  plt.close()
  fig, (ax1, ax2, ax3) = plt.subplots(3)
  print "Iteration " + str(t)

  if firing_rate != None:
    ax1.bar(np.arange(neurons),firing_rate)
    ax1.set_title('Firing rate', {'fontsize': 10})

  if non_lin != None:
    x = np.arange(-10, 10, 0.1)
    X = np.tile(x, (neurons, 1))
    ax2.plot(X.T, non_lin.plot(x).T)

  image = np.ones((side*np.sqrt(neurons)+np.sqrt(neurons),side*np.sqrt(neurons)+np.sqrt(neurons)))
  for i in range(np.sqrt(neurons).astype(int)):
    for j in range(np.sqrt(neurons).astype(int)):
      image[i*side+i:i*side+side+i,j*side+j:j*side+side+j] = np.reshape(W[:,i*np.sqrt(neurons)+j],(side,side))
  cax = ax3.imshow(image, cmap='jet', vmin=np.min(W), vmax=np.max(W), interpolation="nearest")
  fig.colorbar(cax, ticks=[np.min(W), 0, np.max(W)])

  fig.show()

def run(N=81, neurons=128, batch=100, iterations=10000, BUFF=4):
  sz = np.sqrt(N)
  num_images = batch*500
  #IMAGES = extract_patches('../images/vanhateran/', size=N, num_patches=num_images)

  IMAGES = scipy.io.loadmat('../images/IMAGES.mat')['IMAGES']
  (imsize, imsize, num_images) = np.shape(IMAGES)

  I = np.zeros((N,batch))
  rect = NonLinearity(neurons=neurons)
  retina = RetinalGanglionCells(N, neurons, 0.4, 2, 0.1, rect)

  for t in range(iterations+1):
    #I = IMAGES[:,np.random.permutation(arange(0,num_images))[:batch]]
    for i in range(batch):
      # choose a random image
      imi = np.ceil(num_images * random.uniform(0,1))
      r = BUFF + np.ceil((imsize-sz-2*BUFF) * random.uniform(0,1))
      c = BUFF + np.ceil((imsize-sz-2*BUFF) * random.uniform(0,1))
      I[:,i] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi-1],N,1)

    C_x = cov(I)
    # print "Before update: " + str(retina.conditional_entropy(I, C_x))
    retina.update_W(I, C_x)
    # print "After update: " + str(retina.conditional_entropy(I, C_x))

    if np.mod(t,100) == 0:
      display(t, retina.W, neurons, np.sqrt(N), firing_rate=retina.spikes/float(batch*(t+1)), non_lin=rect)
      print "After update -H(X|R) = " + str(retina.conditional_entropy(I, C_x))