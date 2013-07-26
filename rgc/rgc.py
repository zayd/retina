# - H(X|R) = -E[1/2 * ln(2*pi*det(C_(X|R)^i))]
import scipy.io
import numpy as np
from numpy import dot
import random
import theano
import theano.sandbox.linalg as tl
import theano.tensor as T

import matplotlib.pyplot as plt
from matplotlib import cm

"""def theano_test():
  tW = T.dmatrix("tW")
  tG = T.dmatrix("tG")
  tC_x = T.dmatrix("tC_x")
  tC_nx = T.dmatrix("tC_nx")
  tC_nr = T.dmatrix("tC_nr")

  tWtG = T.dot(tW,tG)
  tC_xr = tl.ops.matrix_inverse(tl.ops.matrix_inverse(tC_x)
    + T.dot(T.dot(tWtG,tl.ops.matrix_inverse(T.dot(T.dot(tWtG.T, tC_nx), tWtG) + tC_nr)),
    tWtG.T))
  H_xr = -0.5*(T.log(2*np.pi*np.exp(1)) + T.log(tl.ops.det(tl.ops.psd(tC_xr))))
  gH_xr = T.grad(H_xr, tW)

  ce = theano.function([tW,tG,tC_x,tC_nx,tC_nr],tC_xr)
  gce_w = theano.function([tW,tG,tC_x,tC_nx,tC_nr],gH_xr)
  # Example
  W = np.random.randn(256,100)
  C_nx = 0.4 * np.eye(256)
  C_nr = 2 * np.eye(100)
  G =


  gce_w(W, np.diag(np.ones(100)), np.diag(np.ones(256)), C_nx, C_nr)
"""
class RetinalGanglionCells(object):
  def __init__(self, N=256, neurons=100, sigma_nx=0.4, sigma_nr=2, non_lin=None):
    self.N = N
    self.neurons = neurons
    self.sigma_nx = sigma_nx
    self.sigma_nr = sigma_nr
    self.C_nx = sigma_nx * np.eye(N)
    self.C_nr = sigma_nr * np.eye(neurons)
    self.non_lin = non_lin
    self.W = np.random.randn(self.N, self.neurons)
    self.W = dot(self.W,np.diag(1/np.sqrt(np.sum(self.W * self.W, axis = 0))))
    self.eta = 3

    # Theano Variables
    tW = T.dmatrix("tW")
    tG = T.dmatrix("tG")
    tC_x = T.dmatrix("tC_x")
    tC_nx = T.dmatrix("tC_nx")
    tC_nr = T.dmatrix("tC_nr")
    tlambda = T.dscalar("tlambda")
    tR = T.dvector("tE_r")

    # NEED TO ADD SPARSITY Constraint

    # W*G
    tWtG = T.dot(tW,tG)
    # C_x|r
    tC_xr = tl.ops.matrix_inverse(tl.ops.matrix_inverse(tC_x)
      + T.dot(T.dot(tWtG,tl.ops.matrix_inverse(T.dot(T.dot(tWtG.T, tC_nx), tWtG) + tC_nr)),
      tWtG.T))
    # -H(X|R)
    H_xr = -0.5*T.log(2*np.pi*np.exp(1)*tl.ops.det(tl.ops.psd(tC_xr)))
    #H_xr = -T.sum(T.log(tl.ops.trace(tl.ops.cholesky(tC_xr))))
    gH_xr = T.grad(H_xr, tW)

    self.ce = theano.function([tW,tG,tC_x,tC_nx,tC_nr],H_xr)
    self.gce_w = theano.function([tW,tG,tC_x,tC_nx,tC_nr],gH_xr)

  def r(self, x):
    # Computes neuron's response r to input image x
    return dot(self.G(x),dot(dot(W.T,(x + self.__sample_noise(self.sigma_nx)))) +
      self.__sample_noise(self.sigma_nr)) + f_0

  def G(self, x):
    # Computes first order approximation to input vector x. Individual x's along columns
    return np.diag(self.non_lin.derivative(self.W,x + self.sigma_nx*np.random.randn(self.N)))

  def update_W(self, X, C_x):
    # Computes grad H(X|R) w.r.t. W and updates W over sample of images X
    # X: Sample of images 2-D array. Images by column
    # C_x: Covariance matrix of input data
    (N, M) = np.shape(X) # Dimensions, Images in batch

    update = 0
    for m in range(M):
      update = update + (1.0/M)*self.gce_w(self.W,
        self.G(X[:,m]), C_x, self.C_nx, self.C_nr)

    self.W = self.W + self.eta * update - 0.001 * np.sign(self.W)
    norms = np.apply_along_axis(np.linalg.norm, 0, self.W)
    self.W = self.W / np.tile(norms,(self.W.shape[0],1))

  def conditional_entropy(self, X, C_x):
    (N, M) = np.shape(X)
    H_xr = 0
    for m in range(M):
      #print self.ce(self.W, self.G(X[:,m]), C_x, self.C_nx, self.C_nr)
      H_xr = H_xr + (1.0/M)*self.ce(self.W, self.G(X[:,m]), C_x, self.C_nx, self.C_nr)

    return H_xr

  def __sample_noise(self, sigma):
    return 0

class NonLinearity(object):
  """
  Takes input vector x and returns output y. Stores parameters as vectors, with
  seperate instances for each neuron.

  Exponential:
  y = exp(a*x + b) """

  def __init__(self, type='rectifying', neurons=100, a=None, b=None, c=None):
    self.type = type
    self.neurons = neurons
    self.a = 0.1*np.ones(self.neurons) if a == None else a
    self.b = np.ones(self.neurons) if b == None else b
    self.c = np.zeros(self.neurons) if c == None else c

  def constant(self):
    return c;

  def evaluate(self, W, x):
    # Matrix W. Matrix X (Image patches as columns).
    if self.type =='exponential':
      # output = exp(a*(w^T*x) + b) + c. not vectorized
        return np.exp(self.a * dot(W.T, x) + self.b) + self.c
        # return np.exp(self.a * dot(W.T, X) + self.b) + self.c
      # Returns vector with output of neuron to each image patch

  def derivative(self, W, X):
    # Matrix W. Matrix X (Image patches as columns).
    # derivative with respect to y = w^T*x
    if self.type == 'rectifying':
      # if x <= a: return 0 if x > a: return b. not vectorized
      #G = (dot(W.T,X) <= np.tile(self.a, (1,num_images))) * np.tile(self.b, (1,num_images))
      G = (dot(W.T,X) <= self.a) * self.b
      return G.astype('float64')

    elif self.type == 'exponential':
      # not vectorized.
      return self.a * np.exp(self.a * dot(W.T, x) + self.b)
      # return self.a * np.exp(self.a * dot(W.T, X) + self.b)
    # Returns vector with output of of neuron's derivative to each image patch

def compute_cov(X):
  """Covariance of observation matrix X"""

  [N, neurons] = X.shape
  m = np.mean(X,1).reshape(N, 1)
  C = 1/float(neurons) * dot(X - dot(m, np.ones((1,neurons))),
    (X - dot(m, np.ones((1,neurons)))).T)
  # Add to diagonals to ensure matrix is PD
  return C #+ 0.005*np.eye(N,N)

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
      patch = patch / std(patch)
      I[:,j*num_patches/num_images+i] = patch

  return I

def display(t, W, M, side):
  plt.close()
  print "Iteration " + str(t)
  image = np.ones((side*np.sqrt(M)+np.sqrt(M),side*np.sqrt(M)+np.sqrt(M)))
  for i in range(np.sqrt(M).astype(int)):
    for j in range(np.sqrt(M).astype(int)):
      image[i*side+i:i*side+side+i,j*side+j:j*side+side+j] = np.reshape(W[:,i*np.sqrt(M)+j],(side,side))
  plt.imshow(image, cmap='jet', interpolation="nearest")
  plt.show(block=False)

def run(N=81, neurons=64, batch=100, iterations=10000, BUFF=4):
  sz = np.sqrt(N)

  # num_images = batch_size*500
  # IMAGES = extract_patches('./images/vanhateran/', size=M, num_patches=num_images)

  IMAGES = scipy.io.loadmat('../images/IMAGES.mat')['IMAGES']
  (imsize, imsize, num_images) = np.shape(IMAGES)

  I = np.zeros((N,batch))

  rect = NonLinearity(neurons=neurons)
  retina = RetinalGanglionCells(N, neurons, 0.4, 2, rect)

  for t in range(iterations+1):

    for i in range(batch):
      # choose a random image
      imi = np.ceil(num_images * random.uniform(0,1))
      r = BUFF + np.ceil((imsize-sz-2*BUFF) * random.uniform(0,1))
      c = BUFF + np.ceil((imsize-sz-2*BUFF) * random.uniform(0,1))

      I[:,i] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi-1],N,1)

    C_x = compute_cov(I)

    # print "Before update: " + str(retina.conditional_entropy(I, C_x))
    oldW = retina.W
    retina.update_W(I, C_x)
    # print "After update: " + str(retina.conditional_entropy(I, C_x))

    if np.mod(t,100) == 0:
      display(t, retina.W, neurons, np.sqrt(N))
      print "After update: " + str(retina.conditional_entropy(I, C_x))





