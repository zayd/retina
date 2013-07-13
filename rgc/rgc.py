# - H(X|R) = -E[1/2 * ln(2*pi*det(C_(X|R)^i))]
import scipy.io
import numpy as np
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
    self.W = np.dot(self.W,np.diag(1/np.sqrt(np.sum(self.W * self.W, axis = 0))))
    self.eta = 50.0

    # Theano Variables
    tW = T.dmatrix("tW")
    tG = T.dmatrix("tG")
    tC_x = T.dmatrix("tC_x")
    tC_nx = T.dmatrix("tC_nx")
    tC_nr = T.dmatrix("tC_nr")

    # NEED TO ADD SPARSITY

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
    return np.dot(self.G(x),np.dot(np.dot(W.T,(x + self.__sample_noise(self.sigma_nx)))) +
      self.__sample_noise(self.sigma_nr)) + f_0

  def G(self, x):
    # Computes first order approximation to input x
    return np.diag(self.non_lin.derivative(self.W,x))

  def update_W(self, X, C_x):
    # Computes grad H(X|R) w.r.t. W and updates W over sample of images X
    # X: Sample of images 2-D array. Images by column
    # C_x: Covariance matrix of input data
    (N, M) = np.shape(X) # Dimensions, Images in batch

    update = 0
    for m in range(M):
      update = update + (1.0/M)*self.gce_w(self.W, self.G(X[:,m]),
        C_x, self.C_nx, self.C_nr)

    self.W = self.W + (self.eta/M) * update

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

  def __init__(self, type='exponential', neurons=100, a=None, b=None, c=None):
    self.type = type
    self.neurons = neurons
    self.a = 0.1*np.ones(self.neurons) if a == None else a
    self.b = -1*np.ones(self.neurons) if b == None else b
    self.c = np.zeros(self.neurons) if c == None else c

  def constant(self):
    return c;

  def evaluate(self, W, x):
    if self.type =='exponential':
      # output = exp(a*(w^T*x) + b) + c
      return np.exp(self.a * np.dot(W.T, x) + self.b) + self.c

  def derivative(self, W, x):
    # derivative with respect to y = w^T*x
    if self.type == 'exponential':
      return self.a * np.exp(self.a * np.dot(W.T, x) + self.b)

def compute_cov(X):
  """Covariance of observation matrix X"""

  [N, neurons] = X.shape
  m = np.mean(X,1).reshape(N, 1)
  C = 1/float(neurons) * np.dot(X - np.dot(m, np.ones((1,neurons))),
    (X - np.dot(m, np.ones((1,neurons)))).T)
  # Add to diagonals to ensure matrix is PD
  return C + 0.04*np.eye(N,N)

def run(N=256, neurons=100, batch=100, iterations=10000, BUFF=4):

  sz = np.sqrt(N)

  IMAGES = scipy.io.loadmat('../images/IMAGES.mat')['IMAGES']
  (imsize, imsize, num_images) = np.shape(IMAGES)

  I = np.zeros((N,batch))

  exp = NonLinearity()
  retina = RetinalGanglionCells(N, neurons, 0.4, 2, exp)

  for t in range(iterations):

    for i in range(batch):
      # choose a random image
      imi = np.ceil(num_images * random.uniform(0,1))
      r = BUFF + np.ceil((imsize-sz-2*BUFF) * random.uniform(0,1))
      c = BUFF + np.ceil((imsize-sz-2*BUFF) * random.uniform(0,1))

      I[:,i] = np.reshape(IMAGES[r:r+sz, c:c+sz, imi-1],N,1)

    C_x = compute_cov(I)

    # print "Before update: " + str(retina.conditional_entropy(I, C_x))
    retina.update_W(I, C_x)
    # print "After update: " + str(retina.conditional_entropy(I, C_x))

    if np.mod(t,5) == 0:
      print "Iteration " + str(t)
      print "After update: " + str(retina.conditional_entropy(I, C_x))
      image = np.zeros((sz*np.sqrt(neurons)+np.sqrt(neurons),sz*np.sqrt(neurons)+np.sqrt(neurons)))
      for i in range(np.sqrt(neurons).astype(int)):
        for j in range(np.sqrt(neurons).astype(int)):
          image[i*sz+i:i*sz+sz+i,j*sz+j:j*sz+sz+j] = np.reshape(retina.W[:,i*np.sqrt(neurons)+j],(sz,sz))

      plt.imshow(image, cmap=cm.jet, interpolation="nearest")
      plt.draw()
      plt.show()





