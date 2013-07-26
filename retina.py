import random
import scipy.io
import numpy as np
import array
import matplotlib.pyplot as plt

import skimage.transform as skt
#import skimage.filter as skf

from joblib import Memory
memory = Memory(cachedir='/tmp/retina', verbose=0)

from sparsenet import sparsify

class Retina(object):
  def __init__(self, radius=100, N=144, type='uniform', fovea_cutoff=None):
    self.radius = radius # Eye size (in pixels).
    self.N = N  # Number of receptors
    self.sqrt_N = np.sqrt(self.N).astype(int)
    self.receptors = [] # List of receptor objects
    self.type = type
    self.fovea_cutoff = fovea_cutoff # (optional)

    if self.type == 'uniform':
      # Top left corner
      stride = np.ceil((2.0*radius)/self.sqrt_N)
      x = -radius + np.ceil(stride/2)
      y = radius - np.ceil(stride/2)

      for i in range(self.sqrt_N):
        for j in range(self.sqrt_N):
          outer = ((i == self.sqrt_N-1 or j == self.sqrt_N-1) or
            (i == 0 or j == 0)) # Check if receptor on outer boundaries
          self.receptors.append(Receptor(x + stride*i, y - stride*j,
          'circle', stride.astype(int)/2 + 1, outer))

    elif self.type == 'log-polar':
      """
      Based on Sandini & Tagliasco 1980 model for retinal sampling lattice
      radius = scale*2*pi/3N * eccentricity. Where N is the # elements at a given
      eccentricity. scale factor is own addition to make things work at small level.

      Set N = sqrt(# receptors) per level => levels = sqrt(# receptors)

      fovea_cutoff = 3N/2*pi * sqrt(1/pi)
      In fovea, we sample at every pixel (note, this implies that there
      are actually more than # receptors)
      """
      scale = 2
      encircle = [(2*np.pi/(self.sqrt_N))*e for e in range(self.sqrt_N)]

      eccentricity = []
      size = []
      edge = 180
      for r in range(self.sqrt_N):
        # Fill inwards & RGCs circle edges touch prev edge
        e = edge/(scale*(2.0*np.pi)/(3.0*len(encircle)) + 1.0)
        s = scale*(2.0*np.pi*e)/(3.0*len(encircle))
        eccentricity.append(e)
        size.append(s)
        edge = edge - 1.0*s

      pixel_per_deg = radius/180.0 # pixels per eccentricity degree
      radii = [int(np.ceil(e * pixel_per_deg)) for e in eccentricity]
      size = [int(np.ceil(s * pixel_per_deg)) for s in size]

      if self.fovea_cutoff == None:
        self.fovea_cutoff = (3.0*len(encircle))/(scale*2.0*np.pi) * np.sqrt(1.0/np.pi) * pixel_per_deg

      for r,s in filter(lambda (r,s): r >= self.fovea_cutoff, zip(radii,size)):
          for theta in encircle:
              self.receptors.append(Receptor(np.floor(r*np.cos(theta)), np.floor(r*np.sin(theta)),
              'circle', int(np.ceil(max(1, s))), outer=(r==radii[0])))

      for i in range(-self.fovea_cutoff+1,self.fovea_cutoff):
        for j in range(-self.fovea_cutoff+1,self.fovea_cutoff):
          if np.sqrt(i**2 + j**2) < self.fovea_cutoff:
            self.receptors.append(Receptor(i, j, 'circle', 1, outer='False'))

    self._convert_to_array()

  def _convert_to_array(self):
    """
    Convert fom receptor list representation to array of receptive fields
    """
    self.W = np.zeros((2*self.radius,2*self.radius,len(self.receptors)))
    for i,receptor in enumerate(self.receptors):
      pixels = receptor.sample_pixels_pos(0,0)
      normalizing = len(pixels)
      for p in pixels:
        self.W[p[0]+self.radius-1, p[1]+self.radius-1, i] = 1.0/normalizing * (1/(np.abs(p[0]+p[1])+1))
    # Reshape to make 2-D
    self.W = self.W.reshape(2*self.radius*2*self.radius,len(self.receptors))

  def sample(self, center_x, center_y, image):
    """
    Return vector of samples (averaged photoreceptor values for each RGC)
    """
    output = np.zeros(self.N)
    for i,receptor in enumerate(self.receptors):
      output[i] = receptor.sample_pixels(center_x, center_y, image)
    return output

  def reconstruct(self, image_patch, lambdav, method='lstsq'):
    """
    Assume linear generative & sparse model. Learn coefficients to reconstruct
    image patch.
    image: Image patch
    lambdav: Sparsity constraint
    """

    I = image_patch.reshape(image_patch.shape[0]*image_patch.shape[1],)

    if method == 'lstsq':
      a = np.linalg.lstsq(self.W,I)
    elif method == 'l1':
      a = sparsify(image_patch.reshape(image_patch.shape[0]*image_patch.shape[1],1), self.W, lambdav)

    return np.dot(self.W,a[0]).reshape(image_patch.shape[0],image_patch.shape[1])

  def plot_lattice(self):
    fig, ax = plt.subplots()
    ax.scatter([receptor.abs_pos()[0] for receptor in self.receptors],
      [receptor.abs_pos()[1] for receptor in self.receptors], s=10, color='tomato')

    for receptor in self.receptors:
      pixels = receptor.sample_pixels_pos(0, 0)
      ax.scatter(pixels[:,0], pixels[:,1], s=1, alpha=0.2)

    plt.draw()


  def next_fixation(self, center_x, center_y, saliency_map):
    """
    Sample saliency map and move in *outer* direction with highest saliency.
    Look at saliency for outer level of receptors
    """
    avg_saliency = 0
    for receptor in self.receptors:
      if receptor.outer == True:
        if receptor.sample_pixels(center_x, center_y, saliency_map) > avg_saliency:
          direction = receptor.pos(center_x, center_y)
          avg_saliency = receptor.sample_pixels(self, center_x, center_y, image)

    return direction

class Receptor(object):
  def __init__(self, x, y, type, radius, outer=False, noise=False):
    self.x = x
    self.y = y
    self.type = type # circle
    self.radius = radius
    self.outer = outer # whether receptor on boundary or not
    self.noise = noise
    self.pixels = _circle_pixels(self.radius)
    if self.noise != False:
      self.mean = self.noise[0]
      self.std = self.noise[1] # If there is noise, std and mean of noise as tuple

  def abs_pos(self):
    return (self.x, self.y)

  def rel_pos(self, center_x, center_y):
    return (self.x + center_x, self.y + center_y)

  def sample_pixels_pos(self, center_x, center_y):
    # Returns list of pixels to sample given center_x and center_y
    if self.type == 'circle':
      return self.pixels + np.tile([[self.x + center_x, self.y + center_y]],(self.pixels.shape[0],1))

  def sample_pixels(self, center_x, center_y, image):
    output = 0
    sampling_pixels = self.sample_pixels_pos(center_x, center_y)

    for pixel in sampling_pixels:
      output = output + image[int(pixel[0]), int(pixel[1])]

      if self.noise != False:
        output = output + self.std * np.random.randn() + self.mean

    return output/float(len(sampling_pixels))

  def weights(self):
    if self.type == 'circle':
      return np.ones(len(self.pixels))/float(len(self.pixels))
    else:
      return None

@memory.cache
def _circle_pixels(radius):
  pixels = np.zeros((1,2))
  # Return matrix (pixels by (x,y)) of all pixels in circle from (-radius, radius)
  for i in [r - radius for r in range(2*radius + 1)]:
    for j in [r - radius for r in range(2*radius + 1)]:
      if np.sqrt(i**2 + j**2) < radius:
        pixels = np.append(pixels,[[i,j]], axis = 0)
  return pixels[1:,:]

def preprocess(filename, database='vanhateran',log=False, scale=1):
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

    # Downsample to 128x128
    #if scale != 1:
    #  print "Scaling by factor: " + str(scale)
    #  img = tuple(skt.pyramid_gaussian(img, downscale=scale))[1]

    # Take log10 transform of data to make Gaussian
    #if log == True:
    #  img[img == 0] = 1
    #  img = np.log10(img)

    # Zero mean and unit variance
    img = img.reshape(R*R/scale**2,1)
    #img = img - np.mean(img)
    #img = img/np.std(img)

    return img.reshape(R/scale,R/scale)

def generate_epic(image):
  size = 10
  noise = np.random.random_sample((size*2,size*2)) * 5

  noisy = image.copy()
  # Starting: 64,20
  noisy[64-size:64+size, 20-size:20+size] = noisy[64-size:64+size, 20-size:20+size] + noise

  return(image, noisy)

