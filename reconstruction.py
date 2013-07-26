"""
Learn optimal receptive fields for gazes on natural images using 
"""

from retina import *

def gist(image, scale=16):
  return skt.pyramid_expand(tuple(skt.pyramid_gaussian(image, downscale = scale))[1], upscale = scale)

def saliency(image):
  #from src.utils import OpencvIo
  from src.saliency_map import SaliencyMap

  # Black and white image so copy over to all channels
  image = np.dstack((image, image, image))
  image = image.astype('float32')
  sm = SaliencyMap(image)
  return sm.map

def linear_theano():
  # S(x,y) * norm((I(x,y) - As).^2)) + lambda <s>
  import theano
  import theano.tensor as T

  tW = T.dmatrix("tW")
  tA = T.dmatrix("tA")
  tI = T.dmatrix("tI")
  ts = T.dmatrix("tS")

  E = (tW * (tI - theano.dot(tA,ts))**2).sum() + abs(ts).sum
  gE_W = T.grad(E, tW)

  e = theano.function([tA, ts, tI, tW], E)
  ge_w = theano.function([tA, ts, tI, tW], gE_W)

def linear():
  fixations = 3  # Number of fixations allowed
  radius = 64 # Radius size of eye (in pixels)
  N = 144 # Number of retinal ganglion cells
  logp = Retina(radius, N, 'log-polar', fovea_cutoff=7)
  logp.plot_lattice()

  IMAGES = scipy.io.loadmat('./IMAGES/IMAGES.mat')
  IMAGES = IMAGES['IMAGES']
  (image_size, image_size, num_images) = IMAGES

  num_images = 1

  for i in range(num_images):
    S = saliency(IMAGES[:,:,i])

    # Get starting position randomly
    (x,y) = np.floor(image_size * np.random.rand(2) *
    ((image_size - 2*radius)/float(image_size)) + radius).tolist()

    R = np.zeros((image_size, image_size))

    for j in range(fixations):
      # Inference
      I = IMAGES[x-radius:x+radius,y-radius:y+radius,i]
      (R[x-radius:x+radius,y-radius:y+radius], ) = logp.reconstruct(I, lambdav=0.01)

      # Move in direction with highest saliency
      saliency_samples = logp.sample(x, y, S)
      center = logp.receptors[np.argmax(saliency_samples)].rel_pos(x, y)

    # Compute learning step based on fixation strategy over based on moving towards
    error = S*(I - R)

    # Learning Step
    dW = np.dot(error, a.T)


def main():
  fixations = 1  # Number of fixations allowed
  radius = 64 # Radius size of eye (in pixels)
  N = 144 # Number of retinal ganglion cells
  logp = Retina(radius, N, 'log-polar', fovea_cutoff=7)
  logp.plot_lattice()

  image_size = 512

  #img = preprocess('./images/imk00466.imc')
  IMAGES = scipy.io.loadmat('./IMAGES/IMAGES.mat')
  IMAGES = IMAGES['IMAGES']
  img = IMAGES[:,:,1]
  #plt.imshow(img, cmap = 'gray', interpolation='nearest')
  # img=mpimg.imread('./images/1.jpg')

  #sm = saliency(gist(img))
  #plt.imshow(img, cmap='gray', interpolation='nearest')
  #plt.show()

  start = np.floor(image_size * np.random.rand(2,1) *
    ((image_size - 2*radius)/float(image_size)) + radius)
  for i in range(fixations):
    #S[:,i] = logp.sample(int(start[0]), int(start[1]), img)

    # reconstruct image
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    print str(int(start[0]-radius)) + " to " + str(int(start[0]+radius))
    I = img[int(start[0]-radius):int(start[0]+radius),int(start[1]-radius):int(start[1]+radius)]
    reconstruction = logp.reconstruct(I, lambdav=0.001)
    residual = I - reconstruction
    ax1.imshow(I, cmap='gray', interpolation='nearest')
    ax2.imshow(reconstruction, cmap='gray', interpolation='nearest')
    ax3.imshow(residual, cmap='gray', interpolation='nearest')
    ax4.imshow(logp.W[:,1].reshape(2*radius,2*radius), cmap = 'gray', interpolation='nearest')

    # moving to area with high saliency
    #start = logp.receptors[np.argmax(S[:,i], axis=0)].rel_pos(start[0], start[1])
    print "Next position: " + str(start)
    #print logp.next_fixation(int(start[0]), int(start[1]), sm)

  #logp.plot_lattice()

  # to do fix noise: SNR?
  # log  polar sampling lattice - why so much power in center?
  # fix parameters of log lattice
  # how to move gaze - difference of neighboring samples?
  # non constant weight average in receptive fields

  # optimizing photoreceptors - greedy approach, choose one at a time

  # frank warblen
  # attentional search

  #S = np.zeros((N,fixations)) # Observation matrix of samples


  #logp.plot_lattice()
  #plt.imshow(np.sum(logp.convert_to_array()[:,1].reshape(2*radius,2*radius,1), axis=2), cmap = 'gray', interpolation='nearest')

  #uniform = Retina(radius, 100, 'uniform')
  #uniform.plot_lattice()

  # on\off surround receptive fields


  """
  logp.plot_lattice()

  S = np.zeros((N,fixations)) # Observation matrix of samples
  for i in range(fixations):
    # Choose random start point (buffer = radius/2 + 2 pixels border)
    start = np.floor(image_size * np.random.rand(2,1) * \
    ((image_size - 2*radius)/float(image_size)) + radius)
    S[:,i] = logp.sample(start[0], start[1], img)
    # moving to area with high power
    # start = logp.receptors[np.argmax(S[:,i], axis=0)].rel_pos(start[0], start[1])
    print start

  m = np.mean(S,1).reshape(N, 1)
  C = 1/float(fixations) * np.dot(S - np.dot(m, np.ones((1,fixations))),
    (S - np.dot(m, np.ones((1,fixations)))).T)

  # pg 483, Haykin
  sample_entropy = 0.5 * (N + N * np.log(2*np.pi) + np.linalg.slogdet(C)[1])
  print sample_entropy

  # img_vector = img.reshape(1024*1535)
  # finding eig decomposition of http://www.mathworks.com/matlabcentral/newsreader/view_thread/320655

  # hist, bins = np.histogram(r,bins = 256)
  # width = 0.7*(bins[1]-bins[0])
  # center = (bins[:-1]+bins[1:])/2
  # plt.bar(center, hist, align = 'center', width = width)
  # plt.show()

  """
