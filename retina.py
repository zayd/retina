import random
import numpy as np
import array
import matplotlib.pyplot as plt

import skimage.transform as skt
import skimage.filter as skf

from joblib import Memory
memory = Memory(cachedir='/tmp/retina', verbose=0)

def memoize(f):
	cache= {}
	def memf(*x):
		if x not in cache:
			cache[x] = f(*x)
		return cache[x]
	return memf

# class Gaze(object):
#	def __init__(self):
#		return

class Retina(object):
	def __init__(self, radius=100, N=100, type='uniform'):
		self.radius = radius # Eye size (in pixels).
		self.N = N  # Number of receptors
		self.sqrt_N = np.sqrt(self.N).astype(int)
		self.receptors = [] # List of receptor objects
		self.type = type

		if self.type == 'uniform':
			# Top left corner
			side = np.floor(self.radius * np.cos(np.pi/4))
			stride = np.floor((2*side)/self.sqrt_N)
			x = -side + np.floor(stride/2)
			y = side - np.floor(stride/2)

			for i in range(self.sqrt_N):
				for j in range(self.sqrt_N):
					outer = ((i == self.sqrt_N-1 or j == self.sqrt_N-1) or
						(i == 0 or j == 0)) # Check if receptor on outer boundaries
					self.receptors.append(Receptor(x + stride*i, y - stride*j,
					'circle', stride.astype(int)/2 + 1, outer))

		elif self.type == 'log-polar':
			"""Make less hacky"""
			encircle = [(2*np.pi/self.sqrt_N)*e for e in range(self.sqrt_N)]
			radii = [np.exp(r/float(2)+1) for r in range(self.sqrt_N)]

			radii.insert(0, 0) # Start with radius 0. (i,e, points)
			for q,r in zip(radii[1:], radii): # Difference of prev and cur radii
				for theta in encircle:
					self.receptors.append(Receptor(np.floor(r*np.cos(theta)), np.floor(r*np.sin(theta)),
					'circle', np.round((q-r)/float(2.25)+1).astype(int), outer=(q == radii[-1])))

	def sample(self, center_x, center_y, image):
		output = np.zeros(self.N)
		for i,receptor in enumerate(self.receptors):
			output[i] = receptor.sample_pixels(center_x, center_y, image)
		return output

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

	def convert_to_array(self):
		"""TODO"""
		W = np.zeros((2*self.radius,2*self.radius,len(self.receptors)))
		return None

class Receptor(object):
	def __init__(self, x, y, type, radius, outer):
		self.x = x
		self.y = y
		self.type = type # circle
		self.radius = radius
		self.outer = outer # whether receptor on boundary or not

	def abs_pos(self):
		return (self.x, self.y)

	def rel_pos(self, center_x, center_y):
		return (self.x + center_x, self.y + center_y)

	def sample_pixels_pos(self, center_x, center_y):
		# Returns list of pixels to sample given center_x and center_y
		if self.type == 'point':
			return (self.x + center_x, self.y + center_y)

		elif self.type == 'circle':
			pixels = _circle_pixels(self.radius)
			return pixels + np.tile([[self.x + center_x, self.y + center_y]],(pixels.shape[0],1))

	def sample_pixels(self, center_x, center_y, image):
		# TODO: make work with 'point'
		output = 0
		pixels = self.sample_pixels_pos(center_x, center_y)
		for pixel in pixels:
			output = output + image[int(pixel[0]), int(pixel[1])]

		return output/float(len(pixels))

@memory.cache
def _circle_pixels(radius):
	print "Called for " + str(radius)
	pixels = np.zeros((1,2))
	# Return matrix (pixels by (x,y)) of all pixels in circle from (-radius, radius)
	for i in [r - radius for r in range(2*radius + 1)]:
		for j in [r - radius for r in range(2*radius + 1)]:
			if np.sqrt(i**2 + j**2) < radius:
				pixels = np.append(pixels,[[i,j]], axis = 0)
	return pixels[1:,:]


def preprocess(filename):
	# Preprocess Van Hateran Natural Images
	R = 1024
	C = 1536
	extra = (C-R)/2

	fin = open(filename, 'rb')
	s = fin.read()
	fin.close()
	arr = array.array('H', s)
	arr.byteswap()
	img = np.array(arr, dtype='uint16').reshape(R,C)
	img = img[:,extra-1:C-extra-1] # crop to make square
	return img

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

def main():
	fixations = 5  # Number of fixations allowed
	radius = 400 # Radius size of eye (in pixels)
	N = 100 # Number of retinal ganglion cells

	image_size = 1024

	for i in range(1):
		img = preprocess('./images/imk04212.imc')
		#plt.imshow(img, cmap = 'gray', interpolation='nearest')
		# img=mpimg.imread('./images/1.jpg')

		sm = saliency(gist(img))
		#plt.imshow(img, cmap='gray', interpolation='nearest')
		#plt.show()

	#logp = Retina(radius, N, 'log-polar')
	#print "done"
	#logp.plot_lattice()
	#print "done"

	# to do fix noise: SNR?
	# log  polar sampling lattice - why so much power in center?
	# fix parameters of log lattice
	# how to move gaze - difference of neighboring samples?
	# non constant weight average in receptive fields

	# optimizing photoreceptors - greedy approach, choose one at a time

	# frank warblen
	# attentional search

	S = np.zeros((N,fixations)) # Observation matrix of samples

	logp = Retina(radius, N, 'uniform')
	start = np.floor(image_size * np.random.rand(2,1) * \
	((image_size - 2*radius)/float(image_size)) + radius)
	S[:,i] = logp.sample(int(start[0]), int(start[1]), img)
	# moving to area with high power
	# start = logp.receptors[np.argmax(S[:,i], axis=0)].rel_pos(start[0], start[1])
	print start
	print logp.next_fixation(int(start[0]), int(start[1]), sm)



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
