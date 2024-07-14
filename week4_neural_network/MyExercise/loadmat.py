import sys
import numpy
from scipy.io import loadmat

numpy.set_printoptions(threshold=sys.maxsize)
data1 = loadmat('ex3data1.mat')
print(data1)