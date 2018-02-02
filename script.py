
# coding: utf-8


# import the packages
import numpy as np
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
from numpy import matlib
import math
from scipy import stats
import imageio
from skimage.transform import resize
import skimage
import zlib, sys
import gzip
import matplotlib
import scipy
import copy

# define a function to covert the image to a gray scale image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# define a function to get the proper Haar matrix and permutation matrix
def GetHaarMatrices(N):
    Q = np.matrix("[1,1;1,-1]")
    M = int(N/2)
    T = np.kron(matlib.eye(M),Q)/np.sqrt(2)
    P = np.vstack((matlib.eye(N)[::2,:],matlib.eye(N)[1::2,:]))
    return T,P

# reads in a jpeg image
A = imageio.imread('image.jpg')

# show the original image just read in
plt.imshow(A, cmap = plt.get_cmap('gray'))
plt.title("original image")
plt.show()

# resize the image(before apply gray scale function) as a 256 by 256 matrix
A = skimage.transform.resize(A, [256, 256], mode='constant')

# show the jpeg image in a figure
plt.imshow(A, cmap = plt.get_cmap('gray'))
plt.title("original image after resize")
plt.show()

# Apply the rgb2gray function to the image
A = rgb2gray(A)

# show the jpeg image in a figure
plt.imshow(A, cmap = plt.get_cmap('gray'))
plt.title("Gray-scale after resize")
plt.show()

# make a deep copy of resize&gray-scale image
B = copy.deepcopy(A)

# set size to 256
N = 256

# Doing full-level Encoding (Forward Haar Transform)
for i in range(int(np.log2(N))):
    T,P = GetHaarMatrices(N)
    #print(T.shape)
    B[0:N, 0:N] = P*T*B[0:N, 0:N]*T.T*P.T 
    N = int(N/2)

# show the result of full-level encoding
plt.figure()
plt.imshow(B[127:256,127:256], cmap = plt.get_cmap('gray'))
plt.title("Full-levelForward Haar Transform")
plt.show()

# print the info of B
print(B)

# make 2 deep copy of B
X = copy.deepcopy(B)
Y = copy.deepcopy(B)

# convert X(2D numpy array) into 1D numpy array
Y = Y.ravel()

# print the shape of reshaped X
print(Y.shape)

# create a codebook to store the sign of the numpy array elements
sign = np.ones(len(Y),)

# set the positive 1 to -1 if the correspond element in X is negative
for element in Y:
    if element < 0:
        element = -1
        
# print the sign codebook
print(sign)

# make a deep copy to X to get the threshold but not affect X
Z = copy.deepcopy(Y)


# sort the numpy array by its absolute value
Z = np.sort(abs(Z))

# promopt to ask user what the top percent pixel will retain the same
percent = input('How many percents of smallest elements you want to set to zero?')

# define thresholding function to find the threshold
def thresholding(source, percentage):
    index = 0
    index = math.floor(len(source) * percentage / 100)
    threshold = source[index]
    return threshold


# apply the thresholding function to find the threshold th
th = thresholding(Z, int(percent))
print(th)

# create an empty list to store the pixel which set to zeros
data = []

# implementation of the threshold process to numpy array X
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
            
        if X[i][j] > th:
            continue
        else:
            data.append(X[i][j])
            X[i][j] = 0

#print(len(data))

# show the image after apply to threshold
plt.imshow(X[127:256,127:256], cmap = plt.get_cmap('gray'))
plt.title("After Thresholding")
plt.show()

# print the matrix out the make sure A apply to the threshold function correctly
print(X)

# make a copy of image after thresholding as M
M = copy.deepcopy(X)

# read in M row by row, skip the element of 0
# and take binary log to the nonzero positive element
# set only one element in each container
def log_quantiz(inp):
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            
            if inp[i][j] == 0:
                continue
            else:
                inp[i][j] = math.log2(inp[i][j])

# Apply log_quantiz function to M
log_quantiz(M)

# show the image after apply to log quantization
plt.imshow(M[127:256,127:256], cmap = plt.get_cmap('gray'))
plt.title("After Log-quantization")
plt.show()

#print(M)

# make a copy of image after thresholding as N
N = copy.deepcopy(M)

# start of the lossless compression by using package
# compress the image 
compressed_data = zlib.compress(N, 9)
compress_ratio = float(sys.getsizeof(compressed_data))/sys.getsizeof(N)

# print out the percent of lossless compression
#print("Size before compress:", sys.getsizeof(N))
#print("Size after compress:", sys.getsizeof(compressed_data)) 
print("compress_ratio:", compress_ratio * 100, "%")

# ----------------------------------------------------------------
# start of decompressed image 
# ----------------------------------------------------------------

# convert the lossless compressed image by using zlib  
decompressed_data = zlib.decompress(compressed_data)
print(sys.getsizeof(decompressed_data))

# convert the bytes type to numpy array
decompressed_data = np.frombuffer(decompressed_data)

# to check that we won't loss any info since of compression and decompression
print(decompressed_data == M.ravel())

# convert the 1D decompressed data into a 2D numpy array E
E = np.reshape(decompressed_data, (256, 256))

# show the image before reverse log quantization
plt.imshow(E[127:256,127:256], cmap = plt.get_cmap('gray'))
plt.title("decompress the compressed data")
plt.show()

# reverse log quantization to E
# make a deep copy of E as F
F = copy.deepcopy(E)

# read in F row by row, skip the element of 0
# and take binary power to the nonzero positive element
# set only one element in each container
def reverse_log_quantiz(inp):
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            
            if inp[i][j] == 0:
                continue
            else:
                inp[i][j] = math.pow(2, inp[i][j])
                
# Apply reverse_log_quantiz function to M
reverse_log_quantiz(F)

# show the image after apply to reverse log quantization
plt.imshow(F[127:256,127:256], cmap = plt.get_cmap('gray'))
plt.title("reverse of log-quantization")
plt.show()

print(F)

# reverse threshold to F
# make a deep copy of F as G
G = copy.deepcopy(F)

# read in F row by row, find the min nonzero pixel
# put the number from data codebook before apply thresholding function
# in order to put the data back to nonzero
def reverse_thresholding(source, preplacement):
    index = 0
    
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            
            if source[i][j] == 0:
                source[i][j] = data[index]
                index += 1
            else:
                continue
                
        
# Apply reverse thresholding function to M
reverse_thresholding(G, data)

# show the image after apply to reverse threshold
plt.imshow(G[127:256,127:256], cmap = plt.get_cmap('gray'))
plt.title("Reverse of Thresholding")
plt.show()

print(G)

# make a deep copy of G
J = copy.deepcopy(G)

# get number of times of decoding and the starting point 
N = len(J)
times = int(np.log2(N))
start = 2

# Doing full-level decoding (Backward Haar Transform)
for i in range(times):
    T,P = GetHaarMatrices(start)
    J[0:start, 0:start] = T.T*P.T*J[0:start, 0:start]*P*T 
    start = 2 * start
    

# show the result of full-level decoding
plt.figure()
plt.imshow(J, cmap = plt.get_cmap('gray'))
plt.show()

# print the info of J

print(J)

