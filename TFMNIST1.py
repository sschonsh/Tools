#exec(open("E:/Dropbox/Research/PTCnet/TFMNIST/TFMNIST1.py").read()) 

#exec(open("C:/Users/sscho/Desktop/PTC_Demo/TFMNIST1.py").read()) 


#User Parameters###############################################################################################################

max_its    = 10000  #iterations
alpha      = 1e-4 #learning rate 
nKer1      = 16   #number of Kernels in first layers
gamma1     = .3#l2 regulaization

report =  100#How often to report loss

#Import packages###################################################################################################################
import os
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf 
import scipy.sparse as spsp
import scipy.io as spio
import time

#Funcitons
def weight_var(shape):
	initial = tf.truncated_normal(shape = shape, stddev = .1)
	return tf.Variable(initial)

def FC_var(shape):
	initial = tf.truncated_normal(shape = shape, stddev = .1)
	return tf.Variable(initial)
	
def Bias_var(shape):
	initial = tf.constant(0.01, shape=shape)
	return tf.Variable(initial)

def Cell_to_Sparse(C,flag):
	#Unpack into tensor
	if flag == 1:
		Ctemp = C.tocoo()
		Ctemp = Ctemp.transpose()
	else:
		Ctemp = C.tocoo()
	indices = np.mat([Ctemp.row, Ctemp.col]).transpose()
	return tf.SparseTensorValue(indices, Ctemp.data, Ctemp.shape) 

def WtoTensor(W):
	WF  = W[1]
	Wcoo = WF.tocoo()
	indices = np.mat([Wcoo.row, Wcoo.col]).transpose()
	return tf.SparseTensorValue(indices, Wcoo.data, Wcoo.shape)
	
def m_to_tensor(w):
    wcoo = (2 * w).tocoo()
    indices = np.mat([wcoo.row, wcoo.col]).transpose()
    return tf.SparseTensorValue(indices, wcoo.data, wcoo.shape)
    
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

    
def surface_conv(x, mass, weight, ind, mv, ds, l, kernel, n_ker1, n_ker2):
    z = tf.sparse_tensor_dense_matmul(mass, tf.transpose(tf.nn.relu(x)))
    z_long  = tf.reshape(tf.transpose(z), [784 * n_ker1])
    val     = tf.gather(z_long, mv, 0)
    zmr     = tf.SparseTensor(indices=ind, values=val, dense_shape=ds)
    wk      = tf.transpose(tf.sparse_tensor_dense_matmul(weight, kernel))
    wk_long = tf.reshape(wk, [l * n_ker1, n_ker2])
    c_long  = tf.sparse_tensor_dense_matmul(zmr, wk_long)
    c_full  = tf.reshape(c_long, [n_ker2, 784, n_ker1])
    f = tf.transpose(tf.reduce_sum(c_full, 2))
    return f

#Load Manifold info and Weight Matricies
mat1   = spio.loadmat('C:/Users/sscho/Desktop/PTC_Demo/ExSurf.mat', squeeze_me=True)
surf1  = mat1['Surf']
M_val  = surf1['M'][()]
M_mat  = m_to_tensor(M_val)
W1cell = surf1['Wfull1'][()] 
W1mat  = Cell_to_Sparse(W1cell,0)
R1cell = surf1['Rfull1'][()]
R1mat  = Cell_to_Sparse(R1cell,1)
L1mat  = surf1['L1'][()]
I1val  = np.asarray(surf1['Ifull1'][()])-1
I1RowIndex  = np.asarray(surf1['IRowIndex1'][()])-1#minus 1 for 0 indexing
I1ColIndex  = np.asarray(surf1['IColumnIndex1'][()])-1
#format for TF 
Vals1 = I1val
R1 = np.int64(I1RowIndex)
C1 = np.int64(I1ColIndex)
Indicies1 = np.transpose(np.mat([R1, C1]))
DenseShape1 = np.int64(np.asarray([28**2, len(C1)]))
    
	
	
#Model##########################################################################################################################################################################################################################

#Input
#1
x1   = tf.placeholder(tf.float32)
W1   = tf.sparse_placeholder(tf.float32)
Mass = tf.sparse_placeholder(tf.float32)
MV1  = tf.placeholder(tf.int64)
Ind1 = tf.placeholder(tf.int64)
DS1  = tf.placeholder(tf.int64)
L1   = tf.placeholder(tf.int32)
y    = tf.placeholder(tf.float32)
n_pts = tf.placeholder(tf.int32)

#Intialize
Ker1  = tf.transpose(weight_var([nKer1,13]))	
FC1   = weight_var([28**2*nKer1,10])
B1    = weight_var([10])

##Conv
C1 = surface_conv(x1, Mass, W1, Ind1, MV1, DS1, L1, Ker1, 1, 16)
#FC
C1Flat = tf.reshape(C1,[(28**2)*nKer1,1])
F1 = tf.matmul(tf.transpose(C1Flat),FC1)+B1	

##Loss
reg = tf.norm(Ker1)+tf.norm(B1)+tf.norm(FC1)
output = tf.nn.softmax(F1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y)+gamma1*reg


#Session and training#########################################################################################################


#Session
sess = tf.Session()
#define train method	
train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)
#Initiatlize and begin saver
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#Write info for graph
#writer = tf.summary.FileWriter("E:/Dropbox/Research/PTCnet/TFimplementation/Vis",sess.graph)
	
#Load MNSIT from tutorial data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	
	
#Training Loop
accuracy = []
print('Begining Training')
for i in range(max_its):
	#Choose data
	batch = mnist.train.next_batch(1)
	
	#Train Step
	train_step.run(session=sess, feed_dict={x1:batch[0],y:batch[1],Mass:M_mat,W1:W1mat,Ind1:Indicies1,MV1:Vals1,DS1:DenseShape1,L1:L1mat,n_pts:784})
	
	#Report
	if i%report == 0:
		count = 0;
		for j in range(50):
				batch = mnist.train.next_batch(1)
				ypred = F1.eval(session=sess, feed_dict={x1:batch[0],y:batch[1],W1:W1mat,Mass:M_mat,Ind1:Indicies1,MV1:Vals1,DS1:DenseShape1,L1:L1mat,n_pts:784})
				if np.argmax(ypred) == np.argmax(batch[1]):
					count = count+1
		acc = count/50
		accuracy.append(acc)
		print('i:',i,'acc:',acc)

#Testing####################################################################################
	
print('Begining Testing')
count = 0
for i in range(10000):
    batch = mnist.test.next_batch(1)
    ypred = F1.eval(session=sess, feed_dict={x1:batch[0],y:batch[1],W1:W1mat,Mass:M_mat,Ind1:Indicies1,MV1:Vals1,DS1:DenseShape1,L1:L1mat,n_pts:784})
    if np.argmax(ypred) == np.argmax(mnist.test.labels[i]):
        count = count+1
        if i%1000 == 0:
            print(i, 'Tests Complete')
acc = count/10000
print('Accuacy:', acc)


#Plot
its = np.arange(int(max_its/report))*report
plt.figure(0)
fig0,ax1 = plt.subplots()
ax1.semilogy(its,accuracy,'g-', label ="Loss")
ax1.set_ylabel('Accuracy', color='g')
ax1.tick_params('y', colors='g')
ax1.set_xlabel('Iterations')

plt.show()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	