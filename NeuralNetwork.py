import numpy as np


def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

#Basic dataset to test (Where do I find good data??)
X = np.array([[0,0,0],
           [0,0,1],
           [0,1,0],
           [0,1,1]])
                
y = np.array([[0],
           [0],
           [1],
           [1]])

#seed so every time I have the same random generated values
np.random.seed(1)

# randomly initialize our weights with mean 0 (I made it so the X and y values can be any size)
syn0 = 2*np.random.random((len(X[0]),len(X))) - 1
syn1 = 2*np.random.random((len(X),len(y[0]))) - 1


for a in xrange(1):

    pass

#train the neural network (60000 itinerations)
for j in xrange(60000):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = y - l2
            
    # in what direction is the target value?
    # were we really sure? check it with the derivative of the sigma
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = np.dot(l2_delta,syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? check it with the derivative of the sigma
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += np.dot(l1.T,l2_delta)
    syn0 += np.dot(l0.T,l1_delta)


#test neural network with new data
TestArray=np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])

l0 = TestArray
l1 = nonlin(np.dot(l0,syn0))
l2 = nonlin(np.dot(l1,syn1))

#predict results
print('Predictions:{0}'.format(l2))