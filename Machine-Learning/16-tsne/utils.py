import numpy
import scipy,scipy.spatial
import sklearn,sklearn.datasets

# ----------------------------------------
# Compute the pairwise affinities
# ----------------------------------------

def getaffinity(X,target):

    N = len(X)
    T = 100

    def getp(D2n):
        up = numpy.exp(-0.5*D2n)
        up = up * (1-numpy.identity(len(up))) # do not include diagonal elements
        up = up + 1e-9                        # add numerical stability
        p  = up / up.sum(axis=1,keepdims=True)
        return p

    D2 = scipy.spatial.distance.cdist(X,X,'sqeuclidean')

    sqsigmas = numpy.logspace(-2,3,T)
    err = numpy.zeros([T,N])

    for i in range(T):

        p = getp(D2/sqsigmas[i])
        
        entropy = -(p*numpy.log2(p)).sum(axis=1)

        perp = 2.0**entropy

        err[i] = (perp-target)**2

    sqsigmabest = numpy.array([sqsigmas[numpy.argmin(err[:,j])] for j in range(N)])
    
    p = getp(D2/sqsigmabest[:,numpy.newaxis])
    p = p + p.T
    p = p/p.sum()  
    return p

# ----------------------------------------
# Read a dataset
# ----------------------------------------

def get_data():

    D=sklearn.datasets.load_digits()
    # Read only 500 first examples
    X = D.data[:500]
    T = D.target[:500]

    # Normalize them
    X -= X.mean(axis=0)
    X /= (X**2).mean()**.5

    return X,T

