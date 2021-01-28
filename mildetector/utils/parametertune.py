from __future__ import print_function
import numpy as np
from scipy.spatial.distance import cdist

def tune(gamma_ini, gamma_fin, step, train_bags, test_bags):
    _bags = [np.asmatrix(bag) for bag in train_bags]
    svm_X = np.vstack(_bags)
    text =""
    Y = np.asmatrix(svm_X)

    _bags = [np.asmatrix(bag) for bag in test_bags]
    svm_X = np.vstack(_bags)
    X = np.asmatrix(svm_X)

    step_num = int((gamma_fin - gamma_ini) / step) + 1
    K_list = []
    var_list = []
    prevVar = -9999
    for i in range(step_num):
        g = gamma_ini + step*i
        K = kernel_rbf(gamma=g, x=X, y=Y)
        var = np.var(K)
        print("gamma:{0} --> variance:{1}".format(g, var))
        text += "gamma:{0} --> variance:{1}\n".format(g, var)
        var_list.append(var)
        K_list.append(K)
        if prevVar > var:
            break
        else:
            prevVar = var
        #print(K)
    best_index = np.argmax(np.array(var_list))
    best_g = gamma_ini + step * best_index
    print("the biggest variance is {0}:{1}".format(best_g, var_list[best_index]))
    print("K is:")
    print(K_list[best_index])
    text += "the biggest variance is {0}:{1}\n".format(best_g, var_list[best_index])
    text += "K is:\n"

    return text, K_list[best_index], best_g

def kernel_rbf(gamma, x, y):
    return np.matrix(np.exp(-gamma * cdist(x, y, 'sqeuclidean')))

def polynomial(p):
    """General polynomial kernel (1 + x'*y)^p"""

    def p_kernel(x, y):
        return np.power(1e0 + x * y.T, p)

    return p_kernel