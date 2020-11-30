import numpy as np
import scipy.special as spec



def target_bump(X, delta=0.5, bump_type='single'):
    d = X.shape[1]
    w = np.random.randn(d)
    w = w / np.linalg.norm(w)
    if bump_type == 'single':
        return np.float64(X.dot(w) >= 1. - delta)
    elif bump_type == 'even':
        return np.float64(np.abs(X.dot(w)) >= 1. - delta)
    elif bump_type == 'odd':
        return np.float64(X.dot(w) >= 1. - delta) - np.float64(X.dot(w) <= -1. + delta)
    
def target_cone(X, delta=0.5, bump_type='single'):
    d = X.shape[1]
    w = np.random.randn(d)
    w = w / np.linalg.norm(w)
    if bump_type == 'single':
        return (1. - np.abs(delta * X.dot(w) - 1.) / delta).clip(min=0.)
    elif bump_type == 'even':
        return np.float64(np.abs(X.dot(w)) >= 1. - delta)
    elif bump_type == 'odd':
        return np.float64(X.dot(w) >= 1. - delta) - np.float64(X.dot(w) <= -1. + delta)

def target_lap(X, c=1., gamma=0.5, ty='single'):
    d = X.shape[1]
    w = np.random.randn(d)
    w = w / np.linalg.norm(w)
    if ty == 'single':
        return np.exp(-c * (1 - X.dot(w)) ** gamma)
    elif ty == 'even':
        return np.exp(-c * (1 - X.dot(w)) ** gamma) + np.exp(-c * (1 + X.dot(w)) ** gamma)
    
def target_kappa1(X, ty='single'):
    d = X.shape[1]
    w = np.random.randn(d)
    w = w / np.linalg.norm(w)
    if ty == 'single':
        return kappa1(X.dot(w))
    elif ty == 'even':
        return kappa1(X.dot(w)) + kappa1(-X.dot(w))

def rand_data(n, d):
    X = np.random.randn(n, d)
    return X / np.linalg.norm(X, axis=1)[:,None]

def lin_preds(features, y, lmbda=0.):
    n = features.shape[0]
    return features.dot(np.linalg.solve(features.T.dot(features) + n * lmbda * np.eye(features.shape[1]), features.T.dot(y)))

def lin_model(features, y, lmbda=0.):
    n = features.shape[0]
    return np.linalg.solve(features.T.dot(features) + n * lmbda * np.eye(features.shape[1]), features.T.dot(y))

def preds_rf_relu(X, y, nrf=100, lmbda=0.):
    n, d = X.shape
    weights = np.random.randn(d, nrf)
    features = (X.dot(weights)).clip(min=0) / np.sqrt(nrf)
    
    return lin_preds(features, y, lmbda)

def preds_rf_relu_bias(X, y, nrf=100, lmbda=0.):
    n, d = X.shape
    weights = np.random.randn(d, nrf)
    features = (X.dot(weights)).clip(min=0) / np.sqrt(nrf)
    bfeatures = X[:,:,None] * features[:,None,:]
    features = np.hstack([features, bfeatures.reshape(n, d*nrf)])
    
    return lin_preds(features, y, lmbda)

def preds_rf_relu_swap(X, y, nrf=100, lmbda=0.):
    n, d = X.shape
    weights = np.random.randn(d, nrf)
    features = (X.dot(weights)).clip(min=0) / np.sqrt(nrf)
    features = (X[:,:,None] * features[:,None,:]).reshape(n, d*nrf)
    
    return lin_preds(features, y, lmbda)

def preds_rf_step(X, y, nrf=100, lmbda=0.):
    n, d = X.shape
    weights = np.random.randn(d, nrf)
    features = np.heaviside(X.dot(weights), 0) / np.sqrt(nrf)
    
    return lin_preds(features, y, lmbda)

def preds_rf_step_bias(X, y, nrf=100, lmbda=0.):
    n, d = X.shape
    weights = np.random.randn(d, nrf)
    features = np.heaviside(X.dot(weights), 0) / np.sqrt(nrf)
    bfeatures = X[:,:,None] * features[:,None,:]
    features = np.hstack([features, bfeatures.reshape(n, d*nrf)])
    
    return lin_preds(features, y, lmbda)

def preds_rf_step_swap(X, y, nrf=100, lmbda=0.):
    n, d = X.shape
    weights = np.random.randn(d, nrf)
    features = np.heaviside(X.dot(weights), 0) / np.sqrt(nrf)
    features = (X[:,:,None] * features[:,None,:]).reshape(n, d*nrf)
    
    return lin_preds(features, y, lmbda)

def preds_rf_2layer_relu(X, y, nrf_in=16, nrf_out=100, lmbda=0.):
    n, d = X.shape
    weights_in = np.random.randn(d, nrf_in)
    activations = (X.dot(weights_in)).clip(min=0) / np.sqrt(nrf_in)
    weights_out = np.random.randn(nrf_in, nrf_out)
    features = (activations.dot(weights_out)).clip(min=0) / np.sqrt(nrf_out)
    
    return lin_preds(features, y, lmbda)




####### FEATURES #########
def features_rf_relu(X, nrf=100):
    n, d = X.shape
    weights = np.random.randn(d, nrf)
    features = (X.dot(weights)).clip(min=0) / np.sqrt(nrf)
    return features

def features_rf_relu_bias(X, nrf=100.):
    n, d = X.shape
    weights = np.random.randn(d, nrf)
    features = (X.dot(weights)).clip(min=0) / np.sqrt(nrf)
    bfeatures = X[:,:,None] * features[:,None,:]
    features = np.hstack([features, bfeatures.reshape(n, d*nrf)])
    return features

def features_rf_relu_swap(X, nrf=100):
    n, d = X.shape
    weights = np.random.randn(d, nrf)
    features = (X.dot(weights)).clip(min=0) / np.sqrt(nrf)
    features = (X[:,:,None] * features[:,None,:]).reshape(n, d*nrf)
    return features

def features_rf_step(X, nrf=100):
    n, d = X.shape
    weights = np.random.randn(d, nrf)
    features = np.heaviside(X.dot(weights), 0) / np.sqrt(nrf)
    return features

def features_rf_step_bias(X, nrf=100):
    n, d = X.shape
    weights = np.random.randn(d, nrf)
    features = np.heaviside(X.dot(weights), 0) / np.sqrt(nrf)
    bfeatures = X[:,:,None] * features[:,None,:]
    features = np.hstack([features, bfeatures.reshape(n, d*nrf)])
    return features

def features_rf_step_swap(X, nrf=100):
    n, d = X.shape
    weights = np.random.randn(d, nrf)
    features = np.heaviside(X.dot(weights), 0) / np.sqrt(nrf)
    features = (X[:,:,None] * features[:,None,:]).reshape(n, d*nrf)
    return features

def features_rf_2layer_relu(X, nrf_in=16, nrf_out=100):
    n, d = X.shape
    weights_in = np.random.randn(d, nrf_in)
    activations = (X.dot(weights_in)).clip(min=0) / np.sqrt(nrf_in)
    weights_out = np.random.randn(nrf_in, nrf_out)
    features = (activations.dot(weights_out)).clip(min=0) / np.sqrt(nrf_out)
    return features


####### Kernels #########

def kappa0(u):
    return 1.- np.arccos(u) / np.pi

def kappa1(u):
    return (np.sqrt(1 - u**2) + u * (np.pi - np.arccos(u))) / np.pi

def kappa0b(u):
    return (1. + u) * kappa0(u)

def kappa1b(u):
    return (1. + u) * kappa1(u)

def ck_3layer(u):
    return kappa1(kappa1(u))

def ck_4layer(u):
    return kappa1(ck_3layer(u))

def ntk(u):
    return 0.5 * (u * kappa0(u) + kappa1(u))

def ntkb(u):
    return 0.5 * ((u + 1.) * kappa0(u) + kappa1(u))

def ntk_3layer(u):
    return (2. * ntk(u) * kappa0(kappa1(u)) + kappa1(kappa1(u))) / 3.

def ntk_4layer(u):
    return 0.25 * (3. * ntk_3layer(u) * kappa0(ck_3layer(u)) + kappa1(ck_3layer(u)))

def gamma_exp(c=1., gamma=0.5):
    def kappa(u):
        return np.exp(-c * (1. - u) ** gamma)
    return kappa

def ckl(u, l):
    if l == 2:
        return kappa1(u)
    return kappa1(ckl(u, l - 1))

def ntkl(u, l):
    if l == 2:
        return u * kappa0(u) + kappa1(u)
    K = ckl(u, l-1)
    return ntkl(u, l-1) * kappa0(K) + kappa1(K)
