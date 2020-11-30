import argparse
import numpy as np
import scipy.linalg
import os
import sys


def arccos0(u):
    return (np.pi - np.arccos(u)) / np.pi

def arccos1(u):
    return (u * (np.pi - np.arccos(u)) + np.sqrt(1. - u * u)) / np.pi

def kernel_matrix(X, Y, args):
    xnorm = np.sqrt((X**2).sum(1))
    ynorm = np.sqrt((Y**2).sum(1))
    print('cov')
    K = X.dot(Y.T) / xnorm[:,None] / ynorm[None,:]
    K = K.clip(min=-1, max=1.)

    if args.lap: # laplace kernel
        K = np.exp(-args.lap_c * np.sqrt(1 - K))
        return xnorm[:,None] * ynorm[None,:] * K

    if args.ntk:
        Kntk = K.copy()
        if args.bias:
            Kntk += 1.

    for l in range(args.nlayers - 1):
        print('l = ', l)
        if args.ntk:
            Kntk = Kntk * arccos0(K) + arccos1(K)
            K = arccos1(K)
        else:
            K = arccos1(K)

    if args.ntk:
        K = Kntk

    if args.normalized:
        return K
    else:
        return xnorm[:,None] * ynorm[None,:] * K


def make_dataset():
    import _infimnist as infimnist
    mnist = infimnist.InfimnistGenerator()
    indexes_test = np.arange(10000, dtype=np.int64)
    digits, yte = mnist.gen(indexes_test)
    Xte = digits.astype(np.float32).reshape(indexes_test.shape[0], 28, 28, 1)
    Xte = Xte / 255

    indexes_train = np.arange(10000, 70000, dtype=np.int64)
    digits, ytr = mnist.gen(indexes_train)
    X = digits.astype(np.float32).reshape(indexes_train.shape[0], 28, 28, 1)
    X = X / 255

    np.save('mnist_xtr.npy', X)
    np.save('mnist_ytr.npy', ytr)
    np.save('mnist_xte.npy', Xte)
    np.save('mnist_yte.npy', yte)


def get_dataset(ds):
    if ds == 'mnist':
        from torchvision.datasets import MNIST
        dstr = MNIST('mnist_data', download=True, train=True)
        dste = MNIST('mnist_data', download=True, train=False)
    elif ds == 'fmnist':
        from torchvision.datasets import FashionMNIST
        dstr = FashionMNIST('fmnist_data', download=True, train=True)
        dste = FashionMNIST('fmnist_data', download=True, train=False)

    Xtr = dstr.train_data.numpy().astype(np.float32) / 255
    ytr = dstr.train_labels.numpy()
    Xte = dste.test_data.numpy().astype(np.float32) / 255
    yte = dste.test_labels.numpy()

    return Xtr, ytr, Xte, yte


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mnist fc kernel')
    parser.add_argument('--ntrain', type=int, default=60000)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ntk', action='store_true')
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--fmnist', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--lap', action='store_true')
    parser.add_argument('--lap_c', type=float, default=1.)
    parser.add_argument('--start_lmbda', type=int, default=0)
    args = parser.parse_args()

    if args.fmnist:
        ds = 'fmnist'
    else:
        ds = 'mnist'

    if args.save:
        if args.lap:
            res_fname = f'{ds}{args.seed}_lap_{args.ntrain}_{args.lap_c}.pkl'
        else:
            if args.ntk and args.bias:
                kern = 'ntkb'
            elif args.ntk:
                kern = 'ntk'
            else:
                kern = 'rf'
            res_fname = f'{ds}{args.seed}_{kern}_{args.ntrain}_{args.nlayers}.pkl'
        if os.path.exists(os.path.join('res', res_fname)):
            print('already done!')
            sys.exit(0)

    nval = 10000
    np.random.seed(42 + args.seed)
    idx = np.random.permutation(60000)
    assert(args.ntrain <= 60000 - nval)
    idx_tr, idx_val = idx[:args.ntrain], idx[-nval:]

    # X = np.load(f'{ds}_xtr.npy').astype(np.float32)
    X, ytr, Xte, yte = get_dataset(ds)
    X = X.reshape(X.shape[0], -1)
    X, Xval = X[idx_tr], X[idx_val]
    # Xte = np.load(f'{ds}_xte.npy').astype(np.float32)
    Xte = Xte.reshape(Xte.shape[0], -1)

    # ytr = np.load(f'{ds}_ytr.npy')
    ytr, yval = ytr[idx_tr], ytr[idx_val]
    # yte = np.load(f'{ds}_yte.npy')

    y = np.zeros((args.ntrain, 10))
    y[np.arange(args.ntrain), ytr] = 1.
    y -= 0.1

    print('computing train matrix')
    Ktrain = kernel_matrix(X, X, args)
    print('computing val matrix')
    Kval = kernel_matrix(X, Xval, args)
    print('computing test matrix')
    Ktest = kernel_matrix(X, Xte, args)
    I = np.eye(args.ntrain)

    best_acc = -1
    best_val = -1
    best_val_test = None
    # lmbdas = 10. ** np.arange(0, -9, -1)
    lmbdas = 10. ** np.arange(args.start_lmbda, -4, -1)
    accs_val = []
    accs = []
    for lmbda in lmbdas:
        print(lmbda)
        alpha = scipy.linalg.solve(Ktrain + args.ntrain * lmbda * I, y)
        yhat_val = Kval.T.dot(alpha)
        ypred_val = yhat_val.argmax(axis=1)
        acc_val = np.mean(ypred_val == yval)

        yhat = Ktest.T.dot(alpha)
        ypred = yhat.argmax(axis=1)
        acc = np.mean(ypred == yte)
        print('val, test accuracies:', acc_val, acc)
        accs_val.append(acc_val)
        accs.append(acc)
        if acc_val > best_val:
            best_val = acc_val
            best_val_test = acc
        best_acc = max(acc, best_acc)
    print('best acc:', best_acc, 'best validated acc:', best_val_test)

    if args.save:
        import pickle
        res = {'lmbdas': lmbdas, 'accs': accs, 'accs_val': accs_val, 'best_acc': best_acc, 'best_val_acc': best_val_test}

        pickle.dump(res, open(os.path.join('res', res_fname), 'wb'))
