import numpy as np
import matplotlib.pyplot as plt
import os, pickle
from nn_rf import *

plt.style.use('ggplot')

def relu_rf(recompute=False):
    fname = f'relu_rf_curves.pkl'
    if not recompute and os.path.exists(fname):
        curves = pickle.load(open(fname, 'rb'))
    else:
        curves = []
        np.random.seed(42)
        ns = [int(n) for n in 2. ** np.arange(1., 14., 0.4)]
        lmbdas = 2. ** np.arange(-30, 5, 3)

        nmax = np.max(ns)

        d = 4
        X = rand_data(nmax + 10000, d)

        bump_delta = 0.3
        bump_type = 'even'
        # y = target_bump(X, delta=bump_delta, bump_type=bump_type)
        # y = target_cone(X, delta=bump_delta, bump_type=bump_type)
        y = target_lap(X, gamma=1.5, ty=bump_type)

        Xtr, Xte = X[:nmax], X[nmax:]
        ytr, yte = y[:nmax], y[nmax:]


        experiments = [
            ('2-layer, m=sqrt(n)', lambda X, rtn: features_rf_relu(X, nrf=rtn)),
            ('3-layer (sqrt(n), sqrt(n))', lambda X, rtn: features_rf_2layer_relu(X, nrf_in=rtn, nrf_out=rtn)),
            ('3-layer (10, sqrt(n))', lambda X, rtn: features_rf_2layer_relu(X, nrf_in=10, nrf_out=rtn)),
            ('3-layer (sqrt(n), 10)', lambda X, rtn: features_rf_2layer_relu(X, nrf_in=rtn, nrf_out=10)),
        ]

        for name, feat_fn in experiments:
            err_reps = []
            for rep in range(20):
                errs = []
                for n in ns:
                    rootn = int(np.sqrt(n))
                    features_tr = feat_fn(Xtr, rootn)
                    features_te = feat_fn(Xte, rootn)

                    features = features_tr[:n]
                    targets = ytr[:n]
                    valid_errs = []
                    for lmbda in lmbdas:
                        preds = features_te.dot(lin_model(features, targets, lmbda=lmbda))
                        err = np.mean((preds - yte) ** 2)
                        valid_errs.append(err)
                    errs.append(np.min(valid_errs))

                err_reps.append(errs)
            if name.startswith('2'):
                o = '-'
            elif name.startswith('3'):
                o = '--'
            else:
                o = '-.'
            curves.append((ns, np.mean(err_reps, axis=0), o, name))

        pickle.dump(curves, open(fname, 'wb'))
        
    plt.figure(figsize=(4,3))
    for ns, vals, o, name in curves:
        plt.loglog(ns, vals, o, label=name)

    plt.title('ReLU random features')
    plt.xlabel('n')
    plt.ylabel(r'$\|f_n - f_2^*\|^2$')
    plt.legend()
    plt.savefig(os.path.join('figures', f'relu_rf.pdf'), pad_inches=0, bbox_inches='tight')



def full_kernel(log_lambda_min=-5, recompute=False):
    fname = f'full_kernel_curves_{log_lambda_min}.pkl'
    if not recompute and os.path.exists(fname):
        curves = pickle.load(open(fname, 'rb'))
    else:
        curves = []
        np.random.seed(42)
        n, d = 20000, 4
        X = rand_data(n, d).astype(np.float32)

        bump_delta = 0.3
        bump_type = 'single'
        y = target_bump(X, delta=bump_delta, bump_type=bump_type)
        # y = target_cone(X, delta=bump_delta, bump_type=bump_type)
        # y = target_lap(X, gamma=0.5, ty=bump_type)

        y = y.astype(np.float32)

        Xtr, Xte = X[:10000], X[10000:]
        ytr, yte = y[:10000], y[10000:]


        ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000] #, 10000]
        lmbdas = 10. ** np.arange(-2, log_lambda_min, -1)

        lmbdas = lmbdas.astype(np.float32)

        experiments = [
            ('rf_b 2-layer', kappa1b),
            ('rf 3-layer', ck_3layer),
            ('rf 4-layer', ck_4layer),

            ('ntk_b 2-layer', ntkb),
            ('ntk 3-layer', ntk_3layer),
            ('ntk 4-layer', ntk_4layer),

            ('lap c=1', gamma_exp()),
        ]

        XX = Xtr.dot(Xtr.T).clip(min=-1., max=1.)
        XXte = Xtr.dot(Xte.T).clip(min=-1., max=1.)

        for name, kernel_fn in experiments:
            Ktr_full = kernel_fn(XX)
            Kte_full = kernel_fn(XXte)
            errs = []
            for n in ns:
                K = Ktr_full[:n,:n]
                Kte = Kte_full[:n,:]
                targets = ytr[:n]
                valid_errs = []
                for lmbda in lmbdas:
                    preds = Kte.T.dot(np.linalg.solve(K + n * lmbda * np.eye(n, dtype=K.dtype), targets))
                    err = np.mean((preds - yte) ** 2)
                    valid_errs.append(err)
                errs.append(np.min(valid_errs))

            if name.startswith('r'):
                o = '-'
            elif name.startswith('n'):
                o = '--'
            else:
                o = '-.'
            curves.append((ns, errs, o, name))
            
        pickle.dump(curves, open(fname, 'wb'))
    
    
    plt.figure(figsize=(4,3))
    for ns, errs, o, name in curves:
        plt.loglog(ns, errs, o, label=name)

    plt.title('KRR, $\\lambda_\\min$ = 1e{}'.format(log_lambda_min))
    plt.xlabel('n')
    plt.ylabel(r'$\|f_n - f_1^*\|^2$')
    plt.legend()
    plt.savefig(os.path.join('figures', f'full_kernel_{abs(log_lambda_min)}.pdf'), pad_inches=0, bbox_inches='tight')
    
    
if __name__ == '__main__':
    full_kernel(log_lambda_min=-5, recompute=False)
    full_kernel(log_lambda_min=-9, recompute=False)
    relu_rf(recompute=False)

