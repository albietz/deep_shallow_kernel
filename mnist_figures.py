import numpy as np
import matplotlib.pyplot as plt
import os, pickle
from nn_rf import *

plt.style.use('ggplot')

NS = [500, 1000, 2000, 5000, 10000, 20000, 50000]

def get_res(ds, kern, nlayers, n):
    fname = f'{ds}_{kern}_{n}_{nlayers}.pkl'
    res = pickle.load(open(os.path.join('res', fname), 'rb'))
    return res

def get_curve(ds, kern, nlayers, ns=None):
    best_accs = []
    if ns is None:
        ns = NS
    for n in ns:
        res = get_res(ds, kern, nlayers, n)
        best_accs.append(res['best_acc'])
    return best_accs

def get_curve_with_std(ds, kern, nlayers, ns=None):
    best_accs_mean = []
    best_accs_std = []
    if ns is None:
        ns = NS
    for n in ns:
        vals = []
        for seed in range(5):
            res = get_res(f'{ds}{seed}', kern, nlayers, n)
            vals.append(res['best_val_acc'])
        best_accs_mean.append(np.mean(vals))
        best_accs_std.append(np.std(vals))
    return best_accs_mean, best_accs_std

def acc_curve(ds='mnist'):
    nls = [2, 3, 4, 5]

    plt.figure(figsize=(5,4))

    for nlayers in nls:
        mean, std = get_curve_with_std(ds, 'rf', nlayers)
        plt.semilogx(ns, mean, label=f'RF {nlayers}')
    #     plt.errorbar(ns, mean, yerr=std, label=f'RF {nlayers}')

    for nlayers in nls:
        mean, std = get_curve_with_std(ds, 'ntk', nlayers)
        plt.semilogx(ns, mean, '--', label=f'NTK {nlayers}')
    #     plt.errorbar(ns, mean, yerr=std, label=f'NTK {nlayers}')

    plt.title(ds)
    plt.xlabel('n')
    plt.ylabel('test accuracy')
    plt.legend()
    plt.savefig(os.path.join('figures', f'{ds}_acc_curve.pdf'), pad_inches=0, bbox_inches='tight')

    
def print_acc_table(ds, n):
    nidx = NS.index(n)
    kerns = ['rf', 'ntk']
    kern_labels = ['RF', 'NTK']
    
    print(r'\begin{tabular}{ | c |', ' c | ' * len(kerns), '}')
    print(r'\hline')    
    print('L & ', ' & '.join(kern_labels), r'\\ \hline')


    for nlayers in [2, 3, 4, 5]:
        print(nlayers, end=' ')
        for kern in kerns:
            mean, std = get_curve_with_std(ds, kern, nlayers)
            print(' & {:.2f} $\\pm$ {:.2f} '.format(100. * mean[nidx], 100. * std[nidx]))
        print(r'\\ ')

    print(r'\hline')
    print(r'\end{tabular}')


    
if __name__ == '__main__':
    print('mnist table, 50000')
    print_acc_table('mnist', 50000)
    
    print('fmnist table, 50000')
    print_acc_table('fmnist', 50000)
    
    # print('mnist lap, 50000', get_curve_with_std('mnist', 'lap', 0.1, [50000]))
    
