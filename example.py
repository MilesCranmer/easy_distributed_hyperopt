from copy import deepcopy as copy
from sklearn.model_selection import train_test_split
import helper_functions as h
from bayesian_rcnn import rcnn_model
from sklearn.linear_model import LinearRegression, Ridge
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
import matplotlib as mpl
import numpy as np
mpl.use('agg')
from cranmerplot import density_scatter
from matplotlib import pyplot as plt
from importlib import reload
import torch
from torch import nn
from torch.autograd import Variable
from icecream import ic
import sys

TOTAL_EPOCHS = 30

sample = open(sys.argv[1] + '_log.txt', 'a')

easy = False
h = reload(h)

fullX, fully = ...

# Hide fraction of test
remy, finaly, remX, finalX = train_test_split(fully, fullX, shuffle=False, test_size=1./5)
trainy, testy, trainX, testX = train_test_split(remy, remX, shuffle=False, test_size=1./5)

ssy = StandardScaler()
ssX = StandardScaler() #Power is best

n_t = trainX.shape[1]
n_features = trainX.shape[2]

ssy.fit(trainy[::50])
ssX.fit(trainX.reshape(-1, n_features)[::1539])
ttrainy = ssy.transform(trainy)
ttesty = ssy.transform(testy)
ttrainX = ssX.transform(trainX.reshape(-1, n_features)).reshape(-1, n_t, n_features)
ttestX = ssX.transform(testX.reshape(-1, n_features)).reshape(-1, n_t, n_features)

if easy:
    ssXeasy = PowerTransformer()
    ssXeasy.fit(trainXeasy[::50])
    ttrainXeasy = ssXeasy.transform(trainXeasy)
    ttestXeasy = ssXeasy.transform(testXeasy)


train_len = ttrainX.shape[0]
X = Variable(torch.from_numpy(np.concatenate((ttrainX, ttestX))).type(torch.FloatTensor))
y = Variable(torch.from_numpy(np.concatenate((ttrainy, ttesty))).type(torch.FloatTensor))
N = len(X)

lr = 2e-3
hidden = 300
latent = 120
p = 0.5
batch_size = 350
in_length = 0
out_length = 1
train_fraction = 0.05
act = nn.LeakyReLU

def run_trial(kwargs):
    lr = kwargs['lr']
    p = kwargs['p']
    hidden = int(kwargs['hidden'])
    latent = int(kwargs['latent'])
    batch_size = int(kwargs['batch_size'])
    in_length = int(kwargs['in_length'])
    out_length = int(kwargs['out_length'])
    act = kwargs['act']
    idxes = np.array([i for i in range(n_features) if kwargs['n%d'%(i,)] == 1])
    if len(idxes) == 0:
        return 100

    print('Running with', kwargs)

    effective_n_features = len(idxes)

    def mlp(in_n, out_n, hidden, p, layers):
        if layers == 0:
            return nn.Linear(in_n, out_n)

        result = [nn.Linear(in_n, hidden)]
        for i in reversed(range(layers)):
            result.extend([
                act(),
                nn.Dropout(p)
                ])
            if i > 0:
                result.extend([nn.Linear(hidden, hidden)])

        result.extend([nn.Linear(hidden, out_n)])
        return nn.Sequential(*result)

    dataset = torch.utils.data.TensorDataset(X[:train_len, :, idxes], y[:train_len])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(X[train_len:, :, idxes], y[train_len:])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=3000, shuffle=False)

    class VarModel(nn.Module):
        def __init__(self, hidden=50, latent=5, p=0.1):
            super().__init__()
            self.feature_nn = mlp(effective_n_features, latent, hidden, p, in_length)
            self.regress_nn = mlp(latent*2, 1, hidden, p, out_length)

        def random_sample_with_replacement(self, x): #for bootstrap estimates
            _n = x.shape[1]
            _idx = np.random.randint(0, _n, size=_n)
            x = x[:, _idx]
            return x

        def forward(self, x):
            x = self.random_sample_with_replacement(x)
            x = self.feature_nn(x)
            features = torch.mean(x, dim=1)
            std = torch.sqrt(torch.mean((features[:, np.newaxis, :]-x)**2, dim=1))
            in_regress = torch.cat((features, std), dim=1)
            return self.regress_nn(in_regress)

    clf = VarModel(hidden, latent=latent, p=p).cuda()

    def augment(x):
        #N, Ntime, Nchan
        size = int(np.random.randint(n_t//5, n_t+1))
        start = int(np.random.randint(0, n_t+1-size))
        return x[:, start:start+size]

    from tqdm import tqdm_notebook as tqdm
    loss_fnc = nn.MSELoss(reduction='sum')
    best_state_dict = None
    best_val_loss = np.inf
    test_fraction = 1.0
    val_patience = 30/train_fraction
    opt = torch.optim.Adam(
        clf.parameters(), lr=lr,
        weight_decay=1e-8, amsgrad=True)
#sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=val_patience, verbose=True)
# ### NEED TO FIX LOSS FUNCTION - THE BRCNN OUTPUTS MULTIPLE TIMES!!
    val_bad_count = 0

    for epoch in tqdm(range(int(TOTAL_EPOCHS*0.05/train_fraction))):
        clf.train()
        losses = 0
        val_losses = 0
        num_trained = 0
        num_tested = 0
        while num_trained <= train_fraction * train_len:
            for X_sample, y_sample in dataloader:
                X_sample = X_sample.cuda()
                y_sample = y_sample.cuda()
                X_sample = augment(X_sample)
                y_sample_pred = clf(X_sample)
                loss = loss_fnc(y_sample_pred, y_sample)
                (loss/batch_size).backward()
                losses += loss.item()*ssy.scale_[0]**2
                opt.step()
                opt.zero_grad()
                num_trained += len(y_sample)
                if num_trained > train_fraction * train_len:
                    break
        # Get validation loss
        clf.eval()
        while num_tested <= test_fraction * (len(X) - train_len): 
            for X_sample, y_sample in test_dataloader:
                X_sample = X_sample.cuda()
                y_sample = y_sample.cuda()
                y_sample_pred = clf(X_sample)
                loss = loss_fnc(y_sample_pred, y_sample)
                val_losses += loss.item()*ssy.scale_[0]**2
                num_tested += len(y_sample)
                if num_tested > test_fraction * (len(X) - train_len):
                    break

        avg_loss = losses / num_trained
        avg_val_loss = val_losses / num_tested
        if (epoch) % int(0.2/train_fraction) == 0:
            print('epoch, avg, val, best:', epoch, avg_loss, avg_val_loss, best_val_loss, flush=True)
            print('epoch, avg, val, best:', epoch, avg_loss, avg_val_loss, best_val_loss, file=sample, flush=True)
        #sched.step(avg_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = copy(clf.state_dict())
            val_bad_count = 0
        else:
            val_bad_count += 1
        if val_bad_count > 200:
            break
        opt.zero_grad()

    try:
        clf.load_state_dict(best_state_dict)
    except:
        return 100

    truths = []
    preds = []
#Reset dataloader
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=3000, shuffle=False)
    nc = 0
    losses = 0.0
    clf.eval()
    for X_sample, y_sample in test_dataloader:
        X_sample = X_sample.cuda()
        y_sample = y_sample.cuda()
        y_sample_pred = clf(X_sample)
        loss = loss_fnc(y_sample_pred, y_sample)
        losses += loss.item()*ssy.scale_[0]**2
        nc += len(y_sample)

    losses /= nc

    print('Total validation loss', losses)
    print('Total validation loss', losses, file=sample, flush=True)
    print(lr, hidden, latent, p, batch_size, in_length, out_length, act, losses, ssX, idxes,
          "lr, hidden, latent, p, batch_size, in_length, out_length, act, losses, ssX, idxes",
          file=sample, flush=True)
    if losses > 100 or losses is None or losses is np.inf:
        return 100

    return losses

import pickle as pkl
from hyperopt import hp, fmin, tpe, Trials
import datetime

def merge_trials(trials1, trials2_slice):
    max_tid = 0
    if len(trials1.trials) > 0:
        max_tid = max([trial['tid'] for trial in trials1.trials])

    for trial in trials2_slice:
        tid = trial['tid'] + max_tid + 1
        hyperopt_trial = Trials().new_trial_docs(
                tids=[None],
                specs=[None],
                results=[None],
                miscs=[None])
        hyperopt_trial[0] = trial
        hyperopt_trial[0]['tid'] = tid
        hyperopt_trial[0]['misc']['tid'] = tid
        for key in hyperopt_trial[0]['misc']['idxs'].keys():
            hyperopt_trial[0]['misc']['idxs'][key] = [tid]
        trials1.insert_trial_docs(hyperopt_trial) 
        trials1.refresh()
    return trials1

while True:
    # Load up all runs:
    trials = None
    import glob
    path = 'trials_v5/*.pkl'
    files = 0
    for fname in glob.glob(path):
        trials_obj = pkl.load(open(fname, 'rb'))
        n_trials = trials_obj['n']
        trials_obj = trials_obj['trials']
        if files == 0:
            trials = trials_obj
        else:
            print("Merging trials")
            trials = merge_trials(trials, trials_obj.trials[-n_trials:])
        files += 1

    if files == 0:
        trials = Trials()

    # Expand by 5:
    raw_space = {
        'lr' : hp.lognormal('lr', -5.5, 1.5),
        'p' : hp.uniform('p', 0.0, 0.9),
        'hidden' : hp.qloguniform('hidden', np.log(10), np.log(1000+1), 1),
        'latent' : hp.qloguniform('latent', np.log(5), np.log(1000), 1),
        'batch_size' : hp.qloguniform('batch_size', np.log(256), np.log(3000), 1),
        'in_length' : hp.quniform('in_length', 0, 4+1, 1),
        'out_length' : hp.quniform('out_length', 0, 4+1, 1),
        'act' : hp.choice('act', [nn.LeakyReLU]),
        'epochs': hp.choice('epochs', [TOTAL_EPOCHS])
    }
    #Optimize which features we give.
    feature_space = {
        'n%d'%(i,): hp.choice('n%d'%(i,), [0, 1]) for i in range(n_features)
    }

    space = {**raw_space, **feature_space}

    np.random.seed()
    n = 1
    best = fmin(run_trial,
        space=space,
        algo=tpe.suggest,
        max_evals=n + len(trials.trials),
        trials=trials,
        verbose=1,
        rstate=np.random.RandomState(np.random.randint(1,10**6))
        )
    print('current best', best)
    hyperopt_trial = Trials()
    # Merge with empty trials dataset:
    trials = merge_trials(hyperopt_trial, trials.trials[-n:])
    pkl.dump({'trials': trials, 'n': n}, open('trials_v5/' + str(np.random.rand()).replace('.', '') + '.pkl', 'wb'))
