"""Start a hyperoptimization from a single node"""
import numpy as np
import pickle as pkl
from hyperopt import hp, fmin, tpe, Trials


#Change the following code to your file
################################################################################
# TODO: Declare a folder to hold all trials objects
TRIALS_FOLDER = 'trials'
NUMBER_TRIALS_PER_RUN = 1

def run_trial(args):
    """Run a training iteration based on args.

    :args: A dictionary containing all hyperparameters
    :returns: Average loss from cross-validation

    """

    #TODO: Fill this in.
    ...

    return cross_validation_test_loss


#TODO: Declare your hyperparameter priors here:
space = {
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

################################################################################



def merge_trials(trials1, trials2_slice):
    """Merge two hyperopt trials objects

    :trials1: The primary trials object
    :trials2_slice: A slice of the trials object to be merged,
        obtained with, e.g., trials2.trials[:10]
    :returns: The merged trials object

    """
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

loaded_fnames = []
trials = None
# Run new hyperparameter trials until killed
while True:
    np.random.seed()

    # Load up all runs:
    import glob
    path = TRIALS_FOLDER + '/*.pkl'
    files = 0
    for fname in glob.glob(path):
        if fname in loaded_fnames:
            continue

        loaded_fnames.append(fname)
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

    n = NUMBER_TRIALS_PER_RUN
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
    new_fname = TRIALS_FOLDER + '/' + str(np.random.randint(0, sys.maxsize)) + '.pkl',
    pkl.dump({'trials': trials, 'n': n}, open(new_fname, 'wb'))
    loaded_fnames.append(new_fname)
