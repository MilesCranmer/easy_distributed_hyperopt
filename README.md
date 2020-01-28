# Easy distributed hyperopt

The code in example.py allows you to run distributed
hyperopt with only a shared folder: all trials
will be stored and shared between nodes using that folder.

No head node, no distributed frameworks. Just a shared
folder.

## To use, 
1. Create an empty folder to store all the trial scores. The
folder should be accessible from all your nodes.
2. Fill in example.py with the foldername, your desired
optimization function and hyperparameters to optimize over.
3. Then, run example.py on all your nodes.
4. Kill the script when you want to stop.
5. Set the foldername in print_best_model.py and then run it.

