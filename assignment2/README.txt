# Assignment 2
**GID**: cfarr31

##################
## Instructions ##
##################

Repository: 

Randomized Optimization: https://github.com/ChrisJFarr/OMSCS-ML-CS-7641/tree/main/assignment1
Supervised Learning: https://github.com/ChrisJFarr/OMSCS-ML-CS-7641/tree/main/assignment2

CLI through hydra. 

Example usage: (see details below)
    * Perform Genetic Algorithm optimization on the MaxKColor problem and compute validation curve plots
        Run `python run.py +func=vc experiments=ga-color`
    * Perform Simulated Annealing optimization on the Queens problem and compute iterative curve plots
        Run `python run.py +func=ic experiments=sa-queens`

Step 1: Install requirements.txt into python version 3.8+
    Run `pip install -r requirements.txt`
Step 2: Run experiment as above using the below options while in the `assignment2` folder
    * `experiments` options: "[model]-[dataset]"
        * ga-color
        * ga-peaks
        * ga-queens
        * hill-color
        * hill-peaks
        * hill-queens
        * mim-color
        * mim-peaks
        * mim-queens
        * sa-color
        * sa-peaks
        * sa-queens
    * `+func` options:
        * vc: validation curve
        * fc: fitness curve
        * wall: compute clock time per iteration

For evaluating Part 2, reference assignment1/README.txt and use the below commands
* `experiments`
    * ga-1
    * sa-1
    * hill-1
* `+func`
    * vc: validation curve
    * ip2: iterative plot
