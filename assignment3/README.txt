# Assignment 3
**GID**: cfarr31

##################
## Instructions ##
##################

Repository: 

https://github.com/ChrisJFarr/OMSCS-ML-CS-7641/tree/main/assignment3

CLI through hydra. 

Step 1: Install requirements.txt into python version 3.8+
    Run `pip install -r requirements.txt`
Step 2: Run experiments as shown below; (copy from nohup to &, include to run in background or exclude otherwise)
    kmeans: ds1; nohup python run.py +func=sp experiments=km-1 & 
    em: ds1; nohup python run.py +func=sp experiments=em-1 &
    kmeans: ds2; nohup python run.py +func=sp experiments=km-2 &
    em: ds2; nohup python run.py +func=sp experiments=em-2 &
    pca: ds1; nohup python run.py +func=tsne experiments=pca-1 &
    ica: ds1; nohup python run.py +func=tsne experiments=ica-1 &
    proj: ds1; nohup python run.py +func=tsne experiments=proj-1 &
    pca: ds2; nohup python run.py +func=tsne experiments=pca-2 &
    ica: ds2; nohup python run.py +func=tsne experiments=ica-2 &
    proj: ds2; nohup python run.py +func=tsne experiments=proj-2 &
    feat: ds1; nohup python run.py +func=feat_sel experiments=feat-1 &
    feat: ds2; nohup python run.py +func=feat_sel experiments=feat-2 &
    pca->km: ds1; nohup python run.py +func=sp_auc experiments=pca-km-1 &
    pca->em: ds1; nohup python run.py +func=sp_auc experiments=pca-em-1 &
    ica->km: ds1; nohup python run.py +func=sp_auc experiments=ica-km-1 &
    ica->em: ds1; nohup python run.py +func=sp_auc experiments=ica-em-1 &
    proj->km: ds1; nohup python run.py +func=sp_auc experiments=proj-km-1 &
    proj->em: ds1; nohup python run.py +func=sp_auc experiments=proj-em-1 &
    feat->km: ds1; nohup python run.py +func=feat_sel_auc experiments=feat-km-1 &
    feat->em: ds1; nohup python run.py +func=feat_sel_auc experiments=feat-em-1 &
    pca->km: ds2; nohup python run.py +func=sp experiments=pca-km-2 &
    pca->em: ds2; nohup python run.py +func=sp experiments=pca-em-2 &
    ica->km: ds2; nohup python run.py +func=sp experiments=ica-km-2 &
    ica->em: ds2; nohup python run.py +func=sp experiments=ica-em-2 &
    proj->km: ds2; nohup python run.py +func=sp experiments=proj-km-2 &
    proj->em: ds2; nohup python run.py +func=sp experiments=proj-em-2 &
    feat->km: ds2; nohup python run.py +func=feat_sel experiments=feat-km-2 &
    feat->em: ds2; nohup python run.py +func=feat_sel experiments=feat-em-2 &
    pca->nn: ds1; nohup python run.py +func=vc experiments=pca-nn-1 &
    ica->nn: ds1; nohup python run.py +func=vc experiments=ica-nn-1 & 
    proj->nn: ds1; nohup python run.py +func=vc experiments=proj-nn-1 & 
    feat->nn: ds1; nohup python run.py +func=vc experiments=feat-nn-1 &
    nohup python run.py +func=vc experiments=pca-km-rand-clust-1 &
    nohup python run.py +func=feat_imp experiments=pca-km-rand-clust-1 &
    nohup python run.py +func=vc experiments=ica-km-rand-clust-1 &
    nohup python run.py +func=feat_imp experiments=ica-km-rand-clust-1 &
    nohup python run.py +func=vc experiments=proj-km-rand-clust-1 & 
    nohup python run.py +func=feat_imp experiments=proj-km-rand-clust-1 &