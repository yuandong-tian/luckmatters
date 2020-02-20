# Supplementary Materials
This is supplementary materials for ICML submission "Student Specialization in Deep Rectified Networks With Finite Width and Input Dimension". 

# Required package
Install [hydra](https://github.com/facebookresearch/hydra) by following its instructions.

Install Pytorch and other packages (yaml, json, matplotlib). 


# Usage

## Two-layer

Fig. 9 is generated with the following sweep (note that in the paper we use `theory_suggest_sigma=2` and `theory_suggest_std=2`). Note that `teacher_strength_decay` is parameter `p` in Eqn. 10, and `m` is the number of teacher hidden node, `multi` is the over-realization rate.

```
python two_layer_new2.py -m multi=1,2,5,10 d=100 m=5 teacher_strength_decay=1 lr=0.01 use_sgd=true N_train=30,45,60,75,90,105,120,135,150,165,180,195 num_epoch=20 num_iter_per_epoch=5000 batchsize=16 theory_suggest_train=true theory_suggest_sigma=2 theory_suggest_mean=2
python two_layer_new2.py -m multi=1,2,5,10 d=100 m=5 teacher_strength_decay=1 lr=0.01 use_sgd=true N_train=30,45,60,75,90,105,120,135,150,165,180,195 num_epoch=20 num_iter_per_epoch=5000 batchsize=16 theory_suggest_train=false
python two_layer_new2.py -m multi=1,2,5,10 d=100 m=5 teacher_strength_decay=1 lr=0.01 use_sgd=true N_train=210,270,330,390,510,630,810,1110 num_epoch=20 num_iter_per_epoch=5000 batchsize=16 theory_suggest_train=true theory_suggest_sigma=2 theory_suggest_mean=2
python two_layer_new2.py -m multi=1,2,5,10 d=100 m=5 teacher_strength_decay=1 lr=0.01 use_sgd=true N_train=210,270,330,390,510,630,810,1110 num_epoch=20 num_iter_per_epoch=5000 batchsize=16 theory_suggest_train=false
```

To get one datapointi for teacher-agnostic results, run the following command line to get a datapoint of Fig.9. The idea is to remove `-m` switch and only run one parameter setup. 

```
python two_layer_new2.py multi=2 d=100 m=5 teacher_strength_decay=1 lr=0.01 use_sgd=true N_train=1020 num_epoch=20 num_iter_per_epoch=5000 batchsize=16 theory_suggest_train=false
```

Use the following for teacher-aware results in Fig. 9: 

```
python two_layer_new2.py multi=2 d=100 m=5 teacher_strength_decay=1 lr=0.01 use_sgd=true N_train=60 num_epoch=20 num_iter_per_epoch=5000 batchsize=16 theory_suggest_train=true theory_suggest_sigma=3 theory_suggest_mean=3
```


## Multi-layer

Fig. 6 is generated with the following command:

```
python recon_multilayer.py -m seed=11,12,13,14,15,16,17,18,19,20 num_trial=1 node_multi=2,5 num_epoch=200 random_dataset_size=5000,10000,20000,50000,100000,200000,500000 stats_grad_norm=true optim_method=sgd lr=0-0.2-20-0.1 cross_entropy=true,false
python recon_multilayer.py -m seed=11,12,13,14,15,16,17,18,19,20 num_trial=1 node_multi=2,5 num_epoch=200 random_dataset_size=750000,1000000,2000000 stats_grad_norm=true optim_method=sgd lr=0-0.2-20-0.1 cross_entropy=true

```

To get one datapoint, remove `-m` switch and run one parameter setup. 

