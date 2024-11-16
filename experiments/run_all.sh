#!/bin/bash
# All the tuning runs in the order I did them, one change at a time.
# Each run adds a row to experiments/log.csv.
# run from the project folder:  bash experiments/run_all.sh

T="python3 src/train.py"

# step 0: first thing I tried
$T --name 0_baseline --hidden 32 --batch 0 --init small --no-standardize

# step 1: standardize the pixels
$T --name 1_standardize --hidden 32 --batch 0 --init small

# step 2: He init
$T --name 2_he_init --hidden 32 --batch 0

# step 3: mini-batches instead of full batch
$T --name 3_minibatch --hidden 32 --batch 64

# step 4: learning rate
for lr in 1 0.3 0.03 0.01; do
    $T --name 4_lr_$lr --hidden 32 --lr $lr
done

# step 5: learning rate decay, because val accuracy was jumping around
$T --name 5_decay --hidden 32 --lr 0.1 --decay 0.85

# step 6: hidden layer size
for h in 16 64 128; do
    $T --name 6_hidden_$h --hidden $h --decay 0.85
done

# step 7: a second hidden layer
$T --name 7_two_layers --hidden 64,32 --decay 0.85

# step 8: L2 regularization
for l2 in 0.001 0.01 0.03; do
    $T --name 8_l2_$l2 --hidden 64 --decay 0.85 --l2 $l2
done

# step 9: shifted copies of the training images (5x more data so fewer epochs)
$T --name 9_augment --hidden 64 --decay 0.6 --l2 0.01 --augment --epochs 8
$T --name 9_augment_l2_0.003 --hidden 64 --decay 0.6 --l2 0.003 --augment --epochs 8

# logistic regression (no hidden layer) with the same settings, to compare
$T --name logistic_regression --hidden 0 --decay 0.6 --l2 0.003 --augment --epochs 8

# final model
$T --name final --hidden 64 --decay 0.6 --l2 0.003 --augment --epochs 8 --save
