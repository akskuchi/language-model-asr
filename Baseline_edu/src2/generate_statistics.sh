#!/bin/bash

# Paper refers to rnn + sigmoid activations. This isn't implemented
# in PyTorch, so we can only test on ReLU and tanh
MODELS="GRU LSTM RNN_RELU RNN_TANH"
# Paper only uses a single hidden layer, we can use multiple
LAYERS="1 2 3 4"
# These are the hidden layer sizes used in the paper
HIDDEN="60 90 250 400"
# Dropout probabilities
DROPOUT="0.001 0.005 0.01 0.05 0.1 0.5"
# Datasets
DATASETS="wikitext-2"
# Predict k words
K="3 5 10"
# Embedding size - must be the same size as the hidden layer, apparently
# EMSIZES="100 200 500"
# Backprop through time
BPTT="8 16 32"

mkdir -p models
mkdir -p statistics

for dataset in $DATASETS; do
    echo "Dataset $dataset"
    for model in $MODELS; do
        echo "Model $model"
        for layer in $LAYERS; do
            echo "Layers: $layer"
            for hidden in $HIDDEN; do
                echo "Hidden: $hidden"
                for dropout in $DROPOUT; do
                    echo "Dropout: $dropout"
                    for k in $K; do
                        echo "K: $k"
                        for bptt in $BPTT; do
                            echo "BPTT: $bptt"
                            echo "Training models/ds-$dataset-model-$model-l-$layer-h-$hidden-d-$dropout-k-$k-em-bptt-$bptt.npy"
                            python main.py \
                                --data data/$dataset \
                                --epochs 10 \
                                --cuda \
                                --nlayers $layer \
                                --bptt $bptt \
                                --dropout $dropout \
                                --emsize $hidden \
                                --nhid $hidden \
                                --model $model \
                                --seed 42 \
                                --save models/ds-$dataset-model-$model-l-$layer-h-$hidden-d-$dropout-k-$k-bptt-$bptt.npy \
                                --save-statistics statistics//ds-$dataset-model-$model-l-$layer-h-$hidden-d-$dropout-k-$k-bptt-$bptt.csv;
                        done
                    done
                done
            done
        done
    done
done
