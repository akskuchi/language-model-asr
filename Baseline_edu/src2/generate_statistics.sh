#!/bin/bash

# Paper refers to rnn + sigmoid activations. This isn't implemented
# in PyTorch, so we can only test on ReLU and tanh
MODELS=${MODELS:-"GRU LSTM RNN_RELU RNN_TANH"}
# Paper only uses a single hidden layer, we can use multiple
LAYERS=${LAYERS-"1 2 3 4"}
# These are the hidden layer sizes used in the paper
HIDDEN=${HIDDEN:-"60 90 250 400"}
# Dropout probabilities
DROPOUT=${DROPOUT:-"0.001 0.005 0.01 0.05 0.1 0.5"}
# Datasets
DATASETS=${DATASETS:-"wikitext-2"}
# Predict k words
K=${K:-"3 5 10"}
# Embedding size - must be the same size as the hidden layer, apparently
EMSIZES="0 100 200 500"
# Backprop through time
BPTT=${BPTT:-"8 16 32"}

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
                        for embsize in $EMBSIZES; do
                            echo "Embedding size: $embsize"
                            for bptt in $BPTT; do
                                echo "BPTT: $bptt"
                                echo "Training models/ds-$dataset-model-$model-l-$layer-h-$hidden-d-$dropout-k-$k-em-$embsize-bptt-$bptt.npy"
                                python main.py \
                                    --data data/$dataset \
                                    --epochs 5 \
                                    --cuda \
                                    --nlayers $layer \
                                    --bptt $bptt \
                                    --dropout $dropout \
                                    --emsize $embsize \
                                    --nhid $hidden \
                                    --model $model \
                                    --seed 42 \
                                    --save-statistics statistics_new//ds-$dataset-model-$model-l-$layer-h-$hidden-d-$dropout-k-$k-em-$embsize-bptt-$bptt.csv;
                            done
                        done
                    done
                done
            done
        done
    done
done
