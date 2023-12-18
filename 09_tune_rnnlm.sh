#!/bin/bash

python3 -m transcription.tune_rnnlm \
    --model $SCRATCH/transcription_results/seq2seq/eslo2-cv-pxslu/best_checkpoint_lm \
    --eslo2_dir $SCRATCH/eslo2_processed \
    --output_dir $SCRATCH/transcription_results/tune_rnnlm/
