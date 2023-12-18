#!/bin/bash

python3 -m transcription.tune_kenlm \
    --model $SCRATCH/transcription_results/ctc/eslo2-cv-pxslu/best_checkpoint \
    --kenlm_file $SCRATCH/transcription_results/kenlm/prune-4/model.arpa \
    --eslo2_dir $SCRATCH/eslo2_processed \
    --output_dir $SCRATCH/transcription_results/tune_kenlm
