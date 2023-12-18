#!/bin/bash

python3 -m transcription.train_kenlm  \
    --cv_dir $SCRATCH/cv-corpus-10.0-2022-07-04 \
    --eslo2_dir $SCRATCH/eslo2_processed \
    --pxslu_dir  $SCRATCH/pxslu \
    --os_dir $SCRATCH/OpenSubtitles_processed \
    --prune 4 \
    --output_dir $SCRATCH/transcription_results/kenlm/prune-4
