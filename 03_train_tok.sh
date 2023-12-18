#!/bin/bash

python3 -m transcription.train_tokenizer  \
    transcription/tokenizer_train_hparams/hparams_bpe500.yaml \
    --cv_dir $SCRATCH/cv-corpus-10.0-2022-07-04 \
    --eslo2_dir $SCRATCH/eslo2_processed \
    --pxslu_dir  $SCRATCH/pxslu \
    --os_dir $SCRATCH/OpenSubtitles_processed \
    --output_dir $SCRATCH/transcription_results/tokenizer/bpe500
