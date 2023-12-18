#!/bin/bash

python3 -m transcription.train_rnnlm  \
    transcription/rnnlm_train_hparams/hparams_eslo2-cv-os-pxslu.yaml \
    --cv_dir $SCRATCH/cv-corpus-10.0-2022-07-04 \
    --eslo2_dir $SCRATCH/eslo2_processed \
    --pxslu_dir  $SCRATCH/pxslu \
    --os_dir $SCRATCH/OpenSubtitles_processed \
    --pretrained_tokenizer_file $SCRATCH/transcription_results/tokenizer/bpe500/500_unigram.model \
    --output_dir $SCRATCH/transcription_results/rnnlm/eslo2-cv-os-pxslu

