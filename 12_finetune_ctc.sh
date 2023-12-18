#!/bin/bash

python3 -m transcription.finetune_ctc \
    --hparams_file transcription/ctc_train_hparams/hparams_finetune_simsamu.yml \
    --pretrained_dir $SCRATCH/transcription_results/ctc/eslo2-cv-pxslu/best_checkpoint \
    --simsamu_dir $SCRATCH/simsamu \
    --output_dir $SCRATCH/transcription_results/ctc_finetuned_simsamu
