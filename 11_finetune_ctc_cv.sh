#!/bin/bash

python3 -m transcription.finetune_ctc_cv \
    --hparams_file transcription/ctc_train_hparams/hparams_finetune_simsamu.yml \
    --pretrained_dir $SCRATCH/transcription_results/ctc/eslo2-cv-pxslu/best_checkpoint \
    --simsamu_dir $SCRATCH/simsamu \
    --output_dir $SCRATCH/transcription_results/ctc_cv_speaker_aware \
    --speaker_aware \
    --kenlm_file $SCRATCH/transcription_results/kenlm/prune-4/model.arpa

python3 -m transcription.finetune_ctc_cv \
    --hparams_file transcription/ctc_train_hparams/hparams_finetune_simsamu.yml \
    --pretrained_dir $SCRATCH/transcription_results/ctc/eslo2-cv-pxslu/best_checkpoint \
    --simsamu_dir $SCRATCH/simsamu \
    --output_dir $SCRATCH/transcription_results/ctc_cv_10_folds \
    --nb_folds 10 \
    --kenlm_file $SCRATCH/transcription_results/kenlm/prune-4/model.arpa
