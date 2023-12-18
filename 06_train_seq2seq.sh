#!/bin/bash

python3 -m transcription.train_seq2seq  \
    transcription/seq2seq_train_hparams/hparams_eslo2-cv-pxslu.yaml \
    --cv_dir $SCRATCH/cv-corpus-10.0-2022-07-04 \
    --eslo2_dir $SCRATCH/eslo2_processed \
    --pxslu_dir  $SCRATCH/pxslu \
    --pretrained_tokenizer_file $SCRATCH/transcription_results/tokenizer/bpe500/500_unigram.model \
    --pretrained_rnnlm_file $SCRATCH/transcription_results/rnnlm/eslo2-cv-os-pxslu/best_checkpoint/lm.ckpt \
    --output_dir $SCRATCH/transcription_results/seq2seq/eslo2-cv-pxslu
