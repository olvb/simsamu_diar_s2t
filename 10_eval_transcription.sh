#!/bin/bash

python3 -m transcription.evaluate \
    --model $SCRATCH/transcription_results/ctc/eslo2-cv-pxslu/best_checkpoint \
    --output_dir $SCRATCH/transcription_results/eval/ctc_simsamu \
    --dataset_dir $SCRATCH/simsamu

python3 -m transcription.evaluate \
    --model $SCRATCH/transcription_results/ctc/eslo2-cv-pxslu/best_checkpoint \
    --output_dir $SCRATCH/transcription_results/eval/ctc_lm_simsamu \
    --dataset_dir $SCRATCH/simsamu \
    --kenlm_file $SCRATCH/transcription_results/kenlm/prune-4/model.arpa

python3 -m transcription.evaluate \
    --model $SCRATCH/transcription_results/seq2seq/eslo2-cv-pxslu/best_checkpoint \
    --output_dir $SCRATCH/transcription_results/eval/seq2seq_simsamu \
    --dataset_dir $SCRATCH/simsamu

python3 -m transcription.evaluate \
    --model $SCRATCH/transcription_results/seq2seq/eslo2-cv-pxslu/best_checkpoint_lm \
    --output_dir $SCRATCH/transcription_results/eval/seq2seq_lm_simsamu \
    --dataset_dir $SCRATCH/simsamu

python3 -m transcription.evaluate \
    --model whisper \
    --output_dir $SCRATCH/transcription_results/eval/whisper_simsamu \
    --dataset_dir $SCRATCH/simsamu
