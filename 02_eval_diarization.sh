#!/bin/bash

python3 -m diarization.evaluate \
    --segmentation_model $SCRATCH/diarization_results/seg/best.ckpt \
    --embedding hbredin/wespeaker-voxceleb-resnet34-LM \
    --diarization_params $SCRATCH/diarization_results/diar/best.yml \
    --output_dir $SCRATCH/diarization_results/eval/simsamu \
    --dataset_dir $SCRATCH/simsamu \
    --hf_auth_token $HF_AUTH_TOKEN
