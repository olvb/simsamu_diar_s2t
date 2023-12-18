#!/bin/bash

export PYANNOTE_DATABASE_CONFIG=$SCRATCH/eslo2_processed/diarization/database.yml

python3 -m diarization.train \
    diarization/exp_params.yml \
    --output_dir $SCRATCH/diarization_results \
    --musan_noise_dir $SCRATCH/RIRS_NOISES/pointsource_noises \
    --hf_auth_token $HF_AUTH_TOKEN
