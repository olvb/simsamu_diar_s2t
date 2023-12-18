## simsamu diarization and transcription

This repository contains code to train diarization and transcription models to be used on recordings from emergy dispatch services. The [simsamu](https://huggingface.co/datasets/medkit/simsamu) dataset is used to evaluate them.

# Training datasets

- _ESLO2_: ~150 hours of real life conversations in french annotated for diarization and transcription. http://eslo.huma-num.fr/
- _CommonVoice FR_: ~1000 hours of french voice recordings. https://commonvoice.mozilla.org/en/datasets
- _PXSLU_: ~ 2000 recorings of medical prescriptions read aloud in french https://arxiv.org/pdf/2207.08292.pdf
- _OpenSubtitles FR_: subtitles files, used to build language models for transcription https://opus.nlpl.eu/OpenSubtitles-v2018.php

# Dataset preparation code:

- `eslo2_prepare/`: python code to parse the `.trs` files from _ESLO2_ and generate ``.rttm` files (for diarization) and `.srt` files (for transcripion). Also processes the audio files to simulate phone recordings.
- `opensubtitles_prepare.py`: python code to parse the `.xml` file from _OpenSubtitles_ and generate simpler ``.txt` files

# Diarization

Diarization is based on [pyannote-audio](https://github.com/pyannote/pyannote-audio) 3.0. The code in `diarization/` peforms the following:
- fine-tuning of the PyaNet segmentation model available at https://huggingface.co/pyannote/segmentation-3.0, using _ESLO2_
- optimization of the hyperparameters of complete diarization pipeline (PyaNet segmentation + pretrained wespeaker embeddings + agglomerative clustering), also using _ESLO2_
- evaluation of the final pipeline on _simsamu_

# Transcription

Transcription is based on models from [speechbrain](https://github.com/speechbrain/speechbrain).
The code in `transcription` performs the following:
- construction of a SentencePiece tokenizer
- training of transcription models based on the CTC and seq2seq architectures, using _CommonVoice_, _ESLO2_ and _PXSLU_
- construction of a [KenLM](https://github.com/kpu/kenlm) language model to be used with the CTC model, using _CommonVoice_, _ESLO2_, _PXSLU_ and _OpenSubtitles_
- training of a speechbrain RNNLM language model to be used with the seq2seq model, using _CommonVoice_, _ESLO2_, _PXSLU_ and _OpenSubtitles_
- optimization of the hyperparameters of the KenLM and RNNLM language models
- evaluation of the CTC and seq2seq models on `simsamu`, with and without their language models, and comparison with [https://github.com/openai/whisper](whisper)
- fine-tuning of the CTC model (which performs best) on `simsamu` with cross-validation

# Scripts

Bash scripts from `00_download_data.sh` to `12_finetune_ctc.sh` will call the appropriate python code to perform all of the previously described tasks. They must be called in sequential order. They expect a `$SCRATCH` environment variable pointing to the directory into which the datasets will be downloaded and the results will be stored, as well as an `$HF_AUTH_TOKEN` variable needed to access https://huggingface.co/pyannote/segmentation-3.0
