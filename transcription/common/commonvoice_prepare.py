# generate speechbrain CSV files from commonvoice to use for training
# based on https://github.com/speechbrain/speechbrain/blob/develop/recipes/CommonVoice/common_voice_prepare.py

import csv

import torchaudio

torchaudio.set_audio_backend("soundfile")
from tqdm import tqdm

from .preprocess_text import preprocess_text, prepare_words_for_wer


def prepare_commonvoice(
    data_dir,
    save_dir,
    include_wav_info=False,
    tokenizer=None,
    min_nb_chars=2,
    min_duration_secs=0.01,
    min_nb_tokens=2,
):
    """
    Prepare the CommonVoice FR dataset for ASR model training

    For each split (train, dev, test), a CSV file will be created with
    one row per utterance and the following columns:
     - ID: unique ID for the utterance (prefixed by "cv_")
     - text: groundtruth transcription, preprocessed (cf preprocess_text() )
     - source: always set to "cv" (used for dataset balancing)

    if include_wav_info is True:
     - duration: length of utterance, in seconds
       This field is used for building batches with items of same duration to
       sort them. speechbrain needs the field to be named "duration".
     - start: start in seconds of the utterance in the file
       (always special value -1 for common voice because there is 1 wav_file per utterance)
     - end: end in seconds of the utterance in the file
       (always special value -1 for common voice because there is 1 wav_file per utterance)
     - wav_file: path to the audio file (actually an mp3 file for common voice)

    if tokenizer is provided:
     - duration: number of tokens, if tokenizer was provided For Language Model
       training, we need the number of tokens to batch togethers utterances of
       same length. speechbrain needs the field to be named "duration".

    include_wav_info set to True and tokenizer are mutually exclusive

    Parameters
    ----------
    data_dir:
        Path to the common voice fr subdir (ex: /scratch/cv-corpus-10.0-2022-07-04/fr)
    save_dir:
        Directory into which the train, dev and test csvs will be saved
    include_wav_info:
        Whether to include wav info in csv
    tokenizer:
        Optional tokenizer, used only for Language Model training, to precompute
        the number of tokens for each utterance
    min_nb_chars:
        Min number of chars in transcribed text (utterance is ignored if less)
    min_duration_secs:
        Min duration of utterance in seconds if include_wav_info is True (utterance
        is ignored if less)
    min_nb_tokens:
        Min number of tokens of utterance if tokenizer is provided (utterance is
        ignored if less)
    """

    assert not include_wav_info or tokenizer is None

    splits = ["train", "dev", "test"]
    csv_files = [save_dir / f"{split}.csv" for split in splits]
    if all(f.exists() for f in csv_files):
        print("CommonVoice CSV files already exist, skipping data preparation")
        return csv_files

    save_dir.mkdir(parents=True, exist_ok=True)

    for split, csv_file in zip(splits, csv_files):
        tsv_file = data_dir / f"{split}.tsv"
        _create_csv(
            tsv_file,
            csv_file,
            data_dir,
            include_wav_info=include_wav_info,
            tokenizer=tokenizer,
            min_nb_chars=min_nb_chars,
            min_duration_secs=min_duration_secs,
            min_nb_tokens=min_nb_tokens,
        )

    return csv_files


def _create_csv(
    tsv_file,
    csv_file,
    data_dir,
    include_wav_info,
    tokenizer,
    min_nb_chars,
    min_duration_secs,
    min_nb_tokens,
):
    clips_dir = data_dir / "clips"

    with open(tsv_file) as tsv_fp:
        tsv_reader = csv.reader(tsv_fp, delimiter="\t")
        next(tsv_reader)  # skip tsv header

        with open(csv_file, mode="w", encoding="utf-8") as csv_fp:
            csv_writer = csv.writer(
                csv_fp, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            header = ["ID", "text", "source"]
            if include_wav_info:
                header += ["duration", "start", "stop", "wav_file"]
            elif tokenizer:
                header.append("duration")

            csv_writer.writerow(header)
            for row in tqdm(tsv_reader):
                mp3_file = clips_dir / row[1]
                # build utterance id
                utterance_id = "cv_" + mp3_file.stem.split("_")[-1]

                text = row[2]
                # preprocess text (uppercase, replace numbers, remove punctuation, etc)
                text = preprocess_text(text)
                # skip utterance if text is too short
                if text is None or len(text.replace(" ", "")) < min_nb_chars:
                    continue
                # there is additional processing done on text before computing WER
                # skip utterance if groundtruth text would be empty before computer WER
                if len(prepare_words_for_wer(text)) == 0:
                    continue

                if include_wav_info:
                    # get duration
                    audio_info = torchaudio.info(mp3_file)
                    nb_samples = audio_info.num_frames
                    duration = nb_samples / audio_info.sample_rate
                    # skip utterance if too short
                    if duration < min_duration_secs:
                        continue
                # tokenize text if tokenizer is provided (for Language Model training)
                elif tokenizer:
                    # compute nb_tokens and use as duration
                    nb_tokens = len(tokenizer.encode(text))
                    # skip utterance if not enough tokens
                    if nb_tokens < min_nb_tokens:
                        continue

                row = [utterance_id, text, "cv"]
                if include_wav_info:
                    # use special values -1, -1 as start/stop because mp3_file contains whole utterance
                    row += [duration, -1, -1, mp3_file]
                elif tokenizer:
                    # use nb_tokens as duration
                    row.append(nb_tokens)

                csv_writer.writerow(row)
