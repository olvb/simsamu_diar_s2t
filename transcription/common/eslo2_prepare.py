# generate speechbrain CSV files from ESLO2 to use for training
# based on https://github.com/speechbrain/speechbrain/blob/develop/recipes/CommonVoice/common_voice_prepare.py

import csv

import pysrt
import torchaudio

torchaudio.set_audio_backend("soundfile")
from tqdm import tqdm

from .preprocess_text import preprocess_text, prepare_words_for_wer


def prepare_eslo2(
    srt_dir,
    save_dir,
    wav_dir=None,
    tokenizer=None,
    min_nb_chars=2,
    min_duration_secs=0.01,
    min_nb_tokens=2,
):
    """
    Prepare the ESLO2 dataset for ASR model training

    For each split (train, dev, test), a CSV file will be created with
    one row per utterance and the following columns:
     - ID: unique ID for the utterance (prefixed by "eslo2_")
     - text: groundtruth transcription, preprocessed (cf preprocess_text() )
     - source: always set to "eslo2" (used for dataset balancing)

    if wav_dir is provided:
     - duration: length of utterance, in seconds
       This field is used for building batches with items of same duration to
       sort them. speechbrain needs the field to be named "duration".
     - start: start in seconds of the utterance in the file (eslo2 audio files
       contain several utterances)
     - end: end in seconds of the utterance in the file
     - wav_file: path to the audio file

    if tokenizer is provided:
     - duration: number of tokens, if tokenizer was provided
       For Language Model training, we need the number of tokens to batch
       togethers utterances of same length. speechbrain needs the field to be
       named "duration".


    wav_dir and tokenizer are mutually exclusive.

    Parameters
    ----------
    srt_dir:
        Path to the directory containing the ESLO2 .srt files generated
        from the .trs files
    save_dir:
        Directory into which the train, dev and test csvs will be saved
    wav_dir:
        Optional path to the directory containing the ESLO2 wav files
    tokenizer:
        Optional tokenizer, used only for Language Model training, to precompute
        the number of tokens for each utterance
    min_nb_chars:
        Min number of chars in transcribed text (utterance is ignored if less)
    min_duration_secs:
        Min duration of utterance in seconds if wav_dir is provided (utterance
        is ignored if less)
    min_nb_tokens:
        Min number of tokens of utterance if tokenizer is provided (utterance is
        ignored if less)
    """

    assert wav_dir is None or tokenizer is None

    _check_srt_dir(srt_dir)
    if wav_dir:
        _check_wav_dir(wav_dir)

    splits = ["train", "dev", "test"]
    csv_files = [save_dir / f"{split}.csv" for split in splits]
    if all(f.exists() for f in csv_files):
        print("ESLO2 CSV files already exist, skipping data preparation")
        return csv_files

    save_dir.mkdir(parents=True, exist_ok=True)

    for split, csv_file in zip(splits, csv_files):
        srt_split_dir = srt_dir / split
        _create_csv(
            srt_split_dir,
            csv_file,
            wav_dir=wav_dir,
            tokenizer=tokenizer,
            min_nb_chars=min_nb_chars,
            min_duration_secs=min_duration_secs,
            min_nb_tokens=min_nb_tokens,
        )

    return csv_files


def _check_wav_dir(wav_dir):
    assert (
        len(list(wav_dir.glob("record-*"))) == 265
    ), f"{wav_dir} is missing some records"


def _check_srt_dir(srt_dir):
    train_dir = srt_dir / "train"
    dev_dir = srt_dir / "dev"
    test_dir = srt_dir / "test"
    assert (
        train_dir.exists() and dev_dir.exists() and test_dir.exists()
    ), f"{srt_dir} doesn't contain split folders"


def _create_csv(
    srt_split_dir,
    csv_file,
    wav_dir,
    tokenizer,
    min_nb_chars,
    min_duration_secs,
    min_nb_tokens,
):
    with open(csv_file, mode="w", encoding="utf-8") as fp:
        csv_writer = csv.writer(
            fp, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        header = ["ID", "text", "source"]
        if wav_dir:
            header += ["duration", "start", "stop", "wav_file"]
        elif tokenizer:
            header.append("duration")
        csv_writer.writerow(header)

        for srt_file in tqdm(sorted(srt_split_dir.glob("*.srt"))):
            _add_srt_to_csv(
                srt_file,
                csv_writer,
                wav_dir=wav_dir,
                tokenizer=tokenizer,
                min_nb_chars=min_nb_chars,
                min_duration_secs=min_duration_secs,
                min_nb_tokens=min_nb_tokens,
            )


def _add_srt_to_csv(
    srt_file,
    csv_writer,
    wav_dir,
    tokenizer,
    min_nb_chars,
    min_duration_secs,
    min_nb_tokens,
):
    """Read all utterances in a .srt file and add them to a .csv file"""

    rows = []
    rec_id = srt_file.stem

    if wav_dir:
        wav_file = wav_dir / f"{rec_id}.wav"
        audio_info = torchaudio.info(wav_file)
        sample_rate = audio_info.sample_rate

    for i, srt_item in enumerate(pysrt.open(srt_file)):
        # build utterance id
        utterance_id = f"eslo2_{rec_id}_{i}"

        text = srt_item.text
        # preprocess text (uppercase, replace numbers, remove punctuation, etc)
        text = preprocess_text(text)
        # skip if text is too short
        if text is None or len(text.replace(" ", "")) < min_nb_chars:
            continue
        # there is additional processing done on text before computing WER
        # skip utterance if groundtruth text would be empty before computer WER
        # skip if empty after pre-WER cleanup
        if len(prepare_words_for_wer(text)) == 0:
            continue

        if wav_dir:
            # compute duration in secs
            start_ms = srt_item.start.ordinal
            start = int(start_ms / 1000 * sample_rate)
            stop_ms = srt_item.end.ordinal
            stop = int(stop_ms / 1000 * sample_rate)
            duration = (stop - start) / sample_rate
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

        row = [utterance_id, text, "eslo2"]
        if wav_dir:
            row += [duration, start, stop, wav_file]
        elif tokenizer:
            # use nb_tokens as duration
            row += [nb_tokens]

        rows.append(row)

    csv_writer.writerows(rows)
