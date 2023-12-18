# generate speechbrain CSV files from PXSLU dataset to use for training
# based on https://github.com/speechbrain/speechbrain/blob/develop/recipes/CommonVoice/common_voice_prepare.py

import csv

import torchaudio

torchaudio.set_audio_backend("soundfile")

from .preprocess_text import preprocess_text, prepare_words_for_wer


def prepare_pxslu(
    data_dir,
    save_dir,
    include_wav_info=False,
    tokenizer=None,
    min_nb_chars=2,
    min_duration_secs=0.01,
    min_nb_tokens=2,
):
    """
    Prepare the PXSLU dataset for ASR model training

    A unique CSV file (no train/dev/test split) will be created with
    one row per utterance and the following columns:
     - ID: unique ID for the utterance (prefixed by "pxslu_")
     - text: groundtruth transcription, preprocessed (cf preprocess_text() )
     - source: always set to "pxslu" (used for dataset balancing)

    if include_wav_info is True:
     - duration: length of utterance, in seconds
       This field is used for building batches with items of same duration to
       sort them. speechbrain needs the field to be named "duration".
     - start: start in seconds of the utterance in the file
       (always special value -1 for pxslu because there is 1 wav_file per utterance)
     - end: end in seconds of the utterance in the file
       (always special value -1 for pxslu because there is 1 wav_file per utterance)
     - wav_file: path to the audio file

    if tokenizer is provided:
     - duration: number of tokens, if tokenizer was provided For Language Model
       training, we need the number of tokens to batch togethers utterances of
       same length. speechbrain needs the field to be named "duration".

    include_wav_info set to True and tokenizer are mutually exclusive

    Parameters
    ----------
    data_dir:
        Path to the pxslu dir (ex: /scratch/pxslu)
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

    csv_file = save_dir / "train.csv"
    if csv_file.exists():
        print("PXSLU CSV files already exists, skipping data preparation")
        return csv_file

    save_dir.mkdir(parents=True, exist_ok=True)

    texts = (data_dir / "seq.in").read_text().split("\n")[:-1]
    paths = (data_dir / "paths.txt").read_text().split("\n")[:-1]
    assert len(texts) == len(paths)

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

        for text, path in zip(texts, paths):
            wav_file = data_dir / "recordings" / path
            # build utterance id
            utterance_id = (
                "pxslu_" + wav_file.parent.name + "_" + wav_file.stem.split("_")[1]
            )

            # preprocess text (uppercase, replace numbers, remove punctuation,
            # etc) nb: we remove incomplete words begining/ending with "/" when
            # wave_info is requested because that means we are training a language model
            text = preprocess_text(text, drop_slashes=include_wav_info)
            # skip utterance if text is too short
            if text is None or len(text.replace(" ", "")) < min_nb_chars:
                continue
            # there is additional processing done on text before computing WER
            # skip utterance if groundtruth text would be empty before computer WER
            if len(prepare_words_for_wer(text)) == 0:
                continue

            if include_wav_info:
                # get duration
                audio_info = torchaudio.info(wav_file)
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

            row = [utterance_id, text, "pxslu"]
            if include_wav_info:
                # use special values -1, -1 as start/stop because wav_file contains whole utterance
                row += [duration, -1, -1, wav_file]
            elif tokenizer:
                # use nb_tokens as duration
                row.append(nb_tokens)

            csv_writer.writerow(row)

    return csv_file
