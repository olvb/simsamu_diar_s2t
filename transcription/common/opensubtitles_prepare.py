# generate speechbrain CSV files from OpenSubtitles dataset to use for training
# based on https://github.com/speechbrain/speechbrain/blob/develop/recipes/CommonVoice/common_voice_prepare.py

import csv
import random

from tqdm import tqdm

from .preprocess_text import preprocess_text


def prepare_opensubtitles(
    txt_dir, save_dir, subset: float, tokenizer=None, min_nb_chars=2, min_nb_tokens=2
):
    """
    Prepare the Open Subtitles dataset for Tokenizer or Language Model training

    For each split (train, dev, test), a CSV file will be created with
    one row per utterance and the following columns:
     - ID: unique ID for the utterance (prefixed by "os_")
     - text: groundtruth transcription, preprocessed (cf preprocess_text() )
     - source: always set to "os" (used for dataset balancing)

    if tokenizer is provided:
     - duration: number of tokens, if tokenizer was provided
       For Language Model training, we need the number of tokens to batch togethers
       utterances of same length.
       speechbrain needs the field to be named "duration".

    Parameters
    ----------
    txt_dir:
        Path to the directory containing the .txt files generated from the Open
        Subtitles .srt files
    subset:
        Proportion of all the availables Open Subtitles txt files that should be used.
        (Open Subtitles is very big, we sometimes don't want to use all of it)
    save_dir:
        Directory into which the train, dev and test csvs will be saved
    tokenizer:
        Optional tokenizer, used only for Language Model training, to precompute
        the number of tokens for each utterance
    min_nb_chars:
        Min number of chars in transcribed text (utterance is ignored if less)
    min_nb_tokens:
        Min number of tokens of utterance if tokenizer is provided (utterance is
        ignored if less)
    """

    assert txt_dir.exists()

    files = sorted(txt_dir.glob("*.txt"))

    # take subset of .txt files
    if subset < 1.0:
        rng = random.Random(0)
        rng.shuffle(files)
        files = sorted(files[: int(subset * len(files))])

    files_by_split = _split(files)
    csv_files = [save_dir / f"{split}.csv" for split in files_by_split]
    if all(f.exists() for f in csv_files):
        print("OpenSubtitles CSV files already exist, skipping data preparation")
        return csv_files

    save_dir.mkdir(parents=True, exist_ok=True)

    for csv_file, files in zip(csv_files, files_by_split.values()):
        _create_csv(
            csv_file,
            files,
            tokenizer=tokenizer,
            min_nb_chars=min_nb_chars,
            min_nb_tokens=min_nb_tokens,
        )

    return csv_files


def _split(files, train_ratio=0.9, dev_ratio=0.05, test_ratio=0.05):
    """Generate train/dev/test split of .txt files"""
    assert train_ratio + dev_ratio + test_ratio == 1.0

    rng = random.Random(0)
    files = files.copy()
    rng.shuffle(files)

    nb_files = len(files)
    nb_train = int(train_ratio * nb_files)
    nb_dev = int(dev_ratio * nb_files)
    nb_test = nb_files - (nb_train + nb_dev)

    files_by_split = dict(
        train=files[:nb_train],
        dev=files[nb_train : nb_train + nb_dev],
        test=files[-nb_test:],
    )
    return files_by_split


def _create_csv(csv_file, txt_files, tokenizer, min_nb_chars, min_nb_tokens):
    with open(csv_file, mode="w", encoding="utf-8") as fp:
        csv_writer = csv.writer(
            fp, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        header = ["ID", "text", "source"]
        if tokenizer:
            header.append("duration")
        csv_writer.writerow(header)

        for txt_file in tqdm(txt_files):
            _add_txt_to_csv(
                txt_file,
                csv_writer,
                tokenizer=tokenizer,
                min_nb_chars=min_nb_chars,
                min_nb_tokens=min_nb_tokens,
            )


def _add_txt_to_csv(txt_file, csv_writer, tokenizer, min_nb_chars, min_nb_tokens):
    rows = []

    rec_id = txt_file.stem
    with open(txt_file) as fp:
        for i, line in enumerate(fp.readlines()):
            # build utterance id
            utterance_id = f"os_{rec_id}_{i}"
            # preprocess text (uppercase, replace numbers, remove punctuation, etc)
            text = preprocess_text(line)
            # skip utterance if too short
            if text is None or len(text.replace(" ", "")) < min_nb_chars:
                continue

            # tokenize text if tokenizer is provided (for Language Model training)
            if tokenizer:
                # compute nb_tokens and use as duration
                nb_tokens = len(tokenizer.encode(text))
                # skip utterance if not enough tokens
                if nb_tokens < min_nb_tokens:
                    continue

            row = [utterance_id, text, "os"]
            if tokenizer:
                # use nb_tokens as duration
                row.append(nb_tokens)
            rows.append(row)

    csv_writer.writerows(rows)
