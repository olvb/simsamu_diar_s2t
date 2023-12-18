# generate speechbrain CSV files from simsamu dataset to use for training
# based on https://github.com/speechbrain/speechbrain/blob/develop/recipes/CommonVoice/common_voice_prepare.py

from collections import defaultdict
import csv
import random

import pysrt
import torchaudio

torchaudio.set_audio_backend("soundfile")
from tqdm import tqdm
from sklearn.model_selection import KFold

from .preprocess_text import preprocess_text, prepare_words_for_wer


def fold_simsamu_speaker_aware(data_dir):
    utterance_ids_by_speaker = _get_utterance_ids_by_speaker(data_dir)
    rng = random.Random(1234)
    folds = []
    for test_speaker in utterance_ids_by_speaker:
        test_utterance_ids = utterance_ids_by_speaker[test_speaker]
        train_utterance_ids = [
            utterance_id
            for speaker, utterance_ids in utterance_ids_by_speaker.items()
            if speaker != test_speaker
            for utterance_id in utterance_ids
        ]
        rng.shuffle(train_utterance_ids)
        rng.shuffle(test_utterance_ids)
        fold = {"train": train_utterance_ids, "test": test_utterance_ids}
        folds.append(fold)
    return folds

def fold_simsamu(data_dir, nb_folds):
    utterance_ids_by_speaker = _get_utterance_ids_by_speaker(data_dir)
    utterance_ids = [
        utterance_id
        for utterance_ids in utterance_ids_by_speaker.values()
        for utterance_id in utterance_ids
    ]
    rng = random.Random(1234)
    folds = []
    kfold = KFold(n_splits=nb_folds, shuffle=True, random_state=1234)
    for train_indices, test_indices in kfold.split(utterance_ids):
        train_utterance_ids = [utterance_ids[i] for i in train_indices]
        test_utterance_ids = [utterance_ids[i] for i in test_indices]
        rng.shuffle(train_utterance_ids)
        rng.shuffle(test_utterance_ids)
        fold = {"train": train_utterance_ids, "test": test_utterance_ids}
        folds.append(fold)
    return folds


def _get_speaker_names_by_rec_id(metadata_file):
    with open(metadata_file) as fp:
        rows = csv.reader(fp)
        next(rows)
        speaker_names_by_rec_id = {}
        for row in rows:
            rec_id = _get_rec_id(row[0])
            names = {}
            names["medecin"] = row[1]
            patient_names = row[2].split("-")
            for i, patient_name in enumerate(patient_names):
                names[f"patient_{i+1}"] = patient_name
            # aliases
            names["patient"] = patient_names[0]
            names["appelant"] = patient_names[0]

            speaker_names_by_rec_id[rec_id] = names
    return speaker_names_by_rec_id

def _load_rttm(rttm_file, speaker_names):
    with open(rttm_file) as fp:
        rows = csv.reader(fp, delimiter=" ")
        speakers_by_time = {}
        for row in rows:
            start_secs = float(row[3])
            end_secs = start_secs + float(row[4])
            speaker_role = row[7]
            if speaker_role == "inintelligible":
                continue
            speaker = speaker_names[speaker_role]
            speakers_by_time[(start_secs, end_secs)] = speaker
    return speakers_by_time

def _get_utterance_ids_by_speaker(data_dir):
    speaker_names_by_rec_id = _get_speaker_names_by_rec_id(data_dir / "metadata.csv")
    srt_files = sorted(data_dir.glob("*.srt"))
    utterance_ids_by_speaker = defaultdict(list)
    for srt_file in srt_files:
        rec_id = _get_rec_id(srt_file.stem)
        speaker_names = speaker_names_by_rec_id[rec_id]
        for speaker, utterance_ids in _get_utterance_ids_by_speaker_in_srt_file(srt_file, speaker_names).items():
            utterance_ids_by_speaker[speaker] += utterance_ids
    return utterance_ids_by_speaker

def _get_utterance_ids_by_speaker_in_srt_file(srt_file, speaker_names):
    rttm_file = srt_file.with_suffix(".rttm")
    speakers_by_time = _load_rttm(rttm_file, speaker_names)

    rec_id = _get_rec_id(srt_file.stem)

    utterance_ids_by_speaker = defaultdict(list)

    for i, srt_item in enumerate(pysrt.open(srt_file)):
        if srt_item.text.strip() in ("<NA>", ""):
            continue

        start_ms = srt_item.start.ordinal
        end_ms = srt_item.end.ordinal
        start_secs = start_ms / 1000
        end_secs = end_ms / 1000

        speaker = _get_speaker(start_secs, end_secs, speakers_by_time)
        utterance_id = f"simsamu_{rec_id}_{i}"

        utterance_ids_by_speaker[speaker].append(utterance_id)
    
    return utterance_ids_by_speaker

def _get_speaker(start, end, speakers_by_time):
    for (speaker_start, speaker_end), speaker in speakers_by_time.items():
        if speaker_start <= end and speaker_end >= start:
            return speaker
    raise Exception("Speaker not found")

def _get_rec_id(basename):
    # handle bad simsamu ids
    return (
        basename.replace(" ", "_")
        .replace(".", "_")
        .replace("é", "e")
        .replace("è", "e")
    )

def get_simsamu_utterance_ids(data_dir):
    utterance_ids_by_speaker = _get_utterance_ids_by_speaker(data_dir)
    return [
        utterance_id
        for utterance_ids in utterance_ids_by_speaker.values()
        for utterance_id in utterance_ids
    ]

def prepare_simsamu(
    data_dir,
    save_dir,
    utterance_ids_by_split,
    include_wav_info,
    min_nb_chars=2,
    min_duration_secs=0.01,
):
    """
    Prepare the Simsamu dataset for ASR model training and/or evaluation

    For each split (train, dev, test), a CSV file will be created with
    one row per utterance and the following columns:
     - ID: unique ID for the utterance (prefixed by "simsamu_")
     - text: groundtruth transcription, preprocessed (cf preprocess_text() )
     - source: always set to "simsamu"
     - duration: length of utterance, in seconds
       This field is used for building batches with items of same duration to
       sort them. speechbrain needs the field to be named "duration".
     - start: start in seconds of the utterance in the file (simsamu audio files
       contain several utterances)
     - end: end in seconds of the utterance in the file
     - wav_file: path to the audio file
    """

    csv_files = [save_dir / f"{split}.csv" for split in utterance_ids_by_split.keys()]
    if all(f.exists() for f in csv_files):
        print("Simsamu CSV files already exist, skipping data preparation")
        return csv_files
    
    save_dir.mkdir(parents=True, exist_ok=True)

    srt_files = sorted(data_dir.glob("**/*.srt"))

    for split, utterance_ids in utterance_ids_by_split.items():
        csv_file = save_dir / f"{split}.csv"
        _create_csv(
            csv_file,
            srt_files,
            include_wav_info=include_wav_info,
            utterance_ids=utterance_ids,
            min_nb_chars=min_nb_chars,
            min_duration_secs=min_duration_secs,
        )

    return csv_files


def _create_csv(
    csv_file,
    srt_files,
    include_wav_info,
    utterance_ids,
    min_nb_chars,
    min_duration_secs,
):
    with open(csv_file, mode="w", encoding="utf-8") as fp:
        csv_writer = csv.writer(
            fp, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        header = ["ID", "text", "source"]
        if include_wav_info:
            header += ["duration", "start", "stop", "wav_file"]
        csv_writer.writerow(header)

        for srt_file in srt_files:
            _add_srt_to_csv(
                srt_file,
                csv_writer,
                include_wav_info=include_wav_info,
                utterance_ids=utterance_ids,
                min_nb_chars=min_nb_chars,
                min_duration_secs=min_duration_secs,
            )


def _add_srt_to_csv(
    srt_file, csv_writer, include_wav_info, utterance_ids, min_nb_chars, min_duration_secs
):
    """Read all utterances in a .srt file and add them to a .csv file"""

    rows = []
    
    # handle bad simsamu ids
    rec_id = _get_rec_id(srt_file.stem)

    wav_file = srt_file.with_suffix(".m4a")
    audio_info = torchaudio.info(wav_file)
    sample_rate = audio_info.sample_rate

    for i, srt_item in enumerate(pysrt.open(srt_file)):
        text = srt_item.text
        if text == "<NA>":
            continue

        # build utterance id
        utterance_id = f"simsamu_{rec_id}_{i}"
        # skip non-included utterances
        if utterance_ids is not None and utterance_id not in utterance_ids:
            continue

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

        # compute duration in secs
        start_ms = srt_item.start.ordinal
        stop_ms = srt_item.end.ordinal
        start = int(start_ms / 1000 * sample_rate)
        stop = int(stop_ms / 1000 * sample_rate)
        duration = (stop - start) / sample_rate

        # skip utterance if too short
        if duration < min_duration_secs:
            continue

        row = [utterance_id, text, "simsamu"]
        if include_wav_info:
            row += [duration, start, stop, wav_file]
        rows.append(row)

    csv_writer.writerows(rows)

