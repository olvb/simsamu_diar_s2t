# writing of .RTTM, .SRT and .UEM files for ESLO2 recordings

import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from .recording_store import RecordingStore

# take into account unannotated beginings/ends
REC_IDS_STARTS = {
    "record-137": 594.466,
    "record-139": 53.541,
    "record-254": 13.048,
}
REC_IDS_ENDS = {
    "record-226": 6732.697,
}


def write_rttms(
    rec_ids_by_split: Dict[str, List[str]], rec_store: RecordingStore, rttm_dir: Path
):
    """
    Generate .rttm files (containing diarization segments) for recordings
    """

    for split_name, rec_ids in tqdm(rec_ids_by_split.items(), desc="Writing RTTMs"):
        rttm_subdir = rttm_dir / split_name
        rttm_subdir.mkdir(parents=True, exist_ok=True)

        for rec_id in rec_ids:
            _write_rttm(rec_id, rec_store, rttm_subdir)


def _write_rttm(rec_id, rec_store, rttm_dir):
    rows = rec_store.get_trs_rows(rec_id).copy()

    # add duration and rec_id
    rows["duration"] = rows["end"] - rows["start"]
    rows = rows[["start", "duration", "speaker"]]
    rows.columns = ["start", "duration", "speaker_id"]
    rows.insert(0, "rec_id", rec_id)
    # add placeholder columns
    rows.insert(0, "SPEAKER", "SPEAKER")
    rows.insert(2, "ZERO", 0)
    rows.insert(5, "NA_1", pd.NA)
    rows.insert(6, "NA_2", pd.NA)
    rows.insert(8, "NA_3", pd.NA)

    rttm_file = rttm_dir / f"{rec_id}.rttm"

    rows.to_csv(
        rttm_file,
        mode="w",
        header=False,
        sep=" ",
        na_rep="<NA>",
        index=False,
    )


def write_uems(
    rec_ids_by_split: Dict[str, List[str]], rec_store: RecordingStore, uem_dir: Path
):
    """
    Generate .uem files (indicating which part of a file is annotated) for recordings
    """

    for split_name, rec_ids in tqdm(rec_ids_by_split.items(), desc="Writing UEMs"):
        uem_subdir = uem_dir / split_name
        uem_subdir.mkdir(parents=True, exist_ok=True)

        for rec_id in rec_ids:
            _write_uem(rec_id, rec_store, uem_subdir)


def _write_uem(rec_id, rec_store, uem_dir):
    duration = rec_store.get_duration(rec_id)

    # correct unannotated starts/ends
    start = REC_IDS_STARTS.get(rec_id, 0.0)
    end = REC_IDS_ENDS.get(rec_id, duration)

    uem_file = uem_dir / f"{rec_id}.uem"
    with open(uem_file, mode="w") as fp:
        fp.write(f"{rec_id} 1 {start:.6f} {end:.6f}")


def write_srts(
    rec_ids_by_split: Dict[str, List[str]],
    rec_store: RecordingStore,
    srt_dir: Path,
    ignore_overlap=True,
):
    """
    Generate .srt files (containing transcription of segments) for recordings.

    If ignore_overlap is True, transcription of segments overlaping with other segments is omitted.
    """

    for split_name, rec_ids in tqdm(rec_ids_by_split.items(), desc="Writing SRTs"):
        srt_subdir = srt_dir / split_name
        srt_subdir.mkdir(parents=True, exist_ok=True)

        for rec_id in rec_ids:
            _write_srt(rec_id, rec_store, srt_subdir, ignore_overlap)


def _write_srt(rec_id, rec_store, srt_dir, ignore_overlap):
    srt_file = srt_dir / f"{rec_id}.srt"
    trs_rows = rec_store.get_trs_rows(rec_id)

    def seconds_to_srt_timestamp(seconds: float) -> str:
        td = datetime.timedelta(seconds=seconds)
        minutes, seconds = divmod(td.seconds, 60)
        hours, minutes = divmod(minutes, 60)
        milliseconds = td.microseconds // 1000
        return f"{hours}:{minutes}:{seconds},{milliseconds}"

    with open(srt_file, mode="w") as fp:
        i = 0
        for _, row in trs_rows.iterrows():
            # ign
            if row["overlap"] and ignore_overlap:
                continue
            fp.write(f"{i + 1}\n")
            start = seconds_to_srt_timestamp(row["start"])
            end = seconds_to_srt_timestamp(row["end"])
            fp.write(f"{start} --> {end}\n")
            fp.write(row["text"] + "\n\n")
            i += 1


def write_pyannote_lists(rec_ids_by_split: Dict[str, List[str]], list_dir: Path):
    """
    Generate list of recordings in a .txt file as required by pyannote-database
    """

    list_dir.mkdir(parents=True, exist_ok=True)
    for name, rec_ids in rec_ids_by_split.items():
        with open(list_dir / f"{name}.txt", mode="w") as fp:
            fp.writelines(r + "\n" for r in rec_ids)
        # shorter debug version
        with open(list_dir / f"{name}.debug.txt", mode="w") as fp:
            fp.writelines(r + "\n" for r in rec_ids[0:5])


def write_pyannote_database(audio_dir: Path, diar_dir: Path):
    """Generate and write in diar_dir the database.yml file needed by pyannote"""

    contents = (Path(__file__).parent / "database.yml").read_text()
    contents = contents.replace("/path/to/eslo2_processed/audio", str(audio_dir))
    contents = contents.replace("/path/to/eslo2_processed/diarization", str(diar_dir))
    (diar_dir / "database.yml").write_text(contents)
