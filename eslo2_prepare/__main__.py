# main script for preprocessing an ESLO2 dataset
# before using it with pyannote and speechbrain
# (proces audio, split datasets, generate uem/rttm files)

from pathlib import Path
import warnings

import click

from .process import process
from .recording_store import RecordingStore
from .split import make_stratified_split, describe_split
from .writers import (
    write_rttms,
    write_uems,
    write_srts,
    write_pyannote_lists,
    write_pyannote_database,
)


@click.command()
@click.argument("source_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("dest_dir", type=click.Path(path_type=Path))
def main(source_dir, dest_dir):
    """
    Preprocess ESLO2 dataset (proces audio, split datasets, generate uem/rttm files for diarization and srt files for transcription)

    source_dir: path to main ESLO2 dir containing all record-* folders

    dest_dir: output dir
    """

    dest_dir.mkdir(parents=True, exist_ok=True)

    if len(list(source_dir.glob("record-*"))) == 0:
        raise Exception(f"Could not find any record-* subdir in {source_dir}")

    print("\nPhone-processing audio files...")
    audio_dir = dest_dir / "audio"
    process(source_dir, audio_dir)

    # for diarization:
    # gen rttms (split 0.8/0.1/0.1, ignore mono speaker recs)
    print("\nGenerating diarization split...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec_store = RecordingStore(
            source_dir,
            audio_dir,
            ignore_mono_speakers=True,
            ignore_meals_category=False,
        )
    diar_dest_dir = dest_dir / "diarization"
    diar_dest_dir.mkdir(parents=True, exist_ok=True)
    rec_store.write_summary(diar_dest_dir / "summary.csv")
    rec_ids_by_split = make_stratified_split(
        rec_store,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
    )
    describe_split(rec_ids_by_split, rec_store)
    write_rttms(rec_ids_by_split, rec_store, rttm_dir=diar_dest_dir / "rttms")
    write_uems(rec_ids_by_split, rec_store, uem_dir=diar_dest_dir / "uems")
    write_pyannote_lists(rec_ids_by_split, list_dir=diar_dest_dir / "lists")

    write_pyannote_database(audio_dir=audio_dir, diar_dir=diar_dest_dir)

    # for speech to text:
    # gen srts (split 0.8/0.1/0.1, keep mono speaker recs)
    # ignore overlaping segments
    # TODO: do we really need a stratified split for transcription?
    print("\nGenerating transcription split...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec_store = RecordingStore(
            source_dir,
            audio_dir,
            ignore_mono_speakers=False,
            ignore_meals_category=True,
        )
    s2t_dest_dir = dest_dir / "transcription"
    s2t_dest_dir.mkdir(parents=True, exist_ok=True)
    rec_store.write_summary(s2t_dest_dir / "summary.csv")
    rec_ids_by_split = make_stratified_split(
        rec_store,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
    )
    write_srts(
        rec_ids_by_split, rec_store, srt_dir=s2t_dest_dir / "srts", ignore_overlap=True
    )


if __name__ == "__main__":
    main()
