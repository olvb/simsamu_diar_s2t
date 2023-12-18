# helper class to access ESLO2 recodings

import csv
from pathlib import Path
from typing import List

import pandas as pd
import torchaudio

torchaudio.set_audio_backend("soundfile")
from tqdm import tqdm

from .trs_reader import read_trs


REC_IDS_TO_IGNORE = [
    # no TRS file
    "record-007",
    "record-019",
    "record-026",
    "record-036",
    "record-113",
    "record-159",
    "record-180",
    # # issue with annotation of multiple speakers
    "record-111",
]

MONO_SPEAKER_REC_IDS = [
    "record-179",
    "record-193",
    "record-217",
    "record-221",
]


class RecordingStore:
    """Holds the TRS rows of each ESLO2 recording"""

    def __init__(
        self,
        eslo2_dir: Path,
        audio_dir: Path,
        ignore_mono_speakers: bool = True,
        ignore_meals_category: bool = True,
    ):
        self.eslo2_dir = eslo2_dir
        self.audio_dir = audio_dir
        self.ignore_meals_category = ignore_meals_category
        self.ignore_mono_speakers = ignore_mono_speakers
        self._trs_rows_by_rec_id = {}
        self._load_trs_rows()

    def _load_trs_rows(self):
        # list rec ids and filter them
        rec_ids = [d.name for d in sorted(self.eslo2_dir.glob("record-*"))]
        rec_ids = [rec_id for rec_id in rec_ids if rec_id not in REC_IDS_TO_IGNORE]
        if self.ignore_mono_speakers:
            rec_ids = [
                rec_id for rec_id in rec_ids if rec_id not in MONO_SPEAKER_REC_IDS
            ]
        if self.ignore_meals_category:
            rec_ids = [
                rec_id for rec_id in rec_ids if self.get_category(rec_id) != "REPAS"
            ]

        for rec_id in tqdm(rec_ids, desc="Reading TRS files"):
            trs_file = next((self.eslo2_dir / rec_id).glob("*.trs"))
            rows = pd.DataFrame(read_trs(trs_file))

            # at least 2 speakers
            if self.ignore_mono_speakers:
                assert (
                    len(pd.unique(rows["speaker"].values)) > 1
                ), f"Only 1 speaker found in {trs_file}"

            self._trs_rows_by_rec_id[rec_id] = rows

    @property
    def rec_ids(self) -> List[str]:
        return list(self._trs_rows_by_rec_id.keys())

    def get_trs_rows(self, rec_id: str) -> pd.DataFrame:
        return self._trs_rows_by_rec_id[rec_id]

    def get_category(self, rec_id: str) -> str:
        # extract category from .mp3 file name
        mp3_file = next((self.eslo2_dir / rec_id).glob("*.mp3"))
        return mp3_file.name.split("_")[1]

    def get_duration(self, rec_id: str) -> float:
        # NB: we must use duration of processed .wav file rather than original .mp3 file
        # because they may slightly differ and pyannote is not happy if we use the other one
        audio_file = self.audio_dir / f"{rec_id}.wav"
        info = torchaudio.info(audio_file)
        return info.num_frames / info.sample_rate

    def get_speakers_in_rec(self, rec_id: str) -> List[str]:
        rows = self._trs_rows_by_rec_id[rec_id]
        speakers = list(pd.unique(rows["speaker"].values))
        if self.ignore_mono_speakers:
            # make sure we don't have mono speaker files
            assert len(speakers) >= 2
        return speakers

    def write_summary(self, file: Path):
        all_speakers = set()
        total_duration = 0.0
        recs = []
        for rec_id in self.rec_ids:
            category = self.get_category(rec_id)
            duration = self.get_duration(rec_id) / 60
            speakers = self.get_speakers_in_rec(rec_id)
            recs.append(
                dict(
                    rec_id=rec_id,
                    category=category,
                    duration=duration,
                    nb_speakers=len(speakers),
                ),
            )

            all_speakers.update(speakers)
            total_duration + duration

        recs.append(
            dict(
                rec_id="TOTAL",
                category=None,
                duration=total_duration,
                nb_speakers=len(all_speakers),
            )
        )
        with open(file, mode="w") as fp:
            writer = csv.DictWriter(fp, fieldnames=recs[0].keys())
            writer.writeheader()
            writer.writerows(recs)
