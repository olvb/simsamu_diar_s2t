# evaluation of a diarization pipeline on directory with .wav and .rttm files

import click
from pathlib import Path

import numpy as np
import pandas as pd
from pyannote.core import Segment, Annotation
from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
import torch
from tqdm import tqdm
import yaml


EMBEDDING_EXCLUDE_OVERLAP = True
CLUSTERING = "AgglomerativeClustering"
MIN_NB_SPEAKERS = 1
MAX_NB_SPEAKERS = 3


# workaround bug in pyannote.database.util.load_rttm
# (dropping all rows in rttm file wih <NA> url)
# copy/pasted from https://github.com/pyannote/pyannote-database/blob/develop/pyannote/database/util.py
# with one change
def load_rttm(file_rttm, keep_type="SPEAKER"):
    names = [
        "type",
        "uri",
        "NA2",
        "start",
        "duration",
        "NA3",
        "NA4",
        "speaker",
        "NA5",
        "NA6",
    ]
    dtype = {"uri": str, "start": float, "duration": float, "speaker": str}
    data = pd.read_csv(
        file_rttm,
        names=names,
        dtype=dtype,
        delim_whitespace=True,
        # the change (keep "<NA>" as string otherwise groupby("uri") yields nothing)
        keep_default_na=False,
    )

    annotations = dict()
    for uri, turns in data.groupby("uri"):
        annotation = Annotation(uri=uri)
        for i, turn in turns.iterrows():
            if turn.type != keep_type:
                continue
            segment = Segment(turn.start, turn.start + turn.duration)
            annotation[segment, i] = turn.speaker
        annotations[uri] = annotation

    return annotations


@click.command()
@click.option("--segmentation_model", required=True, type=click.Path(path_type=Path))
@click.option("--embedding", required=True, type=str)
@click.option("--diarization_params", required=True, type=click.Path(path_type=Path))
@click.option("--output_dir", required=True, type=click.Path(path_type=Path))
@click.option(
    "--dataset_dir", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--hf_auth_token",
    type=str,
    help="Optional hugging face hub auth token (to retrieve private pretrained pyannote models)",
)
def main(
    segmentation_model,
    embedding,
    diarization_params,
    output_dir,
    dataset_dir,
    hf_auth_token,
):
    # init pipeline
    seg_model = Model.from_pretrained(
        segmentation_model,
        use_auth_token=hf_auth_token,
        map_location=torch.device("cuda:0"),
    ).to("cuda:0")

    pipeline = SpeakerDiarization(
        segmentation=seg_model,
        embedding=embedding,
        embedding_exclude_overlap=EMBEDDING_EXCLUDE_OVERLAP,
        clustering=CLUSTERING,
    ).to(torch.device("cuda"))
    with open(diarization_params) as fp:
        params = yaml.safe_load(fp)
    params = {
        "segmentation": {
            # "min_duration_off": 0.9559021479110187,
            "min_duration_off": 0.0
        },
        "clustering":{
            "threshold": 0.4179634569901917,
            # "threshold": 0.7045654963945799,
            "method": "centroid",
            # "min_cluster_size": 17,
            "min_cluster_size": 12,
        },
    }
    pipeline = pipeline.instantiate(params)

    metric_strict = GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
    metric_lax = GreedyDiarizationErrorRate(collar=0.5, skip_overlap=True)
    for wav_file in tqdm(sorted(Path(dataset_dir).glob("*.wav"))):
        diarization = pipeline(
            wav_file, 
            min_speakers=MIN_NB_SPEAKERS,
            max_speakers=MAX_NB_SPEAKERS,
        )
        rttm_file = wav_file.with_suffix(".rttm")
        annotation = list(load_rttm(rttm_file).values())[0]
        annotation.uri = wav_file.stem
        metric_strict(annotation, diarization)
        metric_lax(annotation, diarization)

    # save report
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval.txt", mode="w") as fp:
        fp.write("DER collar=0.0, skip_overlap=False\n")
        fp.write(str(metric_strict))
        fp.write("\n\n=======================\n\n")
        fp.write("DER collar=0.5, skip_overlap=True\n")
        fp.write(str(metric_lax))


if __name__ == "__main__":
    main()
