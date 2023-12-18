# evaluation of a full diarization pipeline on a pyannote protocol/dataset

from pathlib import Path
from typing import Optional, Union

from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.database import Protocol
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
import torch
from tqdm import tqdm
import yaml


EMBEDDING_EXCLUDE_OVERLAP = True


def evaluate_diarization_pipeline(
    protocol: Protocol,
    output_dir: Path,
    segmentation: Union[str, Path],
    embedding: Union[str, Path],
    clustering: str,
    params_file: Path,
    min_nb_speakers: Optional[int],
    max_nb_speakers: Optional[int],
    hf_auth_token: Optional[str] = None,
):
    """
    Evaluate a speaker diarization pipeline

    Parameters
    ----------
    protocol:
        Dataset to use for evaluation (dev and test splits will be used)
    output_dir:
        Where to save the txt files with the evaluation metrics
    segmentation:
        Segmentation model (path or name on hugginface hub) to use
    embedding:
        Embeddings model (path or name on hugginface hub) to use
    clustering:
        Clustering method to use, ex: "HiddenMarkovModelClustering"
        See pyannote.audio.pipelines.clustering.Clustering for available options
    params_file:
        Path to yaml file containing the pipeline hyperparams
    min_nb_speakers:
        Min number of speakers per recording
    max_nb_speakers:
        Max number of speakers per recording
    hf_auth_token:
        Optional hugging face hub token, needed to retrieve private models
    """

    # init pipeline
    seg_model = Model.from_pretrained(
        segmentation,
        use_auth_token=hf_auth_token,
        map_location=torch.device("cuda:0"),
    ).to("cuda:0")
    pipeline = SpeakerDiarization(
        segmentation=seg_model,
        embedding=embedding,
        embedding_exclude_overlap=EMBEDDING_EXCLUDE_OVERLAP,
        clustering=clustering,
    ).to(torch.device("cuda"))
    with open(params_file) as fp:
        best_params = yaml.safe_load(fp)
    pipeline.instantiate(best_params)

    # feed dev split to pipeline and compute metrics
    metric_strict = GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
    metric_lax = GreedyDiarizationErrorRate(collar=0.5, skip_overlap=True)
    for file in tqdm(protocol.development()):
        diarization = pipeline(file, min_speakers=min_nb_speakers, max_speakers=max_nb_speakers)
        metric_strict(file["annotation"], diarization, uem=file["annotated"])
        metric_lax(file["annotation"], diarization, uem=file["annotated"])

    # save report
    with open(output_dir / "dev_eval.txt", mode="w") as fp:
        fp.write("DER collar=0.0, skip_overlap=False\n")
        fp.write(str(metric_strict))
        fp.write("\n\n=======================\n\n")
        fp.write("DER collar=0.5, skip_overlap=True\n")
        fp.write(str(metric_lax))

    # feed test split to pipeline and compute metrics
    metric_strict = GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
    metric_lax = GreedyDiarizationErrorRate(collar=0.5, skip_overlap=True)
    for file in tqdm(protocol.test()):
        diarization = pipeline(file)
        metric_strict(file["annotation"], diarization, uem=file["annotated"])
        metric_lax(file["annotation"], diarization, uem=file["annotated"])

    # save report
    with open(output_dir / "test_eval.txt", mode="w") as fp:
        fp.write("DER collar=0.0, skip_overlap=False\n")
        fp.write(str(metric_strict))
        fp.write("\n\n=======================\n\n")
        fp.write("DER collar=0.5, skip_overlap=True\n")
        fp.write(str(metric_lax))
