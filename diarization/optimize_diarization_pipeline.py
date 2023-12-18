# hyperparams optimization of a diarization pipeline

from pathlib import Path
from typing import Any, Dict, Optional, Union

from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.database import Protocol
from pyannote.pipeline import Optimizer
import torch
import yaml

EMBEDDING_EXCLUDE_OVERLAP = True

class ConstrainedSpeakerDiarization(SpeakerDiarization):
    """Subclass of SpeakerDiarization with a constraint on the number of speakers"""

    def __init__(
        self,
        *args,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def apply(self, file, hook=None):
        return super().apply(
            file=file,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            hook=hook,
        )


def optimize_diarization_pipeline(
    protocol: Protocol,
    output_dir: Path,
    segmentation: Union[str, Path],
    embedding: Union[str, Path],
    clustering: str,
    min_nb_speakers: Optional[int],
    max_nb_speakers: Optional[int],
    initial_params: Dict[str, Any],
    frozen_params: Dict[str, Any],
    sampler,
    pruner,
    nb_iters: int,
    hf_auth_token: Optional[str] = None,
) -> Path:
    """
    Optimize the hyper parameters of a speaker diarization pipeline (including clustering parameters),
    using pyannote's Optimizer based on optuna

    Parameters
    ----------
    protocol:
        Dataset to use for optimization (dev split will be used)
    output_dir:
        Where to save the best hyper parameters
    segmentation:
        Segmentation model (path or name on hugginface hub) to use
    embedding:
        Embeddings model (path or name on hugginface hub) to use
    clustering:
        Clustering method to use, ex: "HiddenMarkovModelClustering"
        See pyannote.audio.pipelines.clustering.Clustering for available options
    min_nb_speakers:
        Min number of speakers per recording
    max_nb_speakers:
        Max number of speakers per recording
    initial_params:
        Initial values of parameters to optimize
    frozen_params:
        Fixed values of parameters not to optimize
    sampler:
        Sampling method to use, cf pyannote.pipeline.Optimizer
    pruner:
        Pruning method to use, cf pyannote.pipeline.Optimizer
    nb_iters:
        Number of optmization iterations
    hf_auth_token:
        Optional hugging face hub token, needed to retrieve private models

    Returns
    -------
    Path:
        Path to file with best params
    """

    output_file = output_dir / "output.yml"
    best_params_file = output_dir / "best.yml"

    # retrieve nb of iterations already performed and corresponding loss
    if output_file.exists():
        with open(output_file) as fp:
            output = yaml.safe_load(fp)
            nb_iters_prev = output["nb_iters"]
            # skip if all iterations have been run
            if nb_iters_prev >= nb_iters and best_params_file.exists():
                print(
                    f"Already optimized diarization params for {nb_iters} iterations, skipping"
                )
                # assert best_params_file.exists()
                return best_params_file
            best_loss = output["best_loss"]
    else:
        nb_iters_prev = 0
        best_loss = float("inf")

    # init segmentation model
    seg_model = Model.from_pretrained(
        segmentation,
        use_auth_token=hf_auth_token,
        map_location=torch.device("cuda:0"),
    ).to("cuda:0")
    pipeline = ConstrainedSpeakerDiarization(
        segmentation=seg_model,
        embedding=embedding,
        embedding_exclude_overlap=EMBEDDING_EXCLUDE_OVERLAP,
        clustering=clustering,
        min_speakers=min_nb_speakers,
        max_speakers=max_nb_speakers,
    ).to(torch.device("cuda"))
    pipeline.instantiate(initial_params)
    pipeline.freeze(frozen_params)

    # run optimizer
    optimizer = Optimizer(
        pipeline,
        db=output_dir / "iterations.db",
        study_name=output_dir.parent.name,
        sampler=sampler,
        pruner=pruner,
    )
    optim_iter = optimizer.tune_iter(
        protocol.development(),
        warm_start=initial_params if nb_iters_prev == 0 else None,
        show_progress=True,
    )

    for current_iter in range(nb_iters_prev, nb_iters):
        print(f"\nIteration {current_iter}:", end="")
        _ = next(optim_iter)
        best_loss = min(optimizer.best_loss, best_loss)
        print(f"\nBest loss: {best_loss:.2f}")

        # save progress in output.yml
        output = dict(
            nb_iters=current_iter + 1,
            best_loss=best_loss,
            best_params=optimizer.best_params,
        )
        with open(output_file, mode="w") as fp:
            yaml.safe_dump(output, fp, sort_keys=False)

    # save final best params
    with open(best_params_file, mode="w") as fp:
        yaml.safe_dump(optimizer.best_params, fp, sort_keys=False)

    return best_params_file
