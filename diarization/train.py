# main script for fine-tuning a diarization pipeline

from pathlib import Path
from pprint import pprint
import shutil

import click
from pyannote.audio.core.io import get_torchaudio_info
from pyannote.database import get_protocol, FileFinder
import yaml

from .train_segmentation_model import train_segmentation_model
from .optimize_diarization_pipeline import optimize_diarization_pipeline
from .evaluate_diarization_pipeline import evaluate_diarization_pipeline


@click.command()
@click.argument("exp_params_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output_dir",
    required=True,
    type=click.Path(path_type=Path),
)
@click.option(
    "--musan_noise_dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--hf_auth_token",
    type=str,
    help="Optional hugging face hub auth token (to retrieve private pretrained pyannote models)",
)
def main(exp_params_file, output_dir, musan_noise_dir, hf_auth_token):
    """
    Run diarization training

    exp_params_file: Experiment params file

    output_dir: Experiment output dir

    musan_noise_dir: Folder containing MUSAN noise RIRs to use for audio augmentation  during segmentation model fine-tuning")
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    # copy experiment params to output dir
    shutil.copy(exp_params_file, output_dir / exp_params_file.name)

    # load all experiment params
    with open(exp_params_file) as fp:
        exp_params = yaml.safe_load(fp)

    print("Loading experiment params:")
    pprint(exp_params)
    print("\n")

    protocol_name = exp_params["protocol"]
    segmentation = exp_params["segmentation"]
    embedding = exp_params["embedding"]
    clustering = exp_params["clustering"]
    min_nb_speakers = exp_params["min_nb_speakers"]
    max_nb_speakers = exp_params["max_nb_speakers"]
    initial_params_diar = exp_params["initial_params_diar"]
    nb_epochs_seg_train = exp_params["segmentation_training"]["nb_epochs"]
    validate_period_seg_train = exp_params["segmentation_training"]["validate_period"]
    lr_seg_train = exp_params["segmentation_training"]["lr"]
    # nb_epochs_emb_train = exp_params["embedding_training"]["nb_epochs"]
    # validate_period_emb_train = exp_params["embedding_training"]["validate_period"]
    # lr_emb_train = exp_params["embedding_training"]["lr"]
    nb_iters_diar_optim = exp_params["diarization_optimization"]["nb_iters"]
    frozen_params_diar = exp_params["diarization_optimization"]["frozen_params"]
    sampler_diar_optim = exp_params["diarization_optimization"]["sampler"]
    pruner_diar_optim = exp_params["diarization_optimization"]["pruner"]

    # load dataset
    print(f"\nLoading {protocol_name}...")
    protocol = get_protocol(
        protocol_name,
        preprocessors={
            "audio": FileFinder(),
            "torchaudio.info": get_torchaudio_info,
        },
    )

    # fine-tune segmentation model
    print("\nFine-tuning segmentation model...")
    seg_output_dir = output_dir / "seg"
    seg_output_dir.mkdir(exist_ok=True)
    seg_checkpoint = train_segmentation_model(
        protocol=protocol,
        output_dir=seg_output_dir,
        model_name=segmentation,
        background_noise_dir=musan_noise_dir,
        nb_epochs=nb_epochs_seg_train,
        validate_period=validate_period_seg_train,
        lr=lr_seg_train,
        hf_auth_token=hf_auth_token,
    )

    # optimize diarization pipeline hyperparams
    print("\nOptimizing diarization pipeline hyperparams")
    diar_output_dir = output_dir / "diar"
    diar_output_dir.mkdir(exist_ok=True)
    params_file = optimize_diarization_pipeline(
        protocol=protocol,
        output_dir=diar_output_dir,
        segmentation=seg_checkpoint,
        embedding=embedding,
        clustering=clustering,
        min_nb_speakers=min_nb_speakers,
        max_nb_speakers=max_nb_speakers,
        initial_params=initial_params_diar,
        frozen_params=frozen_params_diar,
        sampler=sampler_diar_optim,
        pruner=pruner_diar_optim,
        nb_iters=nb_iters_diar_optim,
        hf_auth_token=hf_auth_token,
    )

    # evaluate final diarization pipeline
    print("\nEvaluating final diarization pipeline...")
    evaluate_diarization_pipeline(
        protocol=protocol,
        output_dir=diar_output_dir,
        segmentation=seg_checkpoint,
        embedding=embedding,
        clustering=clustering,
        params_file=params_file,
        min_nb_speakers=min_nb_speakers,
        max_nb_speakers=max_nb_speakers,
        hf_auth_token=hf_auth_token,
    )


if __name__ == "__main__":
    main()
