# fine-tuning of the segmentation model (PyaNet) of a diarization pipeline

import os
from pathlib import Path
from types import MethodType
from typing import Optional, Union

from pyannote.audio import Model
from pyannote.audio.tasks import Segmentation
from pyannote.database import Protocol
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch_audiomentations import AddBackgroundNoise
import yaml


MAX_NB_SPEAKERS = 4


def train_segmentation_model(
    protocol: Protocol,
    output_dir: Path,
    model_name: Union[str, Path],
    background_noise_dir: Path,
    nb_epochs: int,
    validate_period: int,
    lr: float,
    hf_auth_token: Optional[str] = None,
) -> Path:
    """
    Fine-tune a segmentation model (to use in a speaker diarization pipeline),
    using pytorch lighting

    Parameters
    ----------
    protocol:
        Dataset to use for training Dataset to use for evaluation (train split will be used)
    output_dir:
        Where to save training results and model weights
    model_name:
        Pretrained model (path or name on hugginface hub) to fine-tune
    background_noise_dir:
        Path to the folder containing noise RIRs for audio augmentation
    nb_epochs:
        Number of training epochs
    valid_period:
        Number of training epoch between each validation epoch
    lr:
        Learning rate
    hf_auth_token:
        Optional hugging face hub token, needed to retrieve private models

    Returns
    -------
    Path:
        Path to the best fine-tuned checkpoint
    """

    output_file = output_dir / "output.yml"

    # skip if already trained
    if output_file.exists():
        with open(output_file) as fp:
            output = yaml.safe_load(fp)
        if output["nb_epochs"] >= nb_epochs:
            print(
                f"Already trained segmentation model for {nb_epochs} epochs, skipping"
            )
            best_checkpoint = Path(output["best_checkpoint"])
            assert best_checkpoint.exists()
            return best_checkpoint

    # init training task with augmentation
    augmentation = AddBackgroundNoise(
        background_noise_dir,
        min_snr_in_db=5.0,
        max_snr_in_db=15.0,
        p=0.9,
    )
    augmentation.output_type = "dict"
    task = Segmentation(
        protocol,
        duration=5.0,
        max_speakers_per_chunk=MAX_NB_SPEAKERS,
        max_speakers_per_frame=MAX_NB_SPEAKERS,
        augmentation=augmentation,
    )

    # init PyanNet model with pretrained weights and assign task
    model = Model.from_pretrained(
        model_name,
        use_auth_token=hf_auth_token,
        map_location=torch.device("cuda:0"),
    ).to("cuda:0")
    model.task = task

    # configure learning rate
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=lr)

    model.configure_optimizers = MethodType(configure_optimizers, model)

    # init checkpoint callback with validation metric to monitor
    monitor, direction = task.val_monitor
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        monitor=monitor,
        mode=direction,
        save_last=True,
        dirpath=checkpoints_dir,
    )

    # init trainer
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=nb_epochs,
        check_val_every_n_epoch=validate_period,
        callbacks=[checkpoint_cb],
        default_root_dir=output_dir,
    )

    last_checkpoint = checkpoints_dir / "last.ckpt"
    if last_checkpoint.exists():
        # restore checkpoint if available
        ckpt_file = last_checkpoint
    else:
        # validate initial model before 1st epoch to make sure we improve
        ckpt_file = None
        model.setup(stage="fit")
        trainer.validate(model)

    # run training
    model.setup(stage="fit")
    trainer.fit(model, ckpt_path=ckpt_file)

    best_checkpoint = output_dir / "best.ckpt"
    # symlink to final best checkpoint (easier to identify and use after)
    best_checkpoint.unlink(missing_ok=True)
    os.symlink(checkpoint_cb.best_model_path, best_checkpoint)

    # save results in output file
    output = {
        "nb_epochs": trainer.current_epoch,
        "best_score": checkpoint_cb.best_model_score.item(),
        "best_checkpoint": checkpoint_cb.best_model_path,
    }
    with open(output_file, mode="w") as fp:
        yaml.safe_dump(output, fp, sort_keys=False)

    return Path(checkpoint_cb.best_model_path)
