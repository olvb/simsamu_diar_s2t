# main script for finetuning CTC model on entire dataset
# (no validation)

import random
import torch

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

from pathlib import Path

import click
import hyperpyyaml as hpy
import speechbrain as sb

from .common.asr_train import init_dataset
from .common.batching import init_batch_sampler
from .common.simsamu_prepare import fold_simsamu, prepare_simsamu, get_simsamu_utterance_ids
from .train_ctc import add_dynamic_items_to_datasets, Trainer, prepare_best_checkpoint_dir


def _finetune(hparams_file, pretrained_dir, simsamu_dir, output_dir):
    with open(hparams_file) as fp:
        hparams = hpy.load_hyperpyyaml(
            fp,
            overrides={
                "output_dir": str(output_dir),
                "pretrained_dir": str(pretrained_dir),
            },
        )

    # create output dir
    sb.create_experiment_directory(
        experiment_directory=output_dir,
        hyperparams_to_save=hparams_file,
    )

    # generate datasets CSVs
    # (test csv will be empty)
    utterance_ids_by_split = {
        "train": get_simsamu_utterance_ids(simsamu_dir),
        "test": []
    }
    train_csv_file, _ = prepare_simsamu(
        data_dir=simsamu_dir,
        save_dir=output_dir / "data",
        utterance_ids_by_split=utterance_ids_by_split,
        include_wav_info=True,
    )

    # load wav2vec and ASR modules
    pretrainer = hparams["pretrainer"]
    pretrainer.set_collect_in(output_dir / "pretrained")
    pretrainer.collect_files()
    pretrainer.load_collected(device="cuda")

    tokenizer = hparams["tokenizer"]
    wav2vec2 = hparams["wav2vec2"]
    enc = hparams["enc"]
    ctc_lin = hparams["ctc_lin"]

    # init dynamic dataset and sampler
    train_dataset = init_dataset(hparams, train_csv_file)
    add_dynamic_items_to_datasets(tokenizer, hparams, [train_dataset])
    train_batch_sampler = init_batch_sampler(hparams, train_dataset)

    # init trainer
    asr_modules = {
        "enc": enc,
        "ctc_lin": ctc_lin,
    }
    trainer = Trainer(
        tokenizer=tokenizer,
        wav2vec2=wav2vec2,
        asr_modules=asr_modules,
        hparams=hparams,
        save_dir=output_dir,
    )

    # train
    nb_data_workers = hparams["nb_data_workers"]
    trainer.fit(
        trainer.epoch_counter,
        train_dataset,
        train_loader_kwargs={
            "batch_sampler": train_batch_sampler,
            "num_workers": nb_data_workers,
        },
    )

    # save last checkpoint
    source_checkpoint_dir = Path(
        trainer.checkpointer.list_checkpoints()[-1].path
    )
    tokenizer_file = pretrained_dir / "tokenizer.ckpt"
    prepare_best_checkpoint_dir(
        output_dir / "best_checkpoint", source_checkpoint_dir, tokenizer_file
    )

@click.command
@click.option("--hparams_file", required=True, type=click.Path(path_type=Path))
@click.option("--pretrained_dir", required=True, type=click.Path(path_type=Path))
@click.option(
    "--simsamu_dir", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--output_dir", required=True, type=click.Path(path_type=Path))
def main(hparams_file, pretrained_dir, simsamu_dir, output_dir):
    _finetune(
        hparams_file=hparams_file,
        pretrained_dir=pretrained_dir,
        simsamu_dir=simsamu_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
