# main script for training of an RNN language model
# based on https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/LM/train.py


import random
import torch

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

from pathlib import Path
import sys

import hyperpyyaml as hpy
import speechbrain as sb

from .common.tok_lm import prepare_csvs


def init_dataset(hparams, csv_file):
    """ "
    Init a speechbrain dynamic dataset from a CSV file,
    sorting it by number of tokens, and filtering out utterances
    longer than the max_nb_tokens hparam
    """

    dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(csv_path=csv_file)

    dataset = dataset.filtered_sorted(
        key_max_value={"duration": hparams["max_nb_tokens"]}
    )

    # for debug, limit nb samples
    max_nb_samples = hparams.get("max_nb_samples")
    if max_nb_samples is not None:
        dataset = dataset.filtered_sorted(select_n=max_nb_samples)

    return dataset


def add_dynamic_items_to_datasets(tokenizer, bos_index, eos_index, datasets):
    """
    Add two entries to a dynamic dataset created with init_dataset():
    - tokens_bos: tokenization of the utterance text with a BOS (begining of speech)
      token prepended at start
    - tokens_eos: tokenization of the utterance text with a EOS (end of speech)
      token appended at end
    """

    @sb.utils.data_pipeline.takes("text")
    @sb.utils.data_pipeline.provides("tokens_bos", "tokens_eos")
    def text_pipeline(text):
        tokens_list = tokenizer.encode(text)
        tokens_bos = torch.LongTensor([bos_index] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [eos_index])
        yield tokens_eos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "tokens_bos", "tokens_eos"],
    )


@sb.utils.checkpoints.register_checkpoint_hooks
class TensorboardLogger(sb.utils.train_logger.TensorboardLogger):
    """
    Add missing global_step to tensorboard logging so that we don't get 2
    separate curves when resuming training
    """

    @sb.utils.checkpoints.mark_as_saver
    def save(self, path):
        torch.save(self.global_step, path)

    @sb.utils.checkpoints.mark_as_loader
    def load(self, path, end_of_epoch, device):
        self.global_step = torch.load(path)


class Trainer(sb.core.Brain):
    def __init__(
        self,
        lm,
        hparams,
        save_dir,
        run_opts=None,
    ):
        for param in lm.parameters():
            param.requires_grad = True

        # init optimizer, scheduler, epoch counter, tensorboard logger
        # before super().__init__() to we can add them to a checkpointer
        # passed to super().__init__()

        optimizer = torch.optim.Adam(
            lr=hparams["lr"],
            params=lm.parameters(),
            betas=(0.9, 0.98),
            eps=0.000000001,
        )

        scheduler = sb.nnet.schedulers.NewBobScheduler(
            initial_value=hparams["lr"],
            improvement_threshold=0.0025,
            annealing_factor=0.8,
            patient=0,
        )

        epoch_counter = sb.utils.epoch_loop.EpochCounter(limit=hparams["nb_epochs"])
        tensorboard_logger = TensorboardLogger(save_dir / "tensorboard")

        checkpointer = sb.utils.checkpoints.Checkpointer(
            checkpoints_dir=save_dir / "checkpoints",
            recoverables={
                "lm": lm,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "epoch_counter": epoch_counter,
                "tensorboard_logger": tensorboard_logger,
            },
        )

        super().__init__(
            modules={"lm": lm},
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
        )

        self.optimizer = optimizer
        self.epoch_counter = epoch_counter
        self.scheduler = scheduler

        self.log_softmax = sb.nnet.activations.Softmax(apply_log=True)

        self.file_logger = sb.utils.train_logger.FileTrainLogger(
            save_dir / "train_log.txt"
        )
        self.tensorboard_logger = tensorboard_logger
        self.stage_stats = {}

    def compute_forward(self, batch, stage):
        """Forward pass"""

        batch = batch.to(self.device)
        tokens_bos, _ = batch.tokens_bos
        logits = self.modules.lm(tokens_bos)
        pred = self.log_softmax(logits)
        return pred

    def compute_objectives(self, predictions, batch, stage):
        """Compute loss"""

        batch = batch.to(self.device)
        tokens_eos, tokens_len = batch.tokens_eos
        loss = sb.nnet.losses.nll_loss(predictions, tokens_eos, length=tokens_len)
        return loss

    def on_stage_start(self, stage, epoch):
        """Reinit metrics on epoch begin"""

        self.stage_stats[stage] = {}

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gather metrics, update LR scheduler, log and checkpoint on epoch end"""

        stage_stats = self.stage_stats[stage]
        stage_stats["loss"] = stage_loss

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.scheduler(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            for logger in [self.file_logger, self.tensorboard_logger]:
                logger.log_stats(
                    stats_meta={
                        "epoch": epoch,
                        "lr": old_lr,
                    },
                    train_stats=self.stage_stats[sb.Stage.TRAIN],
                    valid_stats=self.stage_stats[sb.Stage.VALID],
                )

            self.checkpointer.save_and_keep_only(min_keys=["loss"])
        elif stage == sb.Stage.TEST:
            for logger in [self.file_logger, self.tensorboard_logger]:
                logger.log_stats(
                    stats_meta={},
                    test_stats=self.stage_stats[sb.Stage.TEST],
                )


def main():
    # read hparams file passed to CLI
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    hparams_file = Path(hparams_file)
    with open(hparams_file) as fp:
        hparams = hpy.load_hyperpyyaml(fp, overrides)

    # check that output_dir has same basename as hparams file
    # (to avoid overiding previously saved results by accident)
    output_dir = Path(hparams["output_dir"])
    assert (
        output_dir.stem == hparams_file.stem[8:]
    ), f"Invalid output dir: {str(output_dir)}"

    # create output dir
    sb.create_experiment_directory(
        experiment_directory=output_dir,
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # load (possibly pretrained) tokenizer and LM modules
    # relying on speechbrain's Pretrainer
    pretrainer = hparams["pretrainer"]
    pretrainer.set_collect_in(output_dir / "pretrained")
    pretrainer.collect_files()
    pretrainer.load_collected(device="cuda")

    tokenizer = hparams["tokenizer"]
    lm = hparams["lm"]

    # generate datasets CSVs
    train_csv_file, valid_csv_file, test_csv_file = prepare_csvs(
        cv_dir=Path(hparams["cv_dir"]),
        eslo2_dir=Path(hparams["eslo2_dir"]),
        pxslu_dir=Path(hparams["pxslu_dir"]),
        os_dir=Path(hparams["os_dir"]),
        os_subset=hparams["os_subset"],
        save_dir=output_dir / "data",
        tokenizer=tokenizer,
    )

    # for debugging purposes
    if hparams["train_on_valid"]:
        train_csv_file = valid_csv_file

    # init dynamic datasets
    train_dataset = init_dataset(hparams, train_csv_file)
    valid_dataset = init_dataset(hparams, valid_csv_file)
    test_dataset = init_dataset(hparams, test_csv_file)
    add_dynamic_items_to_datasets(
        tokenizer,
        hparams["bos_index"],
        hparams["eos_index"],
        [train_dataset, valid_dataset, test_dataset],
    )

    # init trainer
    trainer = Trainer(lm=lm, hparams=hparams, save_dir=output_dir, run_opts=run_opts)

    # train
    loader_kwargs = {
        "batch_size": hparams["batch_size"],
        "num_workers": hparams["nb_data_workers"],
        "shuffle": False,
    }
    trainer.fit(
        trainer.epoch_counter,
        train_dataset,
        valid_dataset,
        train_loader_kwargs=loader_kwargs,
        valid_loader_kwargs=loader_kwargs,
    )

    # test
    trainer.evaluate(
        test_dataset,
        min_key="wer",
        test_loader_kwargs=loader_kwargs,
    )

    (output_dir / "best_checkpoint").mkdir(exist_ok=True)
    source_best_checkpoint_dir = Path(trainer.checkpointer.find_checkpoint().path)
    (output_dir / "best_checkpoint/lm.ckpt").symlink_to(
        source_best_checkpoint_dir / "lm.ckpt"
    )


if __name__ == "__main__":
    main()
