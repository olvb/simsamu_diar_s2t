# common classes for speech to text training
# (used by both CTC and seq2seq training)

import itertools
import logging
import random

import speechbrain as sb
from speechbrain.lobes.augment import TimeDomainSpecAugment
import torch
import torch.utils.tensorboard
from tensorboard.compat.proto.summary_pb2 import (
    Summary as TBSummary,
    SummaryMetadata as TBSummaryMetadata,
)

from .preprocess_text import prepare_words_for_wer
from .eslo2_prepare import prepare_eslo2
from .commonvoice_prepare import prepare_commonvoice
from .pxslu_prepare import prepare_pxslu

logging.getLogger("speechbrain.dataio.sampler").setLevel(logging.WARNING)

MIN_NB_CHARS = 5


def prepare_csvs(
    cv_dir,
    eslo2_dir,
    pxslu_dir,
    save_dir,
    # eslo2_subset,
    # cv_subset,
):
    """
    Extract and preprocess utterances from the CommonVoice fr, PXSLU and ESLO2
    datasets, and store them in train/dev/test csvs.

    Note that the results of each function preparing the csv for a dataset are
    cached, so make sure to delete all csvs in save_dir (including in
    subdirectories) if you changed the preprocessing code and you want to
    rebuild them.

    Parameters
    ----------
    cv_dir:
        Path to the common voice dir  containing a "fr" subdir
        (ex: /scratch/cv-corpus-10.0-2022-07-04/)
    eslo2_dir:
        Path to the preprocessed ESLO2 directory containing a "str" subdir with
        .srt files generated from the .trs files, as well as an "audio" with
        phone-processed wav files
    pxslu_dir:
        Path to the PXSLU directory containing the "seq.in" file and the
        "recordings" subdir
    save_dir:
        Directory into which the train, dev and test csvs will be saved
    eslo2_subset:
        Proportion of the available eslo2 utterances that should be used.
    cv_subset:
        Proportion of the available Commmon Voice utterances that should be used.
    """

    # # handle special cases where we just want full eslo2 or full cv
    # if eslo2_subset == 1.0 and cv_subset == 0.0:
    #     return prepare_eslo2(
    #         srt_dir=eslo2_dir / "transcription/srts",
    #         wav_dir=eslo2_dir / "audio",
    #         save_dir=save_dir / "eslo2",
    #         min_nb_chars=MIN_NB_CHARS,
    #     )
    # if eslo2_subset == 0.0 and cv_subset == 1.0:
    #     return prepare_commonvoice(
    #         data_dir=cv_dir / "fr",
    #         save_dir=save_dir / "cv",
    #         include_wav_info=True,
    #         min_nb_chars=MIN_NB_CHARS,
    #     )

    cv_train_csv_file, cv_dev_csv_file, cv_test_csv_file = prepare_commonvoice(
        data_dir=cv_dir / "fr",
        save_dir=save_dir / "cv",
        include_wav_info=True,
        min_nb_chars=MIN_NB_CHARS,
    )
    eslo2_train_csv_file, eslo2_dev_csv_file, eslo2_test_csv_file = prepare_eslo2(
        srt_dir=eslo2_dir / "transcription/srts",
        wav_dir=eslo2_dir / "audio",
        save_dir=save_dir / "eslo2",
        min_nb_chars=MIN_NB_CHARS,
    )
    pxslu_train_csv_file = prepare_pxslu(
        data_dir=pxslu_dir,
        save_dir=save_dir / "pxslu",
        include_wav_info=True,
        min_nb_chars=MIN_NB_CHARS,
    )

    # merge train from eslo2, commmonvoice and pxslu
    # we only eval on eslo2 so also use test and eval from commonvoice
    train_lines = []
    for file in [
        eslo2_train_csv_file,
        cv_train_csv_file,
        cv_dev_csv_file,
        cv_test_csv_file,
        pxslu_train_csv_file,
    ]:
        with open(file) as fp:
            train_lines += fp.readlines()[1:]  # skip header
    # randomize so when we select a subset of all train lines (for debug) we
    # still have a mix of everything
    rng = random.Random(0)
    rng.shuffle(train_lines)

    train_csv_file = save_dir / "train.csv"
    with open(train_csv_file, mode="w") as fp:
        fp.write("ID,text,source,duration,start,stop,wav_file\n")
        fp.writelines(train_lines)

    return train_csv_file, eslo2_dev_csv_file, eslo2_test_csv_file


@sb.utils.checkpoints.register_checkpoint_hooks
class _MultiOptimizer:
    """
    Pack several optimizers in one class that can be used with sb.core.Brain

    We need this because we use different learning rates for wav2vec and for the rest of the model.
    """

    def __init__(self, optimizers):
        self.optimizers = optimizers

    def step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()

    def zero_grad(self, set_to_none=False):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=False)

    @sb.utils.checkpoints.mark_as_saver
    def save(self, path):
        torch.save(self.optimizers, path)

    @sb.utils.checkpoints.mark_as_loader
    def load(self, path, end_of_epoch, device):
        self.optimizers = torch.load(path)


@sb.utils.checkpoints.register_checkpoint_hooks
class TensorboardLogger(sb.utils.train_logger.TensorboardLogger):
    """
    Add text and audio logging capabitilies to tensorboard logger (so we can
    display reference and infered transcription along with corresponding audio
    for a set of example utterances at each epoch)
    """

    def log_text(self, name, value):
        self.writer.add_text(name, value, self.global_step["meta"])

    def log_audio_with_description(self, name, value, sample_rate, description):
        audio_summary = self._build_tb_audio_summary(
            name, value, sample_rate, description
        )
        self.writer._get_file_writer().add_summary(
            audio_summary, self.global_step["meta"]
        )

    @staticmethod
    def _build_tb_audio_summary(tag, tensor, sample_rate, description):
        import numpy as np

        array = tensor.detach().cpu().numpy()
        array = array.squeeze()
        if abs(array).max() > 1:
            print("warning: audio amplitude out of range, auto clipped.")
            array = array.clip(-1, 1)
        assert array.ndim == 1, "input tensor should be 1 dimensional."
        array = (array * np.iinfo(np.int16).max).astype("<i2")

        import io
        import wave

        fio = io.BytesIO()
        with wave.open(fio, "wb") as wave_write:
            wave_write.setnchannels(1)
            wave_write.setsampwidth(2)
            wave_write.setframerate(sample_rate)
            wave_write.writeframes(array.data)
        audio_string = fio.getvalue()
        fio.close()

        audio = TBSummary.Audio(
            sample_rate=sample_rate,
            num_channels=1,
            length_frames=array.shape[-1],
            encoded_audio_string=audio_string,
            content_type="audio/wav",
        )
        metadata = TBSummaryMetadata(summary_description=description)
        audio_value = TBSummary.Value(
            tag=tag,
            audio=audio,
            metadata=metadata,
        )
        return TBSummary(value=[audio_value])

    # next 2 methods: add missing global_step to tensorboard logging so that we
    # don't get 2 separate curves when resuming training

    @sb.utils.checkpoints.mark_as_saver
    def save(self, path):
        torch.save(self.global_step, path)

    @sb.utils.checkpoints.mark_as_loader
    def load(self, path, end_of_epoch, device):
        self.global_step = torch.load(path)


def get_example_batch(dataset, nb_examples=15):
    """
    Return batch of example utterances from a dataset
    (used to display reference and infered transcription along with corresponding audio
    for a set of example utterances at each epoch)

    This will always return the same examples, chosen pseudo-randomly with a fixed seed.
    """

    rng = random.Random(1234)
    samples = [dataset[i] for i in rng.sample(list(range(len(dataset))), k=nb_examples)]
    return sb.dataio.batch.PaddedBatch(samples)


def init_dataset(hparams, csv_file):
    """
    Init a speechbrain dynamic dataset from a CSV file
    """

    dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(csv_path=csv_file)

    # for debug, limit nb samples
    max_nb_samples = hparams.get("max_nb_samples")
    if max_nb_samples is not None:
        dataset = dataset.filtered_sorted(select_n=max_nb_samples)
    return dataset


def _unpad(data, data_lens):
    """Helper function to get unpaded tensors for a PaddedBatch"""
    # FIXME why not use sb.utils.data_utils.undo_padding?

    abs_lens = (data_lens * data.shape[1]).long()
    data_unpadded = [data[i][: abs_lens[i]] for i in range(len(abs_lens))]
    return data_unpadded


class TrainerBase(sb.core.Brain):
    """
    Subclass of the speechbrain "Brain" (ie Trainer)
    class to share common code between training of CTC and seq2seq models
    """

    def __init__(
        self,
        tokenizer,
        wav2vec2,
        asr_modules,
        hparams,
        save_dir,
        run_opts=None,
        example_batch=None,
    ):
        # retrieve modules to fine tune
        wav2vec2.freeze = False
        modules = {"wav2vec2": wav2vec2, **asr_modules}
        for module in modules.values():
            for param in module.parameters():
                param.requires_grad = True

        # init optimizers, schedulers, epoch counter, tensorboard logger
        # before super().__init__() to we can add them to a checkpointer
        # passed to super().__init__()

        # 2 optimizers and schedulers for wav2vec and for the actual ASR model
        # because we use different learning rates
        wav2vec2_optimizer = hparams["wav2vec2_optimizer"](params=wav2vec2.parameters())
        asr_params = itertools.chain(*[m.parameters() for m in asr_modules.values()])
        asr_optimizer = hparams["asr_optimizer"](params=asr_params)
        optimizer = _MultiOptimizer(
            {"wav2vec2": wav2vec2_optimizer, "asr": asr_optimizer}
        )

        asr_scheduler = sb.nnet.schedulers.NewBobScheduler(
            initial_value=hparams["lr_asr"],
            improvement_threshold=0.0025,
            annealing_factor=0.9,
            patient=0,
        )

        wav2vec2_scheduler = sb.nnet.schedulers.NewBobScheduler(
            initial_value=hparams["lr_wav2vec2"],
            improvement_threshold=0.0025,
            annealing_factor=0.9,
            patient=0,
        )

        epoch_counter = sb.utils.epoch_loop.EpochCounter(limit=hparams["nb_epochs"])
        tensorboard_logger = TensorboardLogger(save_dir / "tensorboard")

        # init checkpointer
        checkpointer = sb.utils.checkpoints.Checkpointer(
            checkpoints_dir=save_dir / "checkpoints",
            recoverables={
                **modules,
                "optimizer": optimizer,
                "wav2vec2_scheduler": wav2vec2_scheduler,
                "asr_scheduler": asr_scheduler,
                "epoch_counter": epoch_counter,
                "tensorboard_logger": tensorboard_logger,
            },
        )

        super().__init__(
            modules=modules,
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
        )

        self.optimizer = optimizer
        self.epoch_counter = epoch_counter
        self.asr_scheduler = asr_scheduler
        self.wav2vec2_scheduler = wav2vec2_scheduler

        self.tokenizer = tokenizer
        self.sample_rate = hparams["sample_rate"]
        self.blank_index = hparams["blank_index"]

        # from the spechbrain doc, TimeDomainSpecAugment will do:
        # - Drop chunks of the audio (zero amplitude or white noise)
        # - Drop frequency bands (with band-drop filters)
        # - Speed peturbation (via resampling to slightly different rate)
        self.spec_augmentation = TimeDomainSpecAugment(
            sample_rate=self.sample_rate,
            speeds=hparams["spec_augment_speeds"],
        )
        self.log_softmax = sb.nnet.activations.Softmax(apply_log=True)

        self.file_logger = sb.utils.train_logger.FileTrainLogger(
            save_dir / "train_log.txt"
        )
        self.tensorboard_logger = tensorboard_logger
        self.stage_stats = {}
        self.example_batch = example_batch

    def on_stage_start(self, stage, epoch):
        """Reinit metrics on epoch begin"""

        stage_stats = {}
        if stage != sb.Stage.TRAIN:
            stage_stats["wer"] = sb.utils.metric_stats.ErrorRateStats()
            stage_stats["cer"] = sb.utils.metric_stats.ErrorRateStats(split_tokens=True)
        self.stage_stats[stage] = stage_stats

        if stage == sb.Stage.VALID and epoch == 1:
            # log waveforms of example utterances once at begining
            self.log_example_waveforms()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gather metrics, update LR schedulers, log and checkpoint on epoch end"""

        stage_stats = self.stage_stats[stage]
        stage_stats["loss"] = stage_loss

        if stage != sb.Stage.TRAIN:
            # compute WER/CER only in valid and test
            stage_stats["wer"] = stage_stats["wer"].summarize("error_rate")
            stage_stats["cer"] = stage_stats["cer"].summarize("error_rate")

        if stage == sb.Stage.VALID:
            # update LR wtith schedulers
            old_lr_wav2vec2, new_lr_wav2vec2 = self.wav2vec2_scheduler(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.optimizer.optimizers["wav2vec2"], new_lr_wav2vec2
            )
            old_lr_asr, new_lr_asr = self.asr_scheduler(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(
                self.optimizer.optimizers["asr"], new_lr_asr
            )

            # log metrics and lrs
            for logger in [self.file_logger, self.tensorboard_logger]:
                logger.log_stats(
                    stats_meta={
                        "epoch": epoch,
                        "lr_asr": old_lr_asr,
                        "lr_wav2vec2": old_lr_wav2vec2,
                    },
                    train_stats=self.stage_stats[sb.Stage.TRAIN],
                    valid_stats=self.stage_stats[sb.Stage.VALID],
                )
            # save checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"wer": self.stage_stats[sb.Stage.VALID]["wer"]},
                min_keys=["wer"],
            )
            # log results for examples in tensorboard
            self.log_examples()
        elif stage == sb.Stage.TEST:
            for logger in [self.file_logger, self.tensorboard_logger]:
                logger.log_stats(
                    stats_meta={},
                    test_stats=self.stage_stats[sb.Stage.TEST],
                )

    def compute_wer_cer(self, ids, tokens, predicted_tokens, tokens_lens, stage):
        """
        Decode predicted and reference tokens and compute WER and WER metrics.
        To be called from compute_objectives()
        """

        target_tokens = sb.utils.data_utils.undo_padding(tokens, tokens_lens)
        # perform additional text processing before computing WER/CER
        target_words = [
            prepare_words_for_wer(self.tokenizer.decode_ids(utt_seq))
            for utt_seq in target_tokens
        ]
        predicted_words = [
            prepare_words_for_wer(self.tokenizer.decode_ids(utt_seq))
            for utt_seq in predicted_tokens
        ]

        stage_stats = self.stage_stats[stage]
        stage_stats["wer"].append(ids, predicted_words, target_words)
        stage_stats["cer"].append(ids, predicted_words, target_words)

    def log_example_waveforms(self):
        """
        Log waveforms of example utterances in tensorboard
        (to be called once at begining of training)
        """

        if self.example_batch is None:
            return

        wavs, wav_lens = self.example_batch.waveform
        waveforms = _unpad(wavs, wav_lens)

        for i, waveform in enumerate(waveforms):
            id = self.example_batch.id[i]
            start_secs = int(self.example_batch.start[i]) / self.sample_rate
            stop_secs = int(self.example_batch.stop[i]) / self.sample_rate
            description = f"{id} ({start_secs:.1f} to {stop_secs:.1f})"
            self.tensorboard_logger.log_audio_with_description(
                f"examples/{i}/audio", waveform, self.sample_rate, description
            )

    def log_examples(self):
        """
        Log infered and reference transcriptions of example utterances
        (to be called after each valid epoch)
        """

        if self.example_batch is None:
            return

        self.modules.eval()
        with torch.no_grad():
            predicted_texts = self.infer(self.example_batch)

        texts = self.example_batch.text

        predicted_words = [t.split(" ") for t in predicted_texts]
        target_words = [t.split(" ") for t in texts]
        wer_details = sb.utils.edit_distance.wer_details_for_batch(
            self.example_batch.id, target_words, predicted_words
        )
        wers = [d["WER"] for d in wer_details]

        for i, (text, predicted_text, wer) in enumerate(
            zip(texts, predicted_texts, wers)
        ):
            description = f"pred: {predicted_text}  \nref: {text}  \nwer: {wer:.2f}%"
            self.tensorboard_logger.log_text(f"examples/{i}", description)

    def infer(self, batch):
        """Model-specific inference pass, to be implented for both CTC and seq2seq"""
        raise NotImplementedError()
