# main script for training of a CTC model
# based on https://github.com/speechbrain/speechbrain/blob/develop/recipes/CommonVoice/ASR/CTC/train_with_wav2vec.py


import random
import torch

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

from pathlib import Path
import shutil
import sys

import hyperpyyaml as hpy
import speechbrain as sb
import torchaudio

torchaudio.set_audio_backend("soundfile")

from .common.asr_train import (
    prepare_csvs,
    TrainerBase,
    init_dataset,
    get_example_batch,
)
from .common.batching import init_batch_sampler


def add_dynamic_items_to_datasets(tokenizer, hparams, datasets):
    """
    Add the following entries to a dynamic dataset created with init_dataset():
    - waveform: the signal of the utterance, properly trimmed and resampled
    - tokens: tokenization of the utterance text
    """

    target_sample_rate = hparams["sample_rate"]
    resamplers = {}

    # audio pipeline (=> waveform)
    @sb.utils.data_pipeline.takes("wav_file", "start", "stop")
    @sb.utils.data_pipeline.provides("waveform")
    def audio_pipeline(wav_file, start, stop):
        start = int(start)
        stop = int(stop)
        if start != -1 and stop != -1:
            nb_frames = stop - start
            waveform, sample_rate = torchaudio.load(
                wav_file, num_frames=nb_frames, frame_offset=start
            )
        else:
            waveform, sample_rate = torchaudio.load(wav_file)
        waveform = waveform.squeeze(0)

        if sample_rate != target_sample_rate:
            if sample_rate not in resamplers:
                resamplers[sample_rate] = torchaudio.transforms.Resample(
                    sample_rate, target_sample_rate
                )
            waveform = resamplers[sample_rate](waveform)

        return waveform

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # text pipeline (=> tokens)
    @sb.utils.data_pipeline.takes("text")
    @sb.utils.data_pipeline.provides("tokens")
    def text_pipeline(text):
        yield torch.LongTensor(tokenizer.encode(text))

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "waveform", "start", "stop", "tokens", "text"],
    )


def prepare_best_checkpoint_dir(
    checkpoint_dir, source_best_checkpoint_dir, tokenizer_file=None
):
    """Create a checkpoint directory with proper inference hparams and symlinks to tokenizer and model weights"""

    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    checkpoint_dir.mkdir()

    template_inference_hparams_file = (
        Path(__file__).parent / "ctc_inference_hparams.yaml"
    )
    inference_hparams = template_inference_hparams_file.read_text()
    inference_hparams = inference_hparams.replace(
        "ckpt_dir: null", "ckpt_dir: " + str(checkpoint_dir)
    )
    (checkpoint_dir / "hyperparams.yaml").write_text(inference_hparams)

    # symlink to tokenizer
    (checkpoint_dir / "tokenizer.ckpt").symlink_to(tokenizer_file)
    # checkpoint of non-finetuning wav2vec2 weights (not really needed but speechbrain will redownload it even if we don't use it)
    (checkpoint_dir / "wav2vec2_checkpoint").symlink_to(
        checkpoint_dir.parent / "wav2vec2_checkpoint", target_is_directory=True
    )

    files = ["wav2vec2.ckpt", "enc.ckpt", "ctc_lin.ckpt"]
    for file in files:
        (checkpoint_dir / file).symlink_to(source_best_checkpoint_dir / file)


class Trainer(TrainerBase):
    def __init__(
        self,
        tokenizer,
        wav2vec2,
        asr_modules,
        hparams,
        save_dir,
        example_batch=None,
        run_opts=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            wav2vec2=wav2vec2,
            asr_modules=asr_modules,
            hparams=hparams,
            save_dir=save_dir,
            example_batch=example_batch,
            run_opts=run_opts,
        )

        self.decoding_func = hparams["decoding_function"]

    def compute_forward(self, batch, stage):
        """Forward pass, return CTC logits"""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.waveform

        if stage == sb.Stage.TRAIN:
            wavs = self.spec_augmentation(wavs, wav_lens)

        features = self.modules.wav2vec2(wavs)
        logits = self.modules.ctc_lin(self.modules.enc(features))
        p_ctc = self.log_softmax(logits)
        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Compute CTC loss, and WER/CER metrics if not in train stage"""

        p_ctc, wav_lens = predictions
        tokens, tokens_lens = batch.tokens
        loss = sb.nnet.losses.ctc_loss(
            p_ctc, tokens, wav_lens, tokens_lens, blank_index=self.blank_index
        )

        if stage != sb.Stage.TRAIN:
            predicted_tokens = self.decoding_func(
                p_ctc, wav_lens, blank_id=self.blank_index
            )
            ids = batch.id
            self.compute_wer_cer(ids, tokens, predicted_tokens, tokens_lens, stage)

        return loss

    def infer(self, batch):
        """
        Use CTC logits to predict tokens, use decoding_func
        (which will be a greedy ctc func since that is all speechbrain offers)
        """

        p_ctc, wav_lens = self.compute_forward(batch, sb.Stage.VALID)
        predicted_tokens = self.decoding_func(
            p_ctc, wav_lens, blank_id=self.blank_index
        )
        predicted_texts = [
            self.tokenizer.decode_ids(utt_seq) for utt_seq in predicted_tokens
        ]
        return predicted_texts


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

    # generate datasets CSVs
    train_csv_file, valid_csv_file, test_csv_file = prepare_csvs(
        cv_dir=Path(hparams["cv_dir"]),
        eslo2_dir=Path(hparams["eslo2_dir"]),
        pxslu_dir=Path(hparams["pxslu_dir"]),
        save_dir=output_dir / "data",
    )

    # for debugging purposes
    if hparams["train_on_valid"]:
        train_csv_file = valid_csv_file

    # load (possibly pretrained) wav2vec and ASR modules
    pretrainer = hparams["pretrainer"]
    pretrainer.set_collect_in(output_dir / "pretrained")
    pretrainer.collect_files()
    pretrainer.load_collected(device="cuda")

    tokenizer = hparams["tokenizer"]
    wav2vec2 = hparams["wav2vec2"]
    enc = hparams["enc"]
    ctc_lin = hparams["ctc_lin"]

    # init dynamic datasets and samplers
    train_dataset = init_dataset(hparams, train_csv_file)
    valid_dataset = init_dataset(hparams, valid_csv_file)
    test_dataset = init_dataset(hparams, test_csv_file)
    add_dynamic_items_to_datasets(
        tokenizer,
        hparams,
        [train_dataset, valid_dataset, test_dataset],
    )
    train_batch_sampler = init_batch_sampler(
        hparams,
        train_dataset,
        subset_by_source={
            "eslo2": hparams["eslo2_subset"],
            "cv": hparams["cv_subset"],
            "pxslu": hparams["pxslu_subset"],
        },
    )
    # force deterministic ordering for eval and test
    valid_batch_sampler = init_batch_sampler(
        hparams, valid_dataset, batch_ordering="ascending"
    )
    test_batch_sampler = init_batch_sampler(
        hparams, test_dataset, batch_ordering="ascending"
    )

    # init trainer
    asr_modules = {
        "enc": enc,
        "ctc_lin": ctc_lin,
    }
    example_batch = get_example_batch(valid_dataset)
    trainer = Trainer(
        tokenizer=tokenizer,
        wav2vec2=wav2vec2,
        asr_modules=asr_modules,
        hparams=hparams,
        save_dir=output_dir,
        example_batch=example_batch,
        run_opts=run_opts,
    )

    # train
    nb_data_workers = hparams["nb_data_workers"]
    trainer.fit(
        trainer.epoch_counter,
        train_dataset,
        valid_dataset,
        train_loader_kwargs={
            "batch_sampler": train_batch_sampler,
            "num_workers": nb_data_workers,
        },
        valid_loader_kwargs={
            "batch_sampler": valid_batch_sampler,
            "num_workers": nb_data_workers,
        },
    )

    # test
    trainer.evaluate(
        test_dataset,
        min_key="wer",
        test_loader_kwargs={
            "batch_sampler": test_batch_sampler,
            "num_workers": nb_data_workers,
        },
    )

    source_best_checkpoint_dir = Path(
        trainer.checkpointer.find_checkpoint(min_key="wer").path
    )
    tokenizer_file = Path(hparams["pretrained_tokenizer_file"])
    prepare_best_checkpoint_dir(
        output_dir / "best_checkpoint", source_best_checkpoint_dir, tokenizer_file
    )


if __name__ == "__main__":
    main()
