# main script for training of a seq2seq model
# based on https://github.com/speechbrain/speechbrain/blob/develop/recipes/CommonVoice/ASR/seq2seq/train_with_wav2vec.py

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
    - tokens_bos: tokenization of the utterance text with a BOS (begining of speech)
      token prepended at start
    - tokens_eos: tokenization of the utterance text with a EOS (end of speech)
      token appended at end
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

    bos_index = hparams["bos_index"]
    eos_index = hparams["eos_index"]

    # text pipeline (=> tokens tokens_bos, tokens_eos)
    @sb.utils.data_pipeline.takes("text")
    @sb.utils.data_pipeline.provides("tokens_bos", "tokens_eos", "tokens")
    def text_pipeline(text):
        tokens_list = tokenizer.encode(text)
        tokens_bos = torch.LongTensor([bos_index] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [eos_index])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "waveform",
            "start",
            "stop",
            "tokens_bos",
            "tokens_eos",
            "tokens",
            "text",
        ],
    )


def prepare_best_checkpoint_dir(
    checkpoint_dir, source_best_checkpoint_dir, tokenizer_file, rnnlm_file=None
):
    """Create a checkpoint directory with proper inference hparams and symlinks to tokenizer and model weights"""

    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    checkpoint_dir.mkdir()

    if rnnlm_file is not None:
        template_inference_hparams_file = (
            Path(__file__).parent / "seq2seq_with_rnnlm_inference_hparams.yaml"
        )
    else:
        template_inference_hparams_file = (
            Path(__file__).parent / "seq2seq_inference_hparams.yaml"
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
    if rnnlm_file is not None:
        (checkpoint_dir / "lm.ckpt").symlink_to(rnnlm_file)

    files = [
        "wav2vec2.ckpt",
        "enc.ckpt",
        "dec.ckpt",
        "ctc_lin.ckpt",
        "seq_lin.ckpt",
        "emb.ckpt",
    ]
    for file in files:
        (checkpoint_dir / file).symlink_to(source_best_checkpoint_dir / file)


class Trainer(TrainerBase):
    def __init__(
        self,
        tokenizer,
        wav2vec2,
        asr_modules,
        beam_searcher,
        hparams,
        save_dir,
        run_opts=None,
        example_batch=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            wav2vec2=wav2vec2,
            asr_modules=asr_modules,
            hparams=hparams,
            run_opts=run_opts,
            save_dir=save_dir,
            example_batch=example_batch,
        )

        self.beam_searcher = beam_searcher
        self.nb_ctc_epochs = hparams["nb_ctc_epochs"]
        self.ctc_loss_weight = hparams["ctc_loss_weight"]

    def compute_forward(self, batch, stage):
        """Forward pass, return CTC logits/seq2seq logits/beamsearched tokens depending on stage"""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.waveform

        if stage == sb.Stage.TRAIN:
            wavs = self.spec_augmentation(wavs, wav_lens)

        features = self.modules.wav2vec2(wavs)
        x = self.modules.enc(features)
        tokens_bos, _ = batch.tokens_bos
        e_in = self.modules.emb(tokens_bos)
        h, _ = self.modules.dec(e_in, x, wav_lens)
        # output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.log_softmax(logits)

        if stage == sb.Stage.TRAIN:
            epoch = self.epoch_counter.current
            if epoch <= self.nb_ctc_epochs:
                # output layer for ctc log-probabilities
                logits_ctc = self.modules.ctc_lin(x)
                p_ctc = self.log_softmax(logits_ctc)
                return p_ctc, p_seq, wav_lens
            else:
                return p_seq, wav_lens
        else:
            p_tokens, scores = self.beam_searcher(x, wav_lens)
            return p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Compute CTC loss/seq2seq loss and WER/CER metrics if not in train stage"""

        epoch = self.epoch_counter.current
        if stage == sb.Stage.TRAIN:
            if epoch <= self.nb_ctc_epochs:
                p_ctc, p_seq, wav_lens = predictions
            else:
                p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens
        loss_seq = sb.nnet.losses.nll_loss(
            p_seq,
            tokens_eos,
            length=tokens_eos_lens,
            label_smoothing=0.1,
        )

        # Add ctc loss if necessary
        if stage == sb.Stage.TRAIN and epoch <= self.nb_ctc_epochs:
            loss_ctc = sb.nnet.losses.ctc_loss(
                p_ctc,
                tokens,
                wav_lens,
                tokens_lens,
                blank_index=self.blank_index,
            )
            loss = (
                self.ctc_loss_weight * loss_ctc + (1 - self.ctc_loss_weight) * loss_seq
            )
        else:
            loss = loss_seq

        if stage != sb.Stage.TRAIN:
            ids = batch.id
            self.compute_wer_cer(ids, tokens, predicted_tokens, tokens_lens, stage)

        return loss

    def infer(self, batch):
        """
        Decode and return beamsearched tokens
        """

        _, _, predicted_tokens = self.compute_forward(batch, sb.Stage.VALID)
        predicted_texts = [
            self.tokenizer.decode_ids(utt_seq) for utt_seq in predicted_tokens
        ]
        return predicted_texts


# def apply_drop_out(enc, drop_out):
#     assert len(enc) == 4
#     enc.update({
#         "linear": enc["linear"],
#         "act": enc["act"],
#         "drop_out": torch.nn.Dropout(p=drop_out),
#         "linear_0": enc["linear_0"],
#         "act_0": enc["act_0"],
#         "drop_out_0": torch.nn.Dropout(p=drop_out),
#     })


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

    # load (possibly pretrained) modules
    pretrainer = hparams["pretrainer"]
    pretrainer.set_collect_in(output_dir / "pretrained")
    pretrainer.collect_files()
    pretrainer.load_collected(device="cuda")

    tokenizer = hparams["tokenizer"]
    wav2vec2 = hparams["wav2vec2"]
    enc = hparams["enc"]
    emb = hparams["emb"]
    ctc_lin = hparams["ctc_lin"]
    seq_lin = hparams["seq_lin"]
    dec = hparams["dec"]
    beam_searcher = hparams["beam_searcher"]

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
        "emb": emb,
        "ctc_lin": ctc_lin,
        "seq_lin": seq_lin,
        "dec": dec,
    }
    example_batch = get_example_batch(valid_dataset)
    trainer = Trainer(
        tokenizer=tokenizer,
        wav2vec2=wav2vec2,
        asr_modules=asr_modules,
        beam_searcher=beam_searcher,
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

    rnnlm_file = Path(hparams["pretrained_rnnlm_file"])
    prepare_best_checkpoint_dir(
        output_dir / "best_checkpoint_lm",
        source_best_checkpoint_dir,
        tokenizer_file,
        rnnlm_file,
    )


if __name__ == "__main__":
    main()
