# main script for folded finetuning and cross-validation of a CTC model

import random
import torch

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

from pathlib import Path

import click
import hyperpyyaml as hpy
import speechbrain as sb
from pyctcdecode import build_ctcdecoder
from speechbrain.pretrained import EncoderASR
from speechbrain.utils.metric_stats import ErrorRateStats
from tqdm import tqdm

from .common.asr_train import init_dataset
from .common.batching import init_batch_sampler
from .common.simsamu_prepare import fold_simsamu, fold_simsamu_speaker_aware, prepare_simsamu
from .common.preprocess_text import preprocess_text, prepare_words_for_wer
from .common.infer import gen_utterance_batches, infer_sb
from .train_ctc import add_dynamic_items_to_datasets, Trainer, prepare_best_checkpoint_dir


def _finetune(hparams_file, pretrained_dir, simsamu_dir, output_dir, utterance_ids_by_split):
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
    train_csv_file, test_csv_file = prepare_simsamu(
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

    # init dynamic datasets and samplers
    train_dataset = init_dataset(hparams, train_csv_file)
    test_dataset = init_dataset(hparams, test_csv_file)
    add_dynamic_items_to_datasets(tokenizer, hparams, [train_dataset, test_dataset])
    train_batch_sampler = init_batch_sampler(hparams, train_dataset)
    # force deterministic ordering for test
    test_batch_sampler = init_batch_sampler(
        hparams, test_dataset, batch_ordering="ascending"
    )

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

    # test
    nb_data_workers = hparams["nb_data_workers"]
    trainer.evaluate(
        test_dataset,
        min_key="wer",
        test_loader_kwargs={
            "batch_sampler": test_batch_sampler,
            "num_workers": nb_data_workers,
        },
    )

    # train
    trainer.fit(
        trainer.epoch_counter,
        train_dataset,
        test_dataset,
        train_loader_kwargs={
            "batch_sampler": train_batch_sampler,
            "num_workers": nb_data_workers,
        },
        valid_loader_kwargs={
            "batch_sampler": test_batch_sampler,
            "num_workers": nb_data_workers,
        },
    )

    # save best checkpoint
    source_best_checkpoint_dir = Path(
        trainer.checkpointer.find_checkpoint(min_key="wer").path
    )
    tokenizer_file = pretrained_dir / "tokenizer.ckpt"
    prepare_best_checkpoint_dir(
        output_dir / "best_checkpoint", source_best_checkpoint_dir, tokenizer_file
    )

def _eval(model, output_dir, eval_csv_file, kenlm_file):
    sb_model = EncoderASR.from_hparams(
        source=model,
        savedir=output_dir / f"model/{model}",
        run_opts={"device": "cuda"},
    )
    sb_model.eval()

    tokenizer = sb_model.tokenizer
    labels = [
        tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())
    ]
    labels[0] = ""
    if kenlm_file is not None:
        sb_decoder = build_ctcdecoder(
            labels,
            kenlm_model_path=str(kenlm_file),
            alpha=0.5,
        )
    else:
        sb_decoder = build_ctcdecoder(labels)

    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    transcription_dir = output_dir / "transcription"
    transcription_dir.mkdir(parents=True, exist_ok=True)

    total_wer = ErrorRateStats()
    total_cer = ErrorRateStats(split_tokens=True)
    all_predicted_texts = []
    all_target_texts = []

    for _, batches in tqdm(gen_utterance_batches(eval_csv_file, batch_size=1)):
        for batch in batches:
            predicted_texts = infer_sb(sb_model, batch, decoder=sb_decoder)

            all_predicted_texts += predicted_texts
            all_target_texts += batch.target

            # update WERS
            all_predicted_words = [prepare_words_for_wer(t) for t in predicted_texts]
            all_target_words = [prepare_words_for_wer(t) for t in batch.target]
            all_predicted_words = [
                w1
                for w1, w2 in zip(all_predicted_words, all_target_words)
                if len(w1) and len(w2)
            ]
            all_target_words = [
                w2
                for w1, w2 in zip(all_predicted_words, all_target_words)
                if len(w1) and len(w2)
            ]
            total_wer.append(batch.id, all_predicted_words, all_target_words)
            total_cer.append(batch.id, all_predicted_words, all_target_words)

    (transcription_dir / "REF.txt").write_text(
        "\n\n".join(all_target_texts)
    )
    (transcription_dir / "PRED.txt").write_text(
        "\n\n".join(all_predicted_texts)
    )

    total_wer = total_wer.summarize("error_rate")
    total_cer = total_cer.summarize("error_rate")

    output = (
        f"WER: {total_wer:.1f}%\n"
        f"CER: {total_cer:.1f}%\n"
    )
    print(output)
    output_file = output_dir / "wer.txt"
    output_file.write_text(output)

def _compute_global_wer(output_dir):
    """Compute total WER/CER accross all folds"""

    wer = ErrorRateStats()
    cer = ErrorRateStats(split_tokens=True)

    for fold_dir in sorted(output_dir.glob("fold_*/")):
        predicted_file = fold_dir / "eval/transcription/PRED.txt"
        predicted_texts = [line for line in predicted_file.read_text().split("\n")][::2]
        all_predicted_words = [prepare_words_for_wer(t) for t in predicted_texts]

        target_file = fold_dir / "eval/transcription/REF.txt"
        target_texts = [line for line in target_file.read_text().split("\n")][::2]
        all_target_words = [prepare_words_for_wer(t) for t in target_texts]

        all_predicted_words = [
            w1
            for w1, w2 in zip(all_predicted_words, all_target_words)
            if len(w1) and len(w2)
        ]
        all_target_words = [
            w2
            for w1, w2 in zip(all_predicted_words, all_target_words)
            if len(w1) and len(w2)
        ]
        ids = [f"{fold_dir.stem}_{i}" for i in range(len(all_predicted_words))]
        wer.append(ids, all_predicted_words, all_target_words)
        cer.append(ids, all_predicted_words, all_target_words)

    wer = wer.summarize("error_rate")
    cer = cer.summarize("error_rate")

    output = (
        f"WER: {wer:.1f}%\n"
        f"CER: {cer:.1f}%\n"
    )
    print(output)
    output_file = output_dir / "global_wer.txt"
    output_file.write_text(output)

@click.command
@click.option("--hparams_file", required=True, type=click.Path(path_type=Path))
@click.option("--pretrained_dir", required=True, type=click.Path(path_type=Path))
@click.option(
    "--simsamu_dir", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--output_dir", required=True, type=click.Path(path_type=Path))
@click.option("--speaker_aware", is_flag=True, default=False)
@click.option("--nb_folds", type=int)
@click.option(
    "--kenlm_file", required=False, type=click.Path(exists=False, path_type=Path)
)
def main(hparams_file, pretrained_dir, simsamu_dir, output_dir, speaker_aware, nb_folds, kenlm_file):
    assert speaker_aware == (nb_folds is None)

    if speaker_aware:
        folds = fold_simsamu_speaker_aware(simsamu_dir)
    else:
        folds = fold_simsamu(simsamu_dir, nb_folds)

    for i, utterance_ids_by_split in enumerate(folds):
        print(f"Fold {i + 1} / {len(folds)}")
        fold_dir = output_dir / f"fold_{i + 1}"

        _finetune(
            hparams_file=hparams_file,
            pretrained_dir=pretrained_dir,
            simsamu_dir=simsamu_dir,
            output_dir=fold_dir,
            utterance_ids_by_split=utterance_ids_by_split,
        )
        _eval(
            model=fold_dir / "best_checkpoint",
            output_dir=fold_dir / "eval",
            eval_csv_file=fold_dir / "data/test.csv",
            kenlm_file=kenlm_file,
        )
    
    _compute_global_wer(output_dir)


if __name__ == "__main__":
    main()
