# hyperparameter optimization of a KenLM ARPA language model

import json
from pathlib import Path

import click
import optuna
from pyctcdecode import build_ctcdecoder
from speechbrain.pretrained import EncoderASR
from speechbrain.utils.metric_stats import ErrorRateStats
import torch

from .common.infer import gen_utterance_batches
from .common.eslo2_prepare import prepare_eslo2
from .common.preprocess_text import prepare_words_for_wer

BATCH_SIZE = 4
SAMPLE_RATE = 16000
NB_TRIALS = 30


def _unpad(data, data_lens):
    """Helper function to get unpaded tensors for a PaddedBatch"""
    # FIXME why not use sb.utils.data_utils.undo_padding?

    abs_lens = (data_lens * data.shape[1]).long()
    data_unpadded = [data[i][: abs_lens[i]] for i in range(len(abs_lens))]
    return data_unpadded


def run(model, decode_func, csv_file):
    wer = ErrorRateStats()

    for _, batches in gen_utterance_batches(csv_file, BATCH_SIZE):
        for batch in batches:
            batch = batch.to("cuda")
            with torch.no_grad():
                encoder_out = model.encode_batch(batch.wav.data, batch.wav.lengths)
                encoder_out = encoder_out.cpu().numpy()
                encoder_out = _unpad(encoder_out, batch.wav.lengths)
                predicted_texts = [decode_func(h) for h in encoder_out]

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
            wer.append(batch.id, all_predicted_words, all_target_words)

    return wer.summarize("error_rate")


@click.command()
@click.option("--model", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--kenlm_file", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--eslo2_dir", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--output_dir", required=True, type=click.Path(path_type=Path))
def main(model, kenlm_file, eslo2_dir, output_dir):
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    _, csv_file, _ = prepare_eslo2(
        srt_dir=eslo2_dir / "transcription/srts",
        wav_dir=eslo2_dir / "audio",
        save_dir=output_dir / "data",
    )

    # init model
    model = EncoderASR.from_hparams(
        source=model,
        savedir=output_dir / f"model/{model}",
        run_opts={"device": "cuda"},
    )
    model.eval()

    tokenizer = model.tokenizer
    labels = [tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())]
    labels[0] = ""

    def run_wrapper(trial):
        alpha = trial.suggest_float("alpha", 0.1, 1.0, step=0.1)
        beta = trial.suggest_float("beta", 0.1, 1.0, step=0.1)
        beam_width = trial.suggest_int("beam_width", 50, 130, step=10)
        beam_prune_logp = trial.suggest_float("beam_prune_logp", -40.0, -0.1, step=0.1)
        token_min_logp = trial.suggest_float("token_min_logp", -20.0, -0.1, step=0.1)
        decoder = build_ctcdecoder(
            labels,
            kenlm_model_path=str(kenlm_file),
            alpha=alpha,
            beta=beta,
        )

        def decode_func(data):
            return decoder.decode(
                data,
                beam_width=beam_width,
                beam_prune_logp=beam_prune_logp,
                token_min_logp=token_min_logp,
            )

        return run(model, decode_func, csv_file)

    db_file = output_dir / "study_kenlm.db"
    is_1st_run = not db_file.exists()
    study = optuna.create_study(
        study_name="study", storage="sqlite:///" + str(db_file), load_if_exists=True
    )

    # force 1st trial with default hyperparams
    if is_1st_run:
        study.enqueue_trial(
            {
                "alpha": 0.5,
                "beta": 1.5,
                "beam_width": 100,
                "beam_prune_logp": -10.0,
                "token_min_logp": -5.0,
            }
        )

    study.optimize(run_wrapper, n_trials=NB_TRIALS)
    print(study.best_params)
    with open(output_dir / "best_params.json", "w") as fp:
        json.dump(study.best_params, fp)


if __name__ == "__main__":
    main()
