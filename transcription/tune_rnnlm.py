# hyperparameter optimization of an RNNLM language model

import json
from pathlib import Path

import click
import optuna
import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.utils.metric_stats import ErrorRateStats
import torch

from .common.infer import gen_utterance_batches
from .common.eslo2_prepare import prepare_eslo2
from .common.preprocess_text import prepare_words_for_wer

BATCH_SIZE = 4
SAMPLE_RATE = 16000
NB_TRIALS = 30


def get_beam_searcher(
    hparams,
    beam_size,
    eos_threshold,
    max_attn_shift,
    ctc_weight,
    temperature,
    lm_weight,
    temperature_lm,
):
    beam_searcher = sb.decoders.S2SRNNBeamSearchLM(
        embedding=hparams.emb,
        decoder=hparams.dec,
        linear=hparams.seq_lin,
        ctc_linear=hparams.ctc_lin,
        bos_index=hparams.bos_index,
        eos_index=hparams.eos_index,
        blank_index=hparams.blank_index,
        min_decode_ratio=hparams.min_decode_ratio,
        max_decode_ratio=hparams.max_decode_ratio,
        beam_size=beam_size,
        eos_threshold=eos_threshold,
        using_max_attn_shift=hparams.using_max_attn_shift,
        max_attn_shift=max_attn_shift,
        ctc_weight=ctc_weight,
        temperature=temperature,
        temperature_lm=temperature_lm,
        language_model=hparams.lm.to("cuda"),
        lm_weight=lm_weight,
    )
    return beam_searcher


def run(model, csv_file):
    wer = ErrorRateStats()

    for _, batches in gen_utterance_batches(csv_file, BATCH_SIZE):
        for batch in batches:
            batch = batch.to("cuda")
            with torch.no_grad():
                predicted_texts, _ = model.transcribe_batch(
                    batch.wav.data, batch.wav.lengths
                )

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
@click.option("--output_dir", required=True, type=click.Path(path_type=Path))
@click.option(
    "--eslo2_dir", required=True, type=click.Path(exists=True, path_type=Path)
)
def main(model, output_dir, eslo2_dir):
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    _, csv_file, _ = prepare_eslo2(
        srt_dir=eslo2_dir / "transcription/srts",
        wav_dir=eslo2_dir / "audio",
        save_dir=output_dir / "data",
    )

    # init model
    model = EncoderDecoderASR.from_hparams(
        source=model,
        savedir=output_dir / f"model/{model}",
        run_opts={"device": "cuda"},
    )
    model.eval()

    def run_wrapper(trial):
        model.mods.decoder = get_beam_searcher(
            model.hparams,
            beam_size=trial.suggest_int("beam_size", 50, 130, step=10),
            eos_threshold=trial.suggest_float("eos_threshold", 1.0, 2.0, step=0.1),
            max_attn_shift=trial.suggest_int("max_attn_shift", 60, 160, step=10),
            ctc_weight=trial.suggest_float("ctc_weight", 0.0, 1.0, step=0.1),
            temperature=trial.suggest_float("temperature", 1.0, 1.5, step=0.1),
            lm_weight=trial.suggest_float("lm_weight", 0.0, 1.0, step=0.1),
            temperature_lm=trial.suggest_float("temperature_lm", 1.0, 1.5, step=0.1),
        )
        return run(model, csv_file)

    db_file = output_dir / "study_rnnlm.db"
    is_1st_run = not db_file.exists()
    study = optuna.create_study(
        study_name="study", storage="sqlite:///" + str(db_file), load_if_exists=True
    )

    # force 1st trial with default hyperparams
    if is_1st_run:
        study.enqueue_trial(
            {
                "beam_size": 80,
                "eos_threshold": 1.5,
                "max_attn_shift": 60,
                "ctc_weight": 0.0,
                "temperature": 1.0,
                "lm_weight": 0.5,
                "temperature_lm": 1.0,
            }
        )

    study.optimize(run_wrapper, n_trials=NB_TRIALS)
    print(study.best_params)
    with open(output_dir / "best_params.json", "w") as fp:
        json.dump(study.best_params, fp)


if __name__ == "__main__":
    main()
