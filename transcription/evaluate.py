# main final evaluation script
# (can take a CTC or seq2seq model, with or without a langage model)
# (also support whisper for comparison purposes)

from pathlib import Path

import click
from pyctcdecode import build_ctcdecoder
from speechbrain.pretrained import EncoderASR, EncoderDecoderASR
from speechbrain.utils.metric_stats import ErrorRateStats
import torch
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from .common.infer import gen_utterance_batches, infer_sb, infer_whisper
from .common.eslo2_prepare import prepare_eslo2
from .common.simsamu_prepare import get_simsamu_utterance_ids, prepare_simsamu
from .common.preprocess_text import preprocess_text, prepare_words_for_wer

BATCH_SIZE = 1


@click.command()
@click.option("--model", required=True, type=str)
@click.option("--output_dir", required=True, type=click.Path(path_type=Path))
@click.option(
    "--dataset_dir", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--kenlm_file", required=False, type=click.Path(exists=False, path_type=Path)
)
def main(model, output_dir, dataset_dir, kenlm_file):
    output_dir.mkdir(parents=True, exist_ok=True)

    # init model
    if model == "whisper":
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v2"
        ).to("cuda")
        whisper_model.config.forced_decoder_ids = (
            whisper_processor.get_decoder_prompt_ids(language="fr", task="transcribe")
        )
    else:
        sb_asr_class = EncoderDecoderASR if "seq2seq" in model else EncoderASR
        sb_model = sb_asr_class.from_hparams(
            source=model,
            savedir=output_dir / f"model/{model}",
            run_opts={"device": "cuda"},
        )
        sb_model.eval()

        if sb_asr_class is EncoderASR:
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
        else:
            sb_decoder = None

    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    if "eslo" in str(dataset_dir).lower():
        _, _, csv_file = prepare_eslo2(
            srt_dir=dataset_dir / "transcription/srts",
            wav_dir=dataset_dir / "audio",
            save_dir=output_dir / "data",
        )
    else:
        utterance_ids_by_split = {"eval": get_simsamu_utterance_ids(dataset_dir)}
        (csv_file,) = prepare_simsamu(
            data_dir=dataset_dir,
            save_dir=output_dir / "data",
            utterance_ids_by_split=utterance_ids_by_split,
            include_wav_info=True,
            min_nb_chars=0,
            min_duration_secs=0,
        )

    transcription_dir = output_dir / "transcription"
    transcription_dir.mkdir(parents=True, exist_ok=True)

    total_wer = ErrorRateStats()
    total_cer = ErrorRateStats(split_tokens=True)
    files_wers = {}
    files_cers = {}

    for file, batches in tqdm(gen_utterance_batches(csv_file, BATCH_SIZE)):
        file_wer = ErrorRateStats()
        file_cer = ErrorRateStats(split_tokens=True)
        file_predicted_texts = []
        file_target_texts = []

        for batch in batches:
            if model == "whisper":
                predicted_texts = infer_whisper(whisper_processor, whisper_model, batch)
            else:
                predicted_texts = infer_sb(sb_model, batch, decoder=sb_decoder)

            file_predicted_texts += predicted_texts
            file_target_texts += batch.target

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
            file_wer.append(batch.id, all_predicted_words, all_target_words)
            file_cer.append(batch.id, all_predicted_words, all_target_words)

        files_wers[file.stem] = file_wer.summarize("error_rate")
        files_cers[file.stem] = file_cer.summarize("error_rate")

        (transcription_dir / f"{file.stem}._REF.txt").write_text(
            "\n\n".join(file_target_texts)
        )
        (transcription_dir / f"{file.stem}_PRED.txt").write_text(
            "\n\n".join(file_predicted_texts)
        )

    total_wer = total_wer.summarize("error_rate")
    total_cer = total_cer.summarize("error_rate")

    output = (
        "\n".join(
            [
                f"DATASET: {str(dataset_dir)}",
                f"MODEL: {model}",
                f"WER: {total_wer:.1f}%",
                f"CER: {total_cer:.1f}%",
            ]
            + [
                f"{file_name} WER: {files_wers[file_name]:.1f}%   CER: {files_cers[file_name]:.1f}%"
                for file_name in files_wers
            ]
        )
        + "\n"
    )
    print(output)
    output_file = output_dir / "wer.txt"
    output_file.write_text(output)


if __name__ == "__main__":
    main()
