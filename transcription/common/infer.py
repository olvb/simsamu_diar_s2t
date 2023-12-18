# common inference code (used for final evaluation/comparison)

from pathlib import Path

import pandas as pd
from speechbrain.dataio.batch import PaddedBatch
import torch
import torchaudio

torchaudio.set_audio_backend("soundfile")

from .preprocess_text import preprocess_text

SAMPLE_RATE = 16000


def _batchify_iter(iter, batch_size: int):
    """Generate iterator of batches of items from iterator of items"""

    while True:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(iter))
            except StopIteration:
                if len(batch) > 0:
                    yield batch
                return
        yield batch


class _CachedWavReader:
    def __init__(self):
        self._wav_file = None
        self._full_signal = None

    def gen_utterance_wav(self, wav_file, start, stop):
        if wav_file != self._wav_file:
            self._wav_file = wav_file
            self._full_signal, sr = torchaudio.load(self._wav_file)
            assert self._full_signal.shape[0] == 1
            assert sr == SAMPLE_RATE

        signal = self._full_signal[:, start:stop].reshape((-1,))
        return signal


def _unpad(data, data_lens):
    """Helper function to get unpaded tensors for a PaddedBatch"""
    # FIXME why not use sb.utils.data_utils.undo_padding?

    abs_lens = (data_lens * data.shape[1]).long()
    data_unpadded = [data[i][: abs_lens[i]] for i in range(len(abs_lens))]
    return data_unpadded


def gen_utterance_batches(csv_file, batch_size):
    """Read a srt file and corresponding wave file and generate speechbrain padded batches"""

    wav_reader = _CachedWavReader()

    rows = pd.read_csv(csv_file, keep_default_na=False)
    for wav_file, file_rows in rows.groupby("wav_file"):

        def gen_file_batches():
            for batch_rows in _batchify_iter(file_rows.itertuples(), batch_size):
                batch_data = [
                    {
                        "wav": wav_reader.gen_utterance_wav(
                            row.wav_file, row.start, row.stop
                        ),
                        "target": row.text,
                        "id": row.ID,
                    }
                    for row in batch_rows
                ]
                yield PaddedBatch(batch_data)

        yield Path(wav_file), gen_file_batches()


def infer_sb(model, batch, decoder=None):
    batch = batch.to("cuda")
    with torch.no_grad():
        if decoder is not None:
            encoder_out = model.encode_batch(batch.wav.data, batch.wav.lengths)
            encoder_out = encoder_out.cpu().numpy()
            encoder_out = _unpad(encoder_out, batch.wav.lengths)
            predicted_texts = [
                decoder.decode(
                    h
                )  # , beam_width=beam_width, beam_prune_logp=beam_prune_logp, token_min_logp=token_min_log)
                for h in encoder_out
            ]
        else:
            predicted_texts, _ = model.transcribe_batch(
                batch.wav.data, batch.wav.lengths
            )
        return predicted_texts


def infer_whisper(processor, model, batch):
    wavs = _unpad(batch.wav.data, batch.wav.lengths)
    predicted_texts = []
    for wav in wavs:
        input_features = processor(
            wav, return_tensors="pt", sampling_rate=SAMPLE_RATE
        ).input_features.to("cuda")
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        predicted_text = transcription[0]

        # whisper is lowercase and with digits so we need to reprocess it before computing WER
        predicted_text_preprocessed = preprocess_text(predicted_text)
        if predicted_text_preprocessed is not None:
            predicted_text = predicted_text_preprocessed
        else:
            predicted_text = predicted_text.upper()

        predicted_texts.append(predicted_text)
    return predicted_texts
