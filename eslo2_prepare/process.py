# phone-like processing of ESLO2 audio files

from pathlib import Path

from pqdm.processes import pqdm
import torch
import torchaudio

torchaudio.set_audio_backend("soundfile")

CURRENT_DIR = Path(__file__).parent.resolve()


def _process_audio(audio, sr):
    """
    Apply some phone-like processing to an audio buffer
    Taken from https://pytorch.org/audio/main/tutorials/audio_data_augmentation_tutorial.html#simulating-a-phone-recoding,
    not very realistic.
    gsm codec seems broken and is not applied.
    """

    # mono mixdown
    audio = torch.sum(audio, axis=0, keepdims=True) / 2

    audio, _ = torchaudio.sox_effects.apply_effects_tensor(
        audio,
        sr,
        effects=[
            ["lowpass", "4000"],
            [
                "compand",
                "0.02,0.05",
                "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8",
                "-8",
                "-7",
                "0.05",
            ],
            ["rate", "8000"],
        ],
    )

    audio = torchaudio.functional.apply_codec(audio, 8000, format="gsm")
    audio, _ = torchaudio.sox_effects.apply_effects_tensor(
        audio,
        8000,
        effects=[
            ["rate", "16000"],
        ],
    )
    return audio, 16000


def process_one(rec_dir, dest_dir):
    rec_id = rec_dir.name
    rec_nb = int(rec_id.split("-")[1])
    rec_id = f"record-{rec_nb:03}"

    processed_file = dest_dir / f"{rec_id}.wav"
    if processed_file.exists():
        return

    file = next(rec_dir.glob("*.mp3"))
    audio, sr = torchaudio.load(file)
    audio, sr = _process_audio(audio, sr)
    assert sr == 16000
    torchaudio.save(processed_file, audio, sr)


DEST_DIR = None


# wrapper func to allow calling process_one() with pqdm
def pqdm_wrapper(rec_dir):
    process_one(rec_dir, DEST_DIR)


def process(source_dir: Path, dest_dir: Path):
    """
    Mixdown to mono, apply phone-like processing and reduce sample rate for all audio files,
    and save results in record-* wav files
    (with 3 digits padding to avoids some weird bugs with some speechbrain scripts)
    """

    dest_dir.mkdir(parents=True, exist_ok=True)

    rec_dirs = sorted(source_dir.glob("record-*"))

    global DEST_DIR
    DEST_DIR = dest_dir

    pqdm(rec_dirs, pqdm_wrapper, n_jobs=6)
