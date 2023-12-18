# main script for training of a SentencePiece tokenizer
# based on https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/Tokenizer/train.py


import random
import torch
import sys

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

from pathlib import Path

import hyperpyyaml as hpy
import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece

from .common.tok_lm import prepare_csvs


def main():
    # read hparams file passed to CLI
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    hparams_file = Path(hparams_file)
    with open(hparams_file) as fp:
        hparams = hpy.load_hyperpyyaml(fp, overrides)

    # check that output_dir has same basename as hparams file
    # (to avoid overiding previously save results by accident)
    output_dir = Path(hparams["output_dir"])
    assert (
        output_dir.stem == hparams_file.stem[8:]
    ), f"Invalid output dir: {str(output_dir)}"

    # generate datasets csv
    train_csv_file, valid_csv_file, _ = prepare_csvs(
        cv_dir=Path(hparams["cv_dir"]),
        eslo2_dir=Path(hparams["eslo2_dir"]),
        pxslu_dir=Path(hparams["pxslu_dir"]),
        os_dir=Path(hparams["os_dir"]),
        os_subset=hparams["os_subset"],
        save_dir=output_dir / "data",
    )

    # train tokenizer
    _ = SentencePiece(
        model_dir=hparams["output_dir"],
        vocab_size=hparams["nb_tokens"],
        annotation_train=str(train_csv_file),
        annotation_read="text",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        annotation_list_to_check=[str(train_csv_file), str(valid_csv_file)],
        bos_id=hparams.get("bos_index", -1),
        eos_id=hparams.get("eos_index", -1),
    )


if __name__ == "__main__":
    main()
