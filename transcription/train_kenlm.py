# main script for generation of a kenlm ARPA language model
# requires lmplz kenlm binary, cf https://github.com/kpu/kenlm
# can be install with conda https://anaconda.org/conda-forge/kenlm

from pathlib import Path
import subprocess

import click

from .common.tok_lm import prepare_csvs


@click.command()
@click.option("--cv_dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--eslo2_dir", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--pxslu_dir", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--os_dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--prune", required=True, type=int)
@click.option("--output_dir", required=True, type=click.Path(path_type=Path))
def main(cv_dir, eslo2_dir, pxslu_dir, os_dir, prune, output_dir):
    train_csv_file, _, _ = prepare_csvs(
        cv_dir=cv_dir,
        eslo2_dir=eslo2_dir,
        pxslu_dir=pxslu_dir,
        os_dir=os_dir,
        os_subset=1.0,
        save_dir=output_dir / "data",
    )

    train_txt_file = output_dir / "data/train.txt"
    with open(train_csv_file) as read_fp:
        with open(train_txt_file, mode="w") as write_fp:
            for line in read_fp:
                line = line.split(",")[1] + "\n"
                write_fp.write(line)

    # reconstruct pruning pattern (ex: 0 1 if prune == 2, 0 0 0 1 if prune == 4)
    # cf lmplz doc
    prune_parts = ["0"] * (prune - 1) + ["1"]

    # requires lmplz kenlm binary, cf https://github.com/kpu/kenlm
    # can be install with conda https://anaconda.org/conda-forge/kenlm
    subprocess.run(
        ["lmplz", "-o", "4", "--prune"]
        + prune_parts
        + ["--text", str(train_txt_file), "--arpa", str(output_dir / "model.arpa")]
    )


if __name__ == "__main__":
    main()
