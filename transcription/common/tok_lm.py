# generate CSVs for traning of RNNLM or tokenizer

import random

from .eslo2_prepare import prepare_eslo2
from .commonvoice_prepare import prepare_commonvoice
from .opensubtitles_prepare import prepare_opensubtitles
from .pxslu_prepare import prepare_pxslu


def prepare_csvs(
    cv_dir,
    eslo2_dir,
    pxslu_dir,
    os_dir,
    os_subset,
    save_dir,
    tokenizer=None,
):
    """
    Extract and preprocess utterances from the CommonVoice, ESLO2 and Open Subtitles datasets,
    and store them in train/dev/test csvs.

    Note that the results of each function preparing the csv for a dataset are
    cached, so make sure to delete all csvs in save_dir (including in
    subdirectories) if you changed the preprocessing code and you want to
    rebuild them.

    Parameters
    ----------
    cv_dir:
        Path to the common voice dir  containing a "fr" subdir
        (ex: /scratch/cv-corpus-10.0-2022-07-04/)
    eslo2_dir:
        Path to the preprocessed ESLO2 directory containing a "str" subdir with
        .srt files generated from the .trs files
    pxslu_dir:
        Path to the PXSLU directory containing the "seq.in" file and the
        "recordings" subdir
    os_dir:
        Path to the processed OpenSubtitles directory containing the .txt files
        generated from the Open Subtitles .srt files
    os_subset:
        Proportion of all the availables Open Subtitles txt files that should be used.
        (Open Subtitles is very big, we sometimes don't want to use all of it)
    save_dir:
        Directory into which the train, dev and test csvs will be saved
    tokenizer:
        Optional tokenizer, used only for Language Model training, to precompute
        the number of tokens for each utterance
    """

    cv_train_csv_file, cv_dev_csv_file, cv_test_csv_file = prepare_commonvoice(
        data_dir=cv_dir / "fr",
        save_dir=save_dir / "cv",
        include_wav_info=False,
        tokenizer=tokenizer,
    )
    eslo2_train_csv_file, eslo2_dev_csv_file, eslo2_test_csv_file = prepare_eslo2(
        srt_dir=eslo2_dir / "transcription/srts",
        save_dir=save_dir / "eslo2",
        tokenizer=tokenizer,
    )
    os_train_csv_file, os_dev_csv_file, os_test_csv_file = prepare_opensubtitles(
        txt_dir=os_dir,
        save_dir=save_dir / "os",
        subset=os_subset,
        tokenizer=tokenizer,
    )
    pxslu_train_csv_file = prepare_pxslu(
        data_dir=pxslu_dir,
        save_dir=save_dir / "pxslu",
        include_wav_info=False,
        tokenizer=tokenizer,
    )

    train_csv_file = save_dir / "train.csv"

    # merge train.csv files from commonvoice, opensubtitles, pxslu and eslo2
    # (we only test and eval on eslo2 so also use test and eval from commonvoice/opensubtitles for train)
    train_lines = []
    for file in [
        os_train_csv_file,
        os_dev_csv_file,
        os_test_csv_file,
        cv_train_csv_file,
        cv_dev_csv_file,
        cv_test_csv_file,
        pxslu_train_csv_file,
        eslo2_train_csv_file,
    ]:
        with open(file) as fp:
            train_lines += fp.readlines()[1:]  # skip header
    # randomize so when we select a subset of all train lines (for debug) we
    # still have a mix of everything
    rng = random.Random(0)
    rng.shuffle(train_lines)

    with open(train_csv_file, mode="w") as fp:
        fields = ["ID", "text", "source"]
        # when tokenizer is provided there is an extra "duration" (in nb tokens) field
        if tokenizer is not None:
            fields.append("duration")
        fp.write(",".join(fields) + "\n")
        fp.writelines(train_lines)

    return train_csv_file, eslo2_dev_csv_file, eslo2_test_csv_file
