#!/usr/bin/env python3

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import click
from pqdm.processes import pqdm

MEDICAL_ONLY = False

MIN_YEAR = 1935

# patterns found in entries that are not transcription
BLACKLISTED = [
    "soustitr",
    "sous titr",
    " by ",
    "*",
    "--",
    "subs.fr",
    "sub-way",
    "seriessub",
    "titles",
    "free.fr",
    "hotmail.",
    "{",
    "}",
    " \\ ",
    " - - 0",
    "bitrate",
    "file size",
]
# additional patterns found in entries that are not transcription (stricter)
# check only in first and last entries of each file, which are more likely to
# not be transcription
HEAD_TAIL_LENGTH = 20
BLACKLISTED_HEAD_TAIL = [
    "titr",
    "traduct",
    "adapt" "rip",
    "sub",
    "@",
    " at ",
]
REPLACEMENT_REGEXPS = [
    # fix punctuation spacing
    (re.compile(r"' "), "'"),
    (re.compile(r" \.\.\."), "..."),
    (re.compile(r" \. ?$"), "."),
    (re.compile(r" \. (?=\w)"), ". "),
    (re.compile(r" , ?$"), ","),
    (re.compile(r" , (?=\w)"), ", "),
    (re.compile(r"^ ?- ?"), ""),
    # common mistake
    (re.compile(r"\bLL\b"), "IL"),
    (re.compile(r"\bLl\b"), "Il"),
]

LETTER_REGEXP = re.compile(r"[a-zA-Z]")
ENGLISH_REGEXP = re.compile(r" and ")
ENCODING_ISSUE_REGEX = re.compile(r"Ã © ")


def replace_in_sentence(sentence):
    for regexp, replacement in REPLACEMENT_REGEXPS:
        sentence = regexp.sub(replacement, sentence)
    return sentence


def convert(source_xml_file, dest_txt_file):
    """
    Convert a subtitle xml file to a simpler txt file to use for Language Model learning

    Also filter garbage/non-transcription entries often found in subtitles files
    """

    if dest_txt_file.exists():
        return

    if MEDICAL_ONLY:
        contents = source_xml_file.read_text().lower()
        if not "médecin" in contents and not "docteur" in contents:
            return

    tree = ET.parse(source_xml_file)
    sentences = [
        " ".join(w.text for w in s[:-1] if w.tag == "w")
        for s in tree.getroot()
        if s.tag == "s"
    ]

    # skip whole file if looks like english or bad encoding
    if len([s for s in sentences if ENGLISH_REGEXP.search(s)]) > 2:
        return
    if any(ENCODING_ISSUE_REGEX.search(s) for s in sentences):
        return

    # here we try to skip garbage sentences (subtitles teams mentions, ASCII art, etc)
    # skip first and last sentences
    sentences = sentences[1:-1]
    # skip sentences with no letter
    sentences = [s for s in sentences if LETTER_REGEXP.search(s)]
    # skip sentences containing blacklisted text
    # (stricter on head/tail)
    sentences = [s for s in sentences if not any(t in s.lower() for t in BLACKLISTED)]
    sentences[:HEAD_TAIL_LENGTH] = [
        s
        for s in sentences[:HEAD_TAIL_LENGTH]
        if not any(t in s.lower() for t in BLACKLISTED_HEAD_TAIL)
    ]
    sentences[-HEAD_TAIL_LENGTH:] = [
        s
        for s in sentences[-HEAD_TAIL_LENGTH:]
        if not any(t in s.lower() for t in BLACKLISTED_HEAD_TAIL)
    ]
    # skip whole file if too few sentences left
    if len(sentences) < 2:
        return

    sentences = [replace_in_sentence(s) for s in sentences]

    dest_txt_file.write_text("\n".join(sentences))


DEST_DIR = None


# wrapper func to allow calling convert() with pqdm
def pqdm_wrapper(file):
    file_name = file.stem
    dest_file = DEST_DIR / f"{file_name}.txt"
    convert(source_xml_file=file, dest_txt_file=dest_file)


@click.command()
@click.argument("source_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("dest_dir", type=click.Path(path_type=Path))
def main(source_dir, dest_dir):
    """
    Preprocess OpenSubtitles dataset (generate filtered and cleaned-up .txt files from original .xml files)

    source_dir: path to main OpenSubtitles dir containing the "xml" folder

    dest_dir: output dir
    """

    dest_dir.mkdir(parents=True, exist_ok=True)

    if not (source_dir / "xml/fr").exists():
        raise Exception(f"Could not find xml/fr subdir in {source_dir}")

    print("Gathering all .xml files...")
    # avoid movies older than MIN_YEAR
    years_dirs = [
        year_dir
        for year_dir in sorted(source_dir.glob("xml/fr/*"))
        if int(year_dir.stem) >= MIN_YEAR
    ]
    movie_dirs = [
        movie_dir for year_dir in years_dirs for movie_dir in sorted(year_dir.glob("*"))
    ]

    # pick last subtitle file in each movie folder
    files = [sorted(movie_dir.glob("*.xml"))[-1] for movie_dir in movie_dirs]

    print("\nProcessing all .xml files...")
    global DEST_DIR
    DEST_DIR = dest_dir

    pqdm(files, pqdm_wrapper, n_jobs=6)


if __name__ == "__main__":
    main()
