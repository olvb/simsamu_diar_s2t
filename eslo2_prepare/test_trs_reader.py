# testing of trs_reader.py

from pathlib import Path

import pytest

from trs_reader import read_trs

TESTS_FILE_DIR = Path(__file__).parent / "trs_test_files"

TEST_PARAMS = [
    (
        "basic.trs",
        [
            {
                "speaker": "TM387",
                "start": 0.0,
                "end": 9.256,
                "text": "bonjour en fait on est étudiantes à la fac bah on a juste quelques questions ça vous intéresse ?",
                "overlap": False,
            },
            {
                "speaker": "TM387",
                "start": 9.876,
                "end": 10.314,
                "text": "OK?",
                "overlap": True,
            },
            {
                "speaker": "WK960",
                "start": 9.876,
                "end": 10.314,
                "text": "D'accord",
                "overlap": True,
            },
        ],
    ),
    (
        "no_start_sync.trs",
        [
            {
                "speaker": "TM387",
                "start": 0.0,
                "end": 5.871,
                "text": "bonjour en fait on est étudiantes à la fac",
                "overlap": False,
            },
        ],
    ),
    (
        "event.trs",
        [
            {
                "speaker": "TM387",
                "start": 0.0,
                "end": 9.256,
                "text": "bonjour en fait on est étudiantes à la fac bah on a juste quelques questions ça vous intéresse ?",
                "overlap": False,
            },
            {
                "speaker": "WK960",
                "start": 9.876,
                "end": 12.314,
                "text": "D'accord Je suis à vous",
                "overlap": False,
            },
            {
                "speaker": "TM387",
                "start": 13.1,
                "end": 15.0,
                "text": "Super Merci Commençons",
                "overlap": False,
            },
        ],
    ),
]


@pytest.mark.parametrize("filename,expected_rows", TEST_PARAMS)
def test_output(filename, expected_rows):
    rows = read_trs(TESTS_FILE_DIR / filename)
    assert rows == expected_rows


def test_full_sanity():
    read_trs(TESTS_FILE_DIR / "full.trs")
