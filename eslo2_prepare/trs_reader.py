# reading of XML .trs files containing ESLO2 annotations

from collections import defaultdict
from pathlib import Path
import warnings

import lxml.etree as ET


def read_trs(
    trs_file: Path,
    merge_consecutive: bool = True,
    merge_overlaping: bool = False,
    max_merge_length=10.0,
):
    """Read ESLO style XML .trs annotations file"""

    root = ET.parse(trs_file).getroot()

    speaker_names_by_id = {
        e.get("id"): e.get("name") for e in root.find("Speakers").findall("Speaker")
    }

    rows = []
    for turn in root.findall("Episode/Section/Turn"):
        rows_in_turn = _parse_turn(turn, speaker_names_by_id, trs_file)
        rows += rows_in_turn

    if merge_consecutive:
        if merge_overlaping:
            rows = _merge_consecutive_rows_agressive(rows, max_length=max_merge_length)
        else:
            rows = _merge_consecutive_rows(rows, max_length=max_merge_length)

    assert len(rows) > 0, f"No rows found in {trs_file}"
    _check_rows_sanity(rows, merge_consecutive, max_merge_length)
    return rows


def _parse_turn(turn, speaker_names_by_id, trs_file):
    time = float(turn.get("startTime"))

    speaker_ids = turn.get("speaker")
    # turn with no speaker (is empty or contains non-text event)
    if speaker_ids is None:
        return []

    speaker_ids = speaker_ids.split(" ")
    if len(set(speaker_ids)) != len(speaker_ids):
        warnings.warn(
            f"Found invalid speaker ids in {trs_file} in turn starting at line {turn.sourceline}, discarding whole turn"
        )
        return []

    overlap = len(speaker_ids) > 1
    if overlap:
        speaker_name = None
    else:
        speaker_name = speaker_names_by_id[speaker_ids[0]]

    rows = []

    # handle text immediately inside <Turn> at begining, not preceded by a, element
    text = turn.text.strip()
    if len(text) > 0:
        row = _build_row(speaker_name, time, text, overlap)
        rows.append(row)

    for element in turn:
        assert element.tag in (
            "Sync",
            "Who",
            "Event",
            "Comment",
        ), f"Unexpected element tag {element.tag}"

        # <Sync> updates current time
        if element.tag == "Sync":
            time = float(element.get("time"))
            # current time becomes end time of previous rows
            # (can be several rows when in multispeaker mode
            # so we have to iterate over all previous rows)
            for row in rows:
                if row["end"] is None:
                    row["end"] = time
        # <Who> updates current speaker
        elif element.tag == "Who":
            speaker_index = int(element.get("nb")) - 1
            if speaker_index >= len(speaker_ids):
                warnings.warn(
                    f"Found speaker index greater than number of speakers in {trs_file} in turn starting at line {element.sourceline}, discarding whole turn"
                )
                return []
            speaker_name = speaker_names_by_id[speaker_ids[speaker_index]]

        # move on to next element if no text to process
        text = element.tail.strip()
        if len(text) == 0:
            continue

        # add additional text to previous row (interrupted by Event/Comment node)
        prev_row = rows[-1] if rows else None
        if prev_row is not None and (
            prev_row["speaker"] == speaker_name
            and prev_row["start"] == time
            and prev_row["overlap"] == overlap
        ):
            assert element.tag not in ("Sync", "Who")
            prev_row["text"] += " " + text
        else:
            row = _build_row(speaker_name, time, text, overlap)
            rows.append(row)

    # apply end time to row without end time
    # (can be several rows when in multispeaker mode)
    time = float(turn.get("endTime"))
    for row in rows:
        if row["end"] is None:
            row["end"] = time

    # filter out rows with duration=0
    nb_before = len(rows)
    rows = [r for r in rows if (r["end"] - r["start"]) > 0]
    if len(rows) < nb_before:
        warnings.warn(
            f"Found rows with duration=0.0 in {trs_file} in turn starting at line {turn.sourceline}, discarding rows"
        )

    return rows


def _build_row(speaker_name, time, text, overlap):
    assert (
        speaker_name is not None
    ), "Speaker hasn't been defined yet for turn with multiple speakers"
    row = {
        "speaker": speaker_name,
        "start": time,
        "end": None,
        "text": text,
        "overlap": overlap,
    }
    return row


# convervative merging: merge consecutive when not split by different speaker
def _merge_consecutive_rows(rows, max_length):
    merged_rows = []
    for row in rows:
        prev_rows = [
            r
            for r in merged_rows
            if (
                r["end"] == row["start"]
                and r["speaker"] == row["speaker"]
                and r["overlap"] == row["overlap"]
            )
        ]
        assert len(prev_rows) <= 1
        prev_row = prev_rows[-1] if prev_rows else None
        if prev_row and row["end"] - prev_row["start"] <= max_length:
            prev_row["text"] += " " + row["text"]
            prev_row["end"] = row["end"]
        else:
            merged_rows.append(row)
    return merged_rows


# aggressive merging: merge consecutive even when rows with different speakers are interleaved
def _merge_consecutive_rows_agressive(rows, max_length):
    rows_by_speaker = defaultdict(list)
    for row in rows:
        rows_by_speaker[row["speaker"]].append(row)

    def _merge_consecutive_speaker_rows(speaker_rows, max_length):
        speaker_rows_merged = [speaker_rows[0]]
        for row in speaker_rows[1:]:
            prev_row = speaker_rows_merged[-1]
            assert prev_row["speaker"] == row["speaker"]
            if (
                prev_row["end"] == row["start"]
                and row["end"] - prev_row["start"] <= max_length
            ):
                prev_row["end"] = row["end"]
                prev_row["text"] += " " + row["text"]
                prev_row["overlap"] = True
            else:
                speaker_rows_merged.append(row)
        return speaker_rows_merged

    rows_by_speaker = {
        speaker: _merge_consecutive_speaker_rows(speaker_rows, max_length)
        for speaker, speaker_rows in rows_by_speaker.items()
    }

    rows = sorted(
        (r for speaker_rows in rows_by_speaker.values() for r in speaker_rows),
        key=lambda r: r["start"],
    )
    return rows


def _check_rows_sanity(rows, merge_consecutive, max_merge_length):
    assert all(r["end"] is not None for r in rows)
    assert all(r["start"] < r["end"] for r in rows)
    assert all(r_prev["start"] <= r["start"] for r_prev, r in zip(rows, rows[1:]))

    if not merge_consecutive:
        return

    speakers = set(r["speaker"] for r in rows)
    for speaker in speakers:
        speaker_rows = [r for r in rows if r["speaker"] == speaker]
        prev_row = speaker_rows[0]
        for row in speaker_rows[1:]:
            assert (
                prev_row["end"] < row["start"]
                or (
                    prev_row["end"] == row["start"]
                    and row["end"] - prev_row["start"] > max_merge_length
                )
                or prev_row["overlap"]
                or row["overlap"]
            )
            prev_row = row
