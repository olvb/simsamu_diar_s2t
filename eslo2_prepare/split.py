# simple or stratified (by categories) split of ESLO2 recordings

from collections import Counter, defaultdict
import itertools
import math
from pprint import pprint
import random
from typing import Dict, List

import networkx as nx
from ortools.sat.python import cp_model

from .recording_store import RecordingStore


def make_simple_split(
    rec_store: RecordingStore,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 12354,
) -> Dict[str, List[str]]:
    """
    Simple random split at recording level.

    Ratios refer to the number of recordings, not the recording duration.
    """

    assert train_ratio + dev_ratio + test_ratio == 1.0

    rng = random.Random(seed)
    rec_ids = rec_store.rec_ids.copy()
    rng.shuffle(rec_ids)

    nb_recs = len(rec_ids)
    nb_train = int(train_ratio * nb_recs)
    nb_dev = int(dev_ratio * nb_recs)
    nb_test = nb_recs - (nb_train + nb_dev)

    rec_ids_by_split = dict(
        train=rec_ids[:nb_train],
        dev=rec_ids[nb_train : nb_train + nb_dev],
        test=rec_ids[-nb_test:],
    )

    _check_split(rec_ids_by_split, nb_recs, rec_store, check_speakers_in_common=False)

    return rec_ids_by_split


def make_stratified_split(
    rec_store: RecordingStore,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 12354,
) -> Dict[str, List[str]]:
    """
    Split recordings in train/dev/test subsets:
      - guaranteeing that each subset does not have a speaker in common with another subset
      - trying to keep the same proportion of recordings of each category accross each split

    Ratios refer to the number of recordings, not the recording duration.
    """

    assert train_ratio + dev_ratio + test_ratio == 1.0
    assert train_ratio > dev_ratio and train_ratio > test_ratio

    # split recordings in group where recordings not in the same group
    # do not have common speakers
    groups = _group_recordings_by_common_speakers(rec_store, seed)

    # distribute groups by category
    groups_by_category = defaultdict(list)
    for group in groups:
        # get categories for each file
        categories = [rec_store.get_category(rec_id) for rec_id in group]
        # find most representative category
        # (in almost all cases all recordings in a group belong to the same category)
        category = max(set(categories), key=categories.count)
        groups_by_category[category].append(group)

    print("Recording groups by category:")
    pprint(
        {cat: [len(g) for g in groups] for cat, groups in groups_by_category.items()}
    )
    print("")

    # perform split for each category and merge results
    rec_ids_by_split = dict(train=[], dev=[], test=[])
    for category_groups in groups_by_category.values():
        category_rec_ids_by_split = _make_split_by_category(
            category_groups, train_ratio, dev_ratio, test_ratio, seed
        )
        for name, rec_ids in category_rec_ids_by_split.items():
            rec_ids_by_split[name] += rec_ids

    # shuffle recordings in each split
    rng = random.Random(seed)
    for rec_ids in rec_ids_by_split.values():
        rng.shuffle(rec_ids)

    nb_recs = len(rec_store.rec_ids)
    _check_split(rec_ids_by_split, nb_recs, rec_store, check_speakers_in_common=True)

    return rec_ids_by_split


def _group_recordings_by_common_speakers(
    rec_store: RecordingStore, seed: int
) -> List[List[str]]:
    """
    Return recording ids grouped in a way that recordings not in the same group
    do not have a common speaker
    """

    # build a graph where nodes are recordings
    graph = nx.Graph()
    for rec_id in rec_store.rec_ids:
        graph.add_node(rec_id)

    # retrieve speakers for each record
    rec_ids_by_speaker = defaultdict(list)
    for rec_id in rec_store.rec_ids:
        speakers = rec_store.get_speakers_in_rec(rec_id)
        for speaker in speakers:
            rec_ids_by_speaker[speaker].append(rec_id)

    # connect all nodes/recordings having common speakers
    for connected_rec_ids in rec_ids_by_speaker.values():
        for rec_id, other_rec_id in itertools.combinations(connected_rec_ids, r=2):
            graph.add_edge(rec_id, other_rec_id)

    # use connected components to determine groups
    # (sort results because networkx returns non-deterministic results)
    groups = sorted(sorted(group) for group in nx.connected_components(graph))
    return groups


def _make_split_by_category(
    groups: List[List[str]],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    # set goals in terms of number of recordings for each subset
    nb_recs = sum(len(group) for group in groups)
    nb_train = train_ratio * nb_recs
    nb_dev = dev_ratio * nb_recs
    nb_test = test_ratio * nb_recs

    # assign group to each subset, formulating this as a
    # weighted bin assignment problem
    # (each item/group has a weight/nb_recs and
    # each bin/subset has a weight/nb_recs capacity)
    item_weights = [len(group) for group in groups]
    bin_weights = [nb_train, nb_dev, nb_test]
    bins = _assign_items_to_bins(item_weights, bin_weights, seed)
    train_groups, dev_groups, test_groups = bins

    # extract rec_ids from assigned groups
    train_rec_ids = [
        rec_id for group_index in train_groups for rec_id in groups[group_index]
    ]
    dev_rec_ids = [
        rec_id for group_index in dev_groups for rec_id in groups[group_index]
    ]
    test_rec_ids = [
        rec_id for group_index in test_groups for rec_id in groups[group_index]
    ]

    return dict(train=train_rec_ids, dev=dev_rec_ids, test=test_rec_ids)


def _assign_items_to_bins(
    item_weights: List[int], target_bin_weights: List[float], seed: int
) -> List[List[int]]:
    """
    Solve a weight bin assigment problem.

    We are trying to assign items with weights to bins with
    weight capacities.

    All items must be assigned (no leftover) but can only be
    assigned to one bin. Bins have a target/capacity but it is not
    a hard limit, it is possible to assign more but this is penalized.
    """

    item_indices = list(range(len(item_weights)))
    bin_indices = list(range(len(target_bin_weights)))

    model = cp_model.CpModel()

    # build the item-to-bin assignation matrix
    # (the variables for which we want to find values)
    assignations = {}
    for item_index in item_indices:
        for bin_index in bin_indices:
            assignations[item_index, bin_index] = model.NewBoolVar(
                f"assign_{item_index}_{bin_index}"
            )

    # each item must be assigned to one and only one bin
    for item_index in item_indices:
        model.Add(
            sum(assignations[item_index, bin_index] for bin_index in bin_indices) == 1
        )

    # each bin must have at least one item (if feasible)
    if len(item_indices) >= len(bin_indices):
        for bin_index in bin_indices:
            model.Add(
                sum(assignations[item_index, bin_index] for item_index in item_indices)
                >= 1
            )

    # compute weights for each bin
    bin_weights = [
        sum(
            assignations[item_index, bin_index] * item_weights[item_index]
            for item_index in item_indices
        )
        for bin_index in bin_indices
    ]

    # compute total square diff between target bin weight and actual bin weight
    total_weight = sum(item_weights)
    weight_square_diff_by_bin = []
    for bin_index in bin_indices:
        diff = math.ceil(target_bin_weights[bin_index]) - bin_weights[bin_index]
        abs_diff = model.NewIntVar(0, total_weight, f"bin_weight_abs_diff_{bin_index}")
        model.AddAbsEquality(abs_diff, diff)
        square_diff = model.NewIntVar(
            0, total_weight**2, f"bin_weight_square_diff_{bin_index}"
        )
        model.AddMultiplicationEquality(square_diff, [abs_diff, abs_diff])
        weight_square_diff_by_bin.append(square_diff)
    total_weight_square_diff = sum(weight_square_diff_by_bin)

    # minimize square diff
    model.Minimize(total_weight_square_diff)

    # solve
    solver = cp_model.CpSolver()
    # make solver deterministic
    solver.parameters.random_seed = seed
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]

    # retrieve assignations solution
    bins = [
        [
            item_index
            for item_index in item_indices
            if solver.Value(assignations[item_index, bin_index])
        ]
        for bin_index in bin_indices
    ]
    return bins


def _check_split(
    rec_ids_by_split: Dict[str, List[str]],
    nb_recs: int,
    rec_store: RecordingStore,
    check_speakers_in_common: bool,
):
    train_rec_ids = rec_ids_by_split["train"]
    dev_rec_ids = rec_ids_by_split["dev"]
    test_rec_ids = rec_ids_by_split["test"]

    assert len(train_rec_ids) + len(dev_rec_ids) + len(test_rec_ids) == nb_recs

    assert len(train_rec_ids) > 0
    assert len(dev_rec_ids) > 0
    assert len(test_rec_ids) > 0

    assert len(set(train_rec_ids) & set(dev_rec_ids)) == 0
    assert len(set(train_rec_ids) & set(test_rec_ids)) == 0
    assert len(set(dev_rec_ids) & set(test_rec_ids)) == 0

    if check_speakers_in_common:
        trains_speakers = {
            s for rec_id in train_rec_ids for s in rec_store.get_speakers_in_rec(rec_id)
        }
        dev_speakers = {
            s for rec_id in dev_rec_ids for s in rec_store.get_speakers_in_rec(rec_id)
        }
        test_speakers = {
            s for rec_id in test_rec_ids for s in rec_store.get_speakers_in_rec(rec_id)
        }
        assert len(trains_speakers & dev_speakers) == 0
        assert len(trains_speakers & test_speakers) == 0
        assert len(dev_speakers & test_speakers) == 0


def describe_split(rec_ids_by_split: Dict[str, List[str]], rec_store: RecordingStore):
    total_nb_recs = 0
    total_duration = 0.0

    for split_name, rec_ids in rec_ids_by_split.items():
        print(f"{split_name}:")
        nb_recs = len(rec_ids)
        print(f"  - nb recordings: {nb_recs}")
        categories = []
        durations = []
        speakers = []
        max_rec_speakers = 0
        for rec_id in rec_ids:
            rec_category = rec_store.get_category(rec_id)
            categories.append(rec_category)
            rec_duration = rec_store.get_duration(rec_id) / 60
            durations.append(rec_duration)
            rec_speakers = rec_store.get_speakers_in_rec(rec_id)
            max_rec_speakers = max(max_rec_speakers, len(rec_speakers))
            speakers += rec_speakers

        duration = sum(durations)
        duration_mean = duration / len(durations)
        print(f"  - duration: {duration:.0f} mins (mean: {duration_mean:.0f})")
        print(f"  - categories: {dict(Counter(categories))}")
        print(f"  - speakers: {len(speakers)}")
        print(f"  - max speakers: {max_rec_speakers}", end="\n\n")

        total_nb_recs += nb_recs
        total_duration += duration

    print(f"total nb recordings: {total_nb_recs}")
    print(f"total duration: {total_duration:.0f} mins")
