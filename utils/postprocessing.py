from __future__ import annotations

from typing import List, Tuple

import numpy as np


def moving_average(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores

    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(scores, kernel, mode="same")


def extract_events(binary: np.ndarray) -> List[Tuple[int, int]]:
    events = []
    in_event = False
    start = 0

    for i, value in enumerate(binary.astype(int)):
        if value == 1 and not in_event:
            start = i
            in_event = True

        elif value == 0 and in_event:
            events.append((start, i))
            in_event = False

    if in_event:
        events.append((start, len(binary)))

    return events


def events_to_binary(events: List[Tuple[int, int]], length: int) -> np.ndarray:
    y = np.zeros(length, dtype=np.int64)

    for start, end in events:
        y[start:end] = 1

    return y


def merge_close_events(events: List[Tuple[int, int]], merge_gap: int) -> List[Tuple[int, int]]:
    if not events:
        return []

    merged = [events[0]]

    for start, end in events[1:]:
        last_start, last_end = merged[-1]

        if start - last_end <= merge_gap:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))

    return merged


def remove_short_events(events: List[Tuple[int, int]], min_length: int) -> List[Tuple[int, int]]:
    return [(start, end) for start, end in events if end - start >= min_length]


def postprocess_predictions(
    raw_predictions: np.ndarray,
    min_event_length: int = 20,
    merge_gap: int = 30,
) -> np.ndarray:
    events = extract_events(raw_predictions)
    events = merge_close_events(events, merge_gap=merge_gap)
    events = remove_short_events(events, min_length=min_event_length)
    events = merge_close_events(events, merge_gap=merge_gap)

    return events_to_binary(events, len(raw_predictions))