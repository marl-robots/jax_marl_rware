import csv
import re

import jax.numpy as jnp
import numpy as np
import pandas as pd

MAXF = jnp.finfo(jnp.float32).max


def sanitize_sentinel(values,isCumsum):
    values = values.to_numpy().astype(jnp.float32)
    mask = values != MAXF
    if isCumsum:
        num = np.cumsum(values * mask)
        den=np.sum(values * mask)
    else:
        num = np.sum(values * mask, axis=0)
        den = np.sum(mask, axis=0)

    # keep only columns where den > 0

    valid = den > 0
    
    if valid.any():
        out = num[valid] / den[valid]
        if isCumsum:
            out = pd.Series(np.sort(out.flatten()))
        else:
            out = pd.Series(np.sort(out))
    else:
        # no valid columns → return zeros

        out = pd.Series(np.zeros(values.shape[1], dtype=np.float32))
    return out


def sanitize_masked(values):
    values = values.to_numpy().astype(jnp.float32)
    mask = values != MAXF
    return np.asarray(values[mask])


def expand_agent_metrics(csv_header_line: list[str]):
    """
    Just expose the exact CSV keys that match *_agent_* patterns.
    """
    out: list[str] = []

    pat = re.compile(r"(.*_agent)_(\d+)(?:_(.+))?$")

    for key in csv_header_line:

        if key.startswith("action_histogram_flat") or "_lorenz_y_" in key:
            continue
        m = pat.fullmatch(key)

        if not m:
            continue
        out.append(key)
    return out


def expand_action_histogram(csv_header_line: list[str]):
    pat = re.compile(r"action_histogram_flat_agent_(\d+)_b(\d+)")
    out = []
    for key in csv_header_line:
        m = pat.fullmatch(key)
        if not m:
            continue
        key_per_agent = key[:-3]
        out.append(key_per_agent)
    out = sorted(list(set(out)))
    return out


def build_generic_mapping(expanded_dict: list[str]):
    """
    Map generic names (those in METRIC_GROUPS) to lists of concrete CSV keys.
    """
    generic_to_keys: dict[str, list[str]] = {}

    # agent metrics: *_agent_*,*_agent

    pat_agent = re.compile(
        r"(?!(action_histogram_flat_agent))(.*_agent)_(\d+)(?:_(.+))?$"
    )

    # action_histogram_flat: action_histogram_flat_agent_0_b0 -> action_histogram_flat

    pat_hist = re.compile(r"(action_histogram_flat_agent)_(\d+)(?:_(.+))?$")

    for key in expanded_dict:
        gname = None
        m = pat_agent.fullmatch(key)
        if m:
            base = m.group(2)
            suffix = m.group(4)
            if suffix:
                gname = f"{base}_{suffix}"  # *_agent_i_*
            else:
                gname = base  # *_agent_i
        else:
            m = pat_hist.fullmatch(key)
            if m:
                s1 = m.group(1)  # action_histogram_flat_agent
                gname = s1
        if gname is not None:
            generic_to_keys.setdefault(gname, []).append(key)
    return generic_to_keys


def insert_expanded_metrics(
    grouped_dict: dict[str, dict[str, str]], expanded_dict: list[str]
):

    generic_to_keys = build_generic_mapping(expanded_dict)
    new_groups: dict[str, dict[str, str]] = {}
    for group, metrics in grouped_dict.items():
        new_list: list[tuple[str, str]] = []
        for m, v in metrics.items():
            if m in generic_to_keys.keys():
                for key in generic_to_keys[
                    m
                ]:  # expand generic name to all concrete keys
                    new_list.append((key, v))
            else:
                new_list.append((m, v))
        new_groups[group] = {key: v for (key, v) in new_list}
    return new_groups


def process_metrics(raw_metrics: dict[str, dict[str, str]], csv_path: str):

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        
    expanded: list[str] = []
    expended_metrics_list: list[list[str]] = []

    expended_metrics_list.append(expand_agent_metrics(header))  # Agent metrics
    expended_metrics_list.append(expand_action_histogram(header))  # Action histogram

    for expanded_list in expended_metrics_list:  # flat
        for m in expanded_list:
            expanded.append(m)
    # Grouped metrics with expansion

    grouped_metrics = insert_expanded_metrics(raw_metrics, expanded)
    return grouped_metrics
