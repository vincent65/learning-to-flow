"""
Utilities for parsing and extracting TensorBoard event files.
"""

import os
import numpy as np
from collections import defaultdict


def parse_tensorboard_logs(logdir):
    """
    Parse TensorBoard event files and extract scalar data.

    Args:
        logdir: Path to TensorBoard log directory

    Returns:
        Dict mapping tag -> List[(step, value)] tuples
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("Error: tensorboard package not found.")
        print("Install with: pip install tensorboard")
        return None

    # Find event files
    event_files = []
    for root, dirs, files in os.walk(logdir):
        for f in files:
            if 'events.out.tfevents' in f:
                event_files.append(os.path.join(root, f))

    if not event_files:
        print(f"No TensorBoard event files found in {logdir}")
        return None

    # Load events
    data = defaultdict(list)

    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()

            # Extract all scalar tags
            for tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                for event in events:
                    data[tag].append((event.step, event.value))
        except Exception as e:
            print(f"Warning: Could not parse {event_file}: {e}")
            continue

    # Sort by step
    for tag in data:
        data[tag] = sorted(data[tag], key=lambda x: x[0])

    return dict(data)


def smooth_curve(values, weight=0.6):
    """
    Apply exponential moving average smoothing to a curve.

    Args:
        values: List of values
        weight: Smoothing weight (0 = no smoothing, 1 = maximum smoothing)

    Returns:
        Smoothed values
    """
    if not values:
        return []

    smoothed = []
    last = values[0]

    for val in values:
        smoothed_val = last * weight + val * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def extract_epoch_data(data, steps_per_epoch):
    """
    Convert step-based data to epoch-based data.

    Args:
        data: List of (step, value) tuples
        steps_per_epoch: Number of steps per epoch

    Returns:
        List of (epoch, mean_value) tuples
    """
    if not data:
        return []

    epochs = defaultdict(list)

    for step, value in data:
        epoch = step // steps_per_epoch
        epochs[epoch].append(value)

    # Compute mean per epoch
    epoch_data = []
    for epoch in sorted(epochs.keys()):
        mean_val = np.mean(epochs[epoch])
        epoch_data.append((epoch, mean_val))

    return epoch_data


def get_training_stats(logdir, steps_per_epoch=None):
    """
    Extract training statistics from TensorBoard logs.

    Args:
        logdir: Path to TensorBoard log directory
        steps_per_epoch: Number of training steps per epoch (optional)

    Returns:
        Dict with training curves
    """
    data = parse_tensorboard_logs(logdir)

    if data is None:
        return None

    stats = {}

    # Organize by train/val
    for tag, values in data.items():
        # Extract steps and values
        steps = [s for s, v in values]
        vals = [v for s, v in values]

        stats[tag] = {
            'steps': steps,
            'values': vals,
            'smoothed': smooth_curve(vals)
        }

        # Convert to epochs if steps_per_epoch provided
        if steps_per_epoch:
            epoch_data = extract_epoch_data(values, steps_per_epoch)
            stats[tag]['epochs'] = [e for e, v in epoch_data]
            stats[tag]['epoch_values'] = [v for e, v in epoch_data]

    return stats
