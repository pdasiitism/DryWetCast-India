"""How many CPUs this process may actually use."""

import os

MAX_WORKERS = 16


def available_cpus() -> int:
    """CPUs available to this process — not the whole machine.

    On a cluster, os.cpu_count() reports every core on the node even when the
    scheduler gave this job only a few; starting that many processes on a
    small allocation makes the run far slower, not faster. sched_getaffinity
    respects the job's CPU binding (Linux); SLURM_CPUS_PER_TASK covers
    clusters that don't bind CPUs.
    """
    n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    slurm = os.environ.get('SLURM_CPUS_PER_TASK', '')
    if slurm.isdigit():
        n = min(n, int(slurm))
    return max(1, n)


def default_workers() -> int:
    return min(MAX_WORKERS, available_cpus())
