"""How many processor cores this process may really use.

In a container ``os.cpu_count()`` and ``nproc`` can report the host's cores
(48 on a 12-vCPU cloud machine), and sizing thread pools by them makes the
threads thrash. The cgroup quota is the true limit."""

import os
from pathlib import Path


def available_cpus():
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for path in (Path("/sys/fs/cgroup/cpu.max"), Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")):
        try:
            text = path.read_text().split()
        except OSError:
            continue
        if path.name == "cpu.max" and text and text[0] != "max":
            count = min(count, max(1, int(int(text[0]) / int(text[1]))))
        elif path.name == "cpu.cfs_quota_us" and text and int(text[0]) > 0:
            period = int((path.parent / "cpu.cfs_period_us").read_text())
            count = min(count, max(1, int(text[0]) // period))
        break
    return max(1, count)


def main():
    print(available_cpus())


if __name__ == "__main__":
    main()
