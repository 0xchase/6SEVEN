# Regenerate mining fixtures from the authors' source directory passed as the first argument.
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

reference = Path(sys.argv[1]).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(reference))
import SpacePartition
import PatternMining


def encode(rows):
    return ["".join(format(int(x), "x") for x in row) for row in rows]


def case(name, rows, threshold, iterations, rejoin=False):
    seeds = np.array(rows, dtype=np.uint8)
    remaining = seeds
    patterns = []
    PatternMining.threshold = threshold
    for _ in range(iterations + 1):
        if len(remaining) == 0:
            break
        outliers = []
        for region in SpacePartition.DHC(remaining):
            if len(region) == 1:
                # The reference returns singleton outliers with a different shape.
                outliers.append(region[0])
                continue
            found, rejected = PatternMining.OutlierDetect(region)
            patterns.extend(sorted(encode(pattern)) for pattern in found)
            outliers.extend(rejected)
        if rejoin:
            remaining_outliers = []
            for row in outliers:
                seed = encode([row])[0]
                for pattern in patterns:
                    if all(len({member[i] for member in pattern}) > 1
                           or seed[i] == pattern[0][i] for i in range(32)):
                        pattern.append(seed)
                        pattern.sort()
                        break
                else:
                    remaining_outliers.append(row)
            outliers = remaining_outliers
        remaining = np.array(outliers, dtype=np.uint8)
    return dict(name=name, seeds=encode(seeds), threshold=threshold,
                iterations=iterations, patterns=patterns, outliers=encode(remaining))


rng = random.Random(0x364752415048)
cases = []
for i in range(32):
    # Shared prefixes, dense small neighborhoods, distant points, and equal-weight edges.
    rows = set()
    size = 8 if i < 16 else 80
    while len(rows) < size:
        row = [0] * 32
        for position in (0, 8, 28, 29, 30, 31):
            row[position] = rng.randrange(3 if i % 2 else 16)
        rows.add(tuple(row))
    rows = sorted(rows)
    rng.shuffle(rows)
    threshold = (0, 1, 2, 3, 12, 32)[i % 6]
    released = case(f"structured-{i}", rows, threshold, i % 4)
    paper = case(f"structured-{i}", rows, threshold, i % 4, rejoin=True)
    released["paper_patterns"] = paper["patterns"]
    released["paper_outliers"] = paper["outliers"]
    cases.append(released)

for i in range(16):
    values = rng.sample(range(256), 16)
    rows = [[0] * 28 + [(value // (4 ** position)) % 4 for position in range(3, -1, -1)]
            for value in values]
    released = case(f"dense-{i}", rows, 2, i % 4)
    paper = case(f"dense-{i}", rows, 2, i % 4, rejoin=True)
    released["paper_patterns"] = paper["patterns"]
    released["paper_outliers"] = paper["outliers"]
    cases.append(released)

rows = [[0] * 28 + [int(nibble, 16) for nibble in suffix] for suffix in (
    "3120", "1300", "3333", "1302", "2022", "3032", "2032", "2021",
    "0202", "0010", "1023", "2133", "0111", "1121", "2000", "0100",
)]
for iterations in (0, 3):
    released = case(f"rejoining-{iterations}", rows, 2, iterations)
    paper = case(f"rejoining-{iterations}", rows, 2, iterations, rejoin=True)
    released["paper_patterns"] = paper["patterns"]
    released["paper_outliers"] = paper["outliers"]
    cases.append(released)

assert any(case["patterns"] != case["paper_patterns"] for case in cases)

payload = {
    "source_sha256": {name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
                      for name in ("SpacePartition.py", "PatternMining.py", "main.py")},
    "cases": cases,
}
Path(__file__).with_name("mining.json").write_text(json.dumps(payload, indent=2) + "\n")
