from collections import deque
import ipaddress
import json
from pathlib import Path
import random
import sys
import types

import numpy as np

source = Path(sys.argv[1])
partition = types.ModuleType("SpacePartition")
text = (source / "SpacePartition.py").read_text()
exec(compile(text.replace("from numpy.lib.shape_base import split", ""), str(source / "SpacePartition.py"), "exec"), partition.__dict__)
sys.modules["SpacePartition"] = partition
outlier = types.ModuleType("OutlierDetection")
exec(compile((source / "OutlierDetection.py").read_text(), str(source / "OutlierDetection.py"), "exec"), outlier.__dict__)


def addresses(rows):
    return ["".join(format(int(n), "x") for n in row) for row in rows]


def mine(seeds):
    seeds = sorted(set(seeds))
    queue = deque([np.array([[int(n, 16) for n in address] for address in seeds])])
    regions = []
    while queue:
        region = queue.popleft()
        if len(region) < 16:
            regions.append(region)
            continue
        scores = []
        for column in region.T:
            counts = np.bincount(column, minlength=16)
            scores.append(int(sum(counts[counts > 1])) if np.count_nonzero(counts) > 1 else -1)
        if max(scores) < 0:
            regions.append(region)
            continue
        dimension = scores.index(max(scores))
        queue.extend(region[region[:, dimension] == value] for value in sorted(set(region[:, dimension])))
    patterns = []
    removed = 0
    for region in regions:
        weights = [0.0] * len(region)
        for column in region.T:
            counts = np.bincount(column, minlength=16)
            if np.count_nonzero(counts) > 1:
                outlier.IoslatedForest(weights, counts, column)
        remaining = list(enumerate(weights))
        indices = []
        # The paper leaves the threshold unspecified, so use the released three-sigma rule.
        for weight in outlier.Four_D(weights):
            position = next(i for i, (_, value) in enumerate(remaining) if value == weight)
            indices.append(remaining.pop(position)[0])
        removed += len(indices)
        normal = region[[i for i in range(len(region)) if i not in indices]]
        if len(normal):
            pattern = "".join(format(int(column[0]), "x") if len(set(column)) == 1 else "*" for column in normal.T)
            patterns.append([pattern, len(normal)])
    return {"seeds": seeds, "partitions": [addresses(region) for region in regions], "patterns": patterns, "outliers": removed}


rng = random.Random(67)
seeds = []
for group in range(40):
    prefix = "20010db8" + f"{group:04x}"
    for row in range(rng.randrange(3, 24)):
        tail = f"{rng.randrange(16 ** 3):03x}" + "0" * 14 + f"{row:03x}"
        seeds.append(prefix + tail)
Path(__file__).with_name("mining.json").write_text(json.dumps(mine(seeds), indent=2) + "\n")

if len(sys.argv) > 2:
    fixture = mine([f"{int(ipaddress.IPv6Address(line.strip())):032x}" for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()])

    def fnv(text):
        value = 0xcbf29ce484222325
        for byte in text.encode():
            value = ((value ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
        return value

    print(json.dumps({
        "regions": len(fixture["partitions"]),
        "region_hash": fnv("".join("".join(address + "|" for address in region) + "\n" for region in fixture["partitions"])),
        "patterns": len(fixture["patterns"]),
        "pattern_hash": fnv("".join(f"{pattern}\t{count}\n" for pattern, count in fixture["patterns"])),
        "outliers": fixture["outliers"],
    }))
