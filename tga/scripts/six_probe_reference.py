import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import Construct6ASForest as reference

rng = random.Random(67)
seeds = []
for group in range(8):
    for _ in range(48):
        address = list("20010db8000000000000000000000000")
        address[12] = format(group, "x")
        for dim in [24, 26, 28, 29, 30, 31][:2 + group % 5]:
            address[dim] = format(rng.randrange(4), "x")
        seeds.append("".join(address))
seeds = list(dict.fromkeys(seeds))
data = np.array([[int(nibble, 16) for nibble in seed] for seed in seeds])
cases = []
for beta in [2, 12, 64]:
    for strategy in ["LeftVDPS", "RightVDPS", "MinEntropy", "MaxCover"]:
        reference.allSpaceList.clear()
        reference.allLeafList.clear()
        patterns = reference.construct6ASTreeByDHC(data, strategy, beta)
        cases.append(dict(strategy=strategy, beta=beta, patterns=patterns))
forests = []
for seed in [0, 7, 4294967297]:
    reference.allSpaceList.clear()
    reference.allLeafList.clear()
    random.seed(seed)
    patterns = reference.construct6ASTreeByDHC(data)
    patterns += reference.constructAdditional6ASTrees(data, 8)
    forests.append(dict(seed=seed, patterns=sorted(set(patterns))))
output = Path(__file__).resolve().parents[1] / "tests/fixtures/six_probe/reference.json"
output.write_text(json.dumps(dict(seeds=seeds, trees=cases, forests=forests), indent=2) + "\n")

import GenerateAddress as generation

stage_cases = []
for name, rows in [
    ("outlier", [list(format(i, "032x")) for i in range(8)] + [list("f" * 32)]),
    ("repeated_outliers", [list(format(i, "032x")) for i in range(32)] + [list("f" * 32), list("0" * 16 + "e" * 16)]),
    ("duplicates", [list(format(i, "032x")) for i in range(8)] + [list(format(i, "032x")) for i in range(3)]),
    ("full_width", [[format(rng.randrange(16), "x") for _ in range(32)] for _ in range(64)]),
    ("parallel", [list("20010db8" + format(i // 128, "04x") + "000000000000" + format(i, "08x")) for i in range(2048)]),
]:
    text_seeds = ["".join(row) for row in rows]
    vectors = np.array([[int(value, 16) for value in row] for row in rows])
    weights = [0.0] * len(rows)
    for dim in range(32):
        counts = np.bincount(vectors[:, dim], minlength=16)
        if np.count_nonzero(counts) > 1:
            reference.IoslatedForest(weights, counts, vectors[:, dim])
    splits = [reference.leftmost(vectors), reference.rightmost(vectors, 1),
              reference.minEntropy(vectors), reference.maxcovering(vectors),
              reference.rightmost(vectors, 32)]
    reference.allSpaceList.clear()
    reference.allLeafList.clear()
    random.seed(67)
    patterns = None
    if len(rows) >= 12:
        patterns = reference.construct6ASTreeByDHC(vectors)
        patterns += reference.constructAdditional6ASTrees(vectors, 40)
        patterns = sorted(set(patterns))
    reference.allLeafList.clear()
    leaf = reference.TreeNode(vectors, reference.TreeNode(vectors))
    reference.init_subspace(leaf, vectors)
    mined = reference.narrowDimension([leaf])
    stage_cases.append(dict(name=name, seeds=text_seeds, weights=weights,
                            outliers=reference.Four_D(weights),
                            splits=[[group.tolist() for group in split] for split in splits],
                            forest=patterns, mined=mined))
expansions = []
for pattern in ["2001:0db8:0000:0000:0000:0000:0000:0001",
                "2001:0db8:0000:0000:0000:0000:0000:000*",
                "*001:0db8:0000:0000:0000:0000:0000:000*",
                "2001:0db8:000*:0000:0000:*000:0000:000*",
                "2001:0db8:*000:000*:0000:00*0:0000:000*"]:
    checksum = 14695981039346656037
    count = 0
    for address in generation.expand(pattern, 0, generation.AllScope):
        for byte in bytes.fromhex(address.replace(":", "")):
            checksum = ((checksum ^ byte) * 1099511628211) & ((1 << 64) - 1)
        count += 1
    expansions.append(dict(pattern=pattern, count=count, checksum=checksum))
output.with_name("stages.json").write_text(json.dumps(dict(cases=stage_cases, expansions=expansions), indent=2) + "\n")
