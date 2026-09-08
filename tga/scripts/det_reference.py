import contextlib
import io
import hashlib
import ipaddress
import json
from pathlib import Path
import random
import sys
import types

sys.path.insert(0, sys.argv[1])
scanner = types.ModuleType("ActiveScan")
scanner.Scan = lambda addresses, *args: list(addresses)
sys.modules["ActiveScan"] = scanner

import AddrsToSeq
import DHC
import DynamicScan
import ScanPre


def ranges(addresses):
    result = []
    for value in sorted(int(ipaddress.IPv6Address(a)) for a in addresses):
        if result and value == int(result[-1][1], 16) + 1:
            result[-1][1] = f"{value:032x}"
        else:
            result.append([f"{value:032x}", f"{value:032x}"])
    return result


def snapshot(node, ids):
    return dict(
        id=ids[node],
        parent=ids.get(node.parent),
        children=[ids[n] for n in node.childs],
        stack=[d - 1 for d in node.DS.stack],
        patterns=sorted([[65535 if d == -1 else d for d in p] for p in node.TS]),
        scanned=ranges(node.SS),
        hits=node.NDA,
        density=node.AAD,
    )


def run(name, bits, leaf_max, values, response, rounds=12):
    seeds = [f"{(0x20010db8 << 96) | value:032x}" for value in values]
    vectors = AddrsToSeq.AddrsToSeq(seeds, bits)
    root = DHC.SpaceTreeGen(vectors, 1 << bits, leaf_max)
    nodes = []

    def visit(node):
        nodes.append(node)
        for child in node.childs:
            visit(child)

    visit(root)
    ids = {node: i for i, node in enumerate(nodes)}
    ScanPre.InitializeDS(root, beta=1 << bits)
    ScanPre.InitializeTS(root)
    batch = []
    DynamicScan.InitializeNodeQueue(root, batch)
    queue = []
    result = dict(name=name, bits=bits, leaf_max=leaf_max, seeds=seeds,
                  tree=[snapshot(n, ids) for n in nodes], rounds=[], corrections=[])
    active = set()
    targets = set()
    budget = 10**12
    for iteration in range(rounds):
        total = sum(sum((1 << bits) ** p.count(-1) for p in n.TS) for n in batch)
        if total > 16384:
            break
        candidate = set().union(*(set(AddrsToSeq.SeqToAddrs(n.TS)) for n in batch))
        candidate -= set().union(*(n.SS for n in batch))
        found = {a for a in candidate if response(int(ipaddress.IPv6Address(a)), iteration)}
        DynamicScan.Scan = lambda addresses, *args: list(found.intersection(addresses))
        result['rounds'].append(dict(
            batch=[snapshot(n, ids) for n in batch],
            queue=[snapshot(n, ids) for n in queue],
            targets=ranges(candidate), active=ranges(found),
        ))
        with contextlib.redirect_stdout(io.StringIO()):
            batch, budget, active, targets = DynamicScan.Scan_Feedback(
                batch, 10**12, budget, active, targets, None, None, None)
        queue = DynamicScan.MergeSort(batch, queue)
        batch = DynamicScan.TakeOutFrontSegment(queue, len(queue) // 10 + 1)
        previous = set(batch)
        parents = {n.parent for n in batch if n.parent is not None and n.parent.DS.stack == n.DS.stack}
        saved_children = {n: n.childs for n in parents}
        for parent in sorted(parents, key=ids.get):
            descendants = []
            def descend(node):
                for child in node.childs:
                    descendants.append(child)
                    descend(child)
            descend(parent)
            if any(n in set(batch + queue) for n in descendants if n not in parent.childs):
                result['corrections'].append(dict(round=iteration + 1, kind="retire_all_descendants"))
            parent.childs = descendants
        DynamicScan.ReplaceDescendants(queue, batch)
        for parent, children in saved_children.items():
            parent.childs = children
        promoted = set(batch) - previous
        # The reference leaves the order of promoted nodes unspecified.
        batch = [n for n in batch if n not in promoted] + sorted(promoted, key=ids.get)
    return result


rng = random.Random(67)
cases = []
responses = [lambda a, r: False, lambda a, r: True,
             lambda a, r: (a + r) % 7 < 2]
for bits in [1, 2, 4, 8]:
    for response_index, leaf_max in enumerate([1, 3, 16]):
        values = [rng.randrange(1 << min(bits * 4, 12)) for _ in range(25)]
        cases.append(run(f"base-{1 << bits}-leaf-{leaf_max}", bits, leaf_max,
                         values, responses[response_index]))
cases.append(run("singleton", 4, 16, [1], responses[0]))
cases.append(run("identical-seeds", 2, 1, [7] * 20, responses[1]))
cases.append(run("density-priority", 4, 2, [1, 2, 3, 16, 18, 33, 34],
                 lambda a, r: a % 3 == 0))
output = Path(__file__).resolve().parents[1] / "tests/fixtures/det/reference.json"
output.parent.mkdir(parents=True, exist_ok=True)
sources = {name: hashlib.sha256((Path(sys.argv[1]) / name).read_bytes()).hexdigest()
           for name in ["AddrsToSeq.py", "DHC.py", "Definitions.py", "ScanPre.py", "DynamicScan.py"]}
output.write_text(json.dumps(dict(sources=sources, cases=cases), separators=(",", ":")) + "\n")
print(f"{len(cases)} cases, {sum(len(c['rounds']) for c in cases)} rounds, {output.stat().st_size} bytes")
