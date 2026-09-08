import argparse
import contextlib
import importlib
import io
import ipaddress
import json
from pathlib import Path
import sys
import types


def active(address):
    return int(ipaddress.IPv6Address(address)) % 7 in (1, 2)


def fingerprint(addresses):
    value = 14695981039346656037
    for address in sorted(int(ipaddress.IPv6Address(address)) for address in addresses):
        for byte in address.to_bytes(16, "big"):
            value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def preorder(node):
    yield node
    for child in node.childs:
        yield from preorder(child)


def replace_descendants(queue, batch):
    # Algorithm 3 retires every descendant rather than only immediate children.
    promoted = []
    for node in batch:
        if node.parent is not None and node.parent.DS.stack == node.DS.stack:
            node.parent.TS = node.TS
            if node.parent not in promoted:
                promoted.append(node.parent)
    promoted = [node for node in promoted if not any(node in list(preorder(other))[1:] for other in promoted)]
    for parent in promoted:
        descendants = set(list(preorder(parent))[1:])
        retired = [node for node in batch + queue if node in descendants]
        parent.SS = set().union(*(node.SS for node in retired))
        parent.NDA = sum(node.NDA for node in retired)
        parent.AAD = parent.NDA / len(parent.SS) if parent.SS else 0
        batch[:] = [node for node in batch if node not in descendants]
        queue[:] = [node for node in queue if node not in descendants]
        batch.append(parent)


def run(reference):
    sys.path.insert(0, str(reference))
    scanner = types.ModuleType("ActiveScan")
    scanner.Scan = lambda addresses, *args: {address for address in addresses if active(address)}
    sys.modules["ActiveScan"] = scanner
    sys.modules["AliasDetection"] = types.ModuleType("AliasDetection")
    addresses = importlib.import_module("AddrsToSeq")
    dhc = importlib.import_module("DHC")
    preparation = importlib.import_module("ScanPre")
    dynamic = importlib.import_module("DynamicScan")
    prefix = int(ipaddress.IPv6Address("2001:db8::"))
    cases = []
    for base in (2, 4, 8, 16, 32):
        for name, suffixes in (
            ("single", [1, 2, 3]),
            ("balanced", [256 * i + 16 * j + k for i in range(3) for j in range(3) for k in range(1, 4)]),
            ("uneven", [256 * i + 16 * j + k for i in range(3) for j in range(i + 1) for k in range(1, 5)]),
            ("separated_dimensions", [(i << 48) + (i % 3 << 16) + j for i in range(5) for j in range(1, 4)]),
        ):
            seeds = [f"{prefix + suffix:032x}" for suffix in suffixes]
            bits = base.bit_length() - 1
            width = 128 // bits * bits
            scope = prefix >> width << width
            vectors = addresses.AddrsToSeq(seeds, bits, lamda=width)
            def seq_to_addrs(vectors):
                return [str(ipaddress.IPv6Address(int(ipaddress.IPv6Address(address)) | scope)) for address in addresses.SeqToAddrs(vectors)]
            dynamic.SeqToAddrs = seq_to_addrs
            root = dhc.SpaceTreeGen(vectors, beta=base)
            preparation.InitializeDS(root, vectors, beta=base)
            preparation.InitializeTS(root, vectors)
            nodes = list(preorder(root))
            indices = {node: index for index, node in enumerate(nodes)}
            snapshot = [dict(parent=indices.get(node.parent), children=[indices[child] for child in node.childs], stack=[dimension - 1 for dimension in node.DS.stack]) for node in nodes]
            batch = []
            dynamic.InitializeNodeQueue(root, batch)
            queue = []
            remaining = 10**9
            found = set()
            issued = set()
            rounds = []
            for iteration in range(12):
                count = sum(base ** node.TS[0].count(-1) * len(node.TS) for node in batch)
                if count > 100000:
                    break
                targets = set(address for node in batch for address in seq_to_addrs(node.TS)) - set(address for node in batch for address in node.SS)
                rounds.append(dict(count=len(targets), fingerprint=fingerprint(targets), active=sum(active(address) for address in targets)))
                with contextlib.redirect_stdout(io.StringIO()):
                    batch, remaining, found, issued = dynamic.Scan_Feedback(batch, 10**9, remaining, found, issued, vectors, "", "", "")
                queue = batch if iteration == 0 else dynamic.MergeSort(batch, queue)
                batch = dynamic.TakeOutFrontSegment(queue, max(1, len(queue) // 10))
                replace_descendants(queue, batch)
            cases.append(dict(name=f"{name}_{base}", base=base, seeds=seeds, nodes=snapshot, rounds=rounds))
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[3] / "prior-tgas/tgas/2019-6tree/6Tree")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/sixtree/reference.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.reference), indent=2) + "\n")
