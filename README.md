# 6SEVEN

An IPv6 target generation, scanning, and analysis workspace.

**More usage details and tutorials are forthcoming**

## Build

Run Cargo from the repository root. The default build includes statistical TGAs without ML dependencies.

```bash
cargo build --release -p sixseven
cargo build --release -p sixseven --features ml
cargo build --release -p sixseven --features gpu
```

The `ml` feature enables CPU-backed ML algorithms. The `gpu` feature enables the WGPU backend.

## Use

```bash
6seven train --seeds seeds.txt --output-model model.bin det
6seven generate --model model.bin --count 16 --unique
6seven scan targets.txt --interface eth0 icmp
6seven scan --model model.bin --count 1000 --batch-size 256 --interface eth0 \
  --output-file results.csv --feedback-file feedback.jsonl --output-model updated.bin icmp
6seven feedback --input-model model.bin --scan-results feedback.jsonl --output-model replayed.bin
6seven analyze addresses.txt entropy
```

## Dealiasing

Offline filtering uses a file of known aliased CIDRs and requires no network access:

```bash
6seven dealias results.csv --aliased-prefixes aliases.txt -o clean.csv
```

Active detection uses three randomized ICMPv6 probes per unique prefix at /48,
/64, /96, /112, /116, and /120. Two Echo Replies classify the prefix as aliased.
It can operate on a saved address list/scan CSV or during scanning:

```bash
6seven dealias results.csv --online --interface eth0 \
  --aliased-prefixes aliases.txt --output-aliases combined.txt -o clean.csv
6seven scan --model model.bin --count 1000000 --interface eth0 \
  --dealias --aliased-prefixes aliases.txt --output-aliases combined.txt \
  --output-file results.csv --feedback-file feedback.jsonl icmp
```

## Development

```bash
cargo fmt --all --check
cargo test --workspace
cargo check --workspace --all-targets --all-features
cargo test -p tga --features ml
```

