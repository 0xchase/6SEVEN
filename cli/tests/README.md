# 6SEVEN Integration Tests

## Entropy IP Performance Comparison

This test compares the performance of the entropy-ip-wasm component when:
1. **Native** - Component compiled directly to native Rust code (no WASM runtime overhead)
2. **WASM** - Component running through Wasmtime (with WASM runtime overhead)

Both use the exact same algorithm implementation from the entropy-ip-wasm component,
ensuring an apples-to-apples comparison that isolates just the WASM runtime overhead.

### Prerequisites

Build the WASM component first:

```bash
cd ../../components/entropy-ip-wasm
cargo build --target wasm32-wasip2 --release
```

### Running the Test

The test is marked with `#[ignore]` since it's a longer-running benchmark. Run it explicitly:

```bash
# Run the performance comparison (release mode for accurate results)
cargo test --test entropy_ip_performance --release -- --ignored --nocapture

# Or run both tests
cargo test --test entropy_ip_performance --release -- --ignored --nocapture
```

### Test Scenarios

#### 1. `compare_entropy_ip_performance`

Trains an Entropy IP model on 10,000 random IPv6 addresses, then benchmarks generation of 1,000,000 targets using both native and WASM approaches.

**Expected Output:**
```
=== Entropy IP Performance Comparison ===

Generating 10000 test seed addresses...
Training native Entropy IP model...
Training completed in X.XXs

Serializing model...
Model size: XXXXX bytes (XX.XX KB)

Benchmarking NATIVE generation (1000000 targets)...
  Duration: X.XXs
  Rate: XXXXXX addresses/second

Benchmarking WASM generation (1000000 targets)...
  Duration: X.XXs
  Rate: XXXXXX addresses/second

=== Performance Summary ===
Native:  X.XXs (XXXXXX addr/s)
WASM:    X.XXs (XXXXXX addr/s)
Overhead: X.XXx (XX.X% slower)

✓ Performance within acceptable bounds
```

**Performance Expectations:**
- **Native**: ~2-5M addresses/second
- **WASM**: ~500K-1M addresses/second
- **Overhead**: 2-5x slower (acceptable for portability)

The test asserts that WASM overhead is < 10x (generous threshold for CI stability).

#### 2. `verify_wasm_native_compatibility`

Verifies that both native and WASM implementations:
- Load the same serialized model successfully
- Generate valid IPv6 addresses
- Work interchangeably

### What's Being Measured

The benchmark measures the **hot path performance** of address generation:

**Native Path:**
```
Model, Generator, generate(), [u8; 16]
```

**WASM Path:**
```
Model, Wasmtime Engine, WASM Component, adaptor::Tga, generate(), [u8; 16]
```

Key overhead sources in WASM:
1. Wasmtime interpreter/JIT overhead
2. Host-guest boundary crossings
3. Tuple conversion (WIT interface)

All of these have been optimized in this implementation (see session optimizations).

### Interpreting Results

**Good Results (Expected):**
- WASM is 2-5x slower than native
- Both generate valid addresses
- Model compatibility works

**Concerning Results:**
- WASM is >10x slower (may indicate configuration issue)
- Test panics (likely missing WASM component)
- Addresses are invalid (algorithm bug)

### Troubleshooting

**Error: "WASM component not found"**
```bash
cd ../../components/entropy-ip-wasm
cargo build --target wasm32-wasip2 --release
```

**Error: "Training failed"**
- Check that tga crate dependencies are available
- Ensure sufficient memory (training 10K addresses needs ~100-500MB)

**Slow performance**
- Make sure you're running in `--release` mode
- Check that WASM component was built with `--release`
- Verify Wasmtime engine optimizations are enabled

### Advanced Usage

Customize the benchmark parameters by editing the constants:

```rust
const NUM_TARGETS: usize = 1_000_000;  // Number of addresses to generate
const NUM_SEEDS: usize = 10_000;        // Training dataset size
```

For quick smoke tests, reduce these values:
```rust
const NUM_TARGETS: usize = 10_000;
const NUM_SEEDS: usize = 100;
```

### Continuous Integration

The test can be included in CI with:

```yaml
- name: Build WASM component
  run: |
    cd components/entropy-ip-wasm
    cargo build --target wasm32-wasip2 --release

- name: Run performance tests
  run: |
    cd cli
    cargo test --test entropy_ip_performance --release -- --ignored --nocapture
```

### Related Files

- Implementation: `../../components/entropy-ip-wasm/src/`
- Native reference: `../tga/src/other/entropy.rs`
- Adaptor: `../adaptor/src/tga_adapter.rs`
- CLI commands: `src/commands/generate.rs`, `src/commands/train.rs`
