# Origin
(very) Largely based on SimdSketch
[![crates.io](https://img.shields.io/crates/v/simd-sketch.svg)](https://crates.io/crates/simd-sketch)
[![docs.rs](https://img.shields.io/docsrs/simd-sketch.svg)](https://docs.rs/simd-sketch)

# Quickstart
This repository includes a CLI that can sketch FASTA/FASTQ files, compute Jaccard similarity
between sketches, list intersection k-mers, and emit sourmash-style JSON signatures.

## Build
This crate uses nightly features. Install Rust nightly and build the binary:

```bash
rustup toolchain install nightly
cargo +nightly build --release
```

The binary is named `Comer` and will be at:

```
./target/release/Comer
```



## Common commands
Run the CLI directly via the binary:

```bash
./target/release/Comer <subcommand> [options...]
```

### 1) Sketch inputs to `.ssketch`
Writes sketches next to the input files (unless `--no-save`).

```bash
./target/release/Comer sketch -k 21 -s 64 examples/data/sample_a.fa examples/data/sample_b.fa
```

### 2) Jaccard similarity (unweighted)
Prints a single line:

```
Jaccard: <value>
```

Example:

```bash
./target/release/Comer dist -k 21 -s 64 examples/data/sample_a.fa examples/data/sample_b.fa
```

### 3) Weighted Jaccard using bcalm/logan headers
Bcalm/logan FASTA headers use tokens like `km:f:<abundance>`, `ka:f:<abundance>`,
`km:i:<abundance>`, or `ka:i:<abundance>`.
When `--bcalm` is set, abundance is parsed from the header; if missing, it defaults to 1.
Float abundances are rounded and clamped to `u16`.

Output:
```
Weighted Jaccard: <value>
```

Example:

```bash
./target/release/Comer dist -k 21 -s 64 --bcalm --weighted examples/data/bcalm_a.fa examples/data/bcalm_b.fa
```

### 4) Triangle matrix (Phylip-style)
Outputs a lower-triangular matrix to stdout (or `--output`).

```bash
./target/release/Comer triangle -k 21 -s 64 examples/data/sample_a.fa examples/data/sample_b.fa examples/data/sample_c.fa
```

### 5) List k-mers in the intersection
Prints one packed k-mer per line (`u32` for `k<=16`, `u64` for `17<=k<=32`, `u128` for larger `k`).

```bash
./target/release/Comer intersect -k 21 -s 64 examples/data/sample_a.fa examples/data/sample_b.fa
```

### 6) Sourmash-style JSON signatures
Emits JSON with `mins` and `abundances` derived from the selected k-mers.

```bash
./target/release/Comer signature -k 21 -s 64 examples/data/sample_a.fa
```

## Notes on outputs
- **dist** prints `Jaccard: ...` or `Weighted Jaccard: ...` (when `--weighted`).
- **triangle** prints a Phylip lower-triangular similarity matrix.
- **intersect** prints packed k-mers (2-bit packed DNA).
- **signature** prints a sourmash-style JSON list of signatures.
  For `k<=16`, `mins` are `u32` numbers.
  For `17<=k<=32`, `mins` are `u64` numbers.
  For `k>32`, `mins` are decimal strings and `hash_function` is `simd-sketch-kmer128-decimal`.
- `.ssketch` files are versioned. If you have older sketches, re-run `sketch` to refresh them.
- Bucket sketches store real packed k-mers, so `k <= 63` is supported.

## Input formats
- **FASTA/FASTQ** are supported. For FASTQ, sequences are read and sketched in the same way.
- **Compression**: inputs can be plain text, `gzip` (`.gz`), `bzip2` (`.bz2`), `xz` (`.xz`), `zstd` (`.zst`/`.zstd`), or `.zip`.
  For `.zip`, the first FASTA/FASTQ-like member in the archive is used.
- **Optional sequence filters** (applied per FASTA/FASTQ record before sketching):
  - `--min-seq-len <LEN>`
  - `--max-n-fraction <FLOAT>` in `[0,1]`
  - `--max-homopolymer-len <LEN>`
  - `--min-entropy <BITS_PER_BASE>` (Shannon entropy)
  If a sequence fails multiple filters, it is assigned to the first failing reason in the order above.
  The CLI reports, for each enabled reason, how many sequences and candidate k-mers were removed.
- **Ambiguous bases**:
  - With `--filter-out-n`, any k-mer containing `N` or any non-`ACGT` base is skipped.
  - Without `--filter-out-n`, non-`ACGT` characters are lossy-mapped into 2-bit DNA codes and are not filtered.
  - Sequences are not removed solely because they contain non-`ACGTN` characters.
- **Bcalm/Logan FASTA headers**: abundance can be encoded as `km:f:<float>`, `ka:f:<float>`,
  `km:i:<int>`, or `ka:i:<int>`.
  Example header token: `>u1 LN:i:100 km:f:3.2 L:+:u2:+`


## Simd-sketch

A SIMD-accelerated library to compute two types of sketches:
- Classic bottom $s$ sketch, containing the $s$ smallest distinct k-mer hashes.
- Bucket sketch, which partitions the hashes into $s$ parts and returns the smallest
  hash in each part. (Introduced as *one permutation hashing* in Li, Owen, Zhang 2012.)

See the corresponding [blog post](https://curiouscoding.nl/posts/simd-sketch/)
for background and evaluation.

Sketching takes 2 seconds for a 3Gbp human genome. This library returns 32-bit `u32`
hashes. This means that currently it may not be very suitable for sequences that are
too close to 1Gbp in length, since the bottom hash values will be relatively dense.

**Algorithm.**
For the bottom $s$ sketch, we first collect all ``sufficiently small'' hashes
into a vector. Then, that vector is sorted and deduplicated, and the smallest
$s$ values are returned. This ensures that the runtime is $O(n + s \log s)$ when
the number of duplicate k-mers is limited.

For the bucket sketch, the classic method is to partition hashes linearly, e.g.,
for $s=2$ into the bottom and top half. Then, a single value is kept per part,
and each hash is compared against the rolling minimum of its bucket.

Instead, here we make buckets by the remainder modulo $s$. This way, we can
again pre-filter for ``sufficiently small'' values, and then only scan those for
the minimum.

In both variants, we double the ``smallness'' threshold until either $s$
distinct values are found or all $s$ buckets have a value in them.

**Formulas**
For the bottom sketch, the **Jaccard similarity** `j` is computed as follows:
1. Find the smallest `s` distinct k-mer hashes in the union of two sketches.
2. Return the fraction of these k-mers that occurs in both sketches.

For the bucket sketch, we store the actual selected k-mers (as packed `u64` values)
and compute **real k-mer Jaccard** directly from the intersection/union of those
selected k-mers. When abundances are provided, **weighted Jaccard** is computed as:

```
sum_k min(a_k, b_k) / sum_k max(a_k, b_k)
```

The CLI reports **Jaccard similarity** (weighted or unweighted), not Mash distance.

**Implementation notes.**
Good performance is mostly achieved by using a branch-free implementation: all
hashes are computed using 8 parallel streams using SIMD, and appended to a vector when they
are sufficiently small to likely be part of the sketch.

The underlying streaming and hashing algorithms are described in the following [preprint](https://doi.org/10.1101/2025.01.27.634998):

- SimdMinimizers: Computing random minimizers, fast.
  Ragnar Groot Koerkamp, Igor Martayan
  bioRxiv 2025.01.27 [doi.org/10.1101/2025.01.27.634998](https://doi.org/10.1101/2025.01.27.634998)
