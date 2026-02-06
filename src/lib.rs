#![feature(hash_set_entry)]
//! # SimdSketch
//!
//! This library provides two types of sequence sketches:
//! - the classic bottom-`s` sketch;
//! - the newer bucket sketch, returning the smallest hash in each of `s` buckets.
//!
//! See the corresponding [blogpost](https://curiouscoding.nl/posts/simd-sketch/) for more background and an evaluation.
//!
//! ## Hash function
//! All internal hashes are 32 bits. Either a forward-only hash or
//! reverse-complement-aware (canonical) hash can be used.
//!
//! **TODO:** Current we use (canonical) ntHash. This causes some hash-collisions
//! for `k <= 16`, [which could be avoided](https://curiouscoding.nl/posts/nthash/#is-nthash-injective-on-kmers).
//!
//! ## BucketSketch
//! For classic bottom-sketch, evaluating the similarity is slow because a
//! merge-sort must be done between the two lists.
//!
//! The bucket sketch solves this by partitioning the hashes into `s` partitions.
//! Previous methods partition into ranges of size `u32::MAX/s`, but here we
//! partition by remainder mod `s` instead.
//!
//! We find the smallest hash for each remainder as the sketch.
//! To compute the similarity, we can simply use the hamming distance between
//! two sketches, which is significantly faster.
//!
//! The bucket sketch similarity has a very strong one-to-one correlation with the classic bottom-sketch.
//!
//! **TODO:** A drawback of this method is that some buckets may remain empty
//! when the input sequences are not long enough.  In that case, _densification_
//! could be applied, but this is not currently implemented. If you need this, please reach out.
//! Instead, we currently simply keep a bitvector indicating empty buckets.
//!
//! ## Jaccard similarity
//! For the bottom sketch, we conceptually estimate similarity as follows:
//! 1. Find the smallest `s` distinct k-mer hashes in the union of two sketches.
//! 2. Return the fraction of these k-mers that occurs in both sketches.
//!
//! For the bucket sketch, we simply return the fraction of parts that have
//! the same k-mer for both sequences (out of those that are not both empty).
//!
//! ## b-bit sketches
//!
//! Instead of storing the full 32-bit hashes, it is sufficient to only store the low bits of each hash.
//! In practice, `b=8` is usually fine.
//! When extra fast comparisons are needed, use `b=1` in combination with a 3 to 4x larger `s`.
//!
//! This causes around `1/2^b` matches because of collisions in the lower bits.
//! We correct for this via `j = (j0 - 1/2^b) / (1 - 1/2^b)`.
//! When the fraction of matches is less than `1/2^b`, this is negative, which we explicitly correct to `0`.
//!
//! ## Mash distance
//! We compute the mash distance as `-log( 2*j / (1+j) ) / k`.
//! This is always >=0, but can be as large as `inf` when `j=0` (as is the case for disjoint input sets).
//!
//! ## Usage
//!
//! The main entrypoint of this library is the [`Sketcher`] object.
//! Construct it in either the forward or canonical variant, and give `k` and `s`.
//! Then call either [`Sketcher::bottom_sketch`] or [`Sketcher::sketch`] on it, and use the
//! `similarity` functions on the returned [`BottomSketch`] and [`BucketSketch`] objects.
//!
//! ```
//! use packed_seq::SeqVec;
//!
//! let sketcher = simd_sketch::SketchParams {
//!     alg: simd_sketch::SketchAlg::Bucket,
//!     rc: false,  // Set to `true` for a canonical (reverse-complement-aware) hash.
//!     k: 31,      // Hash 31-mers
//!     s: 8192,    // Sample 8192 hashes
//!     b: 8,       // Store the bottom 8 bits of each hash.
//!     seed: 0,
//!     count: 0,
//!     coverage: 1,
//!     filter_empty: true, // Explicitly filter out empty buckets for BucketSketch.
//!     filter_out_n: false, // Set to true to ignore k-mers with `N` or other non-ACTG bases.
//! }.build();
//!
//! // Generate two random sequences of 2M characters.
//! let n = 2_000_000;
//! let seq1 = packed_seq::PackedSeqVec::random(n);
//! let seq2 = packed_seq::PackedSeqVec::random(n);
//!
//! // Sketch using given algorithm:
//! let sketch1: simd_sketch::Sketch = sketcher.sketch(seq1.as_slice());
//! let sketch2: simd_sketch::Sketch = sketcher.sketch(seq2.as_slice());
//!
//! // Value between 0 and 1, estimating the fraction of shared k-mers.
//! let j = sketch1.jaccard_similarity(&sketch2);
//! assert!(0.0 <= j && j <= 1.0);
//!
//! let d = sketch1.mash_distance(&sketch2);
//! assert!(0.0 <= d);
//! ```
//!
//! **TODO:** Currently there is no support yet for merging sketches, or for
//! sketching multiple sequences into one sketch. It's not hard, I just need to find a good API.
//! Please reach out if you're interested in this.
//!
//! **TODO:** If you would like a binary instead of a library, again, please reach out :)
//!
//! ## Implementation notes
//!
//! This library works by partitioning the input sequence into 8 chunks,
//! and processing those in parallel using SIMD.
//! This is based on the [`packed-seq`](../packed_seq/index.html) and [`seq-hash`](../seq-hash/index.html) crates
//! that were originally developed for [`simd-minimizers`](../simd_minimizers/index.html).
//!
//! For bottom sketch, the largest hash should be around `target = u32::MAX * s / n` (ignoring duplicates).
//! To ensure a branch-free algorithm, we first collect all hashes up to `bound = 1.5 * target`.
//! Then we sort the collected hashes. If there are at least `s` left after deduplicating, we return the bottom `s`.
//! Otherwise, we double the `1.5` multiplication factor and retry. This
//! factor is cached to make the sketching of multiple genomes more efficient.
//!
//! For bucket sketch, we use the same approach, and increase the factor until we find a k-mer hash in every bucket.
//! In expectation, this needs to collect a fraction around `log(n) * s / n` of hashes, rather than `s / n`.
//! In practice this doesn't matter much, as the hashing of all input k-mers is the bottleneck,
//! and the sorting of the small sample of k-mers is relatively fast.
//!
//! For bucket sketch we assign each element to its bucket via its remainder modulo `s`.
//! We compute this efficiently using [fast-mod](https://github.com/lemire/fastmod/blob/master/include/fastmod.h).
//!
//! ## Performance
//!
//! The sketching throughput of this library is around 2 seconds for a 3GB human genome
//! (once the scaling factor is large enough to avoid a second pass).
//! That's typically a few times faster than parsing a Fasta file.
//!
//! [BinDash](https://github.com/zhaoxiaofei/bindash) instead takes 180s (90x
//! more), when running on a single thread.
//!
//! Comparing sketches is relatively fast, but can become a bottleneck when there are many input sequences,
//! since the number of comparisons grows quadratically. In this case, prefer bucket sketch.
//! As an example, when sketching 5MB bacterial genomes using `s=10000`, each sketch takes 4ms.
//! Comparing two sketches takes 1.6us.
//! This starts to be the dominant factor when the number of input sequences is more than 5000.

pub mod classify;
mod intrinsics;

use std::{
    collections::HashMap,
    fmt,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

use itertools::Itertools;
use log::debug;
use packed_seq::{ChunkIt, PackedNSeq, PaddedIt, Seq, u32x8};
use seq_hash::KmerHasher;

/// Use the classic rotate-by-1 for backwards compatibility.
type FwdNtHasher = seq_hash::NtHasher<false, 1>;
type RcNtHasher = seq_hash::NtHasher<true, 1>;

#[derive(bincode::Encode, bincode::Decode, Debug, mem_dbg::MemSize)]
pub enum Sketch {
    BottomSketch(BottomSketch),
    BucketSketch(BucketSketch),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PackedKmer {
    U32(u32),
    U64(u64),
    U128(u128),
}

impl fmt::Display for PackedKmer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PackedKmer::U32(kmer) => write!(f, "{kmer}"),
            PackedKmer::U64(kmer) => write!(f, "{kmer}"),
            PackedKmer::U128(kmer) => write!(f, "{kmer}"),
        }
    }
}

fn compute_mash_distance(j: f32, k: usize) -> f32 {
    assert!(j >= 0.0, "Jaccard similarity {j} should not be negative");
    // See eq. 4 of mash paper.
    let mash_dist = -(2. * j / (1. + j)).ln() / k as f32;
    assert!(
        mash_dist >= 0.0,
        "Bad mash distance {mash_dist} for jaccard similarity {j}"
    );
    // NOTE: Mash distance can be >1 when jaccard similarity is close to 0.
    // assert!(
    //     mash_dist <= 1.0,
    //     "Bad mash distance {mash_dist} for jaccard similarity {j}"
    // );
    // Distance 0 is computed as -log(1) and becomes -0.0.
    // This maximum fixes that.
    mash_dist.max(0.0)
}

impl Sketch {
    pub fn to_params(&self) -> SketchParams {
        match self {
            Sketch::BottomSketch(sketch) => SketchParams {
                alg: SketchAlg::Bottom,
                rc: sketch.rc,
                k: sketch.k,
                s: sketch.bottom.len(),
                b: 0,
                seed: 0,
                count: sketch.count,
                coverage: 1,
                filter_empty: false,
                filter_out_n: false, // FIXME
            },
            Sketch::BucketSketch(sketch) => SketchParams {
                alg: SketchAlg::Bucket,
                rc: sketch.rc,
                k: sketch.k,
                s: sketch.buckets.len(),
                b: sketch.b,
                seed: 0,
                count: sketch.count,
                coverage: 1,
                filter_empty: false,
                filter_out_n: false, // FIXME
            },
        }
    }
    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        match (self, other) {
            (Sketch::BottomSketch(a), Sketch::BottomSketch(b)) => a.jaccard_similarity(b),
            (Sketch::BucketSketch(a), Sketch::BucketSketch(b)) => a.jaccard_similarity(b),
            _ => panic!("Sketches are of different types!"),
        }
    }
    pub fn weighted_jaccard_similarity(&self, other: &Self) -> f32 {
        match (self, other) {
            (Sketch::BucketSketch(a), Sketch::BucketSketch(b)) => a.weighted_jaccard_similarity(b),
            _ => panic!("Weighted Jaccard only supported for bucket sketches."),
        }
    }
    pub fn mash_distance(&self, other: &Self) -> f32 {
        let j = self.jaccard_similarity(other);
        let k = match self {
            Sketch::BottomSketch(sketch) => sketch.k,
            Sketch::BucketSketch(sketch) => sketch.k,
        };
        compute_mash_distance(j, k)
    }
    pub fn mash_distance_weighted(&self, other: &Self) -> f32 {
        let j = self.weighted_jaccard_similarity(other);
        let k = match self {
            Sketch::BottomSketch(sketch) => sketch.k,
            Sketch::BucketSketch(sketch) => sketch.k,
        };
        compute_mash_distance(j, k)
    }

    pub fn intersection_kmers(&self, other: &Self) -> Vec<PackedKmer> {
        match (self, other) {
            (Sketch::BucketSketch(a), Sketch::BucketSketch(b)) => a.intersection_kmers(b),
            _ => panic!("Intersection only supported for bucket sketches."),
        }
    }
}

/// Store only the bottom b bits of each input value.
#[derive(bincode::Encode, bincode::Decode, Debug, mem_dbg::MemSize)]
pub enum BitSketch {
    B32(Vec<u32>),
    B16(Vec<u16>),
    B8(Vec<u8>),
    B1(Vec<u64>),
}

impl BitSketch {
    fn new(b: usize, vals: &Vec<u32>) -> Self {
        match b {
            32 => BitSketch::B32(vals.clone()),
            16 => BitSketch::B16(vals.iter().map(|x| *x as u16).collect()),
            8 => BitSketch::B8(vals.iter().map(|x| *x as u8).collect()),
            1 => BitSketch::B1({
                assert_eq!(vals.len() % 64, 0);
                vals.chunks_exact(64)
                    .map(|xs| {
                        xs.iter()
                            .enumerate()
                            .fold(0u64, |bits, (i, x)| bits | (((x & 1) as u64) << i))
                    })
                    .collect()
            }),
            _ => panic!("Unsupported bit width. Must be 1 or 8 or 16 or 32."),
        }
    }

    fn len(&self) -> usize {
        match self {
            BitSketch::B32(v) => v.len(),
            BitSketch::B16(v) => v.len(),
            BitSketch::B8(v) => v.len(),
            BitSketch::B1(v) => 64 * v.len(),
        }
    }
}

/// A sketch containing the `s` smallest k-mer hashes.
#[derive(bincode::Encode, bincode::Decode, Debug, mem_dbg::MemSize)]
pub struct BottomSketch {
    pub rc: bool,
    pub k: usize,
    pub seed: u32,
    pub count: usize,
    pub bottom: Vec<u32>,
}

impl BottomSketch {
    /// Compute the similarity between two `BottomSketch`es.
    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.rc, other.rc);
        assert_eq!(self.k, other.k);
        let a = &self.bottom;
        let b = &other.bottom;
        assert_eq!(a.len(), b.len());
        let mut intersection_size = 0;
        let mut union_size = 0;
        let mut i = 0;
        let mut j = 0;
        while union_size < a.len() {
            intersection_size += (a[i] == b[j]) as usize;
            let di = (a[i] <= b[j]) as usize;
            let dj = (a[i] >= b[j]) as usize;
            i += di;
            j += dj;
            union_size += 1;
        }

        return intersection_size as f32 / a.len() as f32;
    }

    pub fn mash_distance(&self, other: &Self) -> f32 {
        let j = self.jaccard_similarity(other);
        compute_mash_distance(j, self.k)
    }
}

/// A sketch containing the smallest k-mer hash for each remainder mod `s`.
#[derive(bincode::Encode, bincode::Decode, Debug, mem_dbg::MemSize)]
pub struct BucketSketch {
    pub rc: bool,
    pub k: usize,
    pub b: usize,
    pub seed: u32,
    pub count: usize,
    pub buckets: BitSketch,
    /// Bit-vector indicating empty buckets, so the similarity score can be adjusted accordingly.
    pub empty: Vec<u64>,
    /// Selected k-mers for each bucket (`u32::MAX`/`u64::MAX`/`u128::MAX` when empty).
    pub kmers: BucketKmers,
    /// Abundance of each selected k-mer (defaults to 1).
    pub abundances: Vec<u16>,
}

#[derive(bincode::Encode, bincode::Decode, Debug, mem_dbg::MemSize)]
pub enum BucketKmers {
    U32(Vec<u32>),
    U64(Vec<u64>),
    U128(Vec<u128>),
}

impl BucketKmers {
    fn len(&self) -> usize {
        match self {
            BucketKmers::U32(v) => v.len(),
            BucketKmers::U64(v) => v.len(),
            BucketKmers::U128(v) => v.len(),
        }
    }
}

fn jaccard_similarity_impl<T: Copy + Eq + std::hash::Hash>(a: &[T], b: &[T], empty: T) -> f32 {
    let mut set = std::collections::HashSet::with_capacity(a.len());
    for &kmer in a {
        if kmer != empty {
            set.insert(kmer);
        }
    }
    let mut intersection = 0usize;
    let mut other_size = 0usize;
    for &kmer in b {
        if kmer == empty {
            continue;
        }
        other_size += 1;
        if set.contains(&kmer) {
            intersection += 1;
        }
    }
    let union = set.len() + other_size - intersection;
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

fn weighted_jaccard_similarity_impl<T: Copy + Eq + std::hash::Hash>(
    a: &[T],
    a_weights: &[u16],
    b: &[T],
    b_weights: &[u16],
    empty: T,
) -> f32 {
    let mut weights = std::collections::HashMap::<T, u16>::with_capacity(a.len());
    for (&kmer, &w) in a.iter().zip(a_weights.iter()) {
        if kmer == empty {
            continue;
        }
        weights.entry(kmer).or_insert(w);
    }

    let mut intersection: u64 = 0;
    let mut union: u64 = 0;
    for (&kmer, &w_other) in b.iter().zip(b_weights.iter()) {
        if kmer == empty {
            continue;
        }
        if let Some(w_self) = weights.remove(&kmer) {
            let min = w_self.min(w_other) as u64;
            let max = w_self.max(w_other) as u64;
            intersection += min;
            union += max;
        } else {
            union += w_other as u64;
        }
    }
    for &w in weights.values() {
        union += w as u64;
    }
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

fn intersection_kmers_impl<T: Copy + Ord + Eq + std::hash::Hash>(
    a: &[T],
    b: &[T],
    empty: T,
    wrap: impl Fn(T) -> PackedKmer,
) -> Vec<PackedKmer> {
    let mut set = std::collections::HashSet::with_capacity(a.len());
    for &kmer in a {
        if kmer != empty {
            set.insert(kmer);
        }
    }
    let mut out = Vec::new();
    for &kmer in b {
        if kmer == empty {
            continue;
        }
        if set.contains(&kmer) {
            out.push(kmer);
        }
    }
    out.sort_unstable();
    out.dedup();
    out.into_iter().map(wrap).collect()
}

impl BucketSketch {
    /// Compute the similarity between two `BucketSketch`es.
    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.rc, other.rc);
        assert_eq!(self.k, other.k);
        match (&self.kmers, &other.kmers) {
            (BucketKmers::U32(a), BucketKmers::U32(b)) => jaccard_similarity_impl(a, b, u32::MAX),
            (BucketKmers::U64(a), BucketKmers::U64(b)) => jaccard_similarity_impl(a, b, u64::MAX),
            (BucketKmers::U128(a), BucketKmers::U128(b)) => {
                jaccard_similarity_impl(a, b, u128::MAX)
            }
            _ => {
                panic!("Bucket sketches use different internal k-mer widths.");
            }
        }
    }

    /// Weighted Jaccard similarity using k-mer abundances.
    pub fn weighted_jaccard_similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.rc, other.rc);
        assert_eq!(self.k, other.k);
        match (&self.kmers, &other.kmers) {
            (BucketKmers::U32(a), BucketKmers::U32(b)) => weighted_jaccard_similarity_impl(
                a,
                &self.abundances,
                b,
                &other.abundances,
                u32::MAX,
            ),
            (BucketKmers::U64(a), BucketKmers::U64(b)) => weighted_jaccard_similarity_impl(
                a,
                &self.abundances,
                b,
                &other.abundances,
                u64::MAX,
            ),
            (BucketKmers::U128(a), BucketKmers::U128(b)) => weighted_jaccard_similarity_impl(
                a,
                &self.abundances,
                b,
                &other.abundances,
                u128::MAX,
            ),
            _ => {
                panic!("Bucket sketches use different internal k-mer widths.");
            }
        }
    }

    pub fn intersection_kmers(&self, other: &Self) -> Vec<PackedKmer> {
        assert_eq!(self.rc, other.rc);
        assert_eq!(self.k, other.k);
        match (&self.kmers, &other.kmers) {
            (BucketKmers::U32(a), BucketKmers::U32(b)) => {
                intersection_kmers_impl(a, b, u32::MAX, PackedKmer::U32)
            }
            (BucketKmers::U64(a), BucketKmers::U64(b)) => {
                intersection_kmers_impl(a, b, u64::MAX, PackedKmer::U64)
            }
            (BucketKmers::U128(a), BucketKmers::U128(b)) => {
                intersection_kmers_impl(a, b, u128::MAX, PackedKmer::U128)
            }
            _ => {
                panic!("Bucket sketches use different internal k-mer widths.");
            }
        }
    }

    pub fn mash_distance(&self, other: &Self) -> f32 {
        let j = self.jaccard_similarity(other);
        compute_mash_distance(j, self.k)
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, Eq, PartialEq)]
pub enum SketchAlg {
    Bottom,
    Bottom2,
    Bottom3,
    Bucket,
}

#[derive(clap::Args, Copy, Clone, Debug, Eq, PartialEq)]
pub struct SketchParams {
    /// Sketch algorithm to use. Defaults to bucket because of its much faster comparisons.
    #[arg(long, default_value_t = SketchAlg::Bucket)]
    #[arg(value_enum)]
    pub alg: SketchAlg,
    /// When set, use forward instead of canonical k-mer hashes.
    #[arg(
        long="fwd",
        num_args(0),
        action = clap::builder::ArgAction::Set,
        default_value_t = true,
        default_missing_value = "false",
    )]
    pub rc: bool,
    /// k-mer size.
    #[arg(short, default_value_t = 31)]
    pub k: usize,
    /// Bottom-s sketch, or number of buckets.
    #[arg(short, default_value_t = 10000)]
    pub s: usize,
    /// For bucket-sketch, store only the lower b bits.
    #[arg(short, default_value_t = 8)]
    pub b: usize,
    /// Seed for the hasher.
    #[arg(long, default_value_t = 0)]
    pub seed: u32,

    /// Sketch only kmers with at least this count.
    #[arg(long, default_value_t = 0)]
    pub count: usize,
    /// When sketching read sets of coverage >1, set this for a better initial estimate for the threshold on kmer hashes.
    #[arg(short, long, default_value_t = 1)]
    pub coverage: usize,

    /// For bucket-sketch, store a bitmask of empty buckets, to increase accuracy on small genomes.
    #[arg(skip = true)]
    pub filter_empty: bool,

    #[arg(long)]
    pub filter_out_n: bool,
}

/// An object containing the sketch parameters.
///
/// Contains internal state to optimize the implementation when sketching multiple similar sequences.
pub struct Sketcher {
    params: SketchParams,
    rc_hasher: RcNtHasher,
    fwd_hasher: FwdNtHasher,
    factor: AtomicU64,
}

impl SketchParams {
    pub fn build(&self) -> Sketcher {
        let mut params = *self;
        // factor is pre-multiplied by 10 for a bit more fine-grained resolution.
        let factor;
        match params.alg {
            SketchAlg::Bottom | SketchAlg::Bottom2 | SketchAlg::Bottom3 => {
                // Clear out redundant value.
                params.b = 0;
                factor = 13; // 1.3 * 10
            }
            SketchAlg::Bucket => {
                // To fill s buckets, we need ln(s)*s elements.
                // lg(s) is already a bit larger.
                factor = params.s.ilog2() as u64 * 10; // 1.0 * 10
            }
        }
        if params.alg == SketchAlg::Bottom {}
        Sketcher {
            params,
            rc_hasher: RcNtHasher::new_with_seed(params.k, params.seed),
            fwd_hasher: FwdNtHasher::new_with_seed(params.k, params.seed),
            factor: AtomicU64::new(factor),
        }
    }

    /// Default sketcher that very fast at comparisons, but 20% slower at sketching.
    /// Use for >= 50000 seqs, and safe default when input sequences are > 500'000 characters.
    ///
    /// When sequences are < 100'000 characters, inaccuracies may occur due to empty buckets.
    pub fn default(k: usize) -> Self {
        SketchParams {
            alg: SketchAlg::Bucket,
            rc: true,
            k,
            s: 32768,
            b: 1,
            seed: 0,
            count: 0,
            coverage: 1,
            filter_empty: true,
            filter_out_n: false,
        }
    }

    /// Default sketcher that is fast at sketching, but somewhat slower at comparisons.
    /// Use for <= 5000 seqs, or when input sequences are < 100'000 characters.
    pub fn default_fast_sketching(k: usize) -> Self {
        SketchParams {
            alg: SketchAlg::Bucket,
            rc: true,
            k,
            s: 8192,
            b: 8,
            seed: 0,
            count: 0,
            coverage: 1,
            filter_empty: false,
            filter_out_n: false,
        }
    }
}

impl Sketcher {
    pub fn params(&self) -> &SketchParams {
        &self.params
    }

    /// Sketch a single sequence.
    pub fn sketch(&self, seq: impl Sketchable + KmerSource) -> Sketch {
        self.sketch_seqs(&[seq])
    }

    /// Sketch multiple sequence (fasta records) into a single sketch.
    pub fn sketch_seqs<'s, S: Sketchable + KmerSource>(&self, seqs: &[S]) -> Sketch {
        match self.params.alg {
            SketchAlg::Bottom => Sketch::BottomSketch(self.bottom_sketch(seqs)),
            SketchAlg::Bottom2 => Sketch::BottomSketch(self.bottom_sketch_2(seqs)),
            SketchAlg::Bottom3 => Sketch::BottomSketch(self.bottom_sketch_3(seqs)),
            SketchAlg::Bucket => {
                let abundances = vec![1u16; seqs.len()];
                Sketch::BucketSketch(self.bucket_sketch_with_abundances(seqs, &abundances))
            }
        }
    }

    pub fn sketch_seqs_with_abundances<'s, S: Sketchable + KmerSource>(
        &self,
        seqs: &[S],
        abundances: &[u16],
    ) -> Sketch {
        match self.params.alg {
            SketchAlg::Bottom => Sketch::BottomSketch(self.bottom_sketch(seqs)),
            SketchAlg::Bottom2 => Sketch::BottomSketch(self.bottom_sketch_2(seqs)),
            SketchAlg::Bottom3 => Sketch::BottomSketch(self.bottom_sketch_3(seqs)),
            SketchAlg::Bucket => {
                Sketch::BucketSketch(self.bucket_sketch_with_abundances(seqs, abundances))
            }
        }
    }

    fn num_kmers<'s>(&self, seqs: &[impl Sketchable]) -> usize {
        seqs.iter()
            .map(|seq| seq.len() - self.params.k + 1)
            .sum::<usize>()
    }

    /// Return the `s` smallest `u32` k-mer hashes.
    fn bottom_sketch<'s>(&self, seqs: &[impl Sketchable]) -> BottomSketch {
        // Iterate all kmers and compute 32bit nthashes.
        let n = self.num_kmers(seqs);
        let mut out = vec![];
        loop {
            // The total number of kmers is roughly n/coverage.
            // We want s of those, so scale u32::MAX by s/(n/coverage).
            let target = u32::MAX as usize * self.params.s / (n / self.params.coverage);
            let factor = self.factor.load(Relaxed);
            let bound = (target as u128 * factor as u128 / 10 as u128).min(u32::MAX as u128) as u32;

            self.collect_up_to_bound(seqs, bound, &mut out, usize::MAX, |_| 0);

            if bound == u32::MAX || out.len() >= self.params.s {
                out.sort_unstable();
                let old_len = out.len();
                let new_len = 1 + out
                    .array_windows()
                    .map(|[l, r]| (l != r) as usize)
                    .sum::<usize>();

                debug!("Deduplicated from {old_len} to {new_len}");
                if bound == u32::MAX || new_len >= self.params.s {
                    out.dedup();
                    out.resize(self.params.s, u32::MAX);

                    if log::log_enabled!(log::Level::Debug) {
                        let mut counts = vec![];
                        for g in out.iter().chunk_by(|x| **x).into_iter() {
                            counts.push(g.1.count() as u16);
                        }

                        self.stats(n, &out, counts);
                    }

                    return BottomSketch {
                        rc: self.params.rc,
                        k: self.params.k,
                        seed: self.params.seed,
                        count: self.params.count,
                        bottom: out,
                    };
                }
            }

            let new_factor = factor + factor.div_ceil(4);
            let prev = self.factor.fetch_max(new_factor, Relaxed);
            debug!(
                "Found only {:>10} of {:>10} ({:>6.3}%)) Increasing factor from {factor} to {new_factor} (was already {prev})",
                out.len(),
                self.params.s,
                out.len() as f32 / self.params.s as f32,
            );
        }
    }

    /// Return the `s` smallest `u32` k-mer hashes.
    fn bottom_sketch_2<'s>(&self, seqs: &[impl Sketchable]) -> BottomSketch {
        // Iterate all kmers and compute 32bit nthashes.
        let n = self.num_kmers(seqs);
        let mut out = vec![];

        self.collect_up_to_bound(seqs, u32::MAX, &mut out, 2 * self.params.s, |out| {
            // TODO: Can this be made more efficient?
            let l0 = out.len();
            out.sort_unstable();
            out.dedup();
            let l1 = out.len();
            out.truncate(self.params.s);
            let l2 = out.len();
            let bound = out.get(self.params.s - 1).copied().unwrap_or(u32::MAX);
            debug!("Len before {l0} => dedup {l1} => truncate {l2}. New bound {bound}");
            bound
        });

        let l0 = out.len();
        out.sort_unstable();
        out.dedup();
        let l1 = out.len();
        debug!("Len before {l0} => after {l1}.");
        out.resize(self.params.s, u32::MAX);

        if log::log_enabled!(log::Level::Debug) {
            let mut counts = vec![];
            for g in out.iter().chunk_by(|x| **x).into_iter() {
                counts.push(g.1.count() as u16);
            }

            self.stats(n, &out, counts);
        }

        return BottomSketch {
            rc: self.params.rc,
            k: self.params.k,
            seed: self.params.seed,
            count: self.params.count,
            bottom: out,
        };
    }

    /// Return the `s` smallest `u32` k-mer hashes.
    fn bottom_sketch_3<'s>(&self, seqs: &[impl Sketchable]) -> BottomSketch {
        let out = new_collect(
            self.params.s,
            self.params.count,
            self.params.coverage,
            seqs.iter().map(|seq| seq.hash_kmers(&self.rc_hasher)),
        );

        return BottomSketch {
            rc: self.params.rc,
            k: self.params.k,
            seed: self.params.seed,
            count: self.params.count,
            bottom: out,
        };
    }

    /// s-buckets sketch. Splits the hashes into `s` buckets and returns the smallest hash per bucket.
    /// Buckets are determined via the remainder mod `s`.
    fn bucket_sketch_with_abundances<'s, S: Sketchable + KmerSource>(
        &self,
        seqs: &[S],
        abundances: &[u16],
    ) -> BucketSketch {
        assert_eq!(seqs.len(), abundances.len());
        assert!(
            self.params.k <= 63,
            "k must be <= 63 to store packed k-mers in u128"
        );

        if self.params.k <= 16 {
            self.bucket_sketch_with_abundances_u32(seqs, abundances)
        } else if self.params.k <= 32 {
            self.bucket_sketch_with_abundances_u64(seqs, abundances)
        } else {
            self.bucket_sketch_with_abundances_u128(seqs, abundances)
        }
    }

    #[inline(always)]
    fn for_each_kmer_hash_and_value_u32<S: KmerSource, F: FnMut(u32, u32)>(&self, seq: S, f: F) {
        if self.params.rc {
            seq.for_each_kmer_hash_and_value_u32(
                &self.rc_hasher,
                self.params.k,
                self.params.rc,
                self.params.filter_out_n,
                f,
            );
        } else {
            seq.for_each_kmer_hash_and_value_u32(
                &self.fwd_hasher,
                self.params.k,
                self.params.rc,
                self.params.filter_out_n,
                f,
            );
        }
    }

    #[inline(always)]
    fn for_each_kmer_hash_and_value_u64<S: KmerSource, F: FnMut(u32, u64)>(&self, seq: S, f: F) {
        if self.params.rc {
            seq.for_each_kmer_hash_and_value_u64(
                &self.rc_hasher,
                self.params.k,
                self.params.rc,
                self.params.filter_out_n,
                f,
            );
        } else {
            seq.for_each_kmer_hash_and_value_u64(
                &self.fwd_hasher,
                self.params.k,
                self.params.rc,
                self.params.filter_out_n,
                f,
            );
        }
    }

    #[inline(always)]
    fn for_each_kmer_hash_and_value_u128<S: KmerSource, F: FnMut(u32, u128)>(&self, seq: S, f: F) {
        if self.params.rc {
            seq.for_each_kmer_hash_and_value_u128(
                &self.rc_hasher,
                self.params.k,
                self.params.rc,
                self.params.filter_out_n,
                f,
            );
        } else {
            seq.for_each_kmer_hash_and_value_u128(
                &self.fwd_hasher,
                self.params.k,
                self.params.rc,
                self.params.filter_out_n,
                f,
            );
        }
    }

    fn bucket_sketch_with_abundances_u32<'s, S: Sketchable + KmerSource>(
        &self,
        seqs: &[S],
        abundances: &[u16],
    ) -> BucketSketch {
        let mut buckets = vec![u32::MAX; self.params.s];
        let mut kmers = vec![u32::MAX; self.params.s];
        let mut kmer_abundances = vec![1u16; self.params.s];
        let m = FM32::new(self.params.s as u32);

        if self.params.count <= 1 {
            for (&seq, &abundance) in seqs.iter().zip(abundances.iter()) {
                self.for_each_kmer_hash_and_value_u32(seq, |hash, kmer| {
                    let bucket = m.fastmod(hash);
                    let min = unsafe { buckets.get_unchecked_mut(bucket) };
                    if hash < *min {
                        *min = hash;
                        unsafe {
                            *kmers.get_unchecked_mut(bucket) = kmer;
                            *kmer_abundances.get_unchecked_mut(bucket) = abundance;
                        }
                    }
                });
            }
        } else {
            let mut counts = HashMap::<u32, (u32, u32, u16)>::new();
            for (&seq, &abundance) in seqs.iter().zip(abundances.iter()) {
                self.for_each_kmer_hash_and_value_u32(seq, |hash, kmer| {
                    let entry = counts.entry(hash).or_insert((0, kmer, abundance));
                    if entry.0 == 0 {
                        entry.1 = kmer;
                        entry.2 = abundance;
                    }
                    entry.0 = entry.0.saturating_add(1);
                    if entry.0 == self.params.count as u32 {
                        let bucket = m.fastmod(hash);
                        let min = unsafe { buckets.get_unchecked_mut(bucket) };
                        if hash < *min {
                            *min = hash;
                            unsafe {
                                *kmers.get_unchecked_mut(bucket) = entry.1;
                                *kmer_abundances.get_unchecked_mut(bucket) = entry.2;
                            }
                        }
                    }
                });
            }
        }

        self.finalize_bucket_sketch(buckets, BucketKmers::U32(kmers), kmer_abundances)
    }

    fn bucket_sketch_with_abundances_u64<'s, S: Sketchable + KmerSource>(
        &self,
        seqs: &[S],
        abundances: &[u16],
    ) -> BucketSketch {
        let mut buckets = vec![u32::MAX; self.params.s];
        let mut kmers = vec![u64::MAX; self.params.s];
        let mut kmer_abundances = vec![1u16; self.params.s];
        let m = FM32::new(self.params.s as u32);

        if self.params.count <= 1 {
            for (&seq, &abundance) in seqs.iter().zip(abundances.iter()) {
                self.for_each_kmer_hash_and_value_u64(seq, |hash, kmer| {
                    let bucket = m.fastmod(hash);
                    let min = unsafe { buckets.get_unchecked_mut(bucket) };
                    if hash < *min {
                        *min = hash;
                        unsafe {
                            *kmers.get_unchecked_mut(bucket) = kmer;
                            *kmer_abundances.get_unchecked_mut(bucket) = abundance;
                        }
                    }
                });
            }
        } else {
            let mut counts = HashMap::<u32, (u32, u64, u16)>::new();
            for (&seq, &abundance) in seqs.iter().zip(abundances.iter()) {
                self.for_each_kmer_hash_and_value_u64(seq, |hash, kmer| {
                    let entry = counts.entry(hash).or_insert((0, kmer, abundance));
                    if entry.0 == 0 {
                        entry.1 = kmer;
                        entry.2 = abundance;
                    }
                    entry.0 = entry.0.saturating_add(1);
                    if entry.0 == self.params.count as u32 {
                        let bucket = m.fastmod(hash);
                        let min = unsafe { buckets.get_unchecked_mut(bucket) };
                        if hash < *min {
                            *min = hash;
                            unsafe {
                                *kmers.get_unchecked_mut(bucket) = entry.1;
                                *kmer_abundances.get_unchecked_mut(bucket) = entry.2;
                            }
                        }
                    }
                });
            }
        }

        self.finalize_bucket_sketch(buckets, BucketKmers::U64(kmers), kmer_abundances)
    }

    fn bucket_sketch_with_abundances_u128<'s, S: Sketchable + KmerSource>(
        &self,
        seqs: &[S],
        abundances: &[u16],
    ) -> BucketSketch {
        let mut buckets = vec![u32::MAX; self.params.s];
        let mut kmers = vec![u128::MAX; self.params.s];
        let mut kmer_abundances = vec![1u16; self.params.s];
        let m = FM32::new(self.params.s as u32);

        if self.params.count <= 1 {
            for (&seq, &abundance) in seqs.iter().zip(abundances.iter()) {
                self.for_each_kmer_hash_and_value_u128(seq, |hash, kmer| {
                    let bucket = m.fastmod(hash);
                    let min = unsafe { buckets.get_unchecked_mut(bucket) };
                    if hash < *min {
                        *min = hash;
                        unsafe {
                            *kmers.get_unchecked_mut(bucket) = kmer;
                            *kmer_abundances.get_unchecked_mut(bucket) = abundance;
                        }
                    }
                });
            }
        } else {
            let mut counts = HashMap::<u32, (u32, u128, u16)>::new();
            for (&seq, &abundance) in seqs.iter().zip(abundances.iter()) {
                self.for_each_kmer_hash_and_value_u128(seq, |hash, kmer| {
                    let entry = counts.entry(hash).or_insert((0, kmer, abundance));
                    if entry.0 == 0 {
                        entry.1 = kmer;
                        entry.2 = abundance;
                    }
                    entry.0 = entry.0.saturating_add(1);
                    if entry.0 == self.params.count as u32 {
                        let bucket = m.fastmod(hash);
                        let min = unsafe { buckets.get_unchecked_mut(bucket) };
                        if hash < *min {
                            *min = hash;
                            unsafe {
                                *kmers.get_unchecked_mut(bucket) = entry.1;
                                *kmer_abundances.get_unchecked_mut(bucket) = entry.2;
                            }
                        }
                    }
                });
            }
        }

        self.finalize_bucket_sketch(buckets, BucketKmers::U128(kmers), kmer_abundances)
    }

    fn finalize_bucket_sketch(
        &self,
        mut buckets: Vec<u32>,
        kmers: BucketKmers,
        kmer_abundances: Vec<u16>,
    ) -> BucketSketch {
        let m = FM32::new(self.params.s as u32);
        assert_eq!(kmers.len(), buckets.len());
        assert_eq!(kmer_abundances.len(), buckets.len());

        let num_empty = buckets.iter().filter(|x| **x == u32::MAX).count();
        if num_empty > 0 {
            debug!("Found {num_empty} empty buckets.");
        }
        let empty = if num_empty > 0 && self.params.filter_empty {
            debug!("Found {num_empty} empty buckets. Storing bitmask.");
            buckets
                .chunks(64)
                .map(|xs| {
                    xs.iter()
                        .enumerate()
                        .fold(0u64, |bits, (i, x)| bits | (((*x == u32::MAX) as u64) << i))
                })
                .collect()
        } else {
            vec![]
        };

        // Reduce buckets mod m.
        buckets.iter_mut().for_each(|x| *x = m.fastdiv(*x) as u32);
        log::debug!(
            "Average sketch value: {}",
            buckets.iter().sum::<u32>() as f32 / self.params.s as f32
        );
        BucketSketch {
            rc: self.params.rc,
            k: self.params.k,
            b: self.params.b,
            seed: self.params.seed,
            count: self.params.count,
            empty,
            buckets: BitSketch::new(self.params.b, &buckets),
            kmers,
            abundances: kmer_abundances,
        }
    }

    fn stats(&self, num_kmers: usize, hashes: &Vec<u32>, counts: Vec<u16>) {
        let num_empty = hashes
            .iter()
            .map(|x| (*x == u32::MAX) as usize)
            .sum::<usize>();
        // Statistics
        let mut cc = HashMap::new();
        for c in counts {
            *cc.entry(c).or_insert(0) += 1;
        }
        let mut cc = cc.into_iter().collect::<Vec<_>>();
        cc.sort_unstable();
        debug!("Counts: {:?}", cc);

        // for bottom sketch:
        // - the i'th smallest value is roughly i * MAX/(n+1).
        // - averaging over i in 1..=s => * (1+s)/2.
        let expected_hash =
            u32::MAX as f64 / (num_kmers as f64 + 1.0) * (1.0 + self.params.s as f64) / 2.0;
        let average_hash = hashes
            .iter()
            .filter(|x| **x != u32::MAX)
            .map(|x| *x as f64)
            .sum::<f64>()
            / (self.params.s - num_empty) as f64;
        let harmonic_hash = (self.params.s - num_empty) as f64
            / hashes
                .iter()
                .filter(|x| **x != u32::MAX)
                .map(|x| 1.0 / (*x as f64 + 1.0))
                .sum::<f64>();
        let harmonic2_hash = self.params.s as f64 * (self.params.s - num_empty) as f64
            / hashes
                .iter()
                .filter(|x| **x != u32::MAX)
                .map(|x| 1.0 / ((*x / self.params.s as u32) as f64 + 1.0))
                .sum::<f64>();
        let median_hash = {
            let mut xs = hashes
                .iter()
                .filter(|x| **x != u32::MAX)
                .copied()
                .collect::<Vec<_>>();
            xs.sort_unstable();
            // debug!("Hashes {xs:?}");
            xs[xs.len() / 2] as f64
        };
        debug!("Expected  hash {expected_hash:>11.2}");
        debug!("Average   hash {average_hash:>11.2}");
        debug!("Harmonic  hash {harmonic_hash:>11.2}");
        debug!("Harmonic2 hash {harmonic2_hash:>11.2}");
        debug!("Median    hash {median_hash:>11.2}");
        debug!("Average ratio   {:>7.4}", average_hash / expected_hash);
        debug!("Harmonic ratio  {:>7.4}", harmonic_hash / expected_hash);
        debug!("Harmonic2 ratio {:>7.4}", harmonic2_hash / expected_hash);
        debug!("Median ratio    {:>7.4}", median_hash / expected_hash);
        let average_kmers = u32::MAX as f64 / average_hash;
        let harmonic_kmers = u32::MAX as f64 / harmonic_hash;
        let harmonic2_kmers = u32::MAX as f64 / harmonic2_hash;
        let median_kmers = u32::MAX as f64 / median_hash;
        let real_kmers = num_kmers as f64 / self.params.s as f64;
        let average_coverage = real_kmers / average_kmers;
        let harmonic_coverage = real_kmers / harmonic_kmers;
        let harmonic2_coverage = real_kmers / harmonic2_kmers;
        let median_coverage = real_kmers / median_kmers;
        debug!("Average coverage   {average_coverage:>7.4}");
        debug!("Harmonic coverage  {harmonic_coverage:>7.4}");
        debug!("Harmonic2 coverage {harmonic2_coverage:>7.4}");
        debug!("Median coverage    {median_coverage:>7.4}");
    }

    /// Collect all values `<= bound`.
    #[inline(always)]
    fn collect_up_to_bound<'s>(
        &self,
        seqs: &[impl Sketchable],
        mut bound: u32,
        out: &mut Vec<u32>,
        batch_size: usize,
        mut callback: impl FnMut(&mut Vec<u32>) -> u32,
    ) {
        out.clear();
        let it = &mut 0;
        if self.params.rc {
            for &seq in seqs {
                let hashes = seq.hash_kmers(&self.rc_hasher);
                collect_impl(&mut bound, hashes, out, batch_size, &mut callback, it);
            }
        } else {
            for &seq in seqs {
                let hashes = seq.hash_kmers(&self.fwd_hasher);
                collect_impl(&mut bound, hashes, out, batch_size, &mut callback, it);
            }
        }
        debug!(
            "Collect up to {bound:>10}: {:>9} ({:>6.3}% of all kmers)",
            out.len(),
            out.len() as f32 / self.num_kmers(seqs) as f32 * 100.0
        );
    }
}

/// Collect values <= bound.
///
/// Once `out` reaches `max_len`, `callback` will be called to update the bound.
#[inline(always)]
fn collect_impl(
    bound: &mut u32,
    hashes: PaddedIt<impl ChunkIt<u32x8>>,
    out: &mut Vec<u32>,
    batch_size: usize,
    callback: &mut impl FnMut(&mut Vec<u32>) -> u32,
    it: &mut usize,
) {
    let mut simd_bound = u32x8::splat(*bound);
    let mut write_idx = out.len();
    let lane_len = hashes.it.len();
    let mut idx = u32x8::from(std::array::from_fn(|i| (i * lane_len) as u32));
    let max_idx = (8 * lane_len - hashes.padding) as u32;
    let max_idx = u32x8::splat(max_idx);
    hashes.it.for_each(|hashes| {
        // hashes <= simd_bound
        // let mask = !simd_bound.cmp_gt(hashes);
        let mask = hashes.cmp_lt(simd_bound);
        // TODO: transform to a decreasing loop with >= 0 check.
        let in_bounds = idx.cmp_lt(max_idx);
        if write_idx + 8 > out.capacity() {
            out.reserve(out.capacity() + 8);
        }
        unsafe { intrinsics::append_from_mask(hashes, mask & in_bounds, out, &mut write_idx) };
        idx += u32x8::ONE;
        if write_idx >= batch_size as usize {
            log::trace!("CALLBACK in iteration {it} old bound {bound}");
            unsafe { out.set_len(write_idx) };
            *bound = callback(out);
            simd_bound = u32x8::splat(*bound);
            write_idx = out.len();
        }
        *it += 1;
    });

    unsafe { out.set_len(write_idx) };
    // Final call to callback to process the last batch of values.
    callback(out);
}

/// Collect the `s` smallest values that occur at least `cnt` times.
///
/// Implementation:
/// 1. Keep a bound, and collect batch of ~`s` kmers below `bound`.
/// 2. Increase counts in a hashmap for all kmers in the batch.
/// 3. Kmers with the required count are added to a priority queue, and the current `bound` is updated.
///    - Should we collect these in batches too?
/// 4. (From time to time, the hashmap with counts is pruned.)
#[inline(always)]
fn new_collect(
    s: usize,
    cnt: usize,
    coverage: usize,
    hashes: impl Iterator<Item = PaddedIt<impl ChunkIt<u32x8>>>,
) -> Vec<u32> {
    let initial_bound = u32::MAX / coverage as u32;
    let mut bound = u32x8::splat(initial_bound);

    // 1. Collect values <= bound.
    let mut buf = vec![];
    let mut write_idx = buf.len();
    let batch_size = s;

    // 2. HashMap with counts.
    let mut counts = HashMap::<u32, usize>::new();
    // let mut counts = Vec::<(u32, u32)>::new();
    // let mut counts_cache = vec![];
    // let mut counts = BTreeMap::<u32, u32>::new();

    // 3. Priority queue with smallest s elements with sufficient count.
    // Largest at the top, so they can be removed easily.
    // let mut pq = BinaryHeap::<u32>::from_iter((0..s).map(|_| initial_bound));

    let mut start = std::time::Instant::now();

    let mut process_batch = |buf: &mut Vec<u32>, write_idx: &mut usize, i: usize| {
        unsafe { buf.set_len(*write_idx) };
        buf.sort_unstable();

        let mut num_large = 0;
        let mut top = initial_bound;
        for hash in &*buf {
            let count = counts.entry(*hash).or_insert(0);
            *count += 1;
            if *count >= cnt {
                num_large += 1;
                if num_large == s {
                    top = *hash;
                    break;
                }
            }
        }

        // counts_cache.clear();
        buf.clear();

        // for &hash in &*buf {
        //     if hash < top {
        //         *counts.entry(hash).or_insert(0) += 1;
        //         if counts[&hash] == cnt {
        //             pq.pop();
        //             pq.push(hash);
        //             top = *pq.peek().unwrap();
        //             // info!("Push {hash:>10}; new top {top:>10}");
        //         }
        //     }
        // }
        let now = std::time::Instant::now();
        let duration = now.duration_since(start);
        start = now;
        log::trace!(
            "Batch of size {write_idx:>10} after {i:>10} kmers. Top: {top:>10} = {:5.1}% counts size {:>9} took {duration:?}",
            top as f32 / u32::MAX as f32 * 100.0,
            counts.len()
        );
        *write_idx = 0;

        u32x8::splat(top)
    };

    let mut i = 0;
    for hashes in hashes {
        // Prevent saving out-of-bound kmers.
        let lane_len = hashes.it.len();
        let mut idx = u32x8::from(std::array::from_fn(|i| (i * lane_len) as u32));
        let max_idx = u32x8::splat((8 * lane_len - hashes.padding) as u32);

        hashes.it.for_each(|hashes| {
            i += 8;
            // hashes <= simd_bound
            // let mask = !simd_bound.cmp_gt(hashes);
            let mask = hashes.cmp_lt(bound);
            // TODO: transform to a decreasing loop with >= 0 check.
            let in_bounds = idx.cmp_lt(max_idx);
            if write_idx + 8 > buf.capacity() {
                buf.reserve(buf.capacity() + 8);
            }
            // Note that this does not increase the length of `out`.
            unsafe {
                intrinsics::append_from_mask(hashes, mask & in_bounds, &mut buf, &mut write_idx)
            };
            idx += u32x8::ONE;
            if write_idx >= batch_size as usize {
                bound = process_batch(&mut buf, &mut write_idx, i);
                i = 0;
            }
        });
    }
    process_batch(&mut buf, &mut write_idx, i);

    // pq.into_vec()
    counts
        .iter()
        .filter(|&(_hash, count)| *count >= cnt)
        .map(|(hash, _count)| *hash)
        .collect()
}

struct KmerValueIter32<I> {
    iter: I,
    k: usize,
    rc: bool,
    mask: u32,
    fwd: u32,
    rev: u32,
    valid_len: usize,
    seen: usize,
    rev_shift: u32,
}

impl<I: Iterator<Item = (u8, bool)>> KmerValueIter32<I> {
    fn new(iter: I, k: usize, rc: bool) -> Self {
        let mask = if k >= 16 {
            u32::MAX
        } else {
            (1u32 << (2 * k)) - 1
        };
        let rev_shift = if k == 0 { 0 } else { (2 * (k - 1)) as u32 };
        Self {
            iter,
            k,
            rc,
            mask,
            fwd: 0,
            rev: 0,
            valid_len: 0,
            seen: 0,
            rev_shift,
        }
    }
}

impl<I: Iterator<Item = (u8, bool)>> Iterator for KmerValueIter32<I> {
    type Item = Option<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (base, invalid) = self.iter.next()?;
            self.seen += 1;
            if invalid {
                self.valid_len = 0;
                self.fwd = 0;
                self.rev = 0;
            } else {
                self.valid_len = (self.valid_len + 1).min(self.k);
                self.fwd = ((self.fwd << 2) | base as u32) & self.mask;
                if self.rc {
                    let comp = (base ^ 2) as u32;
                    self.rev = (self.rev >> 2) | (comp << self.rev_shift);
                }
            }

            if self.seen < self.k {
                continue;
            }
            if invalid || self.valid_len < self.k {
                return Some(None);
            }
            let kmer = if self.rc {
                self.fwd.min(self.rev)
            } else {
                self.fwd
            };
            return Some(Some(kmer));
        }
    }
}

struct KmerValueIter64<I> {
    iter: I,
    k: usize,
    rc: bool,
    mask: u64,
    fwd: u64,
    rev: u64,
    valid_len: usize,
    seen: usize,
    rev_shift: u32,
}

impl<I: Iterator<Item = (u8, bool)>> KmerValueIter64<I> {
    fn new(iter: I, k: usize, rc: bool) -> Self {
        let mask = if k >= 32 {
            u64::MAX
        } else {
            (1u64 << (2 * k)) - 1
        };
        let rev_shift = if k == 0 { 0 } else { (2 * (k - 1)) as u32 };
        Self {
            iter,
            k,
            rc,
            mask,
            fwd: 0,
            rev: 0,
            valid_len: 0,
            seen: 0,
            rev_shift,
        }
    }
}

impl<I: Iterator<Item = (u8, bool)>> Iterator for KmerValueIter64<I> {
    type Item = Option<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (base, invalid) = self.iter.next()?;
            self.seen += 1;
            if invalid {
                self.valid_len = 0;
                self.fwd = 0;
                self.rev = 0;
            } else {
                self.valid_len = (self.valid_len + 1).min(self.k);
                self.fwd = ((self.fwd << 2) | base as u64) & self.mask;
                if self.rc {
                    let comp = (base ^ 2) as u64;
                    self.rev = (self.rev >> 2) | (comp << self.rev_shift);
                }
            }

            if self.seen < self.k {
                continue;
            }
            if invalid || self.valid_len < self.k {
                return Some(None);
            }
            let kmer = if self.rc {
                self.fwd.min(self.rev)
            } else {
                self.fwd
            };
            return Some(Some(kmer));
        }
    }
}

struct KmerValueIter128<I> {
    iter: I,
    k: usize,
    rc: bool,
    mask: u128,
    fwd: u128,
    rev: u128,
    valid_len: usize,
    seen: usize,
    rev_shift: u32,
}

impl<I: Iterator<Item = (u8, bool)>> KmerValueIter128<I> {
    fn new(iter: I, k: usize, rc: bool) -> Self {
        let mask = if k >= 64 {
            u128::MAX
        } else {
            (1u128 << (2 * k)) - 1
        };
        let rev_shift = if k == 0 { 0 } else { (2 * (k - 1)) as u32 };
        Self {
            iter,
            k,
            rc,
            mask,
            fwd: 0,
            rev: 0,
            valid_len: 0,
            seen: 0,
            rev_shift,
        }
    }
}

impl<I: Iterator<Item = (u8, bool)>> Iterator for KmerValueIter128<I> {
    type Item = Option<u128>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (base, invalid) = self.iter.next()?;
            self.seen += 1;
            if invalid {
                self.valid_len = 0;
                self.fwd = 0;
                self.rev = 0;
            } else {
                self.valid_len = (self.valid_len + 1).min(self.k);
                self.fwd = ((self.fwd << 2) | base as u128) & self.mask;
                if self.rc {
                    let comp = (base ^ 2) as u128;
                    self.rev = (self.rev >> 2) | (comp << self.rev_shift);
                }
            }

            if self.seen < self.k {
                continue;
            }
            if invalid || self.valid_len < self.k {
                return Some(None);
            }
            let kmer = if self.rc {
                self.fwd.min(self.rev)
            } else {
                self.fwd
            };
            return Some(Some(kmer));
        }
    }
}

#[inline(always)]
fn pack_char_lossy_local(base: u8) -> u8 {
    (base >> 1) & 3
}

pub trait KmerSource: Copy {
    fn for_each_kmer_hash_and_value_u32<H: KmerHasher, F: FnMut(u32, u32)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        filter_out_n: bool,
        f: F,
    );

    fn for_each_kmer_hash_and_value_u64<H: KmerHasher, F: FnMut(u32, u64)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        filter_out_n: bool,
        f: F,
    );

    fn for_each_kmer_hash_and_value_u128<H: KmerHasher, F: FnMut(u32, u128)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        filter_out_n: bool,
        f: F,
    );
}

impl KmerSource for &[u8] {
    fn for_each_kmer_hash_and_value_u32<H: KmerHasher, F: FnMut(u32, u32)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        filter_out_n: bool,
        mut f: F,
    ) {
        let hash_iter = hasher.hash_kmers_scalar(packed_seq::AsciiSeq(self));
        let base_iter = self.iter().map(|&b| {
            if filter_out_n {
                match b {
                    b'A' | b'a' => (0u8, false),
                    b'C' | b'c' => (1u8, false),
                    b'G' | b'g' => (3u8, false),
                    b'T' | b't' => (2u8, false),
                    _ => (0u8, true),
                }
            } else {
                (pack_char_lossy_local(b), false)
            }
        });
        let kmer_iter = KmerValueIter32::new(base_iter, k, rc);
        for (hash, kmer) in hash_iter.zip(kmer_iter) {
            if let Some(kmer) = kmer {
                f(hash, kmer);
            }
        }
    }

    fn for_each_kmer_hash_and_value_u64<H: KmerHasher, F: FnMut(u32, u64)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        filter_out_n: bool,
        mut f: F,
    ) {
        let hash_iter = hasher.hash_kmers_scalar(packed_seq::AsciiSeq(self));
        let base_iter = self.iter().map(|&b| {
            if filter_out_n {
                match b {
                    b'A' | b'a' => (0u8, false),
                    b'C' | b'c' => (1u8, false),
                    b'G' | b'g' => (3u8, false),
                    b'T' | b't' => (2u8, false),
                    _ => (0u8, true),
                }
            } else {
                (pack_char_lossy_local(b), false)
            }
        });
        let kmer_iter = KmerValueIter64::new(base_iter, k, rc);
        for (hash, kmer) in hash_iter.zip(kmer_iter) {
            if let Some(kmer) = kmer {
                f(hash, kmer);
            }
        }
    }

    fn for_each_kmer_hash_and_value_u128<H: KmerHasher, F: FnMut(u32, u128)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        filter_out_n: bool,
        mut f: F,
    ) {
        let hash_iter = hasher.hash_kmers_scalar(packed_seq::AsciiSeq(self));
        let base_iter = self.iter().map(|&b| {
            if filter_out_n {
                match b {
                    b'A' | b'a' => (0u8, false),
                    b'C' | b'c' => (1u8, false),
                    b'G' | b'g' => (3u8, false),
                    b'T' | b't' => (2u8, false),
                    _ => (0u8, true),
                }
            } else {
                (pack_char_lossy_local(b), false)
            }
        });
        let kmer_iter = KmerValueIter128::new(base_iter, k, rc);
        for (hash, kmer) in hash_iter.zip(kmer_iter) {
            if let Some(kmer) = kmer {
                f(hash, kmer);
            }
        }
    }
}

impl KmerSource for packed_seq::AsciiSeq<'_> {
    fn for_each_kmer_hash_and_value_u32<H: KmerHasher, F: FnMut(u32, u32)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        filter_out_n: bool,
        mut f: F,
    ) {
        let hash_iter = hasher.hash_kmers_scalar(self);
        let base_iter = self.0.iter().map(|&b| {
            if filter_out_n {
                match b {
                    b'A' | b'a' => (0u8, false),
                    b'C' | b'c' => (1u8, false),
                    b'G' | b'g' => (3u8, false),
                    b'T' | b't' => (2u8, false),
                    _ => (0u8, true),
                }
            } else {
                (pack_char_lossy_local(b), false)
            }
        });
        let kmer_iter = KmerValueIter32::new(base_iter, k, rc);
        for (hash, kmer) in hash_iter.zip(kmer_iter) {
            if let Some(kmer) = kmer {
                f(hash, kmer);
            }
        }
    }

    fn for_each_kmer_hash_and_value_u64<H: KmerHasher, F: FnMut(u32, u64)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        filter_out_n: bool,
        mut f: F,
    ) {
        let hash_iter = hasher.hash_kmers_scalar(self);
        let base_iter = self.0.iter().map(|&b| {
            if filter_out_n {
                match b {
                    b'A' | b'a' => (0u8, false),
                    b'C' | b'c' => (1u8, false),
                    b'G' | b'g' => (3u8, false),
                    b'T' | b't' => (2u8, false),
                    _ => (0u8, true),
                }
            } else {
                (pack_char_lossy_local(b), false)
            }
        });
        let kmer_iter = KmerValueIter64::new(base_iter, k, rc);
        for (hash, kmer) in hash_iter.zip(kmer_iter) {
            if let Some(kmer) = kmer {
                f(hash, kmer);
            }
        }
    }

    fn for_each_kmer_hash_and_value_u128<H: KmerHasher, F: FnMut(u32, u128)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        filter_out_n: bool,
        mut f: F,
    ) {
        let hash_iter = hasher.hash_kmers_scalar(self);
        let base_iter = self.0.iter().map(|&b| {
            if filter_out_n {
                match b {
                    b'A' | b'a' => (0u8, false),
                    b'C' | b'c' => (1u8, false),
                    b'G' | b'g' => (3u8, false),
                    b'T' | b't' => (2u8, false),
                    _ => (0u8, true),
                }
            } else {
                (pack_char_lossy_local(b), false)
            }
        });
        let kmer_iter = KmerValueIter128::new(base_iter, k, rc);
        for (hash, kmer) in hash_iter.zip(kmer_iter) {
            if let Some(kmer) = kmer {
                f(hash, kmer);
            }
        }
    }
}

impl KmerSource for packed_seq::PackedSeq<'_> {
    fn for_each_kmer_hash_and_value_u32<H: KmerHasher, F: FnMut(u32, u32)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        _filter_out_n: bool,
        mut f: F,
    ) {
        let hash_iter = hasher.hash_kmers_scalar(self);
        let base_iter = self.iter_bp().map(|b| (b, false));
        let kmer_iter = KmerValueIter32::new(base_iter, k, rc);
        for (hash, kmer) in hash_iter.zip(kmer_iter) {
            if let Some(kmer) = kmer {
                f(hash, kmer);
            }
        }
    }

    fn for_each_kmer_hash_and_value_u64<H: KmerHasher, F: FnMut(u32, u64)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        _filter_out_n: bool,
        mut f: F,
    ) {
        let hash_iter = hasher.hash_kmers_scalar(self);
        let base_iter = self.iter_bp().map(|b| (b, false));
        let kmer_iter = KmerValueIter64::new(base_iter, k, rc);
        for (hash, kmer) in hash_iter.zip(kmer_iter) {
            if let Some(kmer) = kmer {
                f(hash, kmer);
            }
        }
    }

    fn for_each_kmer_hash_and_value_u128<H: KmerHasher, F: FnMut(u32, u128)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        _filter_out_n: bool,
        mut f: F,
    ) {
        let hash_iter = hasher.hash_kmers_scalar(self);
        let base_iter = self.iter_bp().map(|b| (b, false));
        let kmer_iter = KmerValueIter128::new(base_iter, k, rc);
        for (hash, kmer) in hash_iter.zip(kmer_iter) {
            if let Some(kmer) = kmer {
                f(hash, kmer);
            }
        }
    }
}

impl KmerSource for PackedNSeq<'_> {
    fn for_each_kmer_hash_and_value_u32<H: KmerHasher, F: FnMut(u32, u32)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        filter_out_n: bool,
        mut f: F,
    ) {
        let hash_iter: Box<dyn Iterator<Item = u32>> = if filter_out_n {
            Box::new(hasher.hash_valid_kmers_scalar(self))
        } else {
            Box::new(hasher.hash_kmers_scalar(self.seq))
        };
        let base_iter: Box<dyn Iterator<Item = (u8, bool)>> = if filter_out_n {
            Box::new(
                self.seq
                    .iter_bp()
                    .zip(self.ambiguous.iter_bp())
                    .map(|(b, amb)| (b, amb != 0)),
            )
        } else {
            Box::new(self.seq.iter_bp().map(|b| (b, false)))
        };

        let kmer_iter = KmerValueIter32::new(base_iter, k, rc);
        for (hash, kmer) in hash_iter.zip(kmer_iter) {
            if hash == u32::MAX {
                continue;
            }
            if let Some(kmer) = kmer {
                f(hash, kmer);
            }
        }
    }

    fn for_each_kmer_hash_and_value_u64<H: KmerHasher, F: FnMut(u32, u64)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        filter_out_n: bool,
        mut f: F,
    ) {
        let hash_iter: Box<dyn Iterator<Item = u32>> = if filter_out_n {
            Box::new(hasher.hash_valid_kmers_scalar(self))
        } else {
            Box::new(hasher.hash_kmers_scalar(self.seq))
        };
        let base_iter: Box<dyn Iterator<Item = (u8, bool)>> = if filter_out_n {
            Box::new(
                self.seq
                    .iter_bp()
                    .zip(self.ambiguous.iter_bp())
                    .map(|(b, amb)| (b, amb != 0)),
            )
        } else {
            Box::new(self.seq.iter_bp().map(|b| (b, false)))
        };

        let kmer_iter = KmerValueIter64::new(base_iter, k, rc);
        for (hash, kmer) in hash_iter.zip(kmer_iter) {
            if hash == u32::MAX {
                continue;
            }
            if let Some(kmer) = kmer {
                f(hash, kmer);
            }
        }
    }

    fn for_each_kmer_hash_and_value_u128<H: KmerHasher, F: FnMut(u32, u128)>(
        self,
        hasher: &H,
        k: usize,
        rc: bool,
        filter_out_n: bool,
        mut f: F,
    ) {
        let hash_iter: Box<dyn Iterator<Item = u32>> = if filter_out_n {
            Box::new(hasher.hash_valid_kmers_scalar(self))
        } else {
            Box::new(hasher.hash_kmers_scalar(self.seq))
        };
        let base_iter: Box<dyn Iterator<Item = (u8, bool)>> = if filter_out_n {
            Box::new(
                self.seq
                    .iter_bp()
                    .zip(self.ambiguous.iter_bp())
                    .map(|(b, amb)| (b, amb != 0)),
            )
        } else {
            Box::new(self.seq.iter_bp().map(|b| (b, false)))
        };

        let kmer_iter = KmerValueIter128::new(base_iter, k, rc);
        for (hash, kmer) in hash_iter.zip(kmer_iter) {
            if hash == u32::MAX {
                continue;
            }
            if let Some(kmer) = kmer {
                f(hash, kmer);
            }
        }
    }
}

pub trait Sketchable: Copy {
    fn len(&self) -> usize;
    fn hash_kmers<H: KmerHasher>(self, hasher: &H) -> PaddedIt<impl ChunkIt<u32x8>>;
}
impl Sketchable for &[u8] {
    fn len(&self) -> usize {
        Seq::len(self)
    }
    #[inline(always)]
    fn hash_kmers<H: KmerHasher>(self, hasher: &H) -> PaddedIt<impl ChunkIt<u32x8>> {
        hasher.hash_kmers_simd(self, 1)
    }
}
impl Sketchable for packed_seq::AsciiSeq<'_> {
    fn len(&self) -> usize {
        Seq::len(self)
    }
    #[inline(always)]
    fn hash_kmers<H: KmerHasher>(self, hasher: &H) -> PaddedIt<impl ChunkIt<u32x8>> {
        hasher.hash_kmers_simd(self, 1)
    }
}
impl Sketchable for packed_seq::PackedSeq<'_> {
    fn len(&self) -> usize {
        Seq::len(self)
    }
    #[inline(always)]
    fn hash_kmers<H: KmerHasher>(self, hasher: &H) -> PaddedIt<impl ChunkIt<u32x8>> {
        hasher.hash_kmers_simd(self, 1)
    }
}
impl<'s> Sketchable for PackedNSeq<'s> {
    fn len(&self) -> usize {
        Seq::len(&self.seq)
    }
    #[inline(always)]
    fn hash_kmers<'h, H: KmerHasher>(
        self,
        hasher: &'h H,
    ) -> PaddedIt<impl ChunkIt<u32x8> + use<'s, 'h, H>> {
        hasher.hash_valid_kmers_simd(self, 1)
    }
}

/// FastMod32, using the low 32 bits of the hash.
/// Taken from https://github.com/lemire/fastmod/blob/master/include/fastmod.h
#[derive(Copy, Clone, Debug)]
struct FM32 {
    d: u64,
    m: u64,
}
impl FM32 {
    #[inline(always)]
    fn new(d: u32) -> Self {
        Self {
            d: d as u64,
            m: u64::MAX / d as u64 + 1,
        }
    }
    #[inline(always)]
    fn fastmod(self, h: u32) -> usize {
        let lowbits = self.m.wrapping_mul(h as u64);
        ((lowbits as u128 * self.d as u128) >> 64) as usize
    }
    #[inline(always)]
    fn fastdiv(self, h: u32) -> usize {
        ((self.m as u128 * h as u128) >> 64) as u32 as usize
    }
}

#[cfg(test)]
mod test {
    use std::hint::black_box;

    use super::*;
    use packed_seq::SeqVec;

    #[test]
    fn test() {
        let b = 16;

        let k = 31;
        for n in 31..100 {
            for f in [0.0, 0.01, 0.03] {
                let s = n - k + 1;
                let seq = packed_seq::PackedNSeqVec::random(n, f);
                let sketcher = crate::SketchParams {
                    alg: SketchAlg::Bottom,
                    rc: false,
                    k,
                    s,
                    b,
                    seed: 0,
                    count: 0,
                    coverage: 1,
                    filter_empty: false,
                    filter_out_n: true,
                }
                .build();
                let bottom = sketcher.bottom_sketch(&[seq.as_slice()]).bottom;
                assert_eq!(bottom.len(), s);
                assert!(bottom.is_sorted());

                let s = s.min(10);
                let seq = packed_seq::PackedNSeqVec::random(n, f);
                let sketcher = crate::SketchParams {
                    alg: SketchAlg::Bottom,
                    rc: true,
                    k,
                    s,
                    b,
                    seed: 0,
                    count: 0,
                    coverage: 1,
                    filter_empty: false,
                    filter_out_n: true,
                }
                .build();
                let bottom = sketcher.bottom_sketch(&[seq.as_slice()]).bottom;
                assert_eq!(bottom.len(), s);
                assert!(bottom.is_sorted());
            }
        }
    }

    #[test]
    fn rc() {
        let b = 32;
        for k in (0..10).map(|_| rand::random_range(1..=64)) {
            for n in (0..10).map(|_| rand::random_range(k..1000)) {
                for s in (0..10).map(|_| rand::random_range(0..n - k + 1)) {
                    for f in [0.0, 0.001, 0.01] {
                        let seq = packed_seq::PackedNSeqVec::random(n, f);
                        let sketcher = crate::SketchParams {
                            alg: SketchAlg::Bottom,
                            rc: true,
                            k,
                            s,
                            b,
                            seed: 0,
                            count: 0,
                            coverage: 1,
                            filter_empty: false,
                            filter_out_n: true,
                        }
                        .build();
                        let bottom = sketcher.bottom_sketch(&[seq.as_slice()]).bottom;
                        assert_eq!(bottom.len(), s);
                        assert!(bottom.is_sorted());

                        let seq_rc = seq.as_slice().to_revcomp();

                        let bottom_rc = sketcher.bottom_sketch(&[seq_rc.as_slice()]).bottom;
                        assert_eq!(bottom, bottom_rc);
                    }
                }
            }
        }
    }

    #[test]
    fn equal_dist() {
        let s = 1000;
        let k = 10;
        let n = 300;
        let b = 8;
        let seq = packed_seq::AsciiSeqVec::random(n);

        for (alg, filter_empty) in [
            (SketchAlg::Bottom, false),
            (SketchAlg::Bucket, false),
            (SketchAlg::Bucket, true),
        ] {
            let sketcher = crate::SketchParams {
                alg,
                rc: false,
                k,
                s,
                b,
                seed: 0,
                count: 0,
                coverage: 1,
                filter_empty,
                filter_out_n: false,
            }
            .build();
            let sketch = sketcher.sketch(seq.as_slice());
            assert_eq!(sketch.mash_distance(&sketch), 0.0);
        }
    }

    #[test]
    fn fuzz_short() {
        let s = 1024;
        let k = 10;
        for b in [1, 8, 16, 32] {
            for n in [10, 20, 40, 80, 150, 300, 500, 1000, 2000] {
                let seq1 = packed_seq::AsciiSeqVec::random(n);
                let seq2 = packed_seq::AsciiSeqVec::random(n);

                for (alg, filter_empty) in [
                    (SketchAlg::Bottom, false),
                    (SketchAlg::Bucket, false),
                    (SketchAlg::Bucket, true),
                ] {
                    let sketcher = crate::SketchParams {
                        alg,
                        rc: false,
                        k,
                        s,
                        b,
                        seed: 0,
                        count: 0,
                        coverage: 1,
                        filter_empty,
                        filter_out_n: false,
                    }
                    .build();
                    let s1 = sketcher.sketch(seq1.as_slice());
                    let s2 = sketcher.sketch(seq2.as_slice());
                    s1.mash_distance(&s2);
                }
            }
        }
    }

    #[test]
    fn bucket_runtime_dispatch_picks_width_from_k() {
        let seq = packed_seq::AsciiSeqVec::random(400);
        for (k, expected) in [(16, 32), (31, 64), (63, 128)] {
            let sketcher = crate::SketchParams {
                alg: SketchAlg::Bucket,
                rc: true,
                k,
                s: 256,
                b: 8,
                seed: 0,
                count: 0,
                coverage: 1,
                filter_empty: false,
                filter_out_n: false,
            }
            .build();
            let sketch = sketcher.sketch(seq.as_slice());
            let Sketch::BucketSketch(bucket) = sketch else {
                panic!("expected bucket sketch")
            };

            match (&bucket.kmers, expected) {
                (BucketKmers::U32(_), 32) => {}
                (BucketKmers::U64(_), 64) => {}
                (BucketKmers::U128(_), 128) => {}
                _ => panic!("unexpected k-mer storage width for k={k}"),
            }
        }
    }

    #[test]
    fn bucket_supports_k16() {
        let seq = packed_seq::AsciiSeqVec::random(600);
        let sketcher = crate::SketchParams {
            alg: SketchAlg::Bucket,
            rc: true,
            k: 16,
            s: 256,
            b: 8,
            seed: 0,
            count: 0,
            coverage: 1,
            filter_empty: false,
            filter_out_n: false,
        }
        .build();
        let sketch = sketcher.sketch(seq.as_slice());

        assert_eq!(sketch.mash_distance(&sketch), 0.0);
        let kmers = sketch.intersection_kmers(&sketch);
        assert!(kmers.iter().all(|kmer| matches!(kmer, PackedKmer::U32(_))));
    }

    #[test]
    fn bucket_supports_k63() {
        let seq = packed_seq::AsciiSeqVec::random(600);
        let sketcher = crate::SketchParams {
            alg: SketchAlg::Bucket,
            rc: true,
            k: 63,
            s: 256,
            b: 8,
            seed: 0,
            count: 0,
            coverage: 1,
            filter_empty: false,
            filter_out_n: false,
        }
        .build();
        let sketch = sketcher.sketch(seq.as_slice());

        assert_eq!(sketch.mash_distance(&sketch), 0.0);
        let kmers = sketch.intersection_kmers(&sketch);
        assert!(kmers.iter().all(|kmer| matches!(kmer, PackedKmer::U128(_))));
    }

    #[test]
    fn test_collect() {
        let mut out = vec![];
        let n = black_box(8000);
        let it = (0..n).map(|x| u32x8::splat((x as u32).wrapping_mul(546786567)));
        let padded_it = PaddedIt { it, padding: 0 };
        let mut bound = black_box(u32::MAX / 10);
        collect_impl(
            &mut bound,
            padded_it,
            &mut out,
            usize::MAX,
            &mut |out| {
                out.clear();
                u32::MAX
            },
            &mut 0,
        );
        eprintln!("{out:?}");
    }
}
