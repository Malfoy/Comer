use std::{
    fs::File,
    io::{Read, Seek},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed},
};

use clap::Parser;
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use log::info;
use packed_seq::{PackedNSeqVec, PackedSeqVec, SeqVec};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::Serialize;
use simd_sketch::SketchParams;
use std::str;

/// Compute the sketch distance between two fasta files.
#[derive(clap::Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

/// TODO: Support for writing sketches to disk.
#[derive(clap::Subcommand)]
enum Command {
    /// Takes paths to fasta files, and writes .ssketch files.
    Sketch {
        #[command(flatten)]
        params: SketchParams,
        #[command(flatten)]
        sequence_filters: SequenceFilterArgs,
        /// Paths to (directories of) fasta files (plain or compressed).
        paths: Vec<PathBuf>,
        /// Parse kmers from bcalm/logan FASTA headers (km:f: or ka:f:)
        #[arg(long)]
        bcalm: bool,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
        #[arg(long)]
        no_save: bool,
    },
    /// Compute the distance between two sequences.
    Dist {
        #[command(flatten)]
        params: SketchParams,
        #[command(flatten)]
        sequence_filters: SequenceFilterArgs,
        /// First input fasta file or .ssketch file.
        path_a: PathBuf,
        /// Second input fasta file or .ssketch file.
        path_b: PathBuf,
        /// Parse kmers from bcalm/logan FASTA headers (km:f: or ka:f:)
        #[arg(long)]
        bcalm: bool,
        /// Use weighted Jaccard (requires abundances; defaults to 1 when missing).
        #[arg(long)]
        weighted: bool,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
    },
    /// Takes paths to fasta files, and outputs a Phylip distance matrix to stdout.
    Triangle {
        #[command(flatten)]
        params: SketchParams,
        #[command(flatten)]
        sequence_filters: SequenceFilterArgs,
        /// Paths to (directories of) fasta files (plain or compressed) or .ssketch files.
        /// If <path>.ssketch exists, it is automatically used.
        paths: Vec<PathBuf>,
        /// Parse kmers from bcalm/logan FASTA headers (km:f: or ka:f:)
        #[arg(long)]
        bcalm: bool,
        /// Use weighted Jaccard (requires abundances; defaults to 1 when missing).
        #[arg(long)]
        weighted: bool,
        /// Write phylip distance matrix here, or default to stdout.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Save missing sketches to disk, as .ssketch files alongside the input.
        #[arg(long)]
        save_sketches: bool,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
    },
    /// Takes paths to fasta files, and writes .ssketch files.
    Classify {
        // Sketch args
        #[command(flatten)]
        params: SketchParams,
        #[command(flatten)]
        sequence_filters: SequenceFilterArgs,
        /// Paths to directory of fasta files (plain or compressed).
        #[arg(long)]
        targets: Vec<PathBuf>,
        /// Parse kmers from bcalm/logan FASTA headers (km:f: or ka:f:)
        #[arg(long)]
        bcalm: bool,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
        #[arg(long)]
        no_save: bool,

        /// Path to metagenomic FASTQ sample (plain or compressed).
        reads: PathBuf,
    },
    /// List k-mers in the intersection of two sketches.
    Intersect {
        #[command(flatten)]
        params: SketchParams,
        #[command(flatten)]
        sequence_filters: SequenceFilterArgs,
        /// First input fasta file or .ssketch file.
        path_a: PathBuf,
        /// Second input fasta file or .ssketch file.
        path_b: PathBuf,
        /// Parse kmers from bcalm/logan FASTA headers (km:f: or ka:f:)
        #[arg(long)]
        bcalm: bool,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
    },
    /// Output sourmash-style JSON signatures to stdout.
    Signature {
        #[command(flatten)]
        params: SketchParams,
        #[command(flatten)]
        sequence_filters: SequenceFilterArgs,
        /// Paths to (directories of) fasta files (plain or compressed) or .ssketch files.
        paths: Vec<PathBuf>,
        /// Parse kmers from bcalm/logan FASTA headers (km:f: or ka:f:)
        #[arg(long)]
        bcalm: bool,
        /// Write JSON output here, or default to stdout.
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
    },
}

#[derive(clap::Args, Copy, Clone, Debug, Default)]
struct SequenceFilterArgs {
    /// Drop sequences shorter than this length.
    #[arg(long)]
    min_seq_len: Option<usize>,
    /// Drop sequences whose N fraction is larger than this value in [0, 1].
    #[arg(long)]
    max_n_fraction: Option<f32>,
    /// Drop sequences containing a homopolymer run longer than this length.
    #[arg(long)]
    max_homopolymer_len: Option<usize>,
    /// Drop sequences with Shannon entropy (bits/base) below this threshold.
    #[arg(long)]
    min_entropy: Option<f32>,
}

impl SequenceFilterArgs {
    fn has_any(self) -> bool {
        self.min_seq_len.is_some()
            || self.max_n_fraction.is_some()
            || self.max_homopolymer_len.is_some()
            || self.min_entropy.is_some()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SequenceFilterReason {
    MinLength,
    MaxNFraction,
    MaxHomopolymerLength,
    MinEntropy,
}

const BINCODE_CONFIG: bincode::config::Configuration<
    bincode::config::LittleEndian,
    bincode::config::Fixint,
> = bincode::config::standard().with_fixed_int_encoding();
const EXTENSION: &str = "ssketch";
const SKETCH_VERSION: usize = 4;

#[derive(bincode::Encode, bincode::Decode)]
pub struct VersionedSketch {
    /// This version of simd-sketch only supports encoding version 4.
    /// This is encoded first, so that it can (hopefully) still be recovered in case decoding fails.
    version: usize,
    /// The sketch itself.
    sketch: simd_sketch::Sketch,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    // Initialize thread pool.
    let (Command::Sketch { threads, .. }
    | Command::Dist { threads, .. }
    | Command::Triangle { threads, .. }
    | Command::Classify { threads, .. }
    | Command::Intersect { threads, .. }
    | Command::Signature { threads, .. }) = &args.command;
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(*threads)
            .build_global()
            .unwrap();
    }

    let (params, paths) = match &args.command {
        Command::Dist {
            params,
            path_a,
            path_b,
            ..
        } => (params, vec![path_a.clone(), path_b.clone()]),
        Command::Intersect {
            params,
            path_a,
            path_b,
            ..
        } => (params, vec![path_a.clone(), path_b.clone()]),
        Command::Sketch { params, paths, .. } | Command::Triangle { params, paths, .. } => {
            (params, collect_paths(&paths))
        }
        Command::Classify {
            params, targets, ..
        } => (params, collect_paths(&targets)),
        Command::Signature { params, paths, .. } => (params, collect_paths(&paths)),
    };

    let save_sketches = match &args.command {
        Command::Sketch { no_save, .. } => !no_save,
        Command::Classify { no_save, .. } => !no_save,
        Command::Dist { .. } => false,
        Command::Intersect { .. } => false,
        Command::Signature { .. } => false,
        Command::Triangle { save_sketches, .. } => *save_sketches,
    };

    let bcalm = match &args.command {
        Command::Sketch { bcalm, .. } => *bcalm,
        Command::Dist { bcalm, .. } => *bcalm,
        Command::Triangle { bcalm, .. } => *bcalm,
        Command::Classify { bcalm, .. } => *bcalm,
        Command::Intersect { bcalm, .. } => *bcalm,
        Command::Signature { bcalm, .. } => *bcalm,
    };
    let sequence_filters = match &args.command {
        Command::Sketch {
            sequence_filters, ..
        } => *sequence_filters,
        Command::Dist {
            sequence_filters, ..
        } => *sequence_filters,
        Command::Triangle {
            sequence_filters, ..
        } => *sequence_filters,
        Command::Classify {
            sequence_filters, ..
        } => *sequence_filters,
        Command::Intersect {
            sequence_filters, ..
        } => *sequence_filters,
        Command::Signature {
            sequence_filters, ..
        } => *sequence_filters,
    };
    validate_sequence_filters(sequence_filters);

    let weighted = match &args.command {
        Command::Dist { weighted, .. } => *weighted,
        Command::Triangle { weighted, .. } => *weighted,
        _ => false,
    };

    let q = paths.len();

    let sketcher = params.build();
    let params = sketcher.params();

    let style = indicatif::ProgressStyle::with_template(
        "{msg:.bold} [{elapsed_precise:.cyan}] {bar} {pos}/{len} ({percent:>3}%)",
    )
    .unwrap()
    .progress_chars("##-");

    let start = std::time::Instant::now();

    let num_sketched = AtomicUsize::new(0);
    let num_read = AtomicUsize::new(0);
    let num_written = AtomicUsize::new(0);
    let total_bytes = AtomicUsize::new(0);
    let filtered_by_min_len = AtomicUsize::new(0);
    let filtered_by_n_fraction = AtomicUsize::new(0);
    let filtered_by_homopolymer = AtomicUsize::new(0);
    let filtered_by_entropy = AtomicUsize::new(0);
    let kmers_filtered_by_min_len = AtomicU64::new(0);
    let kmers_filtered_by_n_fraction = AtomicU64::new(0);
    let kmers_filtered_by_homopolymer = AtomicU64::new(0);
    let kmers_filtered_by_entropy = AtomicU64::new(0);
    let apply_sequence_filters = sequence_filters.has_any();

    let sketches: Vec<_> = paths
        .par_iter()
        .progress_with_style(style.clone())
        .with_message("Sketching")
        .with_finish(indicatif::ProgressFinish::AndLeave)
        .map(|path| {
            let read_sketch = |path| {
                num_read.fetch_add(1, Relaxed);
                let mut file = File::open(path).unwrap();
                // Read the first integer to check the version.
                let version: usize =
                    bincode::decode_from_std_read(&mut file, BINCODE_CONFIG).unwrap();
                if version != SKETCH_VERSION {
                    panic!("Unsupported sketch version: {version}. Only version {SKETCH_VERSION} is supported.");
                }
                file.seek(std::io::SeekFrom::Start(0)).unwrap();
                let VersionedSketch {
                    version,
                    sketch,
                } = bincode::decode_from_std_read(&mut file, BINCODE_CONFIG).unwrap();
                assert_eq!(version, SKETCH_VERSION);

                let mut sketch_params = sketch.to_params();
                sketch_params.filter_empty = params.filter_empty;
                if *params != sketch_params {
                    panic!(
                        "Sketch parameters do not match:\nCommand line: {params:?}\nOn disk:      {sketch_params:?}",
                    );
                }

                return sketch;
            };

            // Input path is a .ssketch file.
            if path.extension().is_some_and(|ext| ext == EXTENSION) {
                return read_sketch(path);
            }

            // Input path is a .fa, and the .fa.ssketch file exists.
            let ssketch_path = path.with_extension(EXTENSION);
            if ssketch_path.exists() {
                return read_sketch(&ssketch_path);
            }

            let mut reader = parse_fastx_input(path);

            let mut sketch;
            if params.filter_out_n {
                let mut ranges = vec![];
                let mut abundances = vec![];
                let mut seq = PackedNSeqVec::default();
                let mut size = 0;
                while let Some(r) = reader.next() {
                    let record = r.unwrap();
                    let seq_bytes = record.seq();
                    if apply_sequence_filters {
                        let eval = evaluate_sequence_filters(
                            seq_bytes.as_ref(),
                            sequence_filters,
                            params.k,
                            params.filter_out_n,
                        );
                        if let Some(reason) = eval.reason {
                            let removed_kmers = eval.candidate_kmers as u64;
                            match reason {
                                SequenceFilterReason::MinLength => {
                                    filtered_by_min_len.fetch_add(1, Relaxed);
                                    kmers_filtered_by_min_len.fetch_add(removed_kmers, Relaxed);
                                }
                                SequenceFilterReason::MaxNFraction => {
                                    filtered_by_n_fraction.fetch_add(1, Relaxed);
                                    kmers_filtered_by_n_fraction.fetch_add(removed_kmers, Relaxed);
                                }
                                SequenceFilterReason::MaxHomopolymerLength => {
                                    filtered_by_homopolymer.fetch_add(1, Relaxed);
                                    kmers_filtered_by_homopolymer
                                        .fetch_add(removed_kmers, Relaxed);
                                }
                                SequenceFilterReason::MinEntropy => {
                                    filtered_by_entropy.fetch_add(1, Relaxed);
                                    kmers_filtered_by_entropy.fetch_add(removed_kmers, Relaxed);
                                }
                            }
                            continue;
                        }
                    }
                    let abundance = if bcalm {
                        parse_bcalm_abundance(record.id())
                    } else {
                        1
                    };
                    let range = seq.push_ascii(seq_bytes.as_ref());
                    size += range.len();
                    ranges.push(range);
                    abundances.push(abundance);
                }
                total_bytes.fetch_add(size, Relaxed);
                let slices = ranges.into_iter().map(|r| seq.slice(r)).collect_vec();
                sketch = sketcher.sketch_seqs_with_abundances(&slices, &abundances);
            } else {
                let mut ranges = vec![];
                let mut abundances = vec![];
                let mut seq = PackedSeqVec::default();
                let mut size = 0;
                while let Some(r) = reader.next() {
                    let record = r.unwrap();
                    let seq_bytes = record.seq();
                    if apply_sequence_filters {
                        let eval = evaluate_sequence_filters(
                            seq_bytes.as_ref(),
                            sequence_filters,
                            params.k,
                            params.filter_out_n,
                        );
                        if let Some(reason) = eval.reason {
                            let removed_kmers = eval.candidate_kmers as u64;
                            match reason {
                                SequenceFilterReason::MinLength => {
                                    filtered_by_min_len.fetch_add(1, Relaxed);
                                    kmers_filtered_by_min_len.fetch_add(removed_kmers, Relaxed);
                                }
                                SequenceFilterReason::MaxNFraction => {
                                    filtered_by_n_fraction.fetch_add(1, Relaxed);
                                    kmers_filtered_by_n_fraction.fetch_add(removed_kmers, Relaxed);
                                }
                                SequenceFilterReason::MaxHomopolymerLength => {
                                    filtered_by_homopolymer.fetch_add(1, Relaxed);
                                    kmers_filtered_by_homopolymer
                                        .fetch_add(removed_kmers, Relaxed);
                                }
                                SequenceFilterReason::MinEntropy => {
                                    filtered_by_entropy.fetch_add(1, Relaxed);
                                    kmers_filtered_by_entropy.fetch_add(removed_kmers, Relaxed);
                                }
                            }
                            continue;
                        }
                    }
                    let abundance = if bcalm {
                        parse_bcalm_abundance(record.id())
                    } else {
                        1
                    };
                    let range = seq.push_ascii(seq_bytes.as_ref());
                    size += range.len();
                    ranges.push(range);
                    abundances.push(abundance);
                }
                total_bytes.fetch_add(size, Relaxed);
                let slices = ranges.into_iter().map(|r| seq.slice(r)).collect_vec();
                sketch = sketcher.sketch_seqs_with_abundances(&slices, &abundances);
            }
            num_sketched.fetch_add(1, Relaxed);

            if save_sketches {
                num_written.fetch_add(1, Relaxed);
                let versioned_sketch = VersionedSketch {
                    version: SKETCH_VERSION,
                    sketch,
                };
                bincode::encode_into_std_write(
                    &versioned_sketch,
                    &mut File::create(ssketch_path).unwrap(),
                    BINCODE_CONFIG,
                )
                .unwrap();
                sketch = versioned_sketch.sketch;
            }

            sketch
        })
        .collect();
    let t_sketch = start.elapsed();

    info!(
        "Sketching {q} seqs took {t_sketch:?} ({:?} avg, {} MiB/s)",
        t_sketch / q as u32,
        total_bytes.into_inner() as f32 / t_sketch.as_secs_f32() / (1 << 20) as f32
    );
    let num_read = num_read.into_inner();
    let num_sketched = num_sketched.into_inner();
    let num_written = num_written.into_inner();
    if num_read > 0 {
        info!("Read {num_read} sketches from disk.");
    }
    if num_sketched > 0 {
        info!("Newly sketched {num_sketched} files.");
    }
    if num_written > 0 {
        info!("Wrote {num_written} sketches to disk.");
    }
    if sequence_filters.has_any() {
        if let Some(threshold) = sequence_filters.min_seq_len {
            info!(
                "Filtered by min length ({threshold}): {} sequences, {} kmers.",
                filtered_by_min_len.load(Relaxed),
                kmers_filtered_by_min_len.load(Relaxed)
            );
        }
        if let Some(threshold) = sequence_filters.max_n_fraction {
            info!(
                "Filtered by max N fraction ({threshold:.6}): {} sequences, {} kmers.",
                filtered_by_n_fraction.load(Relaxed),
                kmers_filtered_by_n_fraction.load(Relaxed)
            );
        }
        if let Some(threshold) = sequence_filters.max_homopolymer_len {
            info!(
                "Filtered by max homopolymer length ({threshold}): {} sequences, {} kmers.",
                filtered_by_homopolymer.load(Relaxed),
                kmers_filtered_by_homopolymer.load(Relaxed)
            );
        }
        if let Some(threshold) = sequence_filters.min_entropy {
            info!(
                "Filtered by min entropy ({threshold:.6} bits/base): {} sequences, {} kmers.",
                filtered_by_entropy.load(Relaxed),
                kmers_filtered_by_entropy.load(Relaxed)
            );
        }
    }

    if matches!(args.command, Command::Sketch { .. }) {
        // If we are sketching, we are done.
        return;
    }
    if let Command::Classify { reads, .. } = &args.command {
        simd_sketch::classify::classify(&sketches, reads);
        return;
    }
    if let Command::Signature { output, .. } = &args.command {
        let sigs = sketches
            .iter()
            .zip(paths.iter())
            .map(|(sketch, path)| sketch_to_sourmash(sketch, path))
            .collect_vec();
        let json = serde_json::to_vec_pretty(&sigs).unwrap();
        match output {
            Some(output) => std::fs::write(output, json).unwrap(),
            None => println!("{}", str::from_utf8(&json).unwrap()),
        }
        return;
    }
    if let Command::Intersect { .. } = &args.command {
        let kmers = sketches[0].intersection_kmers(&sketches[1]);
        for kmer in kmers {
            println!("{kmer}");
        }
        return;
    }

    let num_pairs = q * (q - 1) / 2;
    let mut pairs = Vec::with_capacity(num_pairs);
    for i in 0..q {
        for j in 0..i {
            pairs.push((i, j));
        }
    }
    let start = std::time::Instant::now();
    let sims: Vec<_> = pairs
        .into_par_iter()
        .progress_with_style(style.clone())
        .with_message("Similarities")
        .with_finish(indicatif::ProgressFinish::AndLeave)
        .map(|(i, j)| {
            if weighted {
                sketches[i].weighted_jaccard_similarity(&sketches[j])
            } else {
                sketches[i].jaccard_similarity(&sketches[j])
            }
        })
        .collect();
    let t_dist = start.elapsed();

    let cnt = q * (q - 1) / 2;
    info!(
        "Computing {cnt} similarities took {t_dist:?} ({:?} avg)",
        t_dist / cnt.max(1) as u32
    );

    match &args.command {
        Command::Sketch { .. } => {
            unreachable!();
        }
        Command::Classify { .. } => {
            unreachable!();
        }
        Command::Dist { .. } => {
            if weighted {
                println!("Weighted Jaccard: {:.6}", sims[0]);
            } else {
                println!("Jaccard: {:.6}", sims[0]);
            }
            return;
        }
        Command::Triangle { output, .. } => {
            use std::io::Write;

            // Output Phylip triangle format.
            let mut out = Vec::new();
            writeln!(out, "{q}").unwrap();
            let mut d = sims.iter();
            for i in 0..q {
                write!(out, "{}", paths[i].to_str().unwrap()).unwrap();
                for _ in 0..i {
                    write!(out, "\t{:.7}", d.next().unwrap()).unwrap();
                }
                writeln!(out).unwrap();
            }

            match output {
                Some(output) => std::fs::write(output, out).unwrap(),
                None => println!("{}", str::from_utf8(&out).unwrap()),
            }
        }
        Command::Intersect { .. } | Command::Signature { .. } => {
            unreachable!();
        }
    }
}

#[derive(Serialize)]
struct SourmashSignatureFile {
    class: &'static str,
    email: &'static str,
    filename: String,
    name: String,
    signatures: Vec<SourmashSignature>,
    version: f32,
}

#[derive(Serialize)]
struct SourmashSignature {
    ksize: usize,
    num: usize,
    seed: u32,
    hash_function: &'static str,
    max_hash: u64,
    molecule: &'static str,
    mins: SignatureMins,
    abundances: Vec<u16>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum SignatureMins {
    U32(Vec<u32>),
    U64(Vec<u64>),
    U128(Vec<String>),
}

impl SignatureMins {
    fn len(&self) -> usize {
        match self {
            SignatureMins::U32(v) => v.len(),
            SignatureMins::U64(v) => v.len(),
            SignatureMins::U128(v) => v.len(),
        }
    }
}

fn sketch_to_sourmash(sketch: &simd_sketch::Sketch, path: &PathBuf) -> SourmashSignatureFile {
    let name = path.to_string_lossy().to_string();
    match sketch {
        simd_sketch::Sketch::BucketSketch(bucket) => {
            let (mins, abundances, hash_function) = match &bucket.kmers {
                simd_sketch::BucketKmers::U32(kmers) => {
                    let mut mins = Vec::new();
                    let mut abundances = Vec::new();
                    for (&kmer, &abundance) in kmers.iter().zip(bucket.abundances.iter()) {
                        if kmer == u32::MAX {
                            continue;
                        }
                        mins.push(kmer);
                        abundances.push(abundance);
                    }
                    (SignatureMins::U32(mins), abundances, "simd-sketch-kmer32")
                }
                simd_sketch::BucketKmers::U64(kmers) => {
                    let mut mins = Vec::new();
                    let mut abundances = Vec::new();
                    for (&kmer, &abundance) in kmers.iter().zip(bucket.abundances.iter()) {
                        if kmer == u64::MAX {
                            continue;
                        }
                        mins.push(kmer);
                        abundances.push(abundance);
                    }
                    (SignatureMins::U64(mins), abundances, "simd-sketch-kmer64")
                }
                simd_sketch::BucketKmers::U128(kmers) => {
                    let mut mins = Vec::new();
                    let mut abundances = Vec::new();
                    for (&kmer, &abundance) in kmers.iter().zip(bucket.abundances.iter()) {
                        if kmer == u128::MAX {
                            continue;
                        }
                        mins.push(kmer.to_string());
                        abundances.push(abundance);
                    }
                    (
                        SignatureMins::U128(mins),
                        abundances,
                        "simd-sketch-kmer128-decimal",
                    )
                }
            };
            SourmashSignatureFile {
                class: "sourmash_signature",
                email: "",
                filename: name.clone(),
                name,
                signatures: vec![SourmashSignature {
                    ksize: bucket.k,
                    num: mins.len(),
                    seed: bucket.seed,
                    hash_function,
                    max_hash: 0,
                    molecule: "DNA",
                    mins,
                    abundances,
                }],
                version: 0.4,
            }
        }
        _ => panic!("Sourmash output only supported for bucket sketches."),
    }
}

fn validate_sequence_filters(filters: SequenceFilterArgs) {
    if let Some(max_n_fraction) = filters.max_n_fraction {
        assert!(
            max_n_fraction.is_finite() && (0.0..=1.0).contains(&max_n_fraction),
            "--max-n-fraction must be finite and in [0, 1], got {max_n_fraction}"
        );
    }
    if let Some(min_entropy) = filters.min_entropy {
        assert!(
            min_entropy.is_finite() && min_entropy >= 0.0,
            "--min-entropy must be finite and >= 0, got {min_entropy}"
        );
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct SequenceFilterEvaluation {
    reason: Option<SequenceFilterReason>,
    candidate_kmers: usize,
}

fn evaluate_sequence_filters(
    seq: &[u8],
    filters: SequenceFilterArgs,
    k: usize,
    filter_out_n: bool,
) -> SequenceFilterEvaluation {
    let len = seq.len();
    let need_n_fraction = filters.max_n_fraction.is_some();
    let need_homopolymer = filters.max_homopolymer_len.is_some();
    let need_entropy = filters.min_entropy.is_some();

    let mut n_count = 0usize;
    let mut longest_homopolymer = 0usize;
    let mut current_run = 0usize;
    let mut previous_base: Option<u8> = None;
    let mut entropy_counts = [0usize; 256];

    // Candidate kmers for reporting removed kmers for filtered sequences.
    let mut candidate_kmers_filter_out_n = 0usize;
    let mut valid_run = 0usize;

    for &raw_base in seq {
        let base = raw_base.to_ascii_uppercase();

        if need_n_fraction && base == b'N' {
            n_count += 1;
        }

        if need_homopolymer {
            if Some(base) == previous_base {
                current_run += 1;
            } else {
                current_run = 1;
                previous_base = Some(base);
            }
            longest_homopolymer = longest_homopolymer.max(current_run);
        }

        if need_entropy {
            entropy_counts[base as usize] += 1;
        }

        if filter_out_n && k > 0 {
            if matches!(base, b'A' | b'C' | b'G' | b'T') {
                valid_run += 1;
                if valid_run >= k {
                    candidate_kmers_filter_out_n += 1;
                }
            } else {
                valid_run = 0;
            }
        }
    }

    let candidate_kmers = if filter_out_n {
        candidate_kmers_filter_out_n
    } else if k == 0 || len < k {
        0
    } else {
        len - k + 1
    };

    let n_fraction = if len == 0 {
        0.0
    } else {
        n_count as f32 / len as f32
    };

    let entropy = if need_entropy {
        if len == 0 {
            0.0
        } else {
            let total = len as f32;
            let mut value = 0.0f32;
            for &count in &entropy_counts {
                if count == 0 {
                    continue;
                }
                let p = count as f32 / total;
                value -= p * p.log2();
            }
            value
        }
    } else {
        0.0
    };

    let reason = if let Some(min_len) = filters.min_seq_len {
        if len < min_len {
            Some(SequenceFilterReason::MinLength)
        } else {
            None
        }
    } else {
        None
    }
    .or_else(|| {
        filters.max_n_fraction.and_then(|max_n_fraction| {
            if n_fraction > max_n_fraction {
                Some(SequenceFilterReason::MaxNFraction)
            } else {
                None
            }
        })
    })
    .or_else(|| {
        filters.max_homopolymer_len.and_then(|max_homopolymer_len| {
            if longest_homopolymer > max_homopolymer_len {
                Some(SequenceFilterReason::MaxHomopolymerLength)
            } else {
                None
            }
        })
    })
    .or_else(|| {
        filters.min_entropy.and_then(|min_entropy| {
            if entropy < min_entropy {
                Some(SequenceFilterReason::MinEntropy)
            } else {
                None
            }
        })
    });

    SequenceFilterEvaluation {
        reason,
        candidate_kmers,
    }
}

#[cfg(test)]
fn first_failed_sequence_filter(
    seq: &[u8],
    filters: SequenceFilterArgs,
) -> Option<SequenceFilterReason> {
    evaluate_sequence_filters(seq, filters, 1, false).reason
}

#[cfg(test)]
fn candidate_kmer_count(seq: &[u8], k: usize, filter_out_n: bool) -> usize {
    evaluate_sequence_filters(seq, SequenceFilterArgs::default(), k, filter_out_n).candidate_kmers
}

#[cfg(test)]
fn shannon_entropy_bits_per_base(seq: &[u8]) -> f32 {
    if seq.is_empty() {
        return 0.0;
    }
    let mut counts = [0usize; 256];
    for &base in seq {
        counts[base.to_ascii_uppercase() as usize] += 1;
    }
    let total = seq.len() as f32;
    let mut entropy = 0.0f32;
    for &count in &counts {
        if count == 0 {
            continue;
        }
        let p = count as f32 / total;
        entropy -= p * p.log2();
    }
    entropy
}

fn parse_bcalm_abundance(header: &[u8]) -> u16 {
    let mut abundance = None;

    let parse_token = |token: &[u8]| -> Option<u16> {
        if let Some(rest) = token
            .strip_prefix(b"km:f:")
            .or_else(|| token.strip_prefix(b"ka:f:"))
        {
            let s = str::from_utf8(rest).ok()?;
            let v: f32 = s.parse().ok()?;
            if !v.is_finite() {
                return None;
            }
            let v = v.round().clamp(0.0, u16::MAX as f32);
            return Some(v as u16);
        }
        if let Some(rest) = token
            .strip_prefix(b"km:i:")
            .or_else(|| token.strip_prefix(b"ka:i:"))
        {
            let s = str::from_utf8(rest).ok()?;
            let v: u32 = s.parse().ok()?;
            return Some(v.min(u16::MAX as u32) as u16);
        }
        None
    };

    for token in header.split(|b| b.is_ascii_whitespace()) {
        if token.is_empty() {
            continue;
        }
        if let Some(v) = parse_token(token) {
            abundance = Some(v);
            break;
        }
    }

    abundance.unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        SequenceFilterArgs, SequenceFilterReason, candidate_kmer_count,
        first_failed_sequence_filter, is_fastx_member_name, is_supported_input_path,
        parse_bcalm_abundance, shannon_entropy_bits_per_base,
    };

    #[test]
    fn parses_km_and_ka_float_tags_equivalently() {
        let km = b">u1 LN:i:64 km:f:3.2 L:+:u2:+";
        let ka = b">u1 LN:i:64 ka:f:3.2 L:+:u2:+";
        assert_eq!(parse_bcalm_abundance(km), 3);
        assert_eq!(parse_bcalm_abundance(ka), 3);
    }

    #[test]
    fn parses_km_and_ka_int_tags_equivalently() {
        let km = b">u1 LN:i:64 km:i:7 L:+:u2:+";
        let ka = b">u1 LN:i:64 ka:i:7 L:+:u2:+";
        assert_eq!(parse_bcalm_abundance(km), 7);
        assert_eq!(parse_bcalm_abundance(ka), 7);
    }

    #[test]
    fn defaults_to_one_when_abundance_missing() {
        let header = b">u1 LN:i:64 L:+:u2:+";
        assert_eq!(parse_bcalm_abundance(header), 1);
    }

    #[test]
    fn supports_new_compressed_extensions_in_path_filter() {
        assert!(is_supported_input_path(Path::new("a.fa")));
        assert!(is_supported_input_path(Path::new("a.fa.bz2")));
        assert!(is_supported_input_path(Path::new("a.fa.xz")));
        assert!(is_supported_input_path(Path::new("a.fa.zst")));
        assert!(is_supported_input_path(Path::new("a.fa.zstd")));
        assert!(is_supported_input_path(Path::new("a.zip")));
        assert!(!is_supported_input_path(Path::new("a.txt")));
    }

    #[test]
    fn selects_fastx_members_in_zip_archives() {
        assert!(is_fastx_member_name("reads.fastq"));
        assert!(is_fastx_member_name("reads.fq.xz"));
        assert!(is_fastx_member_name("reads.fa.bz2"));
        assert!(!is_fastx_member_name("README.md"));
    }

    #[test]
    fn sequence_filter_min_length_triggers() {
        let filters = SequenceFilterArgs {
            min_seq_len: Some(5),
            ..Default::default()
        };
        assert_eq!(
            first_failed_sequence_filter(b"ACGT", filters),
            Some(SequenceFilterReason::MinLength)
        );
    }

    #[test]
    fn sequence_filter_max_n_fraction_triggers() {
        let filters = SequenceFilterArgs {
            max_n_fraction: Some(0.25),
            ..Default::default()
        };
        assert_eq!(
            first_failed_sequence_filter(b"ACNN", filters),
            Some(SequenceFilterReason::MaxNFraction)
        );
    }

    #[test]
    fn sequence_filter_homopolymer_triggers() {
        let filters = SequenceFilterArgs {
            max_homopolymer_len: Some(3),
            ..Default::default()
        };
        assert_eq!(
            first_failed_sequence_filter(b"ACGGGGTA", filters),
            Some(SequenceFilterReason::MaxHomopolymerLength)
        );
    }

    #[test]
    fn sequence_filter_entropy_triggers() {
        let filters = SequenceFilterArgs {
            min_entropy: Some(1.0),
            ..Default::default()
        };
        assert_eq!(
            first_failed_sequence_filter(b"AAAAAA", filters),
            Some(SequenceFilterReason::MinEntropy)
        );
    }

    #[test]
    fn sequence_filter_order_is_deterministic() {
        let filters = SequenceFilterArgs {
            min_seq_len: Some(10),
            max_n_fraction: Some(0.0),
            max_homopolymer_len: Some(2),
            min_entropy: Some(1.0),
        };
        assert_eq!(
            first_failed_sequence_filter(b"NNN", filters),
            Some(SequenceFilterReason::MinLength)
        );
    }

    #[test]
    fn candidate_kmer_count_respects_filter_out_n() {
        assert_eq!(candidate_kmer_count(b"ACGTNACGT", 4, false), 6);
        assert_eq!(candidate_kmer_count(b"ACGTNACGT", 4, true), 2);
    }

    #[test]
    fn entropy_helper_matches_expectation() {
        let low = shannon_entropy_bits_per_base(b"AAAA");
        let high = shannon_entropy_bits_per_base(b"ACGT");
        assert!(low < high);
        assert!(low <= 0.01);
        assert!(high >= 1.9);
    }
}

fn parse_fastx_input(path: &Path) -> Box<dyn needletail::FastxReader> {
    if has_extension(path, "zip") {
        return parse_fastx_from_zip(path);
    }

    needletail::parse_fastx_file(path)
        .unwrap_or_else(|err| panic!("Failed to parse FASTA/FASTQ file {}: {err}", path.display()))
}

fn parse_fastx_from_zip(path: &Path) -> Box<dyn needletail::FastxReader> {
    let file = File::open(path)
        .unwrap_or_else(|err| panic!("Failed to open zip archive {}: {err}", path.display()));
    let mut archive = zip::ZipArchive::new(file)
        .unwrap_or_else(|err| panic!("Failed to read zip archive {}: {err}", path.display()));

    for idx in 0..archive.len() {
        let mut entry = archive.by_index(idx).unwrap_or_else(|err| {
            panic!(
                "Failed to access member #{idx} in zip archive {}: {err}",
                path.display()
            )
        });
        if entry.is_dir() || !is_fastx_member_name(entry.name()) {
            continue;
        }

        let member_name = entry.name().to_owned();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).unwrap_or_else(|err| {
            panic!(
                "Failed reading zip member {member_name} in {}: {err}",
                path.display()
            )
        });

        return needletail::parse_fastx_reader(std::io::Cursor::new(bytes)).unwrap_or_else(|err| {
            panic!(
                "Failed to parse FASTA/FASTQ member {member_name} in {}: {err}",
                path.display()
            )
        });
    }

    panic!(
        "No FASTA/FASTQ member found in zip archive {}.",
        path.display()
    );
}

fn is_fastx_member_name(name: &str) -> bool {
    let lowercase = name.to_ascii_lowercase();
    let suffixes = [
        ".fa", ".fasta", ".fq", ".fastq", ".gz", ".bz2", ".xz", ".zst", ".zstd",
    ];
    suffixes.iter().any(|suffix| lowercase.ends_with(suffix))
}

fn has_extension(path: &Path, ext: &str) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

fn is_supported_input_path(path: &Path) -> bool {
    [
        "fa", "fasta", "fq", "fastq", "gz", "bz2", "xz", "zst", "zstd", "zip", EXTENSION,
    ]
    .iter()
    .any(|ext| has_extension(path, ext))
}

fn collect_paths(paths: &Vec<PathBuf>) -> Vec<PathBuf> {
    let mut res = vec![];
    for path in paths {
        if path.is_dir() {
            res.extend(path.read_dir().unwrap().map(|entry| entry.unwrap().path()));
        } else {
            res.push(path.clone());
        }
    }
    res.sort();

    res.retain(|p| is_supported_input_path(p));
    res
}
