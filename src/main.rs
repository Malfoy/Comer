use std::{
    fs::File,
    io::Seek,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
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
        /// Paths to (directories of) (gzipped) fasta files.
        paths: Vec<PathBuf>,
        /// Parse kmers from bcalm/logan FASTA headers (km:f:)
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
        /// First input fasta file or .ssketch file.
        path_a: PathBuf,
        /// Second input fasta file or .ssketch file.
        path_b: PathBuf,
        /// Parse kmers from bcalm/logan FASTA headers (km:f:)
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
        /// Paths to (directories of) (gzipped) fasta files or .ssketch files.
        /// If <path>.ssketch exists, it is automatically used.
        paths: Vec<PathBuf>,
        /// Parse kmers from bcalm/logan FASTA headers (km:f:)
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
        /// Paths to directory of (gzipped) fasta files.
        #[arg(long)]
        targets: Vec<PathBuf>,
        /// Parse kmers from bcalm/logan FASTA headers (km:f:)
        #[arg(long)]
        bcalm: bool,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
        #[arg(long)]
        no_save: bool,

        /// Path to .fastq.gz metagenomic sample
        reads: PathBuf,
    },
    /// List k-mers in the intersection of two sketches.
    Intersect {
        #[command(flatten)]
        params: SketchParams,
        /// First input fasta file or .ssketch file.
        path_a: PathBuf,
        /// Second input fasta file or .ssketch file.
        path_b: PathBuf,
        /// Parse kmers from bcalm/logan FASTA headers (km:f:)
        #[arg(long)]
        bcalm: bool,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
    },
    /// Output sourmash-style JSON signatures to stdout.
    Signature {
        #[command(flatten)]
        params: SketchParams,
        /// Paths to (directories of) (gzipped) fasta files or .ssketch files.
        paths: Vec<PathBuf>,
        /// Parse kmers from bcalm/logan FASTA headers (km:f:)
        #[arg(long)]
        bcalm: bool,
        /// Write JSON output here, or default to stdout.
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, short = 'j')]
        threads: Option<usize>,
    },
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

            let mut reader = needletail::parse_fastx_file(&path).unwrap();

            let mut sketch;
            if params.filter_out_n {
                let mut ranges = vec![];
                let mut abundances = vec![];
                let mut seq = PackedNSeqVec::default();
                let mut size = 0;
                while let Some(r) = reader.next() {
                    let record = r.unwrap();
                    let abundance = if bcalm {
                        parse_bcalm_abundance(record.id())
                    } else {
                        1
                    };
                    let range = seq.push_ascii(&record.seq());
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
                    let abundance = if bcalm {
                        parse_bcalm_abundance(record.id())
                    } else {
                        1
                    };
                    let range = seq.push_ascii(&record.seq());
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

fn parse_bcalm_abundance(header: &[u8]) -> u16 {
    let mut abundance = None;

    let parse_token = |token: &[u8]| -> Option<u16> {
        if let Some(rest) = token.strip_prefix(b"km:f:") {
            let s = str::from_utf8(rest).ok()?;
            let v: f32 = s.parse().ok()?;
            if !v.is_finite() {
                return None;
            }
            let v = v.round().clamp(0.0, u16::MAX as f32);
            return Some(v as u16);
        }
        if let Some(rest) = token.strip_prefix(b"km:i:") {
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

    let extensions = [
        "fa", "fasta", "fq", "fastq", "gz", "fasta.gz", "fq.gz", "fastq.gz",
    ];
    res.retain(|p| extensions.iter().any(|e| p.extension().unwrap() == *e));
    res
}
