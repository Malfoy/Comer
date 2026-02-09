use std::{collections::HashMap, fs::File, io::Read, path::Path};

use itertools::Itertools;
use log::info;
use mem_dbg::{MemSize, SizeFlags};
use packed_seq::PackedNSeqVec;

use crate::{BitSketch, Sketch};

pub fn classify(sketches: &[Sketch], reads: &Path) {
    let mut params = sketches[0].to_params();
    params.filter_out_n = true;

    let size = sketches.mem_size(SizeFlags::default());
    eprintln!("sketches size: {}kB", size / 1024);

    let mut max_sketch = vec![0; params.s];
    let mut counts = vec![HashMap::<u32, u32>::new(); params.s];
    // inverted index
    let mut occ = HashMap::<u32, Vec<u16>>::new();
    for (j, sketch) in sketches.iter().enumerate() {
        let Sketch::BucketSketch(bucket_sketch) = sketch else {
            panic!()
        };
        let BitSketch::B32(buckets) = &bucket_sketch.buckets else {
            panic!()
        };
        assert_eq!(buckets.len(), params.s);
        for i in 0..params.s {
            let v = buckets[i] * params.s as u32 + i as u32;
            max_sketch[i] = max_sketch[i].max(v);
            *counts[i].entry(v).or_default() += 1;
            occ.entry(v).or_default().push(j as u16);
        }
    }
    eprintln!("{max_sketch:?}");
    let avg = max_sketch.iter().map(|x| *x as usize).sum::<usize>() / max_sketch.len();
    let threshold = max_sketch.iter().max().unwrap();
    eprintln!("avg: {avg:?}");
    eprintln!("max: {threshold:?}");

    let lens = counts.iter().map(|s| s.len()).collect_vec();
    eprintln!("#distinct kmers in each bucket: {lens:?}");

    let mut counts = counts[0].values().collect_vec();
    counts.sort();
    eprintln!("kmer counts for bucket 0: {counts:?}");

    // Flatten `occ`
    let mut occ_flat: Vec<u16> = vec![];
    let mut ranges = HashMap::new();
    for (hash, vals) in &occ {
        let l0 = occ_flat.len() as u32;
        occ_flat.extend_from_slice(&vals);
        let l1 = occ_flat.len() as u32;
        ranges.insert(*hash, l0..l1);
    }
    eprintln!("#occ total: {}", occ_flat.len());

    eprintln!(
        "Size of occ: {}MB",
        occ.mem_size(Default::default()) / 1024 / 1024
    );
    eprintln!(
        "Size of occ_flat: {}MB",
        occ_flat.mem_size(Default::default()) / 1024 / 1024
    );
    eprintln!(
        "Size of ranges: {}MB",
        ranges.mem_size(Default::default()) / 1024 / 1024
    );

    let mut counts = vec![0; sketches.len()];
    let mut per_bucket_counts = vec![HashMap::<u32, u32>::new(); sketches.len()];

    info!("Params: {params:?}");
    let sketcher = params.build();

    // input
    info!("Reading..");
    let seq = read_fastq_with_quality(reads, 20);
    info!("Sketching..");

    let mut read_hashes = vec![];
    sketcher.collect_up_to_bound(
        &[seq.as_slice()],
        *threshold,
        &mut read_hashes,
        8000,
        |hashes| {
            for h in &*hashes {
                let Some(range) = ranges.get(h) else {
                    continue;
                };
                for &j in &occ_flat[range.start as usize..range.end as usize] {
                    counts[j as usize] += 1;
                    *per_bucket_counts[j as usize].entry(*h).or_default() += 1;
                }
            }
            hashes.clear();
            *threshold
        },
    );
    counts.sort();
    info!("Number of matching kmers per target: {counts:?}");
    per_bucket_counts.sort_by_cached_key(|hm| hm.values().sum::<u32>());
    let per_bucket_counts = per_bucket_counts
        .iter()
        .map(|x| {
            let mut counts = x.values().copied().collect_vec();
            counts.sort();
            counts
                .iter()
                .chunk_by(|x| **x)
                .into_iter()
                .map(|(key, g)| (key, g.count()))
                .collect_vec()
        })
        .collect_vec();
    info!("Number of matching kmers per target and bucket: {per_bucket_counts:?}");
}

fn read_fastq_with_quality(path: &Path, min_qual: u8) -> PackedNSeqVec {
    if has_extension(path, "zip") {
        return read_fastq_with_quality_from_zip(path, min_qual);
    }
    PackedNSeqVec::from_fastq_with_quality(path, min_qual)
}

fn read_fastq_with_quality_from_zip(path: &Path, min_qual: u8) -> PackedNSeqVec {
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
        if entry.is_dir() || !is_fastq_member_name(entry.name()) {
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

        let mut reader = needletail::parse_fastx_reader(std::io::Cursor::new(bytes))
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to parse FASTQ member {member_name} in {}: {err}",
                    path.display()
                )
            });
        return packed_nseq_from_fastq_reader(&mut reader, min_qual, path, &member_name);
    }

    panic!(
        "No FASTQ member found in zip archive {}. Expected one of: .fq/.fastq (optionally compressed).",
        path.display()
    );
}

fn packed_nseq_from_fastq_reader(
    reader: &mut Box<dyn needletail::FastxReader>,
    min_qual: u8,
    archive_path: &Path,
    member_name: &str,
) -> PackedNSeqVec {
    let mut dna = Vec::new();
    let mut qual = Vec::new();
    let mut seqs = PackedNSeqVec::default();

    while let Some(record) = reader.next() {
        let seqrec = record.unwrap_or_else(|err| {
            panic!(
                "Invalid FASTQ record in zip member {member_name} (archive {}): {err}",
                archive_path.display()
            )
        });
        let quality = seqrec.qual().unwrap_or_else(|| {
            panic!(
                "Expected FASTQ record with qualities in zip member {member_name} (archive {}).",
                archive_path.display()
            )
        });
        dna.extend_from_slice(&seqrec.seq());
        qual.extend_from_slice(quality);
        dna.push(b'N');
        qual.push(0);

        if dna.len() > 16000 {
            seqs.push_from_ascii_and_quality(&dna, &qual, min_qual);
            dna.clear();
            qual.clear();
        }
    }

    if !dna.is_empty() {
        seqs.push_from_ascii_and_quality(&dna, &qual, min_qual);
    }

    seqs
}

fn is_fastq_member_name(name: &str) -> bool {
    let lowercase = name.to_ascii_lowercase();
    let suffixes = [
        ".fq",
        ".fastq",
        ".fq.gz",
        ".fastq.gz",
        ".fq.bz2",
        ".fastq.bz2",
        ".fq.xz",
        ".fastq.xz",
        ".fq.zst",
        ".fastq.zst",
        ".fq.zstd",
        ".fastq.zstd",
    ];
    suffixes.iter().any(|suffix| lowercase.ends_with(suffix))
}

fn has_extension(path: &Path, ext: &str) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}
