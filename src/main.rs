mod data;
mod search;

use anyhow::Result;
use cpu_time::ProcessTime;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use model2vec_rs::model::StaticModel;
use std::time::{Duration, Instant};

#[derive(Debug)]
struct RunStats {
    label: String,
    docs: usize,
    total_bytes: usize,
    wall: Duration,
    cpu: Duration,
    /// Per-doc wall latencies, sorted. Empty for batched runs.
    latencies_sorted: Vec<Duration>,
    dim: usize,
}

impl RunStats {
    fn cpu_pct(&self) -> f64 {
        if self.wall.as_secs_f64() == 0.0 {
            return 0.0;
        }
        (self.cpu.as_secs_f64() / self.wall.as_secs_f64()) * 100.0
    }

    fn per_doc_ms(&self) -> f64 {
        self.wall.as_secs_f64() * 1000.0 / self.docs as f64
    }

    fn cpu_per_mb_ms(&self) -> f64 {
        let mb = self.total_bytes as f64 / (1024.0 * 1024.0);
        if mb == 0.0 {
            return 0.0;
        }
        self.cpu.as_secs_f64() * 1000.0 / mb
    }

    fn pct_latency_ms(&self, p: f64) -> Option<f64> {
        if self.latencies_sorted.is_empty() {
            return None;
        }
        let idx = ((self.latencies_sorted.len() as f64) * p) as usize;
        let idx = idx.min(self.latencies_sorted.len() - 1);
        Some(self.latencies_sorted[idx].as_secs_f64() * 1000.0)
    }
}

fn time_block<T>(f: impl FnOnce() -> T) -> (T, Duration, Duration) {
    let cpu_t0 = ProcessTime::now();
    let wall_t0 = Instant::now();
    let out = f();
    let wall = wall_t0.elapsed();
    let cpu = cpu_t0.elapsed();
    (out, wall, cpu)
}

// ---------- batched runs (one big call) ----------

fn bench_model2vec_batched(label: &str, repo: &str, docs: &[String]) -> Result<RunStats> {
    println!("[model2vec/batched] loading {repo} ...");
    let model = StaticModel::from_pretrained(repo, None, None, None)?;
    let _ = model.encode_single(&docs[0]);

    println!("[model2vec/batched] embedding {} docs in one call ...", docs.len());
    let (vecs, wall, cpu) = time_block(|| model.encode(docs));

    Ok(RunStats {
        label: label.into(),
        docs: docs.len(),
        total_bytes: docs.iter().map(|d| d.len()).sum(),
        wall,
        cpu,
        latencies_sorted: vec![],
        dim: vecs.first().map(|v| v.len()).unwrap_or(0),
    })
}

fn bench_fastembed_batched(label: &str, docs: &[String]) -> Result<RunStats> {
    println!("[fastembed/batched] loading BGESmallENV15 ...");
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(true),
    )?;
    let _ = model.embed(vec![docs[0].as_str()], None)?;

    println!("[fastembed/batched] embedding {} docs in one call ...", docs.len());
    let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
    let (vecs_res, wall, cpu) = time_block(|| model.embed(refs, None));
    let vecs = vecs_res?;

    Ok(RunStats {
        label: label.into(),
        docs: docs.len(),
        total_bytes: docs.iter().map(|d| d.len()).sum(),
        wall,
        cpu,
        latencies_sorted: vec![],
        dim: vecs.first().map(|v| v.len()).unwrap_or(0),
    })
}

// ---------- single-item runs (mirrors goldfish's hot path) ----------
//
// goldfish_d/src/search/vector.rs serializes embed calls through Mutex<TextEmbedding>
// and embeds one snapshot at a time. This is what blocks the capture loop, so it's
// the metric that matters for "will this make my laptop fan spin up".

fn bench_model2vec_single(label: &str, repo: &str, docs: &[String]) -> Result<RunStats> {
    println!("[model2vec/single] loading {repo} ...");
    let model = StaticModel::from_pretrained(repo, None, None, None)?;
    let _ = model.encode_single(&docs[0]);

    println!("[model2vec/single] embedding {} docs one at a time ...", docs.len());
    let mut latencies = Vec::with_capacity(docs.len());
    let cpu_t0 = ProcessTime::now();
    let wall_t0 = Instant::now();
    let mut last_dim = 0;
    for d in docs {
        let t0 = Instant::now();
        let v = model.encode_single(d);
        latencies.push(t0.elapsed());
        last_dim = v.len();
    }
    let wall = wall_t0.elapsed();
    let cpu = cpu_t0.elapsed();
    latencies.sort();

    Ok(RunStats {
        label: label.into(),
        docs: docs.len(),
        total_bytes: docs.iter().map(|d| d.len()).sum(),
        wall,
        cpu,
        latencies_sorted: latencies,
        dim: last_dim,
    })
}

fn bench_fastembed_single(label: &str, docs: &[String]) -> Result<RunStats> {
    println!("[fastembed/single] loading BGESmallENV15 ...");
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(false),
    )?;
    let _ = model.embed(vec![docs[0].as_str()], None)?;

    println!("[fastembed/single] embedding {} docs one at a time ...", docs.len());
    let mut latencies = Vec::with_capacity(docs.len());
    let cpu_t0 = ProcessTime::now();
    let wall_t0 = Instant::now();
    let mut last_dim = 0;
    for d in docs {
        let t0 = Instant::now();
        let v = model.embed(vec![d.as_str()], None)?;
        latencies.push(t0.elapsed());
        last_dim = v.first().map(|v| v.len()).unwrap_or(0);
    }
    let wall = wall_t0.elapsed();
    let cpu = cpu_t0.elapsed();
    latencies.sort();

    Ok(RunStats {
        label: label.into(),
        docs: docs.len(),
        total_bytes: docs.iter().map(|d| d.len()).sum(),
        wall,
        cpu,
        latencies_sorted: latencies,
        dim: last_dim,
    })
}

// ---------- output ----------

fn print_table(title: &str, results: &[RunStats]) {
    println!("\n=== {title} ===");
    println!(
        "{:<26} {:>6} {:>9} {:>9} {:>8} {:>9} {:>9} {:>11} {:>5}",
        "library", "docs", "wall(s)", "cpu(s)", "cpu%", "p50(ms)", "p95(ms)", "cpu_ms/MB", "dim"
    );
    println!("{}", "-".repeat(100));
    for s in results {
        let p50 = s.pct_latency_ms(0.50).map(|v| format!("{v:.2}")).unwrap_or_else(|| "-".into());
        let p95 = s.pct_latency_ms(0.95).map(|v| format!("{v:.2}")).unwrap_or_else(|| "-".into());
        println!(
            "{:<26} {:>6} {:>9.2} {:>9.2} {:>7.1}% {:>9} {:>9} {:>11.1} {:>5}",
            s.label,
            s.docs,
            s.wall.as_secs_f64(),
            s.cpu.as_secs_f64(),
            s.cpu_pct(),
            p50,
            p95,
            s.cpu_per_mb_ms(),
            s.dim,
        );
    }
}

fn main() -> Result<()> {
    let raw = data::synthesize(0xC0FFEE);
    let docs = data::prepare_for_embedding(&raw);
    let (total, avg, p50, p95) = data::corpus_stats(&docs);
    println!(
        "raw corpus: {} docs (full goldfish-shaped distribution)",
        raw.len()
    );
    println!(
        "after goldfish prep (>=32 chars, <=4096 bytes): {} docs, {:.2} MB total, avg={}B, p50={}B, p95={}B",
        docs.len(),
        total as f64 / (1024.0 * 1024.0),
        avg,
        p50,
        p95,
    );
    println!("(prep mirrors vector.rs: MIN_CHARS_TO_EMBED=32, MAX_INPUT_BYTES=4096)\n");

    // The hot path in goldfish: one-at-a-time, behind a Mutex.
    // This is the headline metric.
    let m2v_models: &[(&str, &str)] = &[
        ("m2v/potion-base-8M",         "minishlab/potion-base-8M"),
        ("m2v/potion-base-32M",        "minishlab/potion-base-32M"),
        ("m2v/potion-retrieval-32M",   "minishlab/potion-retrieval-32M"),
        ("m2v/potion-multi-128M",      "minishlab/potion-multilingual-128M"),
    ];
    let mut single = Vec::new();
    for (label, repo) in m2v_models {
        single.push(bench_model2vec_single(label, repo, &docs)?);
    }
    single.push(bench_fastembed_single("fastembed/bge-small-en", &docs)?);
    print_table("single-item embed loop (matches goldfish's indexer)", &single);

    // Batched: only show 8M m2v + fastembed for ceiling reference; the others
    // scale predictably with vocab size.
    let batched = vec![
        bench_model2vec_batched("m2v/potion-base-8M", "minishlab/potion-base-8M", &docs)?,
        bench_fastembed_batched("fastembed/bge-small-en", &docs)?,
    ];
    print_table("batched embed (one big call -- bulk reindex scenario)", &batched);

    println!("\nlegend:");
    println!("  cpu%       = cpu_time / wall_time x 100. >100% means multi-threaded.");
    println!("  p50/p95    = per-doc latency percentiles (single-item runs only).");
    println!("  cpu_ms/MB  = cpu cost normalized by input bytes.");
    println!();
    println!("the single-item table is the one that matters: goldfish embeds one snapshot");
    println!("at a time through Mutex<TextEmbedding>. p95 is your worst-case capture-loop");
    println!("stall; total cpu(s) is the steady-state battery cost over a workday.");

    search::run_search_demo()?;
    Ok(())
}
