//! Side-by-side retrieval demo.
//!
//! Compares all four model2vec "potion" variants and fastembed/BGE-small on
//! the same corpus and queries. The corpus includes a Swedish entry and the
//! queries include a Swedish paraphrase, since real goldfish snapshots are
//! mixed-language.

use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use model2vec_rs::model::StaticModel;

pub fn search_corpus() -> Vec<&'static str> {
    vec![
        // 0: payments incident
        "Slack #incidents — payments service returning 502s for the last 12 minutes, \
         on-call paged, rolling back the deploy that landed at 14:32",
        // 1: stripe dashboard
        "Stripe Dashboard — Disputes — chargeback for $129.00 marked needs_response, \
         due in 6 days, customer claims item never arrived",
        // 2: rust embeddings code
        "vscode src/embeddings.rs — fn embed_batch(model: &StaticModel, docs: &[String]) \
         -> Vec<Vec<f32>> { model.encode(docs) }  // TODO: chunk long docs",
        // 3: react component
        "vscode src/components/SearchBar.tsx — function SearchBar({ onQuery }) { \
         const [q, setQ] = useState(''); return <input value={q} onChange=... />; }",
        // 4: meeting notes
        "Notes — 1:1 with manager — discussed Q2 goals, agreed to focus on \
         indexer performance and shipping the new search experience by end of May",
        // 5: github PR
        "GitHub PR #482 — refactor: replace candle-based embedder with model2vec — \
         drops indexer CPU usage from 90% to under 5%, ready for review",
        // 6: docs page
        "Model2Vec documentation — static embeddings produced by distilling a \
         sentence-transformer; no transformer at inference time, pure CPU lookup + pooling",
        // 7: aws bill
        "AWS Console — Cost Explorer — April spend $4,213, up 18% from March, \
         driven by RDS storage growth on the analytics cluster",
        // 8: linear ticket
        "Linear GOLD-128 — Bug: snapshot indexer occasionally hangs on safari \
         windows with very long accessibility trees, repro on macOS 15.4",
        // 9: jira ticket
        "Jira PROJ-1923 — Feature: add semantic search to chronicle viewer, \
         priority high, assigned to backend team",
        // 10: spotify
        "Spotify — Now Playing — Lo-fi beats to code to — Chillhop Music — paused",
        // 11: terminal
        "ghostty — cargo build --release — Compiling fastembed v5.13.4, \
         Compiling ort v2.0.0-rc.12, Finished `release` profile in 22.49s",
        // 12: postgres query
        "TablePlus — SELECT count(*) FROM snapshots WHERE captured_at > \
         extract(epoch from now() - interval '7 days') — 14223 rows",
        // 13: legal/admin
        "DocuSign — Employment agreement amendment — please review and sign by Friday, \
         updates equity vesting schedule per board approval",
        // 14: nyt
        "Safari — nytimes.com — Federal Reserve holds rates steady amid mixed \
         economic signals, markets close mostly flat",
        // 15: Swedish standup notes (mirrors what shows up in real goldfish data)
        "Notion — Standup — Idag: jobbade vidare på workflow-detektering, \
         förbättrade CPU-användningen i indexer:n, började titta på flerspråkig sökning",
    ]
}

pub fn queries() -> Vec<&'static str> {
    vec![
        // Abstract paraphrase — the one base-8M missed
        "what was that production outage about",
        // Concrete keyword
        "rust code for embedding documents",
        // Topical paraphrase
        "how much are we spending on cloud",
        // Concrete keyword
        "the bug where the indexer freezes",
        // Topical
        "switching from candle to a faster embedder",
        // Swedish — only multilingual model should land this
        "förbättrad prestanda i sökindexet",
    ]
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-9);
    dot / denom
}

fn topk(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine(query, v)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

fn snippet(s: &str, n: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let cut: String = s.chars().take(n).collect();
        format!("{cut}...")
    }
}

/// Per-(model, query) top-1 result, used to print a compact comparison table.
struct Top1 {
    doc_idx: usize,
    score: f32,
}

pub fn run_search_demo() -> Result<()> {
    let corpus = search_corpus();
    let queries = queries();
    let corpus_strings: Vec<String> = corpus.iter().map(|s| s.to_string()).collect();

    println!("\n=== retrieval quality demo ===");
    println!(
        "{} docs (incl. 1 Swedish), {} queries (incl. 1 Swedish).",
        corpus.len(),
        queries.len()
    );

    let m2v_models: &[(&str, &str)] = &[
        ("potion-base-8M",         "minishlab/potion-base-8M"),
        ("potion-base-32M",        "minishlab/potion-base-32M"),
        ("potion-retrieval-32M",   "minishlab/potion-retrieval-32M"),
        ("potion-multi-128M",      "minishlab/potion-multilingual-128M"),
    ];

    // model_label -> Vec<Top1 per query>
    let mut all_top1: Vec<(String, Vec<Top1>)> = Vec::new();
    // Same data, full top-3 for the detailed dump.
    let mut detailed: Vec<(String, Vec<Vec<(usize, f32)>>)> = Vec::new();

    for (label, repo) in m2v_models {
        println!("[{label}] loading...");
        let m = StaticModel::from_pretrained(repo, None, None, None)?;
        let c_vecs = m.encode(&corpus_strings);
        let q_vecs: Vec<Vec<f32>> = queries.iter().map(|q| m.encode_single(q)).collect();

        let mut top1 = Vec::new();
        let mut top3 = Vec::new();
        for qv in &q_vecs {
            let ranked = topk(qv, &c_vecs, 3);
            top1.push(Top1 { doc_idx: ranked[0].0, score: ranked[0].1 });
            top3.push(ranked);
        }
        all_top1.push((label.to_string(), top1));
        detailed.push((label.to_string(), top3));
    }

    // fastembed
    println!("[bge-small-en] loading...");
    let mut fe = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(false),
    )?;
    let fe_corpus = fe.embed(corpus.clone(), None)?;
    let fe_queries = fe.embed(queries.clone(), None)?;
    {
        let mut top1 = Vec::new();
        let mut top3 = Vec::new();
        for qv in &fe_queries {
            let ranked = topk(qv, &fe_corpus, 3);
            top1.push(Top1 { doc_idx: ranked[0].0, score: ranked[0].1 });
            top3.push(ranked);
        }
        all_top1.push(("bge-small-en".into(), top1));
        detailed.push(("bge-small-en".into(), top3));
    }

    // Compact comparison: top-1 doc index per (model, query). Easy to scan.
    // Each query has a "right answer" — annotate so the user can spot misses at a glance.
    let expected_idx: &[usize] = &[
        0,  // production outage  -> Slack incidents
        2,  // rust embedding code -> embeddings.rs
        7,  // cloud spend         -> AWS Cost Explorer
        8,  // indexer freezes     -> Linear GOLD-128
        5,  // candle -> faster    -> GitHub PR #482
        15, // Swedish: prestanda  -> Notion standup
    ];

    println!("\n=== top-1 per query (✓ = expected doc, ✗ = miss) ===");
    print!("{:<24}", "model");
    for qi in 0..queries.len() {
        print!(" {:>9}", format!("Q{}", qi + 1));
    }
    println!();
    println!("{}", "-".repeat(24 + queries.len() * 10));
    for (label, top1) in &all_top1 {
        print!("{:<24}", label);
        for (qi, t1) in top1.iter().enumerate() {
            let mark = if t1.doc_idx == expected_idx[qi] { "✓" } else { "✗" };
            print!(" {:>4}{:<5}", mark, format!("({:.2})", t1.score));
        }
        println!();
    }
    println!("\nlegend (queries):");
    for (qi, q) in queries.iter().enumerate() {
        println!("  Q{}: {}", qi + 1, q);
    }

    // Detailed top-3 for each model (so the user can see *which wrong doc* gets ranked).
    println!("\n=== detailed top-3 results ===");
    for (label, top3_per_q) in &detailed {
        println!("\n[{label}]");
        for (qi, ranked) in top3_per_q.iter().enumerate() {
            println!("  Q{}: \"{}\"", qi + 1, queries[qi]);
            for (rank, (doc_i, score)) in ranked.iter().enumerate() {
                let mark = if rank == 0 && *doc_i == expected_idx[qi] { "✓" }
                    else if rank == 0 { "✗" } else { " " };
                println!(
                    "    {} {}. ({:.3}) {}",
                    mark,
                    rank + 1,
                    score,
                    snippet(corpus[*doc_i], 88)
                );
            }
        }
    }

    Ok(())
}
