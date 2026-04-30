//! Side-by-side retrieval demo so you can eyeball quality, not just speed.
//!
//! Builds a small themed corpus, runs the same queries through both models,
//! and prints top-k by cosine similarity. The docs are deliberately diverse
//! (chat, code, browser, kb) and the queries are paraphrases — not exact
//! keyword matches — so you can see whether the embedding actually captures
//! meaning vs surface form.

use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use model2vec_rs::model::StaticModel;

/// A miniature "what would I actually search for in goldfish" corpus.
/// Each entry imitates a snapshot you might capture during a workday.
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
        // 9: jira ticket (similar app, different ecosystem)
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
    ]
}

pub fn queries() -> Vec<&'static str> {
    vec![
        "what was that production outage about",
        "rust code for embedding documents",
        "how much are we spending on cloud",
        "the bug where the indexer freezes",
        "switching from candle to a faster embedder",
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

pub fn run_search_demo() -> Result<()> {
    let corpus = search_corpus();
    let queries = queries();
    let corpus_strings: Vec<String> = corpus.iter().map(|s| s.to_string()).collect();

    println!("\n=== retrieval quality demo ===");
    println!("{} docs, {} queries; printing top-3 per query for each model.\n", corpus.len(), queries.len());

    // ---- model2vec ----
    let m2v = StaticModel::from_pretrained("minishlab/potion-base-8M", None, None, None)?;
    let m2v_corpus = m2v.encode(&corpus_strings);
    let m2v_queries: Vec<Vec<f32>> = queries
        .iter()
        .map(|q| m2v.encode_single(q))
        .collect();

    // ---- fastembed ----
    let mut fe = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(false),
    )?;
    let fe_corpus = fe.embed(corpus.clone(), None)?;
    let fe_queries = fe.embed(queries.clone(), None)?;

    for (qi, q) in queries.iter().enumerate() {
        println!("Q: \"{}\"", q);
        println!("  [model2vec/potion-8M]");
        for (rank, (doc_i, score)) in topk(&m2v_queries[qi], &m2v_corpus, 3).iter().enumerate() {
            println!("    {}. ({:.3}) {}", rank + 1, score, snippet(corpus[*doc_i], 90));
        }
        println!("  [fastembed/bge-small]");
        for (rank, (doc_i, score)) in topk(&fe_queries[qi], &fe_corpus, 3).iter().enumerate() {
            println!("    {}. ({:.3}) {}", rank + 1, score, snippet(corpus[*doc_i], 90));
        }
        println!();
    }

    Ok(())
}
