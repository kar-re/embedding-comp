use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BROWSER_TAB: &str = "Persons • PostHog
Tab Group picker
uBlock Origin Lite
Bitwarden
smart search field
https://us.posthog.com/project/393858/persons/f6f1e003-e814-5253-84eb-2fbfd82b61c7
Tab Overview
Show Sidebar
Show Tab Bar
Reload This Page
Persons
Person ID
Distinct IDs
First seen
Last seen
Total events
Filters
Add filter
Search persons
person@example.com
Properties
$browser: Chrome 132.0.6834.84
$os: macOS 15.4
$device_type: Desktop
$current_url: https://app.example.com/dashboard
$session_id: 01935d9e-...
$pageview_count: 47
Identified
Activity tab
Sessions tab
Cohorts tab
Recordings tab
";

const VSCODE_DUMP: &str = "TODO: • Untitled-3 — goldfish
Untitled-2
Cargo.lock
macos.rs
panel
explorer
search
source control
extensions
src/main.rs
src/lib.rs
src/db/mod.rs
src/embeddings.rs
src/snapshot.rs
fn capture_window() -> Result<Snapshot> {
    let pid = active_window_pid()?;
    let title = window_title(pid)?;
    let content = extract_axtree(pid)?;
    Ok(Snapshot { pid, title, content })
}
problems
output
debug console
terminal
status bar
ln 142 col 18 spaces: 4 utf-8 lf rust
";

const CHAT_LOG: &str = "Slack
DMs
Channels
#engineering
#product
#general
@kaspian
just pushed the embedding refactor — switching to model2vec for the snapshot indexer
the cpu usage from candle was killing my battery on long sessions
that should bring it down by like 10x i think
also need to figure out a sensible chunking strategy, some of these snapshots are massive
yeah totally
let's pair on it tomorrow morning?
sure, 9am works
";

const TODO_NOTES: &str = "TODO list
- finish the goldfish indexer rewrite
- benchmark model2vec vs fastembed
- look at the long-tail snapshots, p99 is 88kb of mostly repeated UI strings
- dedupe content before embedding to save cycles
- write a tiny rerank step on top of fts5 results
- decide on hnsw vs flat for 506 docs (probably flat, lol)
- move chronicle generation to a background queue
- add a feature flag for the new search path
- ship by EOW
";

const CODE_BLOCK: &str = "use anyhow::Result;
use std::time::Instant;

pub struct Indexer {
    pub model: StaticModel,
    pub store: VectorStore,
}

impl Indexer {
    pub fn new(model: StaticModel, store: VectorStore) -> Self {
        Self { model, store }
    }

    pub fn index(&mut self, snapshot: &Snapshot) -> Result<()> {
        let t0 = Instant::now();
        let chunks = chunk(&snapshot.content, 512);
        let vecs = self.model.encode(&chunks);
        for (chunk, vec) in chunks.iter().zip(vecs.iter()) {
            self.store.insert(snapshot.id, chunk, vec)?;
        }
        tracing::debug!(snapshot = %snapshot.id, dur_ms = ?t0.elapsed().as_millis(), \"indexed\");
        Ok(())
    }
}
";

const TEMPLATES: &[&str] = &[BROWSER_TAB, VSCODE_DUMP, CHAT_LOG, TODO_NOTES, CODE_BLOCK];

/// Build a document of approximately `target_len` bytes by repeating + slicing
/// templates. Mirrors the repetitive nature of accessibility-extracted window text.
fn make_doc(rng: &mut StdRng, target_len: usize) -> String {
    let mut buf = String::with_capacity(target_len + 256);
    while buf.len() < target_len {
        let tpl = TEMPLATES[rng.gen_range(0..TEMPLATES.len())];
        buf.push_str(tpl);
        buf.push('\n');
    }
    // Walk back to a char boundary so truncate doesn't split a multi-byte codepoint.
    let mut cut = target_len.min(buf.len());
    while cut > 0 && !buf.is_char_boundary(cut) {
        cut -= 1;
    }
    buf.truncate(cut);
    buf
}

/// Generate ~500 synthetic docs with a length distribution that mirrors the
/// goldfish snapshots table:
///   count=506, avg=7886, p10=291, p50=2792, p90=18802, p95=24533, p99=87837, max=141243
pub fn synthesize(seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut docs = Vec::with_capacity(506);

    // Approximate the empirical distribution by sampling from buckets that
    // match the observed percentiles. Bucket: (count, min_len, max_len).
    let buckets: &[(usize, usize, usize)] = &[
        (50, 50, 400),       // bottom 10%: tiny window titles, login screens
        (200, 400, 3000),    // 10-50%: short browser/chat windows
        (200, 3000, 19000),  // 50-90%: typical multi-tab browsers, vscode
        (40, 19000, 25000),  // 90-95%: large tab dumps
        (15, 25000, 90000),  // 95-99%: very large
        (1, 90000, 141000),  // top 1%: massive (one outlier)
    ];

    for &(count, lo, hi) in buckets {
        for _ in 0..count {
            let len = rng.gen_range(lo..hi);
            docs.push(make_doc(&mut rng, len));
        }
    }

    docs
}

/// Mirror what `goldfish_d/src/search/vector.rs` does before handing text to
/// the embedder:
///   1. drop anything < 32 chars (falls back to BM25 in the real indexer)
///   2. truncate to 4096 bytes at a UTF-8 boundary
const MIN_CHARS_TO_EMBED: usize = 32;
const MAX_INPUT_BYTES: usize = 4096;

pub fn prepare_for_embedding(docs: &[String]) -> Vec<String> {
    docs.iter()
        .filter(|d| d.chars().count() >= MIN_CHARS_TO_EMBED)
        .map(|d| {
            if d.len() <= MAX_INPUT_BYTES {
                d.clone()
            } else {
                let mut cut = MAX_INPUT_BYTES;
                while cut > 0 && !d.is_char_boundary(cut) {
                    cut -= 1;
                }
                d[..cut].to_string()
            }
        })
        .collect()
}

pub fn corpus_stats(docs: &[String]) -> (usize, usize, usize, usize) {
    let mut lens: Vec<usize> = docs.iter().map(|d| d.len()).collect();
    lens.sort_unstable();
    let total: usize = lens.iter().sum();
    let avg = total / lens.len();
    let p50 = lens[lens.len() / 2];
    let p95 = lens[lens.len() * 95 / 100];
    (total, avg, p50, p95)
}
