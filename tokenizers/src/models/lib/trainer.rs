use crate::models::lib::model::LiBModel;
use crate::utils::progress::{ProgressBar, ProgressStyle};
use crate::{AddedToken, Result, Trainer};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// LiBTrainerBuilder
// ---------------------------------------------------------------------------

/// Builder for [`LiBTrainer`].
pub struct LiBTrainerBuilder {
    vocab_size: usize,
    num_epochs: usize,
    life: i32,
    max_len: usize,
    memory_in: f64,
    memory_out: f64,
    update_rate: f64,
    doc_size: usize,
    seed: Option<u64>,
    deterministic: bool,
    byte_fallback: bool,
    special_tokens: Vec<AddedToken>,
}

impl Default for LiBTrainerBuilder {
    fn default() -> Self {
        Self {
            vocab_size: 30000,
            num_epochs: 10000,
            // `life` is the probation period τ₀ (Yang et al. 2020, §2.3.2): how
            // many document-epochs a tail unit stays on probation before being
            // forgotten unless re-evaluated good. Paper used 10 (BR-phono) and
            // 500 (CTB8); retune per corpus with hparam_search.py.
            life: 10,
            max_len: 12,
            memory_in: 0.25,   // α: candidate-pair sampling probability
            memory_out: 0.0001, // ω: fraction of the tail probated per document
            update_rate: 0.2,  // Δ: ordinal re-ranking rate
            // A "document" per epoch = this many sentences (the paper's epoch
            // unit; §2.3.2). Paper documents held ~24–78 sentences.
            doc_size: 50,
            seed: None,
            deterministic: false,
            byte_fallback: true,
            special_tokens: Vec::new(),
        }
    }
}

impl LiBTrainerBuilder {
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        self
    }
    pub fn num_epochs(mut self, n: usize) -> Self {
        self.num_epochs = n;
        self
    }
    pub fn life(mut self, l: i32) -> Self {
        self.life = l;
        self
    }
    pub fn max_len(mut self, m: usize) -> Self {
        self.max_len = m;
        self
    }
    pub fn memory_in(mut self, m: f64) -> Self {
        self.memory_in = m;
        self
    }
    pub fn memory_out(mut self, m: f64) -> Self {
        self.memory_out = m;
        self
    }
    pub fn update_rate(mut self, r: f64) -> Self {
        self.update_rate = r;
        self
    }
    pub fn doc_size(mut self, n: usize) -> Self {
        self.doc_size = n.max(1);
        self
    }
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }
    pub fn deterministic(mut self, d: bool) -> Self {
        self.deterministic = d;
        self
    }
    pub fn byte_fallback(mut self, b: bool) -> Self {
        self.byte_fallback = b;
        self
    }
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.special_tokens = tokens;
        self
    }
    pub fn build(self) -> LiBTrainer {
        LiBTrainer {
            vocab_size: self.vocab_size,
            num_epochs: self.num_epochs,
            life: self.life,
            max_len: self.max_len,
            memory_in: self.memory_in,
            memory_out: self.memory_out,
            update_rate: self.update_rate,
            doc_size: self.doc_size,
            seed: self.seed,
            deterministic: self.deterministic,
            byte_fallback: self.byte_fallback,
            special_tokens: self.special_tokens,
            word_counts: HashMap::new(),
            char_set: HashSet::new(),
            sentences: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// LiBTrainer
// ---------------------------------------------------------------------------

/// LiB trainer: learns a vocabulary using the "Less is Better" algorithm.
///
/// Training follows the "Less is Better" process of Yang et al. (2020,
/// §2.3.1–2.3.2). Each epoch processes one *document* — a batch of `doc_size`
/// sentences:
/// 1. Initialize the lexicon with the corpus alphabet (and byte tokens).
/// 2. For each sentence: segment larger-first with chunk evaluation; memorize
///    adjacent candidate pairs by sampling with probability α (`memory_in`) and
///    admitting a pair once it has been sampled at least twice; then apply
///    active forgetting — re-rank each evaluated chunk by its good/bad verdict
///    (ordinal move by Δ = `update_rate`), deleting a chunk pushed past |L|.
/// 3. Once per document: passive forgetting — put the last ω (`memory_out`)
///    fraction of the lexicon on probation for τ₀ (`life`) epochs, forgetting
///    units whose probation expires unless they are re-evaluated good.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LiBTrainer {
    pub vocab_size: usize,
    pub num_epochs: usize,
    pub life: i32,
    pub max_len: usize,
    pub memory_in: f64,
    pub memory_out: f64,
    pub update_rate: f64,
    pub doc_size: usize,
    pub seed: Option<u64>,
    pub deterministic: bool,
    pub byte_fallback: bool,
    pub special_tokens: Vec<AddedToken>,

    #[serde(skip)]
    pub(crate) word_counts: HashMap<String, u64>,
    #[serde(skip)]
    pub(crate) char_set: HashSet<char>,
    #[serde(skip)]
    pub(crate) sentences: Vec<String>,
}

impl LiBTrainer {
    pub fn builder() -> LiBTrainerBuilder {
        LiBTrainerBuilder::default()
    }

    /// Ordinal (lexicon position) of a token; a large value if absent.
    fn ord(model: &LiBModel, tok: &str) -> i64 {
        model
            .trie
            .token_to_id(tok)
            .map(|i| i as i64)
            .unwrap_or(i64::MAX / 2)
    }

    /// Evaluate whether the longest chunk `c1` is redundant relative to the
    /// second-longest `c2nd` at `pos` (Yang et al. 2020, §2.3.1;
    /// `model.py::dropout`). Greedily extend both branches until they re-converge,
    /// counting chunks, unknown symbols, and ordinal sums. `c1` is *bad* if its
    /// branch has more unknowns, or (equal unknowns) more chunks, or (equal both)
    /// a higher ordinal sum. Returns true when `c1` is bad.
    fn eval_chunk(model: &LiBModel, sent: &str, pos: usize, c1: &str, c2nd: &str) -> bool {
        let (mut p0, mut nc0, mut nu0, mut ord0) =
            (pos + c1.len(), 1i64, 0i64, Self::ord(model, c1));
        let (mut p1, mut nc1, mut nu1, mut ord1) =
            (pos + c2nd.len(), 1i64, 0i64, Self::ord(model, c2nd));
        let mut guard = 0;
        while p0 != p1 && guard < 4096 {
            guard += 1;
            if p1 < p0 {
                let rest = &sent[p1..];
                match model.trie.match_longest(rest, model.max_len, false) {
                    Some((t, id)) => { p1 += t.len(); nc1 += 1; ord1 += id as i64; }
                    None => { p1 += rest.chars().next().unwrap().len_utf8(); nu1 += 1; nc1 += 1; }
                }
            } else {
                let rest = &sent[p0..];
                match model.trie.match_longest(rest, model.max_len, false) {
                    Some((t, id)) => { p0 += t.len(); nc0 += 1; ord0 += id as i64; }
                    None => { p0 += rest.chars().next().unwrap().len_utf8(); nu0 += 1; nc0 += 1; }
                }
            }
        }
        let redundant = nu0 == nu1 && (nc0 > nc1 || (nc0 == nc1 && ord0 > ord1));
        nu0 > nu1 || redundant
    }

    /// `c1` with its last character removed (UTF-8 safe).
    fn drop_last_char(s: &str) -> String {
        let mut chars: Vec<char> = s.chars().collect();
        chars.pop();
        chars.into_iter().collect()
    }

    /// Read one sentence (`model.py::reading` inner loop): greedy larger-first
    /// segmentation that always takes the longest match, memorizes adjacent
    /// candidate pairs (α-sampled, admitted on the second sighting within the
    /// document), records good/bad evaluations into `reward_list`, and cancels a
    /// used/good chunk's probation.
    #[allow(clippy::too_many_arguments)]
    fn read_sentence(
        model: &mut LiBModel,
        sent: &str,
        reward_list: &mut Vec<(String, i32)>,
        to_memorize: &mut HashSet<String>,
        to_dropout: &mut HashMap<String, i32>,
        rng: &mut Box<dyn RngCore>,
        alpha: f64,
        mini_gap: usize,
        vocab_size: usize,
    ) {
        let max_len = model.max_len;
        let mut last_chunk = String::new();
        let mut last_known = false;
        let mut pos = 0usize;
        while pos < sent.len() {
            let rest = &sent[pos..];
            let (best, second) = model.trie.match_two(rest, max_len, false);
            match best {
                Some((c1_ref, _)) => {
                    let c1 = c1_ref.to_string();
                    let c1_len_chars = c1.chars().count();

                    // Memorize: α-sample, admit on the second sighting.
                    if 2 + c1_len_chars <= max_len && rng.random::<f64>() < alpha {
                        let to_get = if !last_known {
                            let ll = last_chunk.chars().count();
                            if ll > 0 && ll <= mini_gap { Some(last_chunk.clone()) } else { None }
                        } else if last_chunk.chars().count() + c1_len_chars <= max_len {
                            Some(format!("{last_chunk}{c1}"))
                        } else {
                            None
                        };
                        if let Some(g) = to_get {
                            // Prepend-only space convention: a token may start
                            // with a space but never end with one (the reference
                            // trains on space-stripped text; our HF metaspace
                            // setup keeps spaces, so filter trailing spaces here).
                            if !g.is_empty() && !g.ends_with(' ') && !model.trie.search(&g) {
                                if to_memorize.remove(&g) {
                                    if model.trie.len() < vocab_size {
                                        model.trie.append(g, 0);
                                    }
                                } else {
                                    to_memorize.insert(g);
                                }
                            }
                        }
                    }

                    // Evaluate (if there is a shorter alternative) or cancel probation.
                    let c2nd = second.map(|(s, _)| s).filter(|s| !s.is_empty());
                    if c1_len_chars > 1 && c2nd.is_some() {
                        let c2nd = c2nd.unwrap();
                        if Self::eval_chunk(model, sent, pos, &c1, c2nd) {
                            reward_list.push((c1.clone(), 1)); // bad -> demote
                            let trimmed = Self::drop_last_char(&c1);
                            if !trimmed.is_empty() {
                                if let Some((sm, _)) =
                                    model.trie.match_longest(&trimmed, max_len, false)
                                {
                                    reward_list.push((sm.to_string(), -1));
                                }
                            }
                        } else {
                            reward_list.push((c1.clone(), -1)); // good -> promote
                            to_dropout.remove(&c1);
                        }
                    } else {
                        to_dropout.remove(&c1);
                    }

                    pos += c1.len();
                    last_chunk = c1;
                    last_known = true;
                }
                None => {
                    let ch = rest.chars().next().unwrap();
                    if last_known {
                        last_chunk.clear();
                        last_known = false;
                    }
                    last_chunk.push(ch);
                    pos += ch.len_utf8();
                }
            }
        }
    }

}

impl Default for LiBTrainer {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Trainer for LiBTrainer {
    type Model = LiBModel;

    fn should_show_progress(&self) -> bool {
        true
    }

    fn train(&self, model: &mut LiBModel) -> Result<Vec<AddedToken>> {
        model.max_len = self.max_len;
        model.byte_fallback = self.byte_fallback;

        // Phase 1: Initialize vocabulary
        if self.byte_fallback {
            // Add 256 byte tokens (<0x00>..<0xFF>)
            for b in 0..=255u8 {
                let code = format!("<{b:#04X}>");
                if !model.trie.search(&code) {
                    model.trie.append(code, 0);
                }
            }
            // Add Latin/ASCII characters from the corpus
            let mut chars: Vec<char> = self.char_set.iter()
                .copied()
                .filter(|ch| ch.is_ascii())
                .collect();
            chars.sort();
            for ch in &chars {
                let s = ch.to_string();
                if !model.trie.search(&s) {
                    model.trie.append(s, 0);
                }
            }
        } else {
            // Original behavior: seed all unique characters
            let mut chars: Vec<char> = self.char_set.iter().copied().collect();
            chars.sort();
            for ch in &chars {
                let s = ch.to_string();
                if !model.trie.search(&s) {
                    model.trie.append(s, 0);
                }
            }
        }

        // Phase 2: Online learning
        let mut rng: Box<dyn RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(StdRng::from_os_rng()),
        };

        let sentences = &self.sentences;
        if sentences.is_empty() {
            return Ok(self.special_tokens.clone());
        }

        let progress = if self.should_show_progress() {
            let p = ProgressBar::new(self.num_epochs as u64);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<40!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            p.set_message(format!("Vocab: {} tokens", model.trie.len()));
            Some(p)
        } else {
            None
        };

        // Probation watch (`model.py::to_dropout`): chunk -> remaining life.
        // Seeded units start watched, like the reference `init`.
        let mut to_dropout: HashMap<String, i32> = HashMap::new();
        for i in 0..model.trie.len() {
            if let Some(t) = model.trie.id_to_token(i) {
                to_dropout.insert(t.to_string(), self.life);
            }
        }
        let mini_gap = 2usize;

        for epoch in 0..self.num_epochs {
            let vocab_before = model.trie.len();

            // An epoch reads one document = `doc_size` sentences, accumulating a
            // single reward_list and a per-document memorize set (the twice-rule).
            let mut reward_list: Vec<(String, i32)> = Vec::new();
            let mut to_memorize: HashSet<String> = HashSet::new();
            for k in 0..self.doc_size {
                let sentence = if self.deterministic {
                    &sentences[(epoch * self.doc_size + k) % sentences.len()]
                } else {
                    &sentences[rng.random_range(0..sentences.len())]
                };
                Self::read_sentence(
                    model, sentence, &mut reward_list, &mut to_memorize,
                    &mut to_dropout, &mut rng, self.memory_in, mini_gap, self.vocab_size,
                );
            }

            // batch_update_memory: decrement the probation watch and forget the
            // expired; batch-apply the ordinal re-ranking once; then re-arm the
            // tail ω-fraction onto the watch.
            let mut to_del: HashSet<String> = HashSet::new();
            to_dropout.retain(|chunk, life| {
                if *life <= 1 {
                    to_del.insert(chunk.clone());
                    false
                } else {
                    *life -= 1;
                    true
                }
            });
            model.trie.group_remove(&to_del);

            let rewards: Vec<(String, i32)> =
                reward_list.into_iter().filter(|(w, _)| !to_del.contains(w)).collect();
            for r in model.trie.group_move(&rewards, self.update_rate) {
                to_dropout.remove(&r);
            }

            let n = model.trie.len();
            let start = ((1.0 - self.memory_out) * n as f64) as usize;
            for ind in start..n {
                if let Some(t) = model.trie.id_to_token(ind) {
                    to_dropout.entry(t.to_string()).or_insert(self.life);
                }
            }

            let vocab_after = model.trie.len();
            let delta = vocab_after as i64 - vocab_before as i64;

            if let Some(ref p) = progress {
                p.inc(1);
                p.set_message(format!(
                    "Vocab: {} tokens (chg: {:+})",
                    vocab_after, delta
                ));
            }
        }

        // Re-ranking skipped trie rebuilds on pure reorders for speed, so the
        // trie's ordinal ids may be stale; sync once before returning.
        model.trie.sync();

        if let Some(ref p) = progress {
            p.set_message(format!("Vocab: {} tokens", model.trie.len()));
            p.finish();
        }

        Ok(self.special_tokens.clone())
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        for item in iterator {
            let raw = item.as_ref();
            self.sentences.push(raw.to_string());

            let words = process(raw)?;
            for word in &words {
                *self.word_counts.entry(word.clone()).or_insert(0) += 1;
                for ch in word.chars() {
                    self.char_set.insert(ch);
                }
            }
        }
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Model;

    #[test]
    fn test_feed_collects_words() {
        let mut trainer = LiBTrainer::builder().vocab_size(100).build();
        let data = vec!["hello world", "hello there"];
        let process = |s: &str| -> crate::Result<Vec<String>> {
            Ok(s.split_whitespace().map(String::from).collect())
        };
        trainer.feed(data.into_iter(), process).unwrap();
        assert!(trainer.word_counts.contains_key("hello"));
        assert_eq!(trainer.word_counts["hello"], 2);
        assert_eq!(trainer.word_counts["world"], 1);
        assert_eq!(trainer.word_counts["there"], 1);
    }

    #[test]
    fn test_feed_collects_characters() {
        let mut trainer = LiBTrainer::builder().vocab_size(100).build();
        let data = vec!["ab cd"];
        let process = |s: &str| -> crate::Result<Vec<String>> {
            Ok(s.split_whitespace().map(String::from).collect())
        };
        trainer.feed(data.into_iter(), process).unwrap();
        assert!(trainer.char_set.contains(&'a'));
        assert!(trainer.char_set.contains(&'b'));
        assert!(trainer.char_set.contains(&'c'));
        assert!(trainer.char_set.contains(&'d'));
    }

    #[test]
    fn test_feed_stores_sentences() {
        let mut trainer = LiBTrainer::builder().vocab_size(100).build();
        let data = vec!["line one", "line two"];
        let process = |s: &str| -> crate::Result<Vec<String>> {
            Ok(vec![s.to_string()])
        };
        trainer.feed(data.into_iter(), process).unwrap();
        assert_eq!(trainer.sentences.len(), 2);
        assert_eq!(trainer.sentences[0], "line one");
    }

    #[test]
    fn test_train_produces_vocabulary() {
        let mut trainer = LiBTrainer::builder()
            .vocab_size(50)
            .num_epochs(200)
            .seed(42)
            .build();
        let corpus = vec![
            "the cat sat on the mat",
            "the dog sat on the log",
            "the cat and the dog",
        ];
        let process = |s: &str| -> crate::Result<Vec<String>> {
            Ok(vec![s.to_string()])
        };
        trainer.feed(corpus.into_iter(), process).unwrap();

        let mut model = LiBModel::default();
        trainer.train(&mut model).unwrap();

        assert!(model.get_vocab_size() > 0);
        let vocab = model.get_vocab();
        let multi_char: Vec<_> = vocab.keys().filter(|k| k.chars().count() > 1).collect();
        assert!(
            !multi_char.is_empty(),
            "Should have learned multi-character tokens, vocab: {:?}",
            vocab
        );
    }

    #[test]
    fn test_deterministic_training() {
        let make_model = |seed: u64| {
            let mut trainer = LiBTrainer::builder()
                .vocab_size(30)
                .num_epochs(50)
                .seed(seed)
                .build();
            let corpus = vec!["the cat sat", "the dog sat"];
            let process = |s: &str| -> crate::Result<Vec<String>> {
                Ok(vec![s.to_string()])
            };
            trainer.feed(corpus.into_iter(), process).unwrap();
            let mut model = LiBModel::default();
            trainer.train(&mut model).unwrap();
            model.get_vocab()
        };

        let vocab1 = make_model(42);
        let vocab2 = make_model(42);
        assert_eq!(vocab1, vocab2, "Same seed should produce identical vocabularies");
    }

    #[test]
    fn test_builder_defaults() {
        let trainer = LiBTrainer::default();
        assert_eq!(trainer.vocab_size, 30000);
        assert_eq!(trainer.num_epochs, 10000);
        assert_eq!(trainer.life, 10);
        assert_eq!(trainer.max_len, 12);
        assert!((trainer.memory_in - 0.25).abs() < f64::EPSILON);
        assert!((trainer.memory_out - 0.0001).abs() < f64::EPSILON);
        assert!((trainer.update_rate - 0.2).abs() < f64::EPSILON);
        assert!(trainer.seed.is_none());
        assert!(!trainer.deterministic);
        assert!(trainer.byte_fallback);
    }

    #[test]
    fn test_training_produces_no_trailing_space_tokens() {
        let mut trainer = LiBTrainer::builder()
            .vocab_size(200)
            .num_epochs(500)
            .seed(42)
            .byte_fallback(false)
            .build();
        let corpus = vec![
            "the cat sat on the mat",
            "the dog sat on the log",
            "the cat and the dog",
        ];
        let process = |s: &str| -> crate::Result<Vec<String>> {
            Ok(vec![s.to_string()])
        };
        trainer.feed(corpus.into_iter(), process).unwrap();
        let mut model = LiBModel::default();
        model.byte_fallback = false;
        trainer.train(&mut model).unwrap();

        // Standalone " " is a valid fallback seed token; only check multi-char tokens.
        for (token, _id) in model.get_vocab() {
            if token.len() == 1 { continue; }
            assert!(
                !token.ends_with(' '),
                "Token '{}' ends with a space — violates prepend-only convention",
                token
            );
        }
    }
}
