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
    seed: Option<u64>,
    deterministic: bool,
    byte_fallback: bool,
    special_tokens: Vec<AddedToken>,
}

impl Default for LiBTrainerBuilder {
    fn default() -> Self {
        Self {
            vocab_size: 30000,
            num_epochs: 5000,
            life: 10,
            max_len: 12,
            memory_in: 0.25,
            memory_out: 0.0001,
            update_rate: 0.2,
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
/// Training follows a cognitively-inspired online learning process:
/// 1. Initialize vocabulary with characters from the corpus
/// 2. For each epoch, process a sentence:
///    - Segment using current vocabulary (reading phase)
///    - Probabilistically memorize new candidate units
///    - Test candidates via compression comparison
///    - Reorder vocabulary based on rewards/punishments
///    - Prune rarely-used units
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LiBTrainer {
    pub vocab_size: usize,
    pub num_epochs: usize,
    pub life: i32,
    pub max_len: usize,
    pub memory_in: f64,
    pub memory_out: f64,
    pub update_rate: f64,
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

    /// Segment a sentence using the current model, returning chunks.
    /// Each chunk is (token_string, is_known).
    fn segment(model: &LiBModel, sentence: &str) -> Vec<(String, bool)> {
        let mut chunks = Vec::new();
        let mut byte_pos = 0;

        while byte_pos < sentence.len() {
            let rest = &sentence[byte_pos..];
            let window: String = rest.chars().take(model.max_len).collect();

            match model.trie.match_longest(&window, false) {
                Some((token, _id)) => {
                    let len = token.len();
                    chunks.push((token, true));
                    byte_pos += len;
                }
                None => {
                    let ch = rest.chars().next().unwrap();
                    chunks.push((ch.to_string(), false));
                    byte_pos += ch.len_utf8();
                }
            }
        }
        chunks
    }

    /// Generate candidate units from adjacent chunk pairs.
    fn generate_candidates(chunks: &[(String, bool)], max_len: usize) -> Vec<String> {
        let mut candidates = Vec::new();
        for i in 0..chunks.len().saturating_sub(1) {
            let combined = format!("{}{}", chunks[i].0, chunks[i + 1].0);
            if combined.chars().count() <= max_len {
                candidates.push(combined);
            }
        }
        candidates
    }

    /// Segment a sentence but also greedily match a given candidate token.
    fn segment_with_candidate(
        model: &LiBModel,
        candidate: &str,
        sentence: &str,
    ) -> usize {
        let mut count = 0usize;
        let mut byte_pos = 0;

        while byte_pos < sentence.len() {
            let rest = &sentence[byte_pos..];

            // Try the candidate first
            if rest.starts_with(candidate) {
                count += 1;
                byte_pos += candidate.len();
                continue;
            }

            let window: String = rest.chars().take(model.max_len).collect();
            match model.trie.match_longest(&window, false) {
                Some((token, _)) => {
                    count += 1;
                    byte_pos += token.len();
                }
                None => {
                    count += 1;
                    byte_pos += rest.chars().next().unwrap().len_utf8();
                }
            }
        }
        count
    }

    /// Test a candidate: does adding it reduce the chunk count?
    /// Returns +1.0 (improvement), -1.0 (worse), 0.0 (neutral).
    fn test_candidate(model: &LiBModel, candidate: &str, sentence: &str) -> f64 {
        let n_without = Self::segment(model, sentence).len();
        let n_with = Self::segment_with_candidate(model, candidate, sentence);

        if n_with < n_without {
            1.0
        } else if n_with > n_without {
            -1.0
        } else {
            0.0
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
                    model.trie.append(code, self.life);
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
                    model.trie.append(s, self.life);
                }
            }
        } else {
            // Original behavior: seed all unique characters
            let mut chars: Vec<char> = self.char_set.iter().copied().collect();
            chars.sort();
            for ch in &chars {
                let s = ch.to_string();
                if !model.trie.search(&s) {
                    model.trie.append(s, self.life);
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

        for epoch in 0..self.num_epochs {
            if model.trie.len() >= self.vocab_size {
                break;
            }

            let vocab_before = model.trie.len();

            // Sample a sentence
            let sentence = if self.deterministic {
                &sentences[epoch % sentences.len()]
            } else {
                &sentences[rng.random_range(0..sentences.len())]
            };

            // Read: segment with current vocabulary
            let chunks = Self::segment(model, sentence);

            // Memorize: generate and test candidates
            let candidates = Self::generate_candidates(&chunks, self.max_len);
            let mut rewards: Vec<(String, f64)> = Vec::new();

            for candidate in &candidates {
                if model.trie.search(candidate) {
                    continue;
                }
                if model.trie.len() >= self.vocab_size {
                    break;
                }

                let should_consider = if self.deterministic {
                    true
                } else {
                    rng.random::<f64>() < self.memory_in
                };

                if should_consider {
                    let reward = Self::test_candidate(model, candidate, sentence);

                    if reward > 0.0 {
                        model.trie.append(candidate.clone(), self.life);
                    }
                    rewards.push((candidate.clone(), reward));
                }
            }

            // Update: reorder vocabulary based on rewards
            if !rewards.is_empty() {
                model.trie.batch_update(&rewards, self.update_rate);
            }

            // Prune: remove bottom fraction
            model.trie.prune(self.memory_out);

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
        assert_eq!(trainer.num_epochs, 5000);
        assert_eq!(trainer.life, 10);
        assert_eq!(trainer.max_len, 12);
        assert!((trainer.memory_in - 0.25).abs() < f64::EPSILON);
        assert!((trainer.memory_out - 0.0001).abs() < f64::EPSILON);
        assert!((trainer.update_rate - 0.2).abs() < f64::EPSILON);
        assert!(trainer.seed.is_none());
        assert!(!trainer.deterministic);
        assert!(trainer.byte_fallback);
    }
}
