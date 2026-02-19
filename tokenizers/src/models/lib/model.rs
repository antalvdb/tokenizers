use crate::models::lib::trie::TrieList;
use crate::models::lib::trainer::LiBTrainer;
use crate::{Model, Token, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct LiBModel {
    pub(crate) trie: TrieList,
    pub max_len: usize,
    pub unk_token: Option<String>,
    pub use_supra_words: bool,
    pub byte_fallback: bool,
}

impl LiBModel {
    pub fn new(max_len: usize, unk_token: Option<String>) -> Self {
        Self {
            trie: TrieList::new(),
            max_len,
            unk_token,
            use_supra_words: true,
            byte_fallback: true,
        }
    }

    /// Add a token to the vocabulary with the given life value.
    pub fn add_token(&mut self, token: String, life: i32) {
        self.trie.append(token, life);
    }

    /// Lookahead heuristic: when both the longest and second-longest match
    /// exist, peek one step ahead after each choice.  Prefer the choice
    /// whose continuation yields a known token.  If tied, prefer the
    /// longest (greedy default).
    ///
    /// Both `best` and `second` are `(&str, usize)` (Copy), so they are
    /// passed and returned by value.
    fn choose_best<'a>(
        &self,
        sequence: &'a str,
        byte_pos: usize,
        best: (&'a str, usize),
        second: (&'a str, usize),
        skip_spaces: bool,
    ) -> (&'a str, usize) {
        let remaining_after_best = &sequence[byte_pos + best.0.len()..];
        let remaining_after_second = &sequence[byte_pos + second.0.len()..];

        // If choosing best consumes the rest of the input, that is always
        // at least as good as any continuation — prefer the greedy choice.
        if remaining_after_best.is_empty() {
            return best;
        }

        // When skip_spaces is set, leading spaces in the remainder are word
        // boundaries that will be emitted as their own tokens.  Strip them so
        // the lookahead checks the *next word*, not the space itself (which
        // match_longest would reject because it contains a space).
        let lookahead_best = if skip_spaces {
            remaining_after_best.trim_start_matches(' ')
        } else {
            remaining_after_best
        };
        let lookahead_second = if skip_spaces {
            remaining_after_second.trim_start_matches(' ')
        } else {
            remaining_after_second
        };

        // Pass slices directly — match_longest limits the walk to max_len chars internally.
        let has_next_best = self.trie.match_longest(lookahead_best, self.max_len, skip_spaces).is_some();
        let has_next_second = self.trie.match_longest(lookahead_second, self.max_len, skip_spaces).is_some();

        match (has_next_best, has_next_second) {
            // Both have continuations, or neither does: prefer longest (greedy)
            (true, true) | (false, false) => best,
            // Only best has continuation
            (true, false) => best,
            // Only second has continuation — prefer it
            (false, true) => second,
        }
    }
}

impl Default for LiBModel {
    fn default() -> Self {
        Self::new(12, None)
    }
}

impl Model for LiBModel {
    type Trainer = LiBTrainer;

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(Vec::new());
        }

        let skip_spaces = !self.use_supra_words;
        let mut tokens = Vec::new();
        let mut byte_pos: usize = 0;

        while byte_pos < sequence.len() {
            let rest = &sequence[byte_pos..];
            // Pass rest directly — match_two limits the walk to max_len chars internally,
            // so no window String needs to be built.
            let (best, second) = self.trie.match_two(rest, self.max_len, skip_spaces);

            match (best, second) {
                (Some(b), Some(s)) => {
                    let chosen = self.choose_best(sequence, byte_pos, b, s, skip_spaces);
                    let tok_str = chosen.0;
                    let tok_id = chosen.1 as u32;
                    let byte_end = byte_pos + tok_str.len();
                    tokens.push(Token::new(tok_id, tok_str.to_string(), (byte_pos, byte_end)));
                    byte_pos = byte_end;
                }
                (Some(b), None) => {
                    let byte_end = byte_pos + b.0.len();
                    tokens.push(Token::new(b.1 as u32, b.0.to_string(), (byte_pos, byte_end)));
                    byte_pos = byte_end;
                }
                _ => {
                    // No match: emit a single character
                    let ch = rest.chars().next().unwrap();
                    let ch_str = ch.to_string();
                    let byte_end = byte_pos + ch.len_utf8();

                    // Try direct lookup first
                    if let Some(id) = self.trie.token_to_id(&ch_str) {
                        tokens.push(Token::new(id as u32, ch_str, (byte_pos, byte_end)));
                    } else if self.byte_fallback {
                        // Decompose into UTF-8 byte tokens
                        for (i, b) in ch_str.bytes().enumerate() {
                            let code = format!("<{b:#04X}>");
                            let id = self.trie.token_to_id(&code)
                                .map(|id| id as u32)
                                .unwrap_or(0);
                            tokens.push(Token::new(id, code, (byte_pos + i, byte_pos + i + 1)));
                        }
                    } else {
                        tokens.push(Token::new(0, ch_str, (byte_pos, byte_end)));
                    }
                    byte_pos = byte_end;
                }
            }
        }

        Ok(tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.trie.token_to_id(token).map(|id| id as u32)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.trie.id_to_token(id as usize).map(str::to_owned)
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.trie.get_vocab_map()
    }

    fn get_vocab_size(&self) -> usize {
        self.trie.len()
    }

    fn save(&self, folder: &Path, prefix: Option<&str>) -> Result<Vec<PathBuf>> {
        let file_name = match prefix {
            Some(p) => format!("{}-lib-vocab.json", p),
            None => "lib-vocab.json".to_string(),
        };
        let path: PathBuf = [folder, Path::new(&file_name)].iter().collect();
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, content)?;
        Ok(vec![path])
    }

    fn get_trainer(&self) -> Self::Trainer {
        LiBTrainer::default()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a LiBModel with the given tokens (each gets life=10).
    /// byte_fallback is off by default since test models don't include byte tokens.
    fn make_model(tokens: &[&str]) -> LiBModel {
        let mut model = LiBModel::new(12, None);
        model.byte_fallback = false;
        for tok in tokens {
            model.trie.append(tok.to_string(), 10);
        }
        model
    }

    // 1. test_tokenize_known_words
    #[test]
    fn test_tokenize_known_words() {
        let model = make_model(&["h", "e", "l", "o", "hello", "world"]);
        let tokens = model.tokenize("hello").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, "hello");
    }

    // 2. test_tokenize_fallback_to_chars
    #[test]
    fn test_tokenize_fallback_to_chars() {
        let model = make_model(&["h", "e", "l", "o"]);
        let tokens = model.tokenize("hello").unwrap();
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].value, "h");
        assert_eq!(tokens[1].value, "e");
        assert_eq!(tokens[2].value, "l");
        assert_eq!(tokens[3].value, "l");
        assert_eq!(tokens[4].value, "o");
    }

    // 3. test_tokenize_unknown_char
    #[test]
    fn test_tokenize_unknown_char() {
        let model = make_model(&["h", "e"]);
        let tokens = model.tokenize("hex").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].value, "h");
        assert_eq!(tokens[1].value, "e");
        // 'x' is unknown — emitted as a single character with id 0
        assert_eq!(tokens[2].value, "x");
        assert_eq!(tokens[2].id, 0);
    }

    // 4. test_tokenize_supra_word
    #[test]
    fn test_tokenize_supra_word() {
        let model = make_model(&["t", "h", "e", "the", "c", "a", "cat", "the cat"]);
        let tokens = model.tokenize("the cat").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, "the cat");
    }

    // 5. test_tokenize_empty
    #[test]
    fn test_tokenize_empty() {
        let model = make_model(&["a", "b"]);
        let tokens = model.tokenize("").unwrap();
        assert!(tokens.is_empty());
    }

    // 6. test_vocab_accessors
    #[test]
    fn test_vocab_accessors() {
        let model = make_model(&["alpha", "beta", "gamma"]);

        // token_to_id
        assert_eq!(model.token_to_id("alpha"), Some(0));
        assert_eq!(model.token_to_id("beta"), Some(1));
        assert_eq!(model.token_to_id("gamma"), Some(2));
        assert_eq!(model.token_to_id("delta"), None);

        // id_to_token
        assert_eq!(model.id_to_token(0), Some("alpha".to_string()));
        assert_eq!(model.id_to_token(1), Some("beta".to_string()));
        assert_eq!(model.id_to_token(2), Some("gamma".to_string()));
        assert_eq!(model.id_to_token(3), None);

        // get_vocab
        let vocab = model.get_vocab();
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab["alpha"], 0);
        assert_eq!(vocab["beta"], 1);
        assert_eq!(vocab["gamma"], 2);

        // get_vocab_size
        assert_eq!(model.get_vocab_size(), 3);
    }

    // 7. test_offsets_are_correct
    #[test]
    fn test_offsets_are_correct() {
        let model = make_model(&["he", "llo"]);
        let tokens = model.tokenize("hello").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].value, "he");
        assert_eq!(tokens[0].offsets, (0, 2));
        assert_eq!(tokens[1].value, "llo");
        assert_eq!(tokens[1].offsets, (2, 5));
    }

    // 8. test_lookahead_prefers_better_continuation
    #[test]
    fn test_lookahead_prefers_better_continuation() {
        // Vocab: "ab", "a", "bc"
        // Input: "abc"
        // Greedy would pick "ab" + "c" (unknown)
        // Lookahead: "a" + "bc" (both known) — better
        let model = make_model(&["ab", "a", "bc"]);
        let tokens = model.tokenize("abc").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].value, "a");
        assert_eq!(tokens[1].value, "bc");
    }

    // 9. test_save_and_load
    #[test]
    fn test_save_and_load() {
        let model = make_model(&["hello", "world"]);
        let dir = std::env::temp_dir();
        let paths = model.save(&dir, Some("test")).unwrap();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].to_str().unwrap().contains("test-lib-vocab.json"));

        // Read back and verify
        let content = std::fs::read_to_string(&paths[0]).unwrap();
        let loaded: LiBModel = serde_json::from_str(&content).unwrap();
        assert_eq!(loaded.get_vocab_size(), 2);
        assert_eq!(loaded.token_to_id("hello"), Some(0));
        assert_eq!(loaded.token_to_id("world"), Some(1));

        // Clean up
        let _ = std::fs::remove_file(&paths[0]);
    }

    // 10. test_tokenize_supra_word_disabled
    #[test]
    fn test_tokenize_supra_word_disabled() {
        let mut model = make_model(&["t", "h", "e", " ", "c", "a", "the", "cat", "the cat"]);

        // Default: supra-words enabled — "the cat" is one token
        let tokens = model.tokenize("the cat").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, "the cat");

        // Disable supra-words — falls back to word-level + space
        model.use_supra_words = false;
        let tokens = model.tokenize("the cat").unwrap();
        assert_eq!(tokens.len(), 3, "Expected [the, ' ', cat], got: {:?}",
            tokens.iter().map(|t| &t.value).collect::<Vec<_>>());
        assert_eq!(tokens[0].value, "the");
        assert_eq!(tokens[1].value, " ");
        assert_eq!(tokens[2].value, "cat");
    }

    // 11. test_unicode_offsets
    #[test]
    fn test_unicode_offsets() {
        // e-acute is 2 bytes in UTF-8
        let model = make_model(&["\u{00e9}", "t"]);
        let tokens = model.tokenize("\u{00e9}t").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].value, "\u{00e9}");
        assert_eq!(tokens[0].offsets, (0, 2)); // e-acute is 2 bytes
        assert_eq!(tokens[1].value, "t");
        assert_eq!(tokens[1].offsets, (2, 3));
    }

    // 12. test_byte_fallback
    #[test]
    fn test_byte_fallback() {
        // Build a model with byte tokens and byte_fallback enabled
        let mut model = LiBModel::new(12, None);
        model.byte_fallback = true;
        // Add byte tokens for all 256 bytes
        for b in 0..=255u8 {
            model.trie.append(format!("<{b:#04X}>"), 10);
        }
        model.trie.append("h".to_string(), 10);
        model.trie.append("e".to_string(), 10);

        // 'x' is not in vocab — should decompose to <0x78>
        let tokens = model.tokenize("hex").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].value, "h");
        assert_eq!(tokens[1].value, "e");
        assert_eq!(tokens[2].value, "<0x78>");

        // Multi-byte char: é (U+00E9) = 0xC3 0xA9 in UTF-8
        let tokens = model.tokenize("\u{00e9}").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].value, "<0xC3>");
        assert_eq!(tokens[1].value, "<0xA9>");
    }
}
