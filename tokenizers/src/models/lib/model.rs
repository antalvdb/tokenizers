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
}

impl LiBModel {
    pub fn new(max_len: usize, unk_token: Option<String>) -> Self {
        Self {
            trie: TrieList::new(),
            max_len,
            unk_token,
        }
    }

    /// Lookahead heuristic: when both the longest and second-longest match
    /// exist, peek one step ahead after each choice.  Prefer the choice
    /// whose continuation yields a known token.  If tied, prefer the
    /// longest (greedy default).
    fn choose_best<'a>(
        &self,
        sequence: &str,
        byte_pos: usize,
        best: &'a (String, usize),
        second: &'a (String, usize),
    ) -> &'a (String, usize) {
        let remaining_after_best = &sequence[byte_pos + best.0.len()..];
        let remaining_after_second = &sequence[byte_pos + second.0.len()..];

        // If choosing best consumes the rest of the input, that is always
        // at least as good as any continuation — prefer the greedy choice.
        if remaining_after_best.is_empty() {
            return best;
        }

        // Build the lookahead window (up to max_len chars) for each choice
        let window_best: String = remaining_after_best.chars().take(self.max_len).collect();
        let window_second: String = remaining_after_second.chars().take(self.max_len).collect();

        let has_next_best = self.trie.match_longest(&window_best).is_some();
        let has_next_second = self.trie.match_longest(&window_second).is_some();

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

        let mut tokens = Vec::new();
        let mut byte_pos: usize = 0;

        while byte_pos < sequence.len() {
            // Build window of up to max_len *characters* starting at byte_pos
            let rest = &sequence[byte_pos..];
            let window: String = rest.chars().take(self.max_len).collect();

            let (best, second) = self.trie.match_two(&window);

            match (best, second) {
                (Some(b), Some(s)) => {
                    let chosen = self.choose_best(sequence, byte_pos, &b, &s);
                    let tok_str = &chosen.0;
                    let tok_id = chosen.1 as u32;
                    let byte_end = byte_pos + tok_str.len();
                    tokens.push(Token::new(tok_id, tok_str.clone(), (byte_pos, byte_end)));
                    byte_pos = byte_end;
                }
                (Some(b), None) => {
                    let byte_end = byte_pos + b.0.len();
                    tokens.push(Token::new(b.1 as u32, b.0.clone(), (byte_pos, byte_end)));
                    byte_pos = byte_end;
                }
                _ => {
                    // No match: emit a single character
                    let ch = rest.chars().next().unwrap();
                    let ch_str = ch.to_string();
                    let byte_end = byte_pos + ch.len_utf8();

                    // Look up the single character; if not in vocab, use 0 as unknown id
                    let id = self.trie.token_to_id(&ch_str)
                        .map(|id| id as u32)
                        .unwrap_or(0);
                    tokens.push(Token::new(id, ch_str, (byte_pos, byte_end)));
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
        self.trie.id_to_token(id as usize)
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
    fn make_model(tokens: &[&str]) -> LiBModel {
        let mut model = LiBModel::new(12, None);
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

    // 10. test_unicode_offsets
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
}
