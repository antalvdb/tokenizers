use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TrieNode -- internal node of the array-based trie
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TrieNode {
    children: HashMap<char, usize>, // char -> node index in TrieList.nodes
    token_index: Option<usize>,     // Some(id) if this node terminates a token
}

// ---------------------------------------------------------------------------
// VocabEntry -- one entry in the priority-ordered vocabulary
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabEntry {
    pub token: String,
    pub life: i32,
    pub frequency: u64,
}

// ---------------------------------------------------------------------------
// TrieList -- hybrid trie + priority-ordered list
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrieList {
    nodes: Vec<TrieNode>,                   // array-based trie, index 0 = root
    vocab: Vec<VocabEntry>,                 // priority-ordered vocabulary, index = token ID
    token_to_index: HashMap<String, usize>, // reverse lookup: token string -> vocab index
}

// PartialEq compares only the vocab ordering (not internal trie structure).
impl PartialEq for TrieList {
    fn eq(&self, other: &Self) -> bool {
        self.vocab.len() == other.vocab.len()
            && self
                .vocab
                .iter()
                .zip(other.vocab.iter())
                .all(|(a, b)| a.token == b.token)
    }
}

impl Default for TrieList {
    fn default() -> Self {
        Self::new()
    }
}

impl TrieList {
    // -- construction -------------------------------------------------------

    /// Create an empty TrieList with a single root node.
    pub fn new() -> Self {
        Self {
            nodes: vec![TrieNode::default()], // index 0 = root
            vocab: Vec::new(),
            token_to_index: HashMap::new(),
        }
    }

    // -- size queries -------------------------------------------------------

    /// Number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.vocab.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    // -- insertion ----------------------------------------------------------

    /// Append a token at the end of the vocab (lowest priority).
    pub fn append(&mut self, token: String, life: i32) {
        if self.token_to_index.contains_key(&token) {
            return; // no duplicates
        }
        let id = self.vocab.len();
        self.vocab.push(VocabEntry {
            token: token.clone(),
            life,
            frequency: 0,
        });
        self.token_to_index.insert(token.clone(), id);
        self.trie_insert(&token, id);
    }

    /// Insert a token at position `pos`, shifting subsequent items.
    pub fn insert_at(&mut self, pos: usize, token: String, life: i32) {
        if self.token_to_index.contains_key(&token) {
            return; // no duplicates
        }
        let pos = pos.min(self.vocab.len()); // clamp
        self.vocab.insert(
            pos,
            VocabEntry {
                token: token.clone(),
                life,
                frequency: 0,
            },
        );
        self.rebuild_indices();
        self.rebuild_trie();
    }

    // -- lookups ------------------------------------------------------------

    /// Check whether a token exists in the vocabulary.
    pub fn search(&self, token: &str) -> bool {
        self.token_to_index.contains_key(token)
    }

    /// Return the vocabulary index (= token ID) for a given string.
    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.token_to_index.get(token).copied()
    }

    /// Return the token string for a given ID.
    pub fn id_to_token(&self, id: usize) -> Option<String> {
        self.vocab.get(id).map(|e| e.token.clone())
    }

    /// Get an immutable reference to a vocab entry by ID.
    pub fn get_entry(&self, id: usize) -> Option<&VocabEntry> {
        self.vocab.get(id)
    }

    /// Get a mutable reference to a vocab entry by ID.
    pub fn get_entry_mut(&mut self, id: usize) -> Option<&mut VocabEntry> {
        self.vocab.get_mut(id)
    }

    /// Build a HashMap<String, u32> suitable for the Model trait's `get_vocab`.
    pub fn get_vocab_map(&self) -> HashMap<String, u32> {
        self.vocab
            .iter()
            .enumerate()
            .map(|(i, e)| (e.token.clone(), i as u32))
            .collect()
    }

    /// Iterate over (id, &VocabEntry) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &VocabEntry)> {
        self.vocab.iter().enumerate()
    }

    // -- matching -----------------------------------------------------------

    /// Longest-prefix match: walk the trie from the start of `input`,
    /// returning the longest token that matches and its ID.
    pub fn match_longest(&self, input: &str, skip_spaces: bool) -> Option<(String, usize)> {
        let mut node_idx: usize = 0;
        let mut best: Option<(String, usize)> = None;
        let mut consumed = String::new();

        for ch in input.chars() {
            match self.nodes[node_idx].children.get(&ch) {
                Some(&next) => {
                    node_idx = next;
                    consumed.push(ch);
                    if let Some(tok_id) = self.nodes[node_idx].token_index {
                        if !skip_spaces || !consumed.contains(' ') {
                            best = Some((consumed.clone(), tok_id));
                        }
                    }
                }
                None => break,
            }
        }
        best
    }

    /// Return the longest AND second-longest prefix matches.
    /// The first element is the longest, the second is the second-longest.
    pub fn match_two(
        &self,
        input: &str,
        skip_spaces: bool,
    ) -> (Option<(String, usize)>, Option<(String, usize)>) {
        let mut node_idx: usize = 0;
        let mut best: Option<(String, usize)> = None;
        let mut second: Option<(String, usize)> = None;
        let mut consumed = String::new();

        for ch in input.chars() {
            match self.nodes[node_idx].children.get(&ch) {
                Some(&next) => {
                    node_idx = next;
                    consumed.push(ch);
                    if let Some(tok_id) = self.nodes[node_idx].token_index {
                        if !skip_spaces || !consumed.contains(' ') {
                            second = best.clone();
                            best = Some((consumed.clone(), tok_id));
                        }
                    }
                }
                None => break,
            }
        }
        (best, second)
    }

    // -- reordering / pruning -----------------------------------------------

    /// Batch-update token priorities based on rewards.
    ///
    /// For each `(token, reward)`:
    ///   - positive reward  -> move toward front  (lower index = higher priority)
    ///   - negative reward  -> move toward back   (higher index = lower priority)
    ///   - step = `(current_index as f64 * update_rate + 1.0) as usize`
    ///
    /// After all moves the trie and index maps are rebuilt.
    pub fn batch_update(&mut self, rewards: &[(String, f64)], update_rate: f64) {
        for (token, reward) in rewards {
            if let Some(&cur) = self.token_to_index.get(token.as_str()) {
                let step = (cur as f64 * update_rate + 1.0) as usize;
                let new_pos = if *reward > 0.0 {
                    cur.saturating_sub(step)
                } else if *reward < 0.0 {
                    (cur + step).min(self.vocab.len() - 1)
                } else {
                    continue;
                };
                if new_pos != cur {
                    let entry = self.vocab.remove(cur);
                    self.vocab.insert(new_pos, entry);
                    // Rebuild token_to_index after each move so subsequent
                    // lookups in the same batch see correct positions.
                    self.rebuild_indices();
                }
            }
        }
        self.rebuild_trie();
    }

    /// Remove specific tokens from the vocabulary.
    pub fn remove_tokens(&mut self, tokens: &[String]) {
        let to_remove: std::collections::HashSet<&str> =
            tokens.iter().map(|s| s.as_str()).collect();
        self.vocab.retain(|e| !to_remove.contains(e.token.as_str()));
        self.rebuild_indices();
        self.rebuild_trie();
    }

    /// Remove the bottom `memory_out` fraction of the vocab (lowest priority =
    /// highest indices). Returns the list of removed token strings.
    pub fn prune(&mut self, memory_out: f64) -> Vec<String> {
        let total = self.vocab.len();
        let remove_count = (total as f64 * memory_out).round() as usize;
        if remove_count == 0 || total == 0 {
            return Vec::new();
        }
        let keep = total - remove_count;
        let removed: Vec<String> = self.vocab[keep..]
            .iter()
            .map(|e| e.token.clone())
            .collect();
        self.vocab.truncate(keep);
        self.rebuild_indices();
        self.rebuild_trie();
        removed
    }

    // -- internal helpers ---------------------------------------------------

    /// Insert a single token into the trie (does NOT touch vocab/token_to_index).
    fn trie_insert(&mut self, token: &str, vocab_id: usize) {
        let mut node_idx: usize = 0;
        for ch in token.chars() {
            let next = if let Some(&existing) = self.nodes[node_idx].children.get(&ch) {
                existing
            } else {
                let new_idx = self.nodes.len();
                self.nodes.push(TrieNode::default());
                self.nodes[node_idx].children.insert(ch, new_idx);
                new_idx
            };
            node_idx = next;
        }
        self.nodes[node_idx].token_index = Some(vocab_id);
    }

    /// Rebuild `token_to_index` from current `vocab` ordering.
    fn rebuild_indices(&mut self) {
        self.token_to_index.clear();
        for (i, entry) in self.vocab.iter().enumerate() {
            self.token_to_index.insert(entry.token.clone(), i);
        }
    }

    /// Rebuild the entire trie from scratch based on current `vocab`.
    fn rebuild_trie(&mut self) {
        self.nodes.clear();
        self.nodes.push(TrieNode::default()); // root
        // Collect tokens first to avoid borrow-checker conflict with trie_insert.
        let tokens: Vec<String> = self.vocab.iter().map(|e| e.token.clone()).collect();
        for (id, tok) in tokens.iter().enumerate() {
            self.trie_insert(tok, id);
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // 1. test_insert_and_search
    #[test]
    fn test_insert_and_search() {
        let mut tl = TrieList::new();
        tl.append("hello".to_string(), 5);
        tl.append("world".to_string(), 5);
        tl.append("foo".to_string(), 3);

        assert!(tl.search("hello"));
        assert!(tl.search("world"));
        assert!(tl.search("foo"));
        assert!(!tl.search("bar"));
        assert!(!tl.search("hell")); // prefix, but not a token
        assert_eq!(tl.len(), 3);
    }

    // 2. test_token_to_id
    #[test]
    fn test_token_to_id() {
        let mut tl = TrieList::new();
        tl.append("a".to_string(), 1);
        tl.append("b".to_string(), 1);
        tl.append("c".to_string(), 1);

        assert_eq!(tl.token_to_id("a"), Some(0));
        assert_eq!(tl.token_to_id("b"), Some(1));
        assert_eq!(tl.token_to_id("c"), Some(2));
        assert_eq!(tl.token_to_id("d"), None);
    }

    // 3. test_id_to_token
    #[test]
    fn test_id_to_token() {
        let mut tl = TrieList::new();
        tl.append("alpha".to_string(), 1);
        tl.append("beta".to_string(), 1);

        assert_eq!(tl.id_to_token(0), Some("alpha".to_string()));
        assert_eq!(tl.id_to_token(1), Some("beta".to_string()));
        assert_eq!(tl.id_to_token(2), None);

        // roundtrip
        let id = tl.token_to_id("beta").unwrap();
        assert_eq!(tl.id_to_token(id), Some("beta".to_string()));
    }

    // 4. test_match_longest
    #[test]
    fn test_match_longest() {
        let mut tl = TrieList::new();
        tl.append("h".to_string(), 1);
        tl.append("he".to_string(), 1);
        tl.append("hel".to_string(), 1);
        tl.append("hello".to_string(), 1);

        let result = tl.match_longest("hello world", false);
        assert_eq!(result, Some(("hello".to_string(), 3)));

        // partial match should return longest that exists
        let result = tl.match_longest("help me", false);
        assert_eq!(result, Some(("hel".to_string(), 2)));

        // single char
        let result = tl.match_longest("hat", false);
        assert_eq!(result, Some(("h".to_string(), 0)));
    }

    // 5. test_match_two
    #[test]
    fn test_match_two() {
        let mut tl = TrieList::new();
        tl.append("h".to_string(), 1);
        tl.append("he".to_string(), 1);
        tl.append("hel".to_string(), 1);
        tl.append("hello".to_string(), 1);

        let (best, second) = tl.match_two("hello world", false);
        assert_eq!(best, Some(("hello".to_string(), 3)));
        assert_eq!(second, Some(("hel".to_string(), 2)));

        // only two matches
        let (best, second) = tl.match_two("help", false);
        assert_eq!(best, Some(("hel".to_string(), 2)));
        assert_eq!(second, Some(("he".to_string(), 1)));

        // only one match
        let (best, second) = tl.match_two("hat", false);
        assert_eq!(best, Some(("h".to_string(), 0)));
        assert_eq!(second, None);
    }

    // 6. test_match_no_match
    #[test]
    fn test_match_no_match() {
        let tl = TrieList::new();
        assert_eq!(tl.match_longest("anything", false), None);

        let (best, second) = tl.match_two("anything", false);
        assert_eq!(best, None);
        assert_eq!(second, None);

        // also test non-empty trie with no matching prefix
        let mut tl2 = TrieList::new();
        tl2.append("xyz".to_string(), 1);
        assert_eq!(tl2.match_longest("abc", false), None);
    }

    // 7. test_unicode
    #[test]
    fn test_unicode() {
        let mut tl = TrieList::new();
        tl.append("\u{00e9}".to_string(), 1); // e-acute
        tl.append("\u{00e9}t\u{00e9}".to_string(), 1); // ete with accents
        tl.append("\u{4e16}\u{754c}".to_string(), 1); // Chinese: "world"
        tl.append("\u{1f600}".to_string(), 1); // emoji

        assert!(tl.search("\u{00e9}"));
        assert!(tl.search("\u{4e16}\u{754c}"));
        assert!(tl.search("\u{1f600}"));

        let result = tl.match_longest("\u{00e9}t\u{00e9} hello", false);
        assert_eq!(result, Some(("\u{00e9}t\u{00e9}".to_string(), 1)));

        let result = tl.match_longest("\u{4e16}\u{754c}!", false);
        assert_eq!(result, Some(("\u{4e16}\u{754c}".to_string(), 2)));
    }

    // 8. test_batch_update_reward
    #[test]
    fn test_batch_update_reward() {
        let mut tl = TrieList::new();
        // Insert tokens a..j  (IDs 0..9)
        for ch in 'a'..='j' {
            tl.append(ch.to_string(), 1);
        }
        // "j" starts at index 9 (lowest priority)
        assert_eq!(tl.token_to_id("j"), Some(9));

        // Reward "j" positively -> it should move toward front
        tl.batch_update(&[("j".to_string(), 1.0)], 0.5);
        let new_id = tl.token_to_id("j").unwrap();
        assert!(
            new_id < 9,
            "j should have moved toward the front, but is at {}",
            new_id
        );

        // All tokens should still be searchable
        for ch in 'a'..='j' {
            assert!(tl.search(&ch.to_string()));
        }
    }

    // 9. test_prune
    #[test]
    fn test_prune() {
        let mut tl = TrieList::new();
        for i in 0..100 {
            tl.append(format!("tok_{}", i), 1);
        }
        assert_eq!(tl.len(), 100);

        let removed = tl.prune(0.05);
        assert_eq!(removed.len(), 5);
        assert_eq!(tl.len(), 95);

        // The removed tokens should be the last 5
        for t in &removed {
            assert!(!tl.search(t));
        }
        // The remaining tokens should still be searchable
        for i in 0..95 {
            assert!(tl.search(&format!("tok_{}", i)));
        }
    }

    // 10. test_remove_tokens
    #[test]
    fn test_remove_tokens() {
        let mut tl = TrieList::new();
        tl.append("alpha".to_string(), 1);
        tl.append("beta".to_string(), 1);
        tl.append("gamma".to_string(), 1);
        tl.append("delta".to_string(), 1);
        tl.append("epsilon".to_string(), 1);

        tl.remove_tokens(&["beta".to_string(), "delta".to_string()]);

        assert_eq!(tl.len(), 3);
        assert!(tl.search("alpha"));
        assert!(!tl.search("beta"));
        assert!(tl.search("gamma"));
        assert!(!tl.search("delta"));
        assert!(tl.search("epsilon"));

        // IDs should be contiguous 0..2
        assert_eq!(tl.token_to_id("alpha"), Some(0));
        assert_eq!(tl.token_to_id("gamma"), Some(1));
        assert_eq!(tl.token_to_id("epsilon"), Some(2));

        // Trie should still work
        let result = tl.match_longest("gamma ray", false);
        assert_eq!(result, Some(("gamma".to_string(), 1)));
    }

    // 11. test_insert_at
    #[test]
    fn test_insert_at() {
        let mut tl = TrieList::new();
        tl.append("a".to_string(), 1);
        tl.append("b".to_string(), 1);
        tl.append("c".to_string(), 1);

        // Insert "X" at position 1
        tl.insert_at(1, "X".to_string(), 2);

        assert_eq!(tl.len(), 4);
        assert_eq!(tl.id_to_token(0), Some("a".to_string()));
        assert_eq!(tl.id_to_token(1), Some("X".to_string()));
        assert_eq!(tl.id_to_token(2), Some("b".to_string()));
        assert_eq!(tl.id_to_token(3), Some("c".to_string()));

        // Search and match should work
        assert!(tl.search("X"));
        assert_eq!(tl.token_to_id("X"), Some(1));
    }

    // Additional: test_get_vocab_map
    #[test]
    fn test_get_vocab_map() {
        let mut tl = TrieList::new();
        tl.append("foo".to_string(), 1);
        tl.append("bar".to_string(), 1);

        let map = tl.get_vocab_map();
        assert_eq!(map.len(), 2);
        assert_eq!(map["foo"], 0);
        assert_eq!(map["bar"], 1);
    }

    // Additional: test_duplicate_insert
    #[test]
    fn test_duplicate_insert() {
        let mut tl = TrieList::new();
        tl.append("hello".to_string(), 1);
        tl.append("hello".to_string(), 1); // duplicate
        assert_eq!(tl.len(), 1);
    }

    // Additional: test_partial_eq
    #[test]
    fn test_partial_eq() {
        let mut a = TrieList::new();
        a.append("x".to_string(), 1);
        a.append("y".to_string(), 2);

        let mut b = TrieList::new();
        b.append("x".to_string(), 99); // different life, same token order
        b.append("y".to_string(), 99);

        assert_eq!(a, b); // PartialEq only checks token strings & order
    }

    // Additional: test_iter
    #[test]
    fn test_iter() {
        let mut tl = TrieList::new();
        tl.append("one".to_string(), 1);
        tl.append("two".to_string(), 2);
        tl.append("three".to_string(), 3);

        let collected: Vec<(usize, String)> =
            tl.iter().map(|(i, e)| (i, e.token.clone())).collect();
        assert_eq!(
            collected,
            vec![
                (0, "one".to_string()),
                (1, "two".to_string()),
                (2, "three".to_string()),
            ]
        );
    }

    // Additional: test_get_entry_mut
    #[test]
    fn test_get_entry_mut() {
        let mut tl = TrieList::new();
        tl.append("tok".to_string(), 5);

        {
            let entry = tl.get_entry_mut(0).unwrap();
            entry.frequency += 10;
            entry.life -= 1;
        }

        let entry = tl.get_entry(0).unwrap();
        assert_eq!(entry.frequency, 10);
        assert_eq!(entry.life, 4);
    }

    // test_match_longest_skip_spaces
    #[test]
    fn test_match_longest_skip_spaces() {
        let mut tl = TrieList::new();
        tl.append("the".to_string(), 1);
        tl.append("the cat".to_string(), 1);
        tl.append("the cat sat".to_string(), 1);

        let result = tl.match_longest("the cat sat on", false);
        assert_eq!(result, Some(("the cat sat".to_string(), 2)));

        let result = tl.match_longest("the cat sat on", true);
        assert_eq!(result, Some(("the".to_string(), 0)));

        let result = tl.match_longest("the", true);
        assert_eq!(result, Some(("the".to_string(), 0)));

        let result = tl.match_longest("xyz", true);
        assert_eq!(result, None);
    }

    // test_match_two_skip_spaces
    #[test]
    fn test_match_two_skip_spaces() {
        let mut tl = TrieList::new();
        tl.append("the".to_string(), 1);
        tl.append("the cat".to_string(), 1);
        tl.append("the cat sat".to_string(), 1);

        let (best, second) = tl.match_two("the cat sat on", false);
        assert_eq!(best, Some(("the cat sat".to_string(), 2)));
        assert_eq!(second, Some(("the cat".to_string(), 1)));

        let (best, second) = tl.match_two("the cat sat on", true);
        assert_eq!(best, Some(("the".to_string(), 0)));
        assert_eq!(second, None);
    }
}
