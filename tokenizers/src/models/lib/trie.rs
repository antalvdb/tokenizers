use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// TEMPORARY verification switch: when true, matches report the ordinal a
// token had at the last full rebuild, reproducing the old trie's stale ids.
const STALE_COMPAT: bool = true;

/// Position value of a handle whose unit has been forgotten.
const DELETED: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Handle -- stable identity of a vocabulary entry
// ---------------------------------------------------------------------------

/// A unit's token id is its position in the priority order, which changes on
/// every re-ranking. A handle does not, so the trie and the reverse map store
/// handles and stay valid across reorders; a reorder only rewrites `pos_of`
/// for the moved span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Handle(u32);

impl Handle {
    fn idx(self) -> usize {
        self.0 as usize
    }
}

// ---------------------------------------------------------------------------
// TrieNode -- internal node of the array-based trie
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TrieNode {
    children: HashMap<char, usize>, // char -> node index in TrieList.nodes
    handle: Option<Handle>,         // Some(h) if this node terminates a live token
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

/// Forgotten units leave their handle and trie nodes behind until `sync()`,
/// which compacts everything so that handles equal positions again.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrieList {
    nodes: Vec<TrieNode>,                     // array-based trie, index 0 = root
    entries: Vec<VocabEntry>,                 // indexed by handle
    order: Vec<Handle>,                       // priority order: position (= token id) -> handle
    pos_of: Vec<u32>,                         // handle -> position, DELETED if forgotten
    trie_pos: Vec<u32>,                       // STALE_COMPAT only: position at last rebuild
    token_to_handle: HashMap<String, Handle>, // reverse lookup for live tokens
}

// PartialEq compares only the vocab ordering (not internal trie structure).
impl PartialEq for TrieList {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .zip(other.iter())
                .all(|((_, a), (_, b))| a.token == b.token)
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
            entries: Vec::new(),
            order: Vec::new(),
            pos_of: Vec::new(),
            trie_pos: Vec::new(),
            token_to_handle: HashMap::new(),
        }
    }

    // -- size queries -------------------------------------------------------

    /// Number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    // -- insertion ----------------------------------------------------------

    /// Append a token at the end of the vocab (lowest priority).
    pub fn append(&mut self, token: String, life: i32) {
        if self.token_to_handle.contains_key(&token) {
            return; // no duplicates
        }
        let h = Handle(self.entries.len() as u32);
        let pos = self.order.len() as u32;
        // trie_insert borrows &str, so do it before the move into the map/entries.
        self.trie_insert(&token, h);
        self.token_to_handle.insert(token.clone(), h);
        self.entries.push(VocabEntry { token, life, frequency: 0 });
        self.order.push(h);
        self.pos_of.push(pos);
        self.trie_pos.push(pos);
    }

    /// Insert a token at position `pos`, shifting subsequent items.
    pub fn insert_at(&mut self, pos: usize, token: String, life: i32) {
        if self.token_to_handle.contains_key(&token) {
            return; // no duplicates
        }
        let pos = pos.min(self.len()); // clamp
        self.append(token, life);
        let last = self.len() - 1;
        self.order[pos..=last].rotate_right(1);
        self.reindex_range(pos, last);
        self.snapshot_trie_pos();
    }

    // -- lookups ------------------------------------------------------------

    /// Check whether a token exists in the vocabulary.
    pub fn search(&self, token: &str) -> bool {
        self.token_to_handle.contains_key(token)
    }

    /// Return the vocabulary index (= token ID) for a given string.
    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.token_to_handle
            .get(token)
            .map(|h| self.pos_of[h.idx()] as usize)
    }

    /// Return the token string for a given ID.
    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.get_entry(id).map(|e| e.token.as_str())
    }

    /// Get an immutable reference to a vocab entry by ID.
    pub fn get_entry(&self, id: usize) -> Option<&VocabEntry> {
        self.order.get(id).map(|h| &self.entries[h.idx()])
    }

    /// Get a mutable reference to a vocab entry by ID.
    pub fn get_entry_mut(&mut self, id: usize) -> Option<&mut VocabEntry> {
        let h = *self.order.get(id)?;
        self.entries.get_mut(h.idx())
    }

    /// Build a HashMap<String, u32> suitable for the Model trait's `get_vocab`.
    pub fn get_vocab_map(&self) -> HashMap<String, u32> {
        self.iter()
            .map(|(i, e)| (e.token.clone(), i as u32))
            .collect()
    }

    /// Iterate over (id, &VocabEntry) pairs in priority order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &VocabEntry)> {
        self.order
            .iter()
            .enumerate()
            .map(move |(i, h)| (i, &self.entries[h.idx()]))
    }

    // -- matching -----------------------------------------------------------

    /// Token id reported for a trie match.
    fn match_id(&self, h: Handle) -> usize {
        if STALE_COMPAT {
            self.trie_pos[h.idx()] as usize
        } else {
            self.pos_of[h.idx()] as usize
        }
    }

    /// Longest-prefix match: walk the trie from the start of `input`,
    /// returning a slice of `input` for the longest matching token and its ID.
    /// The walk stops after `max_len` characters (matching the training constraint).
    pub fn match_longest<'a>(&self, input: &'a str, max_len: usize, skip_spaces: bool) -> Option<(&'a str, usize)> {
        let mut node_idx: usize = 0;
        let mut best: Option<(usize, Handle)> = None;
        let mut byte_end: usize = 0;
        let mut char_count: usize = 0;
        let mut has_space = false;

        for ch in input.chars() {
            if char_count >= max_len { break; }
            match self.nodes[node_idx].children.get(&ch) {
                Some(&next) => {
                    node_idx = next;
                    byte_end += ch.len_utf8();
                    char_count += 1;
                    if ch == ' ' { has_space = true; }
                    if let Some(h) = self.nodes[node_idx].handle {
                        if !skip_spaces || !has_space {
                            best = Some((byte_end, h));
                        }
                    }
                }
                None => break,
            }
        }
        best.map(|(end, h)| (&input[..end], self.match_id(h)))
    }

    /// Whether any token is a prefix of `input`, under the same `max_len` and
    /// `skip_spaces` rules as `match_longest`. Returns at the first token
    /// found, so it usually walks a single character.
    pub fn has_match(&self, input: &str, max_len: usize, skip_spaces: bool) -> bool {
        let mut node_idx: usize = 0;
        for (char_count, ch) in input.chars().enumerate() {
            if char_count >= max_len || (skip_spaces && ch == ' ') {
                return false;
            }
            match self.nodes[node_idx].children.get(&ch) {
                Some(&next) => {
                    node_idx = next;
                    if self.nodes[node_idx].handle.is_some() {
                        return true;
                    }
                }
                None => return false,
            }
        }
        false
    }

    /// Return the longest AND second-longest prefix matches.
    /// The first element is the longest, the second is the second-longest.
    /// Both are slices into `input`; `(&str, usize)` is Copy so no clone is needed.
    pub fn match_two<'a>(
        &self,
        input: &'a str,
        max_len: usize,
        skip_spaces: bool,
    ) -> (Option<(&'a str, usize)>, Option<(&'a str, usize)>) {
        let mut node_idx: usize = 0;
        let mut best: Option<(usize, Handle)> = None;
        let mut second: Option<(usize, Handle)> = None;
        let mut byte_end: usize = 0;
        let mut char_count: usize = 0;
        let mut has_space = false;

        for ch in input.chars() {
            if char_count >= max_len { break; }
            match self.nodes[node_idx].children.get(&ch) {
                Some(&next) => {
                    node_idx = next;
                    byte_end += ch.len_utf8();
                    char_count += 1;
                    if ch == ' ' { has_space = true; }
                    if let Some(h) = self.nodes[node_idx].handle {
                        if !skip_spaces || !has_space {
                            second = best;
                            best = Some((byte_end, h));
                        }
                    }
                }
                None => break,
            }
        }
        let resolve = |m: Option<(usize, Handle)>| m.map(|(end, h)| (&input[..end], self.match_id(h)));
        (resolve(best), resolve(second))
    }

    // -- reordering / pruning -----------------------------------------------

    /// Batch-update token priorities based on rewards.
    ///
    /// For each `(token, reward)`:
    ///   - positive reward  -> move toward front  (lower index = higher priority)
    ///   - negative reward  -> move toward back   (higher index = lower priority)
    ///   - step = `(current_index as f64 * update_rate + 1.0) as usize`
    pub fn batch_update(&mut self, rewards: &[(&str, f64)], update_rate: f64) {
        for &(token, reward) in rewards {
            let Some(&h) = self.token_to_handle.get(token) else { continue };
            let cur = self.pos_of[h.idx()] as usize;
            let step = (cur as f64 * update_rate + 1.0) as usize;
            if reward > 0.0 {
                let new_pos = cur.saturating_sub(step);
                self.order[new_pos..=cur].rotate_right(1);
                self.reindex_range(new_pos, cur);
            } else if reward < 0.0 {
                let new_pos = (cur + step).min(self.len() - 1);
                self.order[cur..=new_pos].rotate_left(1);
                self.reindex_range(cur, new_pos);
            }
        }
        self.snapshot_trie_pos();
    }

    /// Remove specific tokens from the vocabulary.
    pub fn remove_tokens(&mut self, tokens: &[&str]) {
        let to_remove: HashSet<&str> = tokens.iter().copied().collect();
        self.remove_where(|t| to_remove.contains(t));
        self.snapshot_trie_pos();
    }

    /// Remove the bottom `memory_out` fraction of the vocab (lowest priority =
    /// highest indices). Returns the list of removed token strings.
    pub fn prune(&mut self, memory_out: f64) -> Vec<String> {
        let total = self.len();
        let remove_count = (total as f64 * memory_out).round() as usize;
        if remove_count == 0 || total == 0 {
            return Vec::new();
        }
        let tail = self.order.split_off(total - remove_count);
        let removed = tail.into_iter().map(|h| self.forget(h)).collect();
        self.snapshot_trie_pos();
        removed
    }

    /// Batched ordinal re-ranking = active forgetting (Yang et al. 2020, §2.3.2;
    /// `structures.py::group_move`). Applied once per document with all rewards.
    /// Reward `e`: `-1` (good) moves the unit headward by `step = ⌊Θ·Δ⌋+1`;
    /// `+1` (bad) moves it tailward by the same step, and if that pushes it past
    /// the tail (`Θ+step ≥ |L|`) the unit is **deleted**. The head unit (Θ=0)
    /// never moves. Atomic units are never deleted. Returns deleted tokens.
    pub fn group_move(&mut self, rewards: &[(String, i32)], update_rate: f64) -> Vec<String> {
        let mut removed = Vec::new();
        for (w, e) in rewards {
            let h = match self.token_to_handle.get(w.as_str()) {
                Some(&h) => h,
                None => continue,
            };
            let cur = self.pos_of[h.idx()] as usize;
            if cur == 0 {
                continue;
            }
            let step = (cur as f64 * update_rate) as usize + 1;
            if *e < 0 {
                let dest = cur.saturating_sub(step);
                if dest != cur {
                    // Rotating the span moves only the entries between dest and
                    // cur, and only their positions need rewriting.
                    self.order[dest..=cur].rotate_right(1);
                    self.reindex_range(dest, cur);
                }
            } else if *e > 0 {
                let dest = cur + step;
                if dest >= self.len() && !Self::is_atomic(w) {
                    self.order.remove(cur); // Θ > |L| -> forget
                    self.reindex_range(cur, self.len().saturating_sub(1));
                    removed.push(self.forget(h));
                } else {
                    let d = dest.min(self.len() - 1);
                    if d != cur {
                        self.order[cur..=d].rotate_left(1);
                        self.reindex_range(cur, d);
                    }
                }
            }
        }
        if !removed.is_empty() {
            self.snapshot_trie_pos();
        }
        removed
    }

    /// Batch-remove tokens (passive forgetting's expired probation; the tail
    /// deletion of `structures.py::group_remove`). Atomic units are kept.
    pub fn group_remove(&mut self, tokens: &HashSet<String>) {
        if tokens.is_empty() {
            return;
        }
        if self.remove_where(|t| tokens.contains(t) && !Self::is_atomic(t)) > 0 {
            self.snapshot_trie_pos();
        }
    }

    /// Compact the structure: drop forgotten handles and dead trie nodes so
    /// that handles equal positions again. Call once after training.
    pub fn sync(&mut self) {
        let mut entries = std::mem::take(&mut self.entries);
        let live: Vec<VocabEntry> = self
            .order
            .iter()
            .map(|h| std::mem::replace(&mut entries[h.idx()], VocabEntry { token: String::new(), life: 0, frequency: 0 }))
            .collect();
        *self = Self::new();
        for e in live {
            self.append(e.token, e.life);
            if let Some(last) = self.entries.last_mut() {
                last.frequency = e.frequency;
            }
        }
    }

    /// Atomic units form the irreducible alphabet and are never forgotten:
    /// single characters and byte-fallback tokens (`<0x..>`).
    fn is_atomic(token: &str) -> bool {
        token.chars().count() <= 1 || (token.starts_with("<0x") && token.ends_with('>'))
    }

    // -- internal helpers ---------------------------------------------------

    /// Rewrite `pos_of` for the positions in `[lo, hi]`, the span affected by
    /// an ordinal move.
    fn reindex_range(&mut self, lo: usize, hi: usize) {
        if self.order.is_empty() {
            return;
        }
        let end = hi.min(self.order.len() - 1);
        for i in lo..=end {
            self.pos_of[self.order[i].idx()] = i as u32;
        }
    }

    /// STALE_COMPAT bookkeeping: the old implementation rebuilt its trie here,
    /// refreshing the ids that matches report.
    fn snapshot_trie_pos(&mut self) {
        if STALE_COMPAT {
            self.trie_pos.clone_from(&self.pos_of);
        }
    }

    /// Drop every unit for which `drop` holds from the order, then forget it.
    /// Returns how many units were removed.
    fn remove_where(&mut self, mut drop: impl FnMut(&str) -> bool) -> usize {
        let entries = &self.entries;
        let mut gone = Vec::new();
        self.order.retain(|&h| {
            let d = drop(&entries[h.idx()].token);
            if d {
                gone.push(h);
            }
            !d
        });
        let n = gone.len();
        for h in gone {
            self.forget(h);
        }
        if n > 0 {
            self.reindex_range(0, self.len());
        }
        n
    }

    /// Forget a unit already taken out of `order`: clear its trie terminal and
    /// reverse-map entry, mark its handle deleted, and return its token.
    fn forget(&mut self, h: Handle) -> String {
        let token = std::mem::take(&mut self.entries[h.idx()].token);
        let mut node_idx = 0;
        for ch in token.chars() {
            match self.nodes[node_idx].children.get(&ch) {
                Some(&next) => node_idx = next,
                None => break,
            }
        }
        if self.nodes[node_idx].handle == Some(h) {
            self.nodes[node_idx].handle = None;
        }
        self.token_to_handle.remove(token.as_str());
        self.pos_of[h.idx()] = DELETED;
        token
    }

    /// Insert a single token into the trie (does NOT touch the order or maps).
    fn trie_insert(&mut self, token: &str, h: Handle) {
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
        self.nodes[node_idx].handle = Some(h);
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

        assert_eq!(tl.id_to_token(0), Some("alpha"));
        assert_eq!(tl.id_to_token(1), Some("beta"));
        assert_eq!(tl.id_to_token(2), None);

        // roundtrip
        let id = tl.token_to_id("beta").unwrap();
        assert_eq!(tl.id_to_token(id), Some("beta"));
    }

    // 4. test_match_longest
    #[test]
    fn test_match_longest() {
        let mut tl = TrieList::new();
        tl.append("h".to_string(), 1);
        tl.append("he".to_string(), 1);
        tl.append("hel".to_string(), 1);
        tl.append("hello".to_string(), 1);

        let result = tl.match_longest("hello world", 12, false);
        assert_eq!(result, Some(("hello", 3)));

        // partial match should return longest that exists
        let result = tl.match_longest("help me", 12, false);
        assert_eq!(result, Some(("hel", 2)));

        // single char
        let result = tl.match_longest("hat", 12, false);
        assert_eq!(result, Some(("h", 0)));
    }

    // 5. test_match_two
    #[test]
    fn test_match_two() {
        let mut tl = TrieList::new();
        tl.append("h".to_string(), 1);
        tl.append("he".to_string(), 1);
        tl.append("hel".to_string(), 1);
        tl.append("hello".to_string(), 1);

        let (best, second) = tl.match_two("hello world", 12, false);
        assert_eq!(best, Some(("hello", 3)));
        assert_eq!(second, Some(("hel", 2)));

        // only two matches
        let (best, second) = tl.match_two("help", 12, false);
        assert_eq!(best, Some(("hel", 2)));
        assert_eq!(second, Some(("he", 1)));

        // only one match
        let (best, second) = tl.match_two("hat", 12, false);
        assert_eq!(best, Some(("h", 0)));
        assert_eq!(second, None);
    }

    // 6. test_match_no_match
    #[test]
    fn test_match_no_match() {
        let tl = TrieList::new();
        assert_eq!(tl.match_longest("anything", 12, false), None);

        let (best, second) = tl.match_two("anything", 12, false);
        assert_eq!(best, None);
        assert_eq!(second, None);

        // also test non-empty trie with no matching prefix
        let mut tl2 = TrieList::new();
        tl2.append("xyz".to_string(), 1);
        assert_eq!(tl2.match_longest("abc", 12, false), None);
    }

    // 7. test_unicode
    #[test]
    fn test_unicode() {
        let mut tl = TrieList::new();
        tl.append("é".to_string(), 1);   // e-acute
        tl.append("été".to_string(), 1); // ete with accents
        tl.append("世界".to_string(), 1); // Chinese: "world"
        tl.append("😀".to_string(), 1);  // emoji

        assert!(tl.search("é"));
        assert!(tl.search("世界"));
        assert!(tl.search("😀"));

        let result = tl.match_longest("été hello", 12, false);
        assert_eq!(result, Some(("été", 1)));

        let result = tl.match_longest("世界!", 12, false);
        assert_eq!(result, Some(("世界", 2)));
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
        tl.batch_update(&[("j", 1.0)], 0.5);
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

    #[test]
    fn test_group_move() {
        let mut tl = TrieList::new();
        for i in 0..10 {
            tl.append(format!("u{i}"), 0);
        }
        // "u9" (tail) rewarded good (-1) -> moves toward the head.
        let before = tl.token_to_id("u9").unwrap();
        tl.group_move(&[("u9".to_string(), -1)], 0.5);
        assert!(tl.token_to_id("u9").unwrap() < before, "good unit moves headward");

        // "u1" punished (+1) repeatedly -> pushed past |L| and deleted.
        for _ in 0..30 {
            tl.group_move(&[("u1".to_string(), 1)], 0.9);
        }
        assert!(!tl.search("u1"), "a repeatedly-bad unit is eventually forgotten");
    }

    #[test]
    fn test_group_remove() {
        use std::collections::HashSet;
        let mut tl = TrieList::new();
        tl.append("a".to_string(), 0); // atomic
        tl.append("keep me".to_string(), 0);
        tl.append("drop me".to_string(), 0);
        let del: HashSet<String> =
            vec!["drop me".to_string(), "a".to_string()].into_iter().collect();
        tl.group_remove(&del);
        assert!(!tl.search("drop me"), "listed multi-char unit is removed");
        assert!(tl.search("keep me"), "unlisted unit stays");
        assert!(tl.search("a"), "atomic unit is never removed");
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

        tl.remove_tokens(&["beta", "delta"]);

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
        let result = tl.match_longest("gamma ray", 12, false);
        assert_eq!(result, Some(("gamma", 1)));
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
        assert_eq!(tl.id_to_token(0), Some("a"));
        assert_eq!(tl.id_to_token(1), Some("X"));
        assert_eq!(tl.id_to_token(2), Some("b"));
        assert_eq!(tl.id_to_token(3), Some("c"));

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

        let result = tl.match_longest("the cat sat on", 12, false);
        assert_eq!(result, Some(("the cat sat", 2)));

        let result = tl.match_longest("the cat sat on", 12, true);
        assert_eq!(result, Some(("the", 0)));

        let result = tl.match_longest("the", 12, true);
        assert_eq!(result, Some(("the", 0)));

        let result = tl.match_longest("xyz", 12, true);
        assert_eq!(result, None);
    }

    #[test]
    fn test_has_match_agrees_with_match_longest() {
        let mut tl = TrieList::new();
        for t in ["the", "the cat", " cat", "c", "ca", "x y"] {
            tl.append(t.to_string(), 1);
        }
        for input in ["the cat sat", " cat", "cat", "x y", "x", "", "zzz", "th"] {
            for skip in [false, true] {
                for max_len in [1, 2, 12] {
                    assert_eq!(
                        tl.has_match(input, max_len, skip),
                        tl.match_longest(input, max_len, skip).is_some(),
                        "input {input:?} skip {skip} max_len {max_len}"
                    );
                }
            }
        }
    }

    // test_match_two_skip_spaces
    #[test]
    fn test_match_two_skip_spaces() {
        let mut tl = TrieList::new();
        tl.append("the".to_string(), 1);
        tl.append("the cat".to_string(), 1);
        tl.append("the cat sat".to_string(), 1);

        let (best, second) = tl.match_two("the cat sat on", 12, false);
        assert_eq!(best, Some(("the cat sat", 2)));
        assert_eq!(second, Some(("the cat", 1)));

        let (best, second) = tl.match_two("the cat sat on", 12, true);
        assert_eq!(best, Some(("the", 0)));
        assert_eq!(second, None);
    }
}
