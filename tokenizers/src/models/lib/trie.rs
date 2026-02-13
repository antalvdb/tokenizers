use serde::{Serialize, Deserialize};

/// Placeholder for the TrieList data structure.
/// Full implementation in Task 3.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrieList;

impl TrieList {
    pub fn new() -> Self { Self }
}
