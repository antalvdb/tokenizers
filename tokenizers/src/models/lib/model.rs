use crate::models::lib::trie::TrieList;
use crate::models::lib::trainer::LiBTrainer;
use crate::{Model, Token, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LiBModel {
    pub(crate) trie: TrieList,
    #[serde(default = "default_max_len")]
    pub max_len: usize,
    #[serde(default)]
    pub unk_token: Option<String>,
}

fn default_max_len() -> usize { 12 }

impl LiBModel {
    pub fn new(max_len: usize, unk_token: Option<String>) -> Self {
        Self {
            trie: TrieList::new(),
            max_len,
            unk_token,
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

    fn tokenize(&self, _sequence: &str) -> Result<Vec<Token>> {
        todo!("implement in Task 4")
    }
    fn token_to_id(&self, _token: &str) -> Option<u32> {
        todo!()
    }
    fn id_to_token(&self, _id: u32) -> Option<String> {
        todo!()
    }
    fn get_vocab(&self) -> HashMap<String, u32> {
        todo!()
    }
    fn get_vocab_size(&self) -> usize {
        todo!()
    }
    fn save(&self, _folder: &Path, _prefix: Option<&str>) -> Result<Vec<PathBuf>> {
        todo!()
    }
    fn get_trainer(&self) -> Self::Trainer {
        LiBTrainer::default()
    }
}
