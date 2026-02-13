use crate::models::lib::model::LiBModel;
use crate::{Trainer, AddedToken, Result};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LiBTrainer {}

impl Default for LiBTrainer {
    fn default() -> Self { Self {} }
}

impl Trainer for LiBTrainer {
    type Model = LiBModel;

    fn should_show_progress(&self) -> bool { true }

    fn train(&self, _model: &mut LiBModel) -> Result<Vec<AddedToken>> {
        todo!("implement in Task 5")
    }

    fn feed<I, S, F>(&mut self, _iterator: I, _process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        todo!("implement in Task 5")
    }
}
