use super::model::LiBModel;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::{self, MapAccess, Visitor};
use std::fmt;

impl Serialize for LiBModel {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let vocab: Vec<(&str, i32, u64)> = self
            .trie
            .iter()
            .map(|(_, entry)| (entry.token.as_str(), entry.life, entry.frequency))
            .collect();

        let mut s = serializer.serialize_struct("LiBModel", 5)?;
        s.serialize_field("type", "LiB")?;
        s.serialize_field("max_len", &self.max_len)?;
        s.serialize_field("unk_token", &self.unk_token)?;
        s.serialize_field("use_supra_words", &self.use_supra_words)?;
        s.serialize_field("vocab", &vocab)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for LiBModel {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Type,
            MaxLen,
            UnkToken,
            UseSupraWords,
            Vocab,
        }

        struct LiBModelVisitor;

        impl<'de> Visitor<'de> for LiBModelVisitor {
            type Value = LiBModel;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct LiBModel")
            }

            fn visit_map<V>(self, mut map: V) -> std::result::Result<LiBModel, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut max_len: Option<usize> = None;
                let mut unk_token: Option<Option<String>> = None;
                let mut use_supra_words: Option<bool> = None;
                let mut vocab: Option<Vec<(String, i32, u64)>> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Type => {
                            let value: String = map.next_value()?;
                            if value != "LiB" {
                                return Err(de::Error::custom(format!(
                                    "expected type \"LiB\", got \"{}\"",
                                    value
                                )));
                            }
                        }
                        Field::MaxLen => {
                            max_len = Some(map.next_value()?);
                        }
                        Field::UnkToken => {
                            unk_token = Some(map.next_value()?);
                        }
                        Field::UseSupraWords => {
                            use_supra_words = Some(map.next_value()?);
                        }
                        Field::Vocab => {
                            vocab = Some(map.next_value()?);
                        }
                    }
                }

                let max_len = max_len.unwrap_or(12);
                let unk_token = unk_token.unwrap_or(None);
                let vocab = vocab.ok_or_else(|| de::Error::missing_field("vocab"))?;

                let mut model = LiBModel::new(max_len, unk_token);
                for (token, life, _freq) in vocab {
                    model.trie.append(token, life);
                }
                model.use_supra_words = use_supra_words.unwrap_or(true);

                Ok(model)
            }
        }

        const FIELDS: &[&str] = &["type", "max_len", "unk_token", "use_supra_words", "vocab"];
        deserializer.deserialize_struct("LiBModel", FIELDS, LiBModelVisitor)
    }
}

#[cfg(test)]
mod tests {
    use crate::models::lib::model::LiBModel;
    use crate::Model;

    #[test]
    fn test_serialization_roundtrip() {
        let mut model = LiBModel::new(12, None);
        model.trie.append("hello".to_string(), 10);
        model.trie.append("world".to_string(), 10);
        model.trie.append("hello world".to_string(), 10);

        let json = serde_json::to_string(&model).unwrap();
        let deserialized: LiBModel = serde_json::from_str(&json).unwrap();

        assert_eq!(model.get_vocab_size(), deserialized.get_vocab_size());
        assert_eq!(model.get_vocab(), deserialized.get_vocab());
        assert_eq!(model.max_len, deserialized.max_len);

        let tokens_orig = model.tokenize("hello world").unwrap();
        let tokens_deser = deserialized.tokenize("hello world").unwrap();
        assert_eq!(tokens_orig.len(), tokens_deser.len());
        for (a, b) in tokens_orig.iter().zip(tokens_deser.iter()) {
            assert_eq!(a.value, b.value);
            assert_eq!(a.id, b.id);
        }
    }

    #[test]
    fn test_json_has_type_field() {
        let model = LiBModel::default();
        let json = serde_json::to_string(&model).unwrap();
        assert!(
            json.contains("\"type\":\"LiB\"") || json.contains("\"type\": \"LiB\""),
            "JSON should contain type discriminator: {}",
            json
        );
    }

    #[test]
    fn test_serialization_use_supra_words() {
        let mut model = LiBModel::new(12, None);
        model.trie.append("hello".to_string(), 10);
        model.trie.append("hello world".to_string(), 10);
        model.use_supra_words = false;

        let json = serde_json::to_string(&model).unwrap();
        assert!(json.contains("\"use_supra_words\":false") || json.contains("\"use_supra_words\": false"),
            "JSON should contain use_supra_words: {}", json);

        let deserialized: LiBModel = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.use_supra_words, false);

        // Backward compat: JSON without the field defaults to true
        let old_json = r#"{"type":"LiB","max_len":12,"unk_token":null,"vocab":[["a",10,0]]}"#;
        let old_model: LiBModel = serde_json::from_str(old_json).unwrap();
        assert_eq!(old_model.use_supra_words, true);
    }

    #[test]
    fn test_deserialization_rejects_wrong_type() {
        let json = r#"{"type":"BPE","max_len":12,"unk_token":null,"vocab":[]}"#;
        let result: std::result::Result<LiBModel, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }
}
