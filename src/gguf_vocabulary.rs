//! Reads a model's vocabulary from the metadata at the start of its GGUF file.

use crate::Error;

const GGUF_MAGIC: &[u8] = b"GGUF";
// Versions 2 and later give lengths and counts as 64-bit integers.
const MINIMUM_GGUF_VERSION: u32 = 2;
const TOKEN_TEXTS_KEY: &str = "tokenizer.ggml.tokens";
const TOKEN_TYPES_KEY: &str = "tokenizer.ggml.token_type";
// llama.cpp's `llama_token_type`; tokenizing with special tokens parsed matches the text of these types.
const UNKNOWN_TOKEN_TYPE: i32 = 2;
const CONTROL_TOKEN_TYPE: i32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GgufValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

impl GgufValueType {
    fn from_code(code: u32) -> Result<Self, Error> {
        match code {
            0 => Ok(GgufValueType::Uint8),
            1 => Ok(GgufValueType::Int8),
            2 => Ok(GgufValueType::Uint16),
            3 => Ok(GgufValueType::Int16),
            4 => Ok(GgufValueType::Uint32),
            5 => Ok(GgufValueType::Int32),
            6 => Ok(GgufValueType::Float32),
            7 => Ok(GgufValueType::Bool),
            8 => Ok(GgufValueType::String),
            9 => Ok(GgufValueType::Array),
            10 => Ok(GgufValueType::Uint64),
            11 => Ok(GgufValueType::Int64),
            12 => Ok(GgufValueType::Float64),
            _ => Err(Error::new(&format!("the model file has a metadata value of unknown type {code}"))),
        }
    }

}

pub struct GgufVocabulary {
    token_texts: Vec<String>,
    token_types: Vec<i32>,
}

impl GgufVocabulary {
    /// The texts that tokenizing a templated prompt reads as control tokens, so message text must not contain them.
    pub fn control_token_texts(&self, end_of_generation_token_ids: &[u32]) -> Vec<String> {
        let mut control_token_texts: Vec<String> = Vec::new();

        for (token_id, token_text) in self.token_texts.iter().enumerate() {
            let token_type: Option<i32> = self.token_types.get(token_id).copied();
            let control_type: bool = matches!(token_type, Some(UNKNOWN_TOKEN_TYPE | CONTROL_TOKEN_TYPE));
            // llama.cpp treats a model's end-of-generation tokens as control tokens even when the file types them otherwise.
            let ends_generation: bool = u32::try_from(token_id).is_ok_and(|token_id| end_of_generation_token_ids.contains(&token_id));

            if (control_type || ends_generation) && !token_text.is_empty() {
                control_token_texts.push(token_text.clone());
            }
        }

        control_token_texts
    }
}

pub enum VocabularyRead {
    Complete(GgufVocabulary),
    /// The vocabulary continues past the end of the bytes given.
    NeedsLongerPrefix,
}

enum ReadStop {
    EndOfPrefix,
    Invalid(Error),
}

impl From<Error> for ReadStop {
    fn from(error: Error) -> Self {
        ReadStop::Invalid(error)
    }
}

/// Reads from the first bytes of a GGUF file; the vocabulary sits in its metadata, before the tensors.
pub fn read_vocabulary(file_prefix: &[u8]) -> Result<VocabularyRead, Error> {
    let mut reader: PrefixReader = PrefixReader {
        bytes: file_prefix,
        position: 0,
    };
    let read_outcome: Result<GgufVocabulary, ReadStop> = read_vocabulary_from(&mut reader);

    match read_outcome {
        Ok(vocabulary) => Ok(VocabularyRead::Complete(vocabulary)),
        Err(ReadStop::EndOfPrefix) => Ok(VocabularyRead::NeedsLongerPrefix),
        Err(ReadStop::Invalid(error)) => Err(error),
    }
}

fn read_vocabulary_from(reader: &mut PrefixReader) -> Result<GgufVocabulary, ReadStop> {
    read_header(reader)?;

    let metadata_entry_count: u64 = reader.read_u64()?;
    let mut token_texts: Option<Vec<String>> = None;
    let mut token_types: Option<Vec<i32>> = None;

    for _ in 0..metadata_entry_count {
        let key: String = reader.read_string()?;
        let value_type: GgufValueType = GgufValueType::from_code(reader.read_u32()?)?;

        match key.as_str() {
            TOKEN_TEXTS_KEY => token_texts = Some(read_string_array(reader, value_type)?),
            TOKEN_TYPES_KEY => token_types = Some(read_int32_array(reader, value_type)?),
            _ => skip_value(reader, value_type)?,
        }

        if token_texts.is_some() && token_types.is_some() {
            break;
        }
    }

    match (token_texts, token_types) {
        (Some(token_texts), Some(token_types)) => Ok(GgufVocabulary {
            token_texts,
            token_types,
        }),
        _ => Err(ReadStop::Invalid(Error::new("the model file's metadata has no token texts and types"))),
    }
}

fn read_header(reader: &mut PrefixReader) -> Result<(), ReadStop> {
    let magic: &[u8] = reader.take(GGUF_MAGIC.len())?;

    if magic != GGUF_MAGIC {
        return Err(ReadStop::Invalid(Error::new("the model file is not a GGUF file")));
    }

    let version: u32 = reader.read_u32()?;

    if version < MINIMUM_GGUF_VERSION {
        return Err(ReadStop::Invalid(Error::new(&format!("the model file uses GGUF version {version}, which is not supported"))));
    }

    let _tensor_count: u64 = reader.read_u64()?;
    Ok(())
}

fn read_array_header(reader: &mut PrefixReader, value_type: GgufValueType, expected_element_type: GgufValueType) -> Result<usize, ReadStop> {
    let element_type: GgufValueType = GgufValueType::from_code(reader.read_u32()?)?;

    if value_type != GgufValueType::Array || element_type != expected_element_type {
        return Err(ReadStop::Invalid(Error::new("the model file's vocabulary has an unexpected type")));
    }

    let element_count: usize = usize::try_from(reader.read_u64()?).map_err(Error::from)?;
    Ok(element_count)
}

fn read_string_array(reader: &mut PrefixReader, value_type: GgufValueType) -> Result<Vec<String>, ReadStop> {
    let element_count: usize = read_array_header(reader, value_type, GgufValueType::String)?;
    let mut strings: Vec<String> = Vec::new();

    for _ in 0..element_count {
        strings.push(reader.read_string()?);
    }

    Ok(strings)
}

fn read_int32_array(reader: &mut PrefixReader, value_type: GgufValueType) -> Result<Vec<i32>, ReadStop> {
    let element_count: usize = read_array_header(reader, value_type, GgufValueType::Int32)?;
    let mut integers: Vec<i32> = Vec::new();

    for _ in 0..element_count {
        integers.push(reader.read_i32()?);
    }

    Ok(integers)
}

fn skip_value(reader: &mut PrefixReader, value_type: GgufValueType) -> Result<(), ReadStop> {
    match value_type {
        GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool => reader.skip(1),
        GgufValueType::Uint16 | GgufValueType::Int16 => reader.skip(2),
        GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32 => reader.skip(4),
        GgufValueType::Uint64 | GgufValueType::Int64 | GgufValueType::Float64 => reader.skip(8),
        GgufValueType::String => reader.read_string().map(|_| ()),
        GgufValueType::Array => skip_array(reader),
    }
}

fn skip_array(reader: &mut PrefixReader) -> Result<(), ReadStop> {
    let element_type: GgufValueType = GgufValueType::from_code(reader.read_u32()?)?;
    let element_count: u64 = reader.read_u64()?;

    for _ in 0..element_count {
        skip_value(reader, element_type)?;
    }

    Ok(())
}

/// GGUF stores integers little-endian.
struct PrefixReader<'bytes> {
    bytes: &'bytes [u8],
    position: usize,
}

impl<'bytes> PrefixReader<'bytes> {
    fn take(&mut self, byte_count: usize) -> Result<&'bytes [u8], ReadStop> {
        let end: usize = self.position.checked_add(byte_count).ok_or(ReadStop::EndOfPrefix)?;
        let taken: &'bytes [u8] = self.bytes.get(self.position..end).ok_or(ReadStop::EndOfPrefix)?;

        self.position = end;
        Ok(taken)
    }

    fn skip(&mut self, byte_count: usize) -> Result<(), ReadStop> {
        self.take(byte_count).map(|_| ())
    }

    fn take_array<const BYTE_COUNT: usize>(&mut self) -> Result<[u8; BYTE_COUNT], ReadStop> {
        let taken: &[u8] = self.take(BYTE_COUNT)?;
        let array: [u8; BYTE_COUNT] = taken.try_into().map_err(Error::from)?;
        Ok(array)
    }

    fn read_u32(&mut self) -> Result<u32, ReadStop> {
        Ok(u32::from_le_bytes(self.take_array()?))
    }

    fn read_i32(&mut self) -> Result<i32, ReadStop> {
        Ok(i32::from_le_bytes(self.take_array()?))
    }

    fn read_u64(&mut self) -> Result<u64, ReadStop> {
        Ok(u64::from_le_bytes(self.take_array()?))
    }

    fn read_string(&mut self) -> Result<String, ReadStop> {
        let byte_count: usize = usize::try_from(self.read_u64()?).map_err(Error::from)?;
        let string_bytes: &[u8] = self.take(byte_count)?;
        Ok(String::from_utf8_lossy(string_bytes).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Read;
    use std::path::PathBuf;

    use crate::gguf_vocabulary;
    use crate::gguf_vocabulary::VocabularyRead;

    const TEST_MODEL_RELATIVE_PATH: &str = "build/models/gemma-3-270m-it-Q8_0.gguf";
    const TEST_MODEL_PREFIX_BYTE_COUNT: u64 = 16 * 1024 * 1024;

    fn read_test_model_prefix(byte_count: u64) -> Vec<u8> {
        let model_path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TEST_MODEL_RELATIVE_PATH);
        let mut prefix: Vec<u8> = Vec::new();
        File::open(model_path).unwrap().take(byte_count).read_to_end(&mut prefix).unwrap();
        prefix
    }

    #[test]
    fn control_token_texts_include_turn_markers_but_not_ordinary_words() {
        let prefix: Vec<u8> = read_test_model_prefix(TEST_MODEL_PREFIX_BYTE_COUNT);

        let vocabulary_read: VocabularyRead = gguf_vocabulary::read_vocabulary(&prefix).unwrap();
        let VocabularyRead::Complete(vocabulary) = vocabulary_read
        else {
            panic!("the vocabulary continues past the prefix");
        };
        let control_token_texts: Vec<String> = vocabulary.control_token_texts(&[]);

        assert!(control_token_texts.iter().any(|text| text == "<start_of_turn>"));
        assert!(control_token_texts.iter().any(|text| text == "<end_of_turn>"));
        assert!(!control_token_texts.iter().any(|text| text == "hello"));
    }

    #[test]
    fn read_vocabulary_asks_for_longer_prefix_when_cut_inside_vocabulary() {
        let prefix: Vec<u8> = read_test_model_prefix(64 * 1024);

        let vocabulary_read: VocabularyRead = gguf_vocabulary::read_vocabulary(&prefix).unwrap();

        assert!(matches!(vocabulary_read, VocabularyRead::NeedsLongerPrefix));
    }
}
