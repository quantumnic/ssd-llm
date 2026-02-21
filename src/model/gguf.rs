//! GGUF file format parser
//! Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46475547; // "GGUF" in little-endian

/// GGUF tensor types
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    Unknown(u32),
}

impl GgmlType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            19 => Self::IQ1S,
            20 => Self::IQ4NL,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            23 => Self::IQ4XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1M,
            30 => Self::BF16,
            x => Self::Unknown(x),
        }
    }

    /// Block size for quantized types (number of elements per block)
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 1,
            Self::F16 | Self::BF16 | Self::I16 => 1,
            Self::I8 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            _ => 32, // sensible default
        }
    }

    /// Bytes per block for this type
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 => 1,
            Self::Q4_0 => 18,             // 2 + 32/2
            Self::Q4_1 => 20,             // 2 + 2 + 32/2
            Self::Q5_0 => 22,             // 2 + 4 + 32/2
            Self::Q5_1 => 24,             // 2 + 2 + 4 + 32/2
            Self::Q8_0 => 34,             // 2 + 32
            Self::Q8_1 => 40,             // 4 + 4 + 32
            Self::Q2K => 2 + 2 + 16 + 64, // f16 d + f16 dmin + 16B scales + 64B qs = 84
            Self::Q3K => 256 / 8 * 3 + 256 / 4 + 12 + 2,
            Self::Q4K => 2 + 2 + 12 + 256 / 2,
            Self::Q5K => 2 + 2 + 12 + 256 / 8 + 256 / 2,
            Self::Q6K => 256 / 2 + 256 / 4 + 256 / 16 + 2,
            Self::Q8K => 4 + 256 + 16 * 2, // f32 d + 256B qs + 16×i16 bsums = 292
            Self::I64 | Self::F64 => 8,
            _ => 2, // fallback
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
    Array(Vec<MetadataValue>),
}

impl fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UInt8(v) => write!(f, "{}", v),
            Self::Int8(v) => write!(f, "{}", v),
            Self::UInt16(v) => write!(f, "{}", v),
            Self::Int16(v) => write!(f, "{}", v),
            Self::UInt32(v) => write!(f, "{}", v),
            Self::Int32(v) => write!(f, "{}", v),
            Self::Float32(v) => write!(f, "{}", v),
            Self::Bool(v) => write!(f, "{}", v),
            Self::String(v) => write!(f, "{}", v),
            Self::UInt64(v) => write!(f, "{}", v),
            Self::Int64(v) => write!(f, "{}", v),
            Self::Float64(v) => write!(f, "{}", v),
            Self::Array(v) => write!(f, "[array of {} elements]", v.len()),
        }
    }
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::UInt32(v) => Some(*v),
            Self::Int32(v) => Some(*v as u32),
            Self::UInt64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::UInt64(v) => Some(*v),
            Self::UInt32(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64, // offset from start of data section
    pub size_bytes: u64,
}

pub struct GgufFile {
    pub header: GgufHeader,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64, // absolute file offset where tensor data begins
    pub file_size: u64,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).context("Failed to open GGUF file")?;
        let file_size = file.metadata()?.len();
        let mut reader = BufReader::new(file);

        // Read magic
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            bail!(
                "Not a GGUF file (magic: {:#x}, expected {:#x})",
                magic,
                GGUF_MAGIC
            );
        }

        // Read version
        let version = reader.read_u32::<LittleEndian>()?;
        if !(2..=3).contains(&version) {
            bail!("Unsupported GGUF version: {} (supported: 2-3)", version);
        }

        // Read counts
        let tensor_count = reader.read_u64::<LittleEndian>()?;
        let metadata_kv_count = reader.read_u64::<LittleEndian>()?;

        let header = GgufHeader {
            version,
            tensor_count,
            metadata_kv_count,
        };

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut reader)?;
            let value = read_metadata_value(&mut reader)?;
            metadata.insert(key, value);
        }

        // Read tensor infos
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut reader)?;
            let n_dims = reader.read_u32::<LittleEndian>()?;
            let mut dimensions = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dimensions.push(reader.read_u64::<LittleEndian>()?);
            }
            let dtype = GgmlType::from_u32(reader.read_u32::<LittleEndian>()?);
            let offset = reader.read_u64::<LittleEndian>()?;

            // Calculate size
            let n_elements: u64 = dimensions.iter().product();
            let block_size = dtype.block_size() as u64;
            let type_size = dtype.type_size() as u64;
            let n_blocks = n_elements.div_ceil(block_size);
            let size_bytes = n_blocks * type_size;

            tensors.push(TensorInfo {
                name,
                dimensions,
                dtype,
                offset,
                size_bytes,
            });
        }

        // Data section starts at aligned position after header
        let current_pos = reader.stream_position()?;
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as u64;
        let data_offset = current_pos.div_ceil(alignment) * alignment;

        Ok(GgufFile {
            header,
            metadata,
            tensors,
            data_offset,
            file_size,
        })
    }

    /// Get model architecture name
    pub fn architecture(&self) -> &str {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_string())
            .unwrap_or("unknown")
    }

    /// Get number of layers
    pub fn n_layers(&self) -> u32 {
        let arch = self.architecture();
        self.metadata
            .get(&format!("{}.block_count", arch))
            .and_then(|v| v.as_u32())
            .unwrap_or(0)
    }

    /// Get embedding dimension
    pub fn n_embd(&self) -> u32 {
        let arch = self.architecture();
        self.metadata
            .get(&format!("{}.embedding_length", arch))
            .and_then(|v| v.as_u32())
            .unwrap_or(0)
    }

    /// Get number of attention heads
    pub fn n_head(&self) -> u32 {
        let arch = self.architecture();
        self.metadata
            .get(&format!("{}.attention.head_count", arch))
            .and_then(|v| v.as_u32())
            .unwrap_or(0)
    }

    /// Get number of KV heads
    pub fn n_head_kv(&self) -> u32 {
        let arch = self.architecture();
        self.metadata
            .get(&format!("{}.attention.head_count_kv", arch))
            .and_then(|v| v.as_u32())
            .unwrap_or(self.n_head())
    }

    /// Get context length
    pub fn n_ctx(&self) -> u32 {
        let arch = self.architecture();
        self.metadata
            .get(&format!("{}.context_length", arch))
            .and_then(|v| v.as_u32())
            .unwrap_or(2048)
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> u32 {
        self.metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                MetadataValue::Array(arr) => Some(arr.len() as u32),
                _ => None,
            })
            .unwrap_or(0)
    }

    /// Get number of experts (0 if not a MoE model)
    pub fn n_experts(&self) -> u32 {
        let arch = self.architecture();
        self.metadata
            .get(&format!("{}.expert_count", arch))
            .and_then(|v| v.as_u32())
            .unwrap_or(0)
    }

    /// Get number of experts used per token (0 if not MoE)
    pub fn n_experts_used(&self) -> u32 {
        let arch = self.architecture();
        self.metadata
            .get(&format!("{}.expert_used_count", arch))
            .and_then(|v| v.as_u32())
            .unwrap_or(0)
    }

    /// Check if this is a Mixture of Experts model
    pub fn is_moe(&self) -> bool {
        self.n_experts() > 0 && self.n_experts_used() > 0
    }

    /// Find tensor by name
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get tensors for a specific layer
    pub fn layer_tensors(&self, layer_idx: u32) -> Vec<&TensorInfo> {
        let prefix = format!("blk.{}.", layer_idx);
        self.tensors
            .iter()
            .filter(|t| t.name.starts_with(&prefix))
            .collect()
    }

    /// Calculate total size of a layer's tensors
    pub fn layer_size(&self, layer_idx: u32) -> u64 {
        self.layer_tensors(layer_idx)
            .iter()
            .map(|t| t.size_bytes)
            .sum()
    }

    /// Get a u32 metadata value by key
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32())
    }

    /// Get a f32 metadata value by key
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.iter().map(|t| t.name.clone()).collect()
    }
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    if len > 1024 * 1024 {
        bail!("String too long: {} bytes", len);
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_metadata_value<R: Read>(reader: &mut R) -> Result<MetadataValue> {
    let value_type = reader.read_u32::<LittleEndian>()?;
    read_metadata_value_typed(reader, value_type)
}

fn read_metadata_value_typed<R: Read>(reader: &mut R, value_type: u32) -> Result<MetadataValue> {
    match value_type {
        0 => Ok(MetadataValue::UInt8(reader.read_u8()?)),
        1 => Ok(MetadataValue::Int8(reader.read_i8()?)),
        2 => Ok(MetadataValue::UInt16(reader.read_u16::<LittleEndian>()?)),
        3 => Ok(MetadataValue::Int16(reader.read_i16::<LittleEndian>()?)),
        4 => Ok(MetadataValue::UInt32(reader.read_u32::<LittleEndian>()?)),
        5 => Ok(MetadataValue::Int32(reader.read_i32::<LittleEndian>()?)),
        6 => Ok(MetadataValue::Float32(reader.read_f32::<LittleEndian>()?)),
        7 => Ok(MetadataValue::Bool(reader.read_u8()? != 0)),
        8 => Ok(MetadataValue::String(read_gguf_string(reader)?)),
        9 => {
            // Array
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()? as usize;
            // For large arrays (like token lists), limit what we store
            let store_limit = 1000;
            let mut values = Vec::with_capacity(count.min(store_limit));
            for i in 0..count {
                let val = read_metadata_value_typed(reader, elem_type)?;
                if i < store_limit {
                    values.push(val);
                }
            }
            Ok(MetadataValue::Array(values))
        }
        10 => Ok(MetadataValue::UInt64(reader.read_u64::<LittleEndian>()?)),
        11 => Ok(MetadataValue::Int64(reader.read_i64::<LittleEndian>()?)),
        12 => Ok(MetadataValue::Float64(reader.read_f64::<LittleEndian>()?)),
        _ => bail!("Unknown metadata value type: {}", value_type),
    }
}
