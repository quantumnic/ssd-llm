//! Async SSD → RAM streaming engine with prefetching

use crate::model::cache::CachedLayer;
use crate::model::gguf::{GgmlType, GgufFile, TensorInfo};
use crate::model::loader::MmapLoader;
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

use tracing::{debug, info};

/// SSD streaming engine that loads tensor data on-demand via mmap
pub struct SsdStreamer {
    loader: MmapLoader,
    budget: usize,
}

impl SsdStreamer {
    pub fn new(path: &Path, budget: usize) -> Result<Self> {
        // We need to parse the GGUF header to get data_offset
        let gguf = GgufFile::open(path)?;
        let loader = MmapLoader::new(path, gguf.data_offset)?;

        info!(
            "SSD Streamer initialized: file={:.2}GB, budget={:.2}GB",
            loader.file_size() as f64 / (1024.0 * 1024.0 * 1024.0),
            budget as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        Ok(Self { loader, budget })
    }

    /// Load a specific tensor's data and dequantize to f32
    pub fn load_tensor_f32(&self, tensor: &TensorInfo) -> Result<Vec<f32>> {
        let data = self
            .loader
            .get_tensor_data(tensor.offset, tensor.size_bytes)?;
        let n_elements: u64 = tensor.dimensions.iter().product();
        dequantize_to_f32(data, &tensor.dtype, n_elements as usize)
    }

    /// Load all tensors for a given layer
    pub fn load_layer(&self, gguf: &GgufFile, layer_idx: u32) -> Result<CachedLayer> {
        let tensors_info = gguf.layer_tensors(layer_idx);
        let mut tensors = HashMap::new();
        let mut total_size = 0usize;

        for ti in &tensors_info {
            let data = self.load_tensor_f32(ti)?;
            total_size += data.len() * std::mem::size_of::<f32>();
            tensors.insert(ti.name.clone(), data);
        }

        debug!(
            "Loaded layer {} ({} tensors, {:.2} MB)",
            layer_idx,
            tensors.len(),
            total_size as f64 / (1024.0 * 1024.0)
        );

        Ok(CachedLayer {
            layer_idx,
            tensors,
            size_bytes: total_size,
        })
    }

    /// Prefetch a layer — issue madvise WILLNEED for all its tensors
    pub fn prefetch_layer(&self, gguf: &GgufFile, layer_idx: u32) {
        for tensor in gguf.layer_tensors(layer_idx) {
            self.loader.prefetch(tensor.offset, tensor.size_bytes);
        }
        debug!("Prefetched layer {} from SSD", layer_idx);
    }

    /// Evict a layer's pages from OS cache
    pub fn evict_layer(&self, gguf: &GgufFile, layer_idx: u32) {
        for tensor in gguf.layer_tensors(layer_idx) {
            self.loader.evict(tensor.offset, tensor.size_bytes);
        }
    }

    /// Load a named tensor directly
    pub fn load_named_tensor_f32(&self, gguf: &GgufFile, name: &str) -> Result<Vec<f32>> {
        let tensor = gguf
            .find_tensor(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found: {}", name))?;
        self.load_tensor_f32(tensor)
    }
}

/// Dequantize raw bytes to f32
fn dequantize_to_f32(data: &[u8], dtype: &GgmlType, n_elements: usize) -> Result<Vec<f32>> {
    match dtype {
        GgmlType::F32 => {
            let floats: &[f32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_elements) };
            Ok(floats.to_vec())
        }
        GgmlType::F16 => {
            let halfs: &[u16] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, n_elements) };
            Ok(halfs
                .iter()
                .map(|&h| half::f16::from_bits(h).to_f32())
                .collect())
        }
        GgmlType::BF16 => {
            let halfs: &[u16] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, n_elements) };
            Ok(halfs
                .iter()
                .map(|&h| f32::from_bits((h as u32) << 16))
                .collect())
        }
        GgmlType::Q8_0 => dequantize_q8_0(data, n_elements),
        GgmlType::Q4_0 => dequantize_q4_0(data, n_elements),
        GgmlType::Q6K => dequantize_q6_k(data, n_elements),
        _ => {
            // For unsupported quantization types, return zeros with a warning
            tracing::warn!("Unsupported quantization type {:?}, returning zeros", dtype);
            Ok(vec![0.0f32; n_elements])
        }
    }
}

/// Dequantize Q8_0: block_size=32, layout: f16 scale + 32 x i8
fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let block_size = 32;
    let block_bytes = 34; // 2 (f16 scale) + 32 (int8 values)
    let n_blocks = n_elements / block_size;
    let mut output = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block_start = i * block_bytes;
        if block_start + block_bytes > data.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();

        for j in 0..block_size {
            let val = data[block_start + 2 + j] as i8;
            output.push(val as f32 * scale);
        }
    }

    output.resize(n_elements, 0.0);
    Ok(output)
}

/// Dequantize Q4_0: block_size=32, layout: f16 scale + 16 bytes (32 nibbles)
fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let block_size = 32;
    let block_bytes = 18; // 2 (f16 scale) + 16 (packed nibbles)
    let n_blocks = n_elements / block_size;
    let mut output = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block_start = i * block_bytes;
        if block_start + block_bytes > data.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();

        for j in 0..16 {
            let byte = data[block_start + 2 + j];
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
            output.push(lo as f32 * scale);
            output.push(hi as f32 * scale);
        }
    }

    output.resize(n_elements, 0.0);
    Ok(output)
}

/// Dequantize Q6_K: block_size=256
/// Layout per block: ql[128] + qh[64] + scales[16] + f16 d
fn dequantize_q6_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>> {
    let block_size: usize = 256;
    let block_bytes: usize = 210; // 128 + 64 + 16 + 2
    let n_blocks = n_elements / block_size;
    let mut output = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let bs = i * block_bytes;
        if bs + block_bytes > data.len() {
            break;
        }
        let ql = &data[bs..bs + 128];
        let qh = &data[bs + 128..bs + 192];
        let scales = &data[bs + 192..bs + 208];
        let d = half::f16::from_bits(u16::from_le_bytes([data[bs + 208], data[bs + 209]])).to_f32();

        for j in 0..256usize {
            let ql_idx = j / 2;
            let ql_byte = ql[ql_idx];
            let ql_val = if j % 2 == 0 { ql_byte & 0x0F } else { ql_byte >> 4 };

            let qh_idx = j / 4;
            let qh_shift = (j % 4) * 2;
            let qh_val = (qh[qh_idx] >> qh_shift) & 0x03;

            let q = (ql_val | (qh_val << 4)) as i8 - 32;
            let scale = scales[j / 16] as i8;

            output.push(d * scale as f32 * q as f32);
        }
    }

    output.resize(n_elements, 0.0);
    Ok(output)
}
