//! Quantized KV Cache — INT8 per-row quantization for 4x memory reduction
//!
//! Stores key/value vectors as INT8 with per-row absmax scale factors.
//! This reduces KV cache memory from 4 bytes/element (f32) to ~1 byte/element,
//! enabling 4x longer context windows within the same memory budget.
//!
//! Quantization scheme: symmetric per-row absmax
//!   scale = max(|x|) / 127
//!   quantized[i] = round(x[i] / scale)
//!   dequantized[i] = quantized[i] * scale

use tracing::debug;

/// A single quantized vector: INT8 data + f32 scale factor
#[derive(Clone)]
struct QuantizedVec {
    data: Vec<i8>,
    scale: f32,
}

impl QuantizedVec {
    /// Quantize an f32 vector to INT8 with absmax scaling
    fn quantize(values: &[f32]) -> Self {
        let absmax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        if absmax == 0.0 {
            return Self {
                data: vec![0i8; values.len()],
                scale: 0.0,
            };
        }

        let scale = absmax / 127.0;
        let inv_scale = 127.0 / absmax;
        let data: Vec<i8> = values
            .iter()
            .map(|&v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8)
            .collect();

        Self { data, scale }
    }

    /// Dequantize back to f32
    fn dequantize(&self) -> Vec<f32> {
        self.data.iter().map(|&q| q as f32 * self.scale).collect()
    }

    /// Dequantize a slice into a pre-allocated buffer
    fn dequantize_into(&self, start: usize, len: usize, out: &mut [f32]) {
        for (i, &q) in self.data[start..start + len].iter().enumerate() {
            out[i] = q as f32 * self.scale;
        }
    }

    /// Compute dot product with an f32 vector without full dequantization
    /// This is the key optimization: avoids allocating a dequantized vector
    fn dot_f32(&self, other: &[f32], start: usize, len: usize) -> f32 {
        let mut sum = 0i32;
        let data = &self.data[start..start + len];

        // Process in chunks of 4 for better ILP
        let chunks = len / 4;
        let remainder = len % 4;

        for i in 0..chunks {
            let base = i * 4;
            // Accumulate in i32 to avoid overflow
            sum += data[base] as i32 * (other[base] * 1000.0) as i32;
            sum += data[base + 1] as i32 * (other[base + 1] * 1000.0) as i32;
            sum += data[base + 2] as i32 * (other[base + 2] * 1000.0) as i32;
            sum += data[base + 3] as i32 * (other[base + 3] * 1000.0) as i32;
        }

        for i in (chunks * 4)..(chunks * 4 + remainder) {
            sum += data[i] as i32 * (other[i] * 1000.0) as i32;
        }

        (sum as f32 * self.scale) / 1000.0
    }

    /// Memory usage in bytes
    fn size_bytes(&self) -> usize {
        self.data.len() + std::mem::size_of::<f32>() // data + scale
    }
}

/// Per-layer quantized KV cache
#[derive(Clone)]
pub struct QuantizedLayerKvCache {
    keys: Vec<QuantizedVec>,
    values: Vec<QuantizedVec>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl QuantizedLayerKvCache {
    pub fn new(n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            n_kv_heads,
            head_dim,
        }
    }

    /// Append key/value for the current position, quantizing on the fly
    pub fn append(&mut self, key: Vec<f32>, value: Vec<f32>) {
        self.keys.push(QuantizedVec::quantize(&key));
        self.values.push(QuantizedVec::quantize(&value));
    }

    /// Current sequence length
    pub fn seq_len(&self) -> usize {
        self.keys.len()
    }

    /// Get dequantized key vector at position for a specific KV head
    pub fn key_at(&self, pos: usize, kv_head: usize) -> Vec<f32> {
        let start = kv_head * self.head_dim;
        let mut out = vec![0.0f32; self.head_dim];
        self.keys[pos].dequantize_into(start, self.head_dim, &mut out);
        out
    }

    /// Get dequantized value vector at position for a specific KV head
    pub fn value_at(&self, pos: usize, kv_head: usize) -> Vec<f32> {
        let start = kv_head * self.head_dim;
        let mut out = vec![0.0f32; self.head_dim];
        self.values[pos].dequantize_into(start, self.head_dim, &mut out);
        out
    }

    /// Compute dot product of query with cached key at position, without dequantizing
    pub fn key_dot_query(&self, pos: usize, kv_head: usize, query: &[f32]) -> f32 {
        let start = kv_head * self.head_dim;
        self.keys[pos].dot_f32(query, start, self.head_dim)
    }

    /// Memory usage in bytes (much less than f32 cache)
    pub fn size_bytes(&self) -> usize {
        let key_bytes: usize = self.keys.iter().map(|k| k.size_bytes()).sum();
        let val_bytes: usize = self.values.iter().map(|v| v.size_bytes()).sum();
        key_bytes + val_bytes
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }

    /// Rollback to a given sequence length
    pub fn rollback(&mut self, new_len: usize) {
        self.keys.truncate(new_len);
        self.values.truncate(new_len);
    }
}

/// Full quantized KV cache across all layers
pub struct QuantizedKvCache {
    layers: Vec<QuantizedLayerKvCache>,
    max_seq_len: usize,
}

impl QuantizedKvCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let layers = (0..n_layers)
            .map(|_| QuantizedLayerKvCache::new(n_kv_heads, head_dim))
            .collect();
        debug!(
            "Quantized KV cache (INT8) initialized: layers={}, kv_heads={}, head_dim={}, max_seq={}",
            n_layers, n_kv_heads, head_dim, max_seq_len
        );
        Self {
            layers,
            max_seq_len,
        }
    }

    pub fn layer_mut(&mut self, layer_idx: usize) -> &mut QuantizedLayerKvCache {
        &mut self.layers[layer_idx]
    }

    pub fn layer(&self, layer_idx: usize) -> &QuantizedLayerKvCache {
        &self.layers[layer_idx]
    }

    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len()).unwrap_or(0)
    }

    pub fn size_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.size_bytes()).sum()
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    pub fn rollback(&mut self, new_len: usize) {
        for layer in &mut self.layers {
            layer.rollback(new_len);
        }
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn is_full(&self) -> bool {
        self.seq_len() >= self.max_seq_len
    }

    /// Compression ratio vs f32 cache (should be ~4x)
    pub fn compression_ratio(&self, n_kv_heads: usize, head_dim: usize) -> f32 {
        let seq = self.seq_len();
        if seq == 0 {
            return 0.0;
        }
        let f32_bytes =
            seq * self.layers.len() * n_kv_heads * head_dim * std::mem::size_of::<f32>() * 2;
        f32_bytes as f32 / self.size_bytes().max(1) as f32
    }
}

/// Configuration for KV cache quantization
#[derive(Clone, Debug, Default)]
pub struct KvQuantConfig {
    /// Enable INT8 quantization
    pub enabled: bool,
    /// Minimum sequence length before quantization kicks in
    /// (short sequences don't benefit much from quantization)
    pub min_seq_len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let values: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let qvec = QuantizedVec::quantize(&values);
        let deq = qvec.dequantize();

        // INT8 quantization should have < 1% relative error for most values
        for (orig, deq_val) in values.iter().zip(deq.iter()) {
            let abs_err = (orig - deq_val).abs();
            // Absolute error should be bounded by scale / 2
            assert!(
                abs_err < qvec.scale + 0.01,
                "Too much quantization error: orig={}, deq={}, err={}",
                orig,
                deq_val,
                abs_err
            );
        }
    }

    #[test]
    fn test_quantize_zero_vector() {
        let values = vec![0.0f32; 64];
        let qvec = QuantizedVec::quantize(&values);
        assert_eq!(qvec.scale, 0.0);
        let deq = qvec.dequantize();
        assert!(deq.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_quantized_cache_basic() {
        let mut cache = QuantizedKvCache::new(2, 4, 64, 2048);
        assert_eq!(cache.seq_len(), 0);

        let key = vec![1.0f32; 4 * 64];
        let val = vec![2.0f32; 4 * 64];
        cache.layer_mut(0).append(key, val);
        assert_eq!(cache.layer(0).seq_len(), 1);

        let k = cache.layer(0).key_at(0, 0);
        assert_eq!(k.len(), 64);
        // Should be close to 1.0 (quantization error expected)
        assert!(
            (k[0] - 1.0).abs() < 0.05,
            "Dequantized key should be close to original: got {}",
            k[0]
        );
    }

    #[test]
    fn test_quantized_cache_memory_savings() {
        let n_kv_heads = 8;
        let head_dim = 128;
        let n_layers = 32;
        let seq_len = 1000;

        let mut cache = QuantizedKvCache::new(n_layers, n_kv_heads, head_dim, 4096);

        for _ in 0..seq_len {
            let kv_dim = n_kv_heads * head_dim;
            let key: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.01).sin()).collect();
            let val: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.02).cos()).collect();
            for layer in 0..n_layers {
                cache.layer_mut(layer).append(key.clone(), val.clone());
            }
        }

        let quantized_bytes = cache.size_bytes();
        let f32_bytes = seq_len * n_layers * n_kv_heads * head_dim * 4 * 2;

        let ratio = f32_bytes as f64 / quantized_bytes as f64;
        assert!(
            ratio > 3.0,
            "Quantized cache should be at least 3x smaller: ratio={}",
            ratio
        );
    }

    #[test]
    fn test_quantized_cache_rollback() {
        let mut cache = QuantizedKvCache::new(1, 2, 32, 512);
        for i in 0..5 {
            cache
                .layer_mut(0)
                .append(vec![i as f32; 64], vec![i as f32; 64]);
        }
        assert_eq!(cache.seq_len(), 5);
        cache.rollback(3);
        assert_eq!(cache.seq_len(), 3);
    }

    #[test]
    fn test_quantized_cache_clear() {
        let mut cache = QuantizedKvCache::new(2, 2, 32, 512);
        cache.layer_mut(0).append(vec![1.0; 64], vec![1.0; 64]);
        cache.layer_mut(1).append(vec![1.0; 64], vec![1.0; 64]);
        assert_eq!(cache.seq_len(), 1);
        cache.clear();
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_dequantize_into_slice() {
        let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.05).collect();
        let qvec = QuantizedVec::quantize(&values);

        let mut out = vec![0.0f32; 64];
        qvec.dequantize_into(64, 64, &mut out);

        let full_deq = qvec.dequantize();
        for i in 0..64 {
            assert!(
                (out[i] - full_deq[64 + i]).abs() < 1e-6,
                "Slice dequantize mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_compression_ratio() {
        let n_kv_heads = 4;
        let head_dim = 64;
        let mut cache = QuantizedKvCache::new(2, n_kv_heads, head_dim, 1024);

        for _ in 0..100 {
            let dim = n_kv_heads * head_dim;
            for layer in 0..2 {
                cache
                    .layer_mut(layer)
                    .append(vec![1.0; dim], vec![1.0; dim]);
            }
        }

        let ratio = cache.compression_ratio(n_kv_heads, head_dim);
        assert!(
            ratio > 3.0,
            "Compression ratio should be > 3x: got {}",
            ratio
        );
    }
}
