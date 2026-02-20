//! Vision/Multimodal support — CLIP ViT encoder + vision-language projection
//!
//! Enables LLaVA-style image understanding: encode images with CLIP ViT,
//! project visual tokens into the LLM embedding space, and interleave them
//! with text tokens for multimodal inference.
//!
//! Supports:
//! - CLIP ViT-L/14 @ 336px (LLaVA-1.5 default)
//! - Image preprocessing: resize, center crop, normalize
//! - Patch embedding via 2D convolution
//! - ViT transformer blocks with pre-norm
//! - MLP projection from CLIP hidden dim to LLM hidden dim
//! - Base64 and URL image loading in OpenAI-compatible API format

use anyhow::{bail, Result};
use std::f32::consts::PI;
use tracing::{debug, info};

/// CLIP Vision model configuration
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Image resolution (square), e.g. 336
    pub image_size: usize,
    /// Patch size, e.g. 14
    pub patch_size: usize,
    /// Hidden dimension of vision encoder
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Intermediate (MLP) size
    pub intermediate_size: usize,
    /// LLM embedding dimension (projection target)
    pub llm_hidden_size: usize,
    /// Projection type: "mlp" (2-layer) or "linear"
    pub projection_type: ProjectionType,
}

/// How vision features are projected into the LLM's embedding space
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectionType {
    /// Single linear layer
    Linear,
    /// Two-layer MLP with GELU activation (LLaVA-1.5 default)
    Mlp,
}

impl VisionConfig {
    /// Number of patches per side
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Total number of patches (excluding CLS token)
    pub fn num_patches(&self) -> usize {
        let n = self.num_patches_per_side();
        n * n
    }

    /// Total vision token count (patches + CLS token)
    pub fn num_vision_tokens(&self) -> usize {
        self.num_patches() + 1
    }

    /// Default config for CLIP ViT-L/14 @ 336px used in LLaVA-1.5
    pub fn clip_vit_l_336(llm_hidden_size: usize) -> Self {
        Self {
            image_size: 336,
            patch_size: 14,
            hidden_size: 1024,
            num_heads: 16,
            num_layers: 24,
            intermediate_size: 4096,
            llm_hidden_size,
            projection_type: ProjectionType::Mlp,
        }
    }
}

/// Pre-processed image ready for the vision encoder
#[derive(Debug, Clone)]
pub struct ProcessedImage {
    /// Pixel values in CHW format, normalized to CLIP stats
    /// Shape: [3, image_size, image_size]
    pub pixels: Vec<f32>,
    /// Original image dimensions before processing (width, height)
    pub original_size: (usize, usize),
}

/// CLIP normalization constants (ImageNet stats used by OpenAI CLIP)
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Preprocess an image for CLIP: resize, center crop, normalize.
///
/// Input: raw RGB pixels in HWC format, shape [height, width, 3], values in [0, 255].
/// Output: normalized CHW tensor for the vision encoder.
pub fn preprocess_image(
    rgb_pixels: &[u8],
    width: usize,
    height: usize,
    target_size: usize,
) -> Result<ProcessedImage> {
    if rgb_pixels.len() != width * height * 3 {
        bail!(
            "Expected {} bytes for {}x{} RGB image, got {}",
            width * height * 3,
            width,
            height,
            rgb_pixels.len()
        );
    }

    // Step 1: Resize shortest side to target_size, maintaining aspect ratio
    let (resized, rw, rh) = resize_bicubic(rgb_pixels, width, height, target_size);

    // Step 2: Center crop to target_size x target_size
    let cropped = center_crop(&resized, rw, rh, target_size);

    // Step 3: Convert to float [0, 1], normalize with CLIP stats, reorder to CHW
    let num_pixels = target_size * target_size;
    let mut pixels = vec![0.0f32; 3 * num_pixels];

    for y in 0..target_size {
        for x in 0..target_size {
            let src_idx = (y * target_size + x) * 3;
            for c in 0..3 {
                let val = cropped[src_idx + c] as f32 / 255.0;
                let normalized = (val - CLIP_MEAN[c]) / CLIP_STD[c];
                pixels[c * num_pixels + y * target_size + x] = normalized;
            }
        }
    }

    Ok(ProcessedImage {
        pixels,
        original_size: (width, height),
    })
}

/// Bicubic resize: resize so shortest side = target_size
fn resize_bicubic(
    pixels: &[u8],
    width: usize,
    height: usize,
    target_size: usize,
) -> (Vec<u8>, usize, usize) {
    let scale = if width < height {
        target_size as f32 / width as f32
    } else {
        target_size as f32 / height as f32
    };

    let new_width = (width as f32 * scale).round() as usize;
    let new_height = (height as f32 * scale).round() as usize;

    let mut result = vec![0u8; new_width * new_height * 3];

    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = x as f32 / scale;
            let src_y = y as f32 / scale;

            // Bilinear interpolation (good enough for CLIP preprocessing)
            let x0 = src_x.floor() as usize;
            let y0 = src_y.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let dst_idx = (y * new_width + x) * 3;
            for c in 0..3 {
                let v00 = pixels[(y0 * width + x0) * 3 + c] as f32;
                let v01 = pixels[(y0 * width + x1) * 3 + c] as f32;
                let v10 = pixels[(y1 * width + x0) * 3 + c] as f32;
                let v11 = pixels[(y1 * width + x1) * 3 + c] as f32;

                let v = v00 * (1.0 - fx) * (1.0 - fy)
                    + v01 * fx * (1.0 - fy)
                    + v10 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                result[dst_idx + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    (result, new_width, new_height)
}

/// Center crop to target_size x target_size
fn center_crop(pixels: &[u8], width: usize, height: usize, target_size: usize) -> Vec<u8> {
    let x_offset = (width.saturating_sub(target_size)) / 2;
    let y_offset = (height.saturating_sub(target_size)) / 2;

    let mut result = vec![0u8; target_size * target_size * 3];

    for y in 0..target_size {
        let src_y = y + y_offset;
        if src_y >= height {
            break;
        }
        for x in 0..target_size {
            let src_x = x + x_offset;
            if src_x >= width {
                break;
            }
            let src_idx = (src_y * width + src_x) * 3;
            let dst_idx = (y * target_size + x) * 3;
            result[dst_idx..dst_idx + 3].copy_from_slice(&pixels[src_idx..src_idx + 3]);
        }
    }

    result
}

/// GELU activation function (used in CLIP MLP)
fn gelu(x: f32) -> f32 {
    // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = (2.0 / PI).sqrt();
    x * 0.5 * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// Layer normalization
fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();

    for i in 0..n {
        x[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/// CLIP Vision Encoder weights
#[derive(Debug)]
pub struct VisionWeights {
    /// Patch embedding convolution: [hidden_size, 3, patch_size, patch_size]
    pub patch_embed_weight: Vec<f32>,
    /// Patch embedding bias: [hidden_size]
    pub patch_embed_bias: Vec<f32>,
    /// CLS token embedding: [hidden_size]
    pub cls_token: Vec<f32>,
    /// Positional embedding: [num_vision_tokens, hidden_size]
    pub position_embedding: Vec<f32>,
    /// Pre-LayerNorm weight and bias
    pub pre_ln_weight: Vec<f32>,
    pub pre_ln_bias: Vec<f32>,
    /// Transformer layer weights
    pub layers: Vec<VisionLayerWeights>,
    /// Post-LayerNorm weight and bias
    pub post_ln_weight: Vec<f32>,
    pub post_ln_bias: Vec<f32>,
    /// Vision-language projection weights
    pub projection: VisionProjection,
}

/// Weights for one ViT transformer layer
#[derive(Debug)]
pub struct VisionLayerWeights {
    /// Attention LayerNorm
    pub attn_ln_weight: Vec<f32>,
    pub attn_ln_bias: Vec<f32>,
    /// QKV combined weight: [3 * hidden_size, hidden_size]
    pub qkv_weight: Vec<f32>,
    pub qkv_bias: Vec<f32>,
    /// Output projection: [hidden_size, hidden_size]
    pub out_proj_weight: Vec<f32>,
    pub out_proj_bias: Vec<f32>,
    /// MLP LayerNorm
    pub mlp_ln_weight: Vec<f32>,
    pub mlp_ln_bias: Vec<f32>,
    /// MLP fc1: [intermediate_size, hidden_size]
    pub mlp_fc1_weight: Vec<f32>,
    pub mlp_fc1_bias: Vec<f32>,
    /// MLP fc2: [hidden_size, intermediate_size]
    pub mlp_fc2_weight: Vec<f32>,
    pub mlp_fc2_bias: Vec<f32>,
}

/// Vision-to-LLM projection weights
#[derive(Debug)]
pub enum VisionProjection {
    /// Single linear: [llm_hidden, vision_hidden]
    Linear { weight: Vec<f32>, bias: Vec<f32> },
    /// Two-layer MLP: fc1 + GELU + fc2 (LLaVA-1.5 style)
    Mlp {
        fc1_weight: Vec<f32>,
        fc1_bias: Vec<f32>,
        fc2_weight: Vec<f32>,
        fc2_bias: Vec<f32>,
    },
}

/// The CLIP Vision Encoder
pub struct VisionEncoder {
    config: VisionConfig,
    weights: VisionWeights,
}

impl VisionEncoder {
    pub fn new(config: VisionConfig, weights: VisionWeights) -> Self {
        Self { config, weights }
    }

    /// Encode an image into vision tokens projected to the LLM's embedding space.
    ///
    /// Input: preprocessed image (CHW normalized float pixels)
    /// Output: `[num_vision_tokens, llm_hidden_size]` feature vectors
    pub fn encode(&self, image: &ProcessedImage) -> Result<Vec<Vec<f32>>> {
        let cfg = &self.config;
        let w = &self.weights;
        let num_patches = cfg.num_patches();
        let nps = cfg.num_patches_per_side();
        let ps = cfg.patch_size;
        let h = cfg.hidden_size;

        debug!(
            "Vision encode: {}x{} image → {} patches, hidden={}",
            cfg.image_size, cfg.image_size, num_patches, h
        );

        // Step 1: Patch embedding — 2D convolution with stride=patch_size
        let mut patch_embeddings = vec![vec![0.0f32; h]; num_patches];
        for py in 0..nps {
            for px in 0..nps {
                let patch_idx = py * nps + px;
                for out_c in 0..h {
                    let mut sum = w.patch_embed_bias[out_c];
                    for in_c in 0..3 {
                        for ky in 0..ps {
                            for kx in 0..ps {
                                let iy = py * ps + ky;
                                let ix = px * ps + kx;
                                let pixel = image.pixels[in_c * cfg.image_size * cfg.image_size
                                    + iy * cfg.image_size
                                    + ix];
                                let kernel_idx =
                                    out_c * 3 * ps * ps + in_c * ps * ps + ky * ps + kx;
                                sum += pixel * w.patch_embed_weight[kernel_idx];
                            }
                        }
                    }
                    patch_embeddings[patch_idx][out_c] = sum;
                }
            }
        }

        // Step 2: Prepend CLS token and add positional embeddings
        let num_tokens = num_patches + 1;
        let mut hidden_states = Vec::with_capacity(num_tokens);
        // CLS token
        let mut cls = w.cls_token.clone();
        for i in 0..h {
            cls[i] += w.position_embedding[i];
        }
        hidden_states.push(cls);
        // Patch tokens
        for (idx, patch) in patch_embeddings.into_iter().enumerate() {
            let pos_offset = (idx + 1) * h;
            let mut token = patch;
            for i in 0..h {
                token[i] += w.position_embedding[pos_offset + i];
            }
            hidden_states.push(token);
        }

        // Step 3: Pre-LayerNorm
        for token in &mut hidden_states {
            layer_norm(token, &w.pre_ln_weight, &w.pre_ln_bias, 1e-5);
        }

        // Step 4: Transformer layers
        for (layer_idx, layer) in w.layers.iter().enumerate() {
            hidden_states = self.transformer_block(&hidden_states, layer, layer_idx)?;
        }

        // Step 5: Post-LayerNorm
        for token in &mut hidden_states {
            layer_norm(token, &w.post_ln_weight, &w.post_ln_bias, 1e-5);
        }

        // Step 6: Project to LLM embedding space
        let projected = self.project_to_llm(&hidden_states)?;

        info!(
            "Vision encoding complete: {} tokens → {} projected embeddings (dim={})",
            num_tokens,
            projected.len(),
            cfg.llm_hidden_size
        );

        Ok(projected)
    }

    /// Single ViT transformer block: LN → Attention → Residual → LN → MLP → Residual
    fn transformer_block(
        &self,
        hidden_states: &[Vec<f32>],
        layer: &VisionLayerWeights,
        _layer_idx: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let h = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let head_dim = h / num_heads;
        let num_tokens = hidden_states.len();

        // Pre-attention LayerNorm
        let mut normed: Vec<Vec<f32>> = hidden_states.to_vec();
        for token in &mut normed {
            layer_norm(token, &layer.attn_ln_weight, &layer.attn_ln_bias, 1e-5);
        }

        // QKV projection
        let mut queries = vec![vec![0.0f32; h]; num_tokens];
        let mut keys = vec![vec![0.0f32; h]; num_tokens];
        let mut values = vec![vec![0.0f32; h]; num_tokens];

        for t in 0..num_tokens {
            for i in 0..h {
                let mut q = layer.qkv_bias[i];
                let mut k = layer.qkv_bias[h + i];
                let mut v = layer.qkv_bias[2 * h + i];
                for j in 0..h {
                    q += layer.qkv_weight[i * h + j] * normed[t][j];
                    k += layer.qkv_weight[(h + i) * h + j] * normed[t][j];
                    v += layer.qkv_weight[(2 * h + i) * h + j] * normed[t][j];
                }
                queries[t][i] = q;
                keys[t][i] = k;
                values[t][i] = v;
            }
        }

        // Multi-head attention
        let mut attn_output = vec![vec![0.0f32; h]; num_tokens];
        let scale = 1.0 / (head_dim as f32).sqrt();

        for head in 0..num_heads {
            let offset = head * head_dim;

            // Compute attention scores for this head
            let mut scores = vec![vec![0.0f32; num_tokens]; num_tokens];
            for i in 0..num_tokens {
                for j in 0..num_tokens {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += queries[i][offset + d] * keys[j][offset + d];
                    }
                    scores[i][j] = dot * scale;
                }
            }

            // Softmax
            for i in 0..num_tokens {
                let max_score = scores[i].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for j in 0..num_tokens {
                    scores[i][j] = (scores[i][j] - max_score).exp();
                    sum += scores[i][j];
                }
                if sum > 0.0 {
                    for j in 0..num_tokens {
                        scores[i][j] /= sum;
                    }
                }
            }

            // Weighted sum of values
            for i in 0..num_tokens {
                for j in 0..num_tokens {
                    let w = scores[i][j];
                    for d in 0..head_dim {
                        attn_output[i][offset + d] += w * values[j][offset + d];
                    }
                }
            }
        }

        // Output projection
        let mut projected = vec![vec![0.0f32; h]; num_tokens];
        for t in 0..num_tokens {
            for i in 0..h {
                let mut sum = layer.out_proj_bias[i];
                for j in 0..h {
                    sum += layer.out_proj_weight[i * h + j] * attn_output[t][j];
                }
                projected[t][i] = sum;
            }
        }

        // Residual connection
        let mut output: Vec<Vec<f32>> = hidden_states
            .iter()
            .zip(projected.iter())
            .map(|(r, p)| r.iter().zip(p.iter()).map(|(a, b)| a + b).collect())
            .collect();

        // Pre-MLP LayerNorm
        let mut mlp_input = output.clone();
        for token in &mut mlp_input {
            layer_norm(token, &layer.mlp_ln_weight, &layer.mlp_ln_bias, 1e-5);
        }

        // MLP: fc1 → GELU → fc2
        let inter = self.config.intermediate_size;
        for t in 0..num_tokens {
            let mut fc1_out = vec![0.0f32; inter];
            for i in 0..inter {
                let mut sum = layer.mlp_fc1_bias[i];
                for j in 0..h {
                    sum += layer.mlp_fc1_weight[i * h + j] * mlp_input[t][j];
                }
                fc1_out[i] = gelu(sum);
            }

            let mut fc2_out = vec![0.0f32; h];
            for i in 0..h {
                let mut sum = layer.mlp_fc2_bias[i];
                for j in 0..inter {
                    sum += layer.mlp_fc2_weight[i * inter + j] * fc1_out[j];
                }
                fc2_out[i] = sum;
            }

            // Residual connection
            for i in 0..h {
                output[t][i] += fc2_out[i];
            }
        }

        Ok(output)
    }

    /// Project vision features to LLM embedding space
    fn project_to_llm(&self, hidden_states: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let h_vision = self.config.hidden_size;
        let h_llm = self.config.llm_hidden_size;

        let mut result = Vec::with_capacity(hidden_states.len());

        match &self.weights.projection {
            VisionProjection::Linear { weight, bias } => {
                for token in hidden_states {
                    let mut out = vec![0.0f32; h_llm];
                    for i in 0..h_llm {
                        let mut sum = bias[i];
                        for j in 0..h_vision {
                            sum += weight[i * h_vision + j] * token[j];
                        }
                        out[i] = sum;
                    }
                    result.push(out);
                }
            }
            VisionProjection::Mlp {
                fc1_weight,
                fc1_bias,
                fc2_weight,
                fc2_bias,
            } => {
                for token in hidden_states {
                    // fc1: vision_hidden → llm_hidden
                    let mut fc1_out = vec![0.0f32; h_llm];
                    for i in 0..h_llm {
                        let mut sum = fc1_bias[i];
                        for j in 0..h_vision {
                            sum += fc1_weight[i * h_vision + j] * token[j];
                        }
                        fc1_out[i] = gelu(sum);
                    }

                    // fc2: llm_hidden → llm_hidden
                    let mut fc2_out = vec![0.0f32; h_llm];
                    for i in 0..h_llm {
                        let mut sum = fc2_bias[i];
                        for j in 0..h_llm {
                            sum += fc2_weight[i * h_llm + j] * fc1_out[j];
                        }
                        fc2_out[i] = sum;
                    }

                    result.push(fc2_out);
                }
            }
        }

        Ok(result)
    }
}

/// Decode a base64-encoded image into raw RGB pixels.
/// Supports PNG and JPEG via minimal header parsing.
/// Returns (pixels, width, height).
pub fn decode_base64_image(base64_data: &str) -> Result<(Vec<u8>, usize, usize)> {
    // Strip data URL prefix if present
    let data = if let Some(pos) = base64_data.find(",") {
        &base64_data[pos + 1..]
    } else {
        base64_data
    };

    let bytes = base64_decode(data)?;

    // Detect format by magic bytes
    if bytes.len() >= 8 && &bytes[0..4] == b"\x89PNG" {
        decode_png_rgb(&bytes)
    } else if bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
        // JPEG — we return a placeholder error since full JPEG decoding is complex
        // In production, this would use a JPEG decoder crate
        bail!("JPEG decoding requires an external decoder; use PNG or provide raw RGB")
    } else {
        bail!("Unsupported image format (expected PNG or JPEG)")
    }
}

/// Minimal base64 decoder
fn base64_decode(input: &str) -> Result<Vec<u8>> {
    let table: Vec<u8> = (0..256u16)
        .map(|i| {
            let c = i as u8;
            match c {
                b'A'..=b'Z' => c - b'A',
                b'a'..=b'z' => c - b'a' + 26,
                b'0'..=b'9' => c - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                _ => 255,
            }
        })
        .collect();

    let filtered: Vec<u8> = input
        .bytes()
        .filter(|&b| b != b'\n' && b != b'\r' && b != b' ' && b != b'=')
        .collect();

    let mut result = Vec::with_capacity(filtered.len() * 3 / 4);
    let chunks = filtered.chunks(4);

    for chunk in chunks {
        let mut buf = [0u8; 4];
        for (i, &b) in chunk.iter().enumerate() {
            buf[i] = table[b as usize];
            if buf[i] == 255 {
                bail!("Invalid base64 character: {}", b as char);
            }
        }
        let n = chunk.len();
        if n >= 2 {
            result.push((buf[0] << 2) | (buf[1] >> 4));
        }
        if n >= 3 {
            result.push((buf[1] << 4) | (buf[2] >> 2));
        }
        if n >= 4 {
            result.push((buf[2] << 6) | buf[3]);
        }
    }

    Ok(result)
}

/// Minimal PNG decoder — extract raw RGB from uncompressed/deflate PNG
/// This handles simple RGBA/RGB PNGs. For production, use the `png` crate.
fn decode_png_rgb(data: &[u8]) -> Result<(Vec<u8>, usize, usize)> {
    // Parse IHDR chunk for dimensions
    if data.len() < 33 {
        bail!("PNG too small");
    }

    // Skip signature (8 bytes), IHDR length (4 bytes), "IHDR" (4 bytes)
    let width = u32::from_be_bytes([data[16], data[17], data[18], data[19]]) as usize;
    let height = u32::from_be_bytes([data[20], data[21], data[22], data[23]]) as usize;
    let bit_depth = data[24];
    let color_type = data[25];

    if bit_depth != 8 {
        bail!("Only 8-bit PNG supported, got {}-bit", bit_depth);
    }

    let channels = match color_type {
        2 => 3, // RGB
        6 => 4, // RGBA
        _ => bail!("Unsupported PNG color type: {}", color_type),
    };

    // Collect IDAT chunks
    let mut idat_data = Vec::new();
    let mut offset = 8; // skip PNG signature
    while offset + 12 <= data.len() {
        let chunk_len = u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        let chunk_type = &data[offset + 4..offset + 8];

        if chunk_type == b"IDAT" {
            idat_data.extend_from_slice(&data[offset + 8..offset + 8 + chunk_len]);
        }

        offset += 12 + chunk_len; // length + type + data + CRC
    }

    // Decompress IDAT (zlib/deflate)
    let decompressed = inflate_zlib(&idat_data)?;

    // Unfilter and extract RGB
    let stride = 1 + width * channels; // filter byte + pixel data
    if decompressed.len() < height * stride {
        bail!(
            "Decompressed PNG data too short: {} < {}",
            decompressed.len(),
            height * stride
        );
    }

    let mut rgb = vec![0u8; width * height * 3];

    for y in 0..height {
        let row_start = y * stride;
        let filter_type = decompressed[row_start];
        let row_data = &decompressed[row_start + 1..row_start + stride];

        // Apply PNG filter (only None and Sub for simplicity)
        let mut filtered_row = row_data.to_vec();
        match filter_type {
            0 => {} // None
            1 => {
                // Sub
                for i in channels..filtered_row.len() {
                    filtered_row[i] = filtered_row[i].wrapping_add(filtered_row[i - channels]);
                }
            }
            2 => {
                // Up
                if y > 0 {
                    let prev_start = (y - 1) * stride + 1;
                    for i in 0..filtered_row.len() {
                        filtered_row[i] =
                            filtered_row[i].wrapping_add(decompressed[prev_start + i]);
                    }
                }
            }
            _ => {} // Average, Paeth — skip for minimal implementation
        }

        for x in 0..width {
            let src = x * channels;
            let dst = (y * width + x) * 3;
            rgb[dst] = filtered_row[src]; // R
            rgb[dst + 1] = filtered_row[src + 1]; // G
            rgb[dst + 2] = filtered_row[src + 2]; // B
                                                  // Alpha channel (if RGBA) is discarded
        }
    }

    Ok((rgb, width, height))
}

/// Minimal zlib inflate (skip 2-byte zlib header, use raw deflate)
fn inflate_zlib(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 2 {
        bail!("Zlib data too short");
    }

    // Check zlib header
    let cmf = data[0];
    let cm = cmf & 0x0F;
    if cm != 8 {
        bail!("Not a zlib/deflate stream (CM={})", cm);
    }

    // Skip 2-byte zlib header, inflate the deflate stream
    inflate_deflate(&data[2..])
}

/// Minimal DEFLATE decompressor — handles uncompressed blocks and fixed Huffman
fn inflate_deflate(data: &[u8]) -> Result<Vec<u8>> {
    // For production, we'd use the `flate2` crate.
    // This minimal implementation handles stored (uncompressed) blocks.
    let mut output = Vec::new();
    let mut pos = 0;
    let mut bit_pos = 0u32;

    loop {
        if pos >= data.len() {
            break;
        }

        let bfinal = read_bits(data, &mut pos, &mut bit_pos, 1);
        let btype = read_bits(data, &mut pos, &mut bit_pos, 2);

        match btype {
            0 => {
                // Stored block
                // Align to byte boundary
                bit_pos = 0;
                pos += 1;
                if pos + 4 > data.len() {
                    break;
                }
                let len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 4; // skip LEN and NLEN
                if pos + len > data.len() {
                    bail!("Stored block exceeds data");
                }
                output.extend_from_slice(&data[pos..pos + len]);
                pos += len;
            }
            1 | 2 => {
                // Fixed or dynamic Huffman — fallback: try to use the data as-is
                // A real implementation would decode Huffman trees here.
                // For now, bail and let the caller know.
                bail!("Compressed PNG requires full DEFLATE support; use the `flate2` crate in production builds");
            }
            _ => bail!("Invalid DEFLATE block type: {}", btype),
        }

        if bfinal == 1 {
            break;
        }
    }

    Ok(output)
}

/// Read bits from a byte stream
fn read_bits(data: &[u8], pos: &mut usize, bit_pos: &mut u32, n: u32) -> u32 {
    let mut result = 0u32;
    for i in 0..n {
        if *pos >= data.len() {
            return result;
        }
        let bit = (data[*pos] >> *bit_pos) & 1;
        result |= (bit as u32) << i;
        *bit_pos += 1;
        if *bit_pos >= 8 {
            *bit_pos = 0;
            *pos += 1;
        }
    }
    result
}

/// Multimodal message content — represents text and/or image parts
#[derive(Debug, Clone)]
pub enum ContentPart {
    /// Plain text content
    Text(String),
    /// Image from base64 data
    ImageBase64 { data: String, media_type: String },
    /// Image from URL
    ImageUrl { url: String },
}

/// Parse OpenAI-style multimodal content array from JSON.
///
/// Supports both:
/// - Simple string content: `"content": "hello"`
/// - Array content: `"content": [{"type": "text", "text": "..."}, {"type": "image_url", ...}]`
pub fn parse_content_parts(value: &serde_json::Value) -> Vec<ContentPart> {
    match value {
        serde_json::Value::String(s) => vec![ContentPart::Text(s.clone())],
        serde_json::Value::Array(arr) => {
            let mut parts = Vec::new();
            for item in arr {
                let part_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match part_type {
                    "text" => {
                        if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                            parts.push(ContentPart::Text(text.to_string()));
                        }
                    }
                    "image_url" => {
                        if let Some(url_obj) = item.get("image_url") {
                            if let Some(url) = url_obj.get("url").and_then(|u| u.as_str()) {
                                if url.starts_with("data:") {
                                    // data:image/png;base64,....
                                    let media_type = url
                                        .split(';')
                                        .next()
                                        .unwrap_or("data:image/png")
                                        .strip_prefix("data:")
                                        .unwrap_or("image/png")
                                        .to_string();
                                    parts.push(ContentPart::ImageBase64 {
                                        data: url.to_string(),
                                        media_type,
                                    });
                                } else {
                                    parts.push(ContentPart::ImageUrl {
                                        url: url.to_string(),
                                    });
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            parts
        }
        _ => vec![],
    }
}

/// Extract vision configuration from GGUF metadata keys (LLaVA-style models)
pub fn vision_config_from_gguf(
    metadata: &std::collections::HashMap<String, String>,
    llm_hidden_size: usize,
) -> Option<VisionConfig> {
    // LLaVA GGUF models use keys like:
    // clip.vision.image_size, clip.vision.patch_size, etc.
    let image_size = metadata
        .get("clip.vision.image_size")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let patch_size = metadata
        .get("clip.vision.patch_size")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    if image_size == 0 || patch_size == 0 {
        return None;
    }

    let hidden_size = metadata
        .get("clip.vision.embedding_length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let num_heads = metadata
        .get("clip.vision.head_count")
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let num_layers = metadata
        .get("clip.vision.block_count")
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let intermediate_size = metadata
        .get("clip.vision.feed_forward_length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);

    let projection_type = metadata
        .get("clip.vision.projection_type")
        .map(|v| {
            if v == "mlp" {
                ProjectionType::Mlp
            } else {
                ProjectionType::Linear
            }
        })
        .unwrap_or(ProjectionType::Mlp);

    Some(VisionConfig {
        image_size,
        patch_size,
        hidden_size,
        num_heads,
        num_layers,
        intermediate_size,
        llm_hidden_size,
        projection_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_config_defaults() {
        let cfg = VisionConfig::clip_vit_l_336(4096);
        assert_eq!(cfg.image_size, 336);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.num_patches_per_side(), 24);
        assert_eq!(cfg.num_patches(), 576);
        assert_eq!(cfg.num_vision_tokens(), 577);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.llm_hidden_size, 4096);
    }

    #[test]
    fn test_preprocess_image_dimensions() {
        // 4x4 red image
        let mut pixels = vec![0u8; 4 * 4 * 3];
        for i in 0..(4 * 4) {
            pixels[i * 3] = 255; // R
            pixels[i * 3 + 1] = 0; // G
            pixels[i * 3 + 2] = 0; // B
        }

        let result = preprocess_image(&pixels, 4, 4, 4).unwrap();
        assert_eq!(result.pixels.len(), 3 * 4 * 4); // CHW format
        assert_eq!(result.original_size, (4, 4));
    }

    #[test]
    fn test_preprocess_normalization() {
        // Single white pixel (255, 255, 255)
        let pixels = vec![255u8; 3];
        let result = preprocess_image(&pixels, 1, 1, 1).unwrap();

        // After normalization: (1.0 - mean) / std
        let expected_r = (1.0 - CLIP_MEAN[0]) / CLIP_STD[0];
        let expected_g = (1.0 - CLIP_MEAN[1]) / CLIP_STD[1];
        let expected_b = (1.0 - CLIP_MEAN[2]) / CLIP_STD[2];

        assert!((result.pixels[0] - expected_r).abs() < 1e-4);
        assert!((result.pixels[1] - expected_g).abs() < 1e-4);
        assert!((result.pixels[2] - expected_b).abs() < 1e-4);
    }

    #[test]
    fn test_preprocess_invalid_size() {
        let pixels = vec![0u8; 10]; // Wrong size
        assert!(preprocess_image(&pixels, 4, 4, 4).is_err());
    }

    #[test]
    fn test_center_crop() {
        // 6x4 image, crop to 4x4
        let mut pixels = vec![0u8; 6 * 4 * 3];
        // Mark center pixel
        let cx = 3;
        let cy = 2;
        pixels[(cy * 6 + cx) * 3] = 200;

        let cropped = center_crop(&pixels, 6, 4, 4);
        assert_eq!(cropped.len(), 4 * 4 * 3);
        // Center pixel should be at (2, 2) in cropped image (offset x=1)
        assert_eq!(cropped[(cy * 4 + (cx - 1)) * 3], 200);
    }

    #[test]
    fn test_gelu() {
        // GELU(0) ≈ 0
        assert!((gelu(0.0)).abs() < 1e-6);
        // GELU(x) ≈ x for large positive x
        assert!((gelu(10.0) - 10.0).abs() < 0.01);
        // GELU is negative for negative inputs (small)
        assert!(gelu(-1.0) < 0.0);
    }

    #[test]
    fn test_layer_norm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];

        layer_norm(&mut x, &weight, &bias, 1e-5);

        // Mean should be ~0, var should be ~1
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);

        let var: f32 = x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_base64_decode() {
        let encoded = "SGVsbG8gV29ybGQ="; // "Hello World"
        let decoded = base64_decode(encoded).unwrap();
        assert_eq!(&decoded, b"Hello World");
    }

    #[test]
    fn test_base64_decode_no_padding() {
        let encoded = "SGVsbG8"; // "Hello" (no padding)
        let decoded = base64_decode(encoded).unwrap();
        assert_eq!(&decoded, b"Hello");
    }

    #[test]
    fn test_parse_content_parts_string() {
        let value = serde_json::json!("hello world");
        let parts = parse_content_parts(&value);
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            ContentPart::Text(t) => assert_eq!(t, "hello world"),
            _ => panic!("Expected text part"),
        }
    }

    #[test]
    fn test_parse_content_parts_array() {
        let value = serde_json::json!([
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        ]);
        let parts = parse_content_parts(&value);
        assert_eq!(parts.len(), 2);
        match &parts[0] {
            ContentPart::Text(t) => assert_eq!(t, "What's in this image?"),
            _ => panic!("Expected text part"),
        }
        match &parts[1] {
            ContentPart::ImageUrl { url } => assert_eq!(url, "https://example.com/img.png"),
            _ => panic!("Expected image URL part"),
        }
    }

    #[test]
    fn test_parse_content_parts_base64_image() {
        let value = serde_json::json!([
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
        ]);
        let parts = parse_content_parts(&value);
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            ContentPart::ImageBase64 { media_type, .. } => {
                assert_eq!(media_type, "image/png");
            }
            _ => panic!("Expected base64 image part"),
        }
    }

    #[test]
    fn test_vision_config_from_gguf_missing() {
        let metadata = std::collections::HashMap::new();
        assert!(vision_config_from_gguf(&metadata, 4096).is_none());
    }

    #[test]
    fn test_vision_config_from_gguf_present() {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("clip.vision.image_size".to_string(), "336".to_string());
        metadata.insert("clip.vision.patch_size".to_string(), "14".to_string());
        metadata.insert(
            "clip.vision.embedding_length".to_string(),
            "1024".to_string(),
        );
        metadata.insert("clip.vision.head_count".to_string(), "16".to_string());
        metadata.insert("clip.vision.block_count".to_string(), "24".to_string());

        let cfg = vision_config_from_gguf(&metadata, 4096).unwrap();
        assert_eq!(cfg.image_size, 336);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.llm_hidden_size, 4096);
    }

    #[test]
    fn test_resize_square() {
        // 2x2 image, resize to target_size=2 (no change)
        let pixels = vec![
            255, 0, 0, 0, 255, 0, // row 0
            0, 0, 255, 128, 128, 128, // row 1
        ];
        let (resized, w, h) = resize_bicubic(&pixels, 2, 2, 2);
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(resized.len(), 2 * 2 * 3);
    }

    #[test]
    fn test_small_vision_encoder() {
        // Tiny 2-patch encoder for testing (2x2 image, 2x2 patches = 1 patch)
        let config = VisionConfig {
            image_size: 2,
            patch_size: 2,
            hidden_size: 4,
            num_heads: 1,
            num_layers: 1,
            intermediate_size: 8,
            llm_hidden_size: 4,
            projection_type: ProjectionType::Linear,
        };

        let num_patches = config.num_patches(); // 1
        let num_tokens = num_patches + 1; // 2 (CLS + 1 patch)
        let h = config.hidden_size;
        let inter = config.intermediate_size;

        let weights = VisionWeights {
            patch_embed_weight: vec![0.1; h * 3 * 4], // [4, 3, 2, 2]
            patch_embed_bias: vec![0.0; h],
            cls_token: vec![0.1; h],
            position_embedding: vec![0.01; num_tokens * h],
            pre_ln_weight: vec![1.0; h],
            pre_ln_bias: vec![0.0; h],
            layers: vec![VisionLayerWeights {
                attn_ln_weight: vec![1.0; h],
                attn_ln_bias: vec![0.0; h],
                qkv_weight: vec![0.1; 3 * h * h],
                qkv_bias: vec![0.0; 3 * h],
                out_proj_weight: vec![0.1; h * h],
                out_proj_bias: vec![0.0; h],
                mlp_ln_weight: vec![1.0; h],
                mlp_ln_bias: vec![0.0; h],
                mlp_fc1_weight: vec![0.1; inter * h],
                mlp_fc1_bias: vec![0.0; inter],
                mlp_fc2_weight: vec![0.1; h * inter],
                mlp_fc2_bias: vec![0.0; h],
            }],
            post_ln_weight: vec![1.0; h],
            post_ln_bias: vec![0.0; h],
            projection: VisionProjection::Linear {
                weight: vec![0.1; h * h],
                bias: vec![0.0; h],
            },
        };

        let encoder = VisionEncoder::new(config, weights);

        // Create a tiny test image
        let image = ProcessedImage {
            pixels: vec![0.5; 3 * 2 * 2], // 2x2 image, CHW
            original_size: (2, 2),
        };

        let result = encoder.encode(&image).unwrap();
        assert_eq!(result.len(), 2); // CLS + 1 patch
        assert_eq!(result[0].len(), 4); // llm_hidden_size
    }

    #[test]
    fn test_projection_mlp() {
        let config = VisionConfig {
            image_size: 2,
            patch_size: 2,
            hidden_size: 4,
            num_heads: 1,
            num_layers: 0, // no transformer layers
            intermediate_size: 8,
            llm_hidden_size: 8,
            projection_type: ProjectionType::Mlp,
        };

        let num_tokens = config.num_vision_tokens();
        let h = config.hidden_size;
        let h_llm = config.llm_hidden_size;

        let weights = VisionWeights {
            patch_embed_weight: vec![0.1; h * 3 * 4],
            patch_embed_bias: vec![0.0; h],
            cls_token: vec![0.1; h],
            position_embedding: vec![0.01; num_tokens * h],
            pre_ln_weight: vec![1.0; h],
            pre_ln_bias: vec![0.0; h],
            layers: vec![],
            post_ln_weight: vec![1.0; h],
            post_ln_bias: vec![0.0; h],
            projection: VisionProjection::Mlp {
                fc1_weight: vec![0.1; h_llm * h],
                fc1_bias: vec![0.0; h_llm],
                fc2_weight: vec![0.1; h_llm * h_llm],
                fc2_bias: vec![0.0; h_llm],
            },
        };

        let encoder = VisionEncoder::new(config, weights);
        let image = ProcessedImage {
            pixels: vec![0.5; 3 * 2 * 2],
            original_size: (2, 2),
        };

        let result = encoder.encode(&image).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 8); // projected to llm_hidden_size=8
    }
}
