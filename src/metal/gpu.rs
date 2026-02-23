//! Metal GPU dispatch via metal-rs
//!
//! v0.4: Real GPU compute pipeline using the metal crate.
//! Dispatches matmul, RMSNorm, softmax, RoPE, and SiLU to the GPU
//! for tensors above the dispatch threshold.

#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize,
};

use tracing::{info, warn};

/// Minimum elements to justify GPU dispatch overhead
const MIN_GPU_ELEMENTS: usize = 4096;

/// Metal GPU context holding device, command queue, and compiled pipelines
pub struct MetalGpu {
    #[cfg(target_os = "macos")]
    device: Device,
    #[cfg(target_os = "macos")]
    queue: CommandQueue,
    #[cfg(target_os = "macos")]
    pipelines: GpuPipelines,
    available: bool,
}

#[cfg(target_os = "macos")]
struct GpuPipelines {
    matvec_f32: ComputePipelineState,
    matvec_q4_0: ComputePipelineState,
    matvec_q3_k: ComputePipelineState,
    matvec_q4_k: ComputePipelineState,
    matvec_q5_k: ComputePipelineState,
    matvec_q6_k: ComputePipelineState,
    matvec_q2_k: ComputePipelineState,
    matvec_q8_k: ComputePipelineState,
    matvec_q8_0: ComputePipelineState,
    matvec_iq4_nl: ComputePipelineState,
    matvec_iq4_xs: ComputePipelineState,
    matvec_iq3_xxs: ComputePipelineState,
    matvec_iq3_s: ComputePipelineState,
    matvec_iq2_xxs: ComputePipelineState,
    matvec_iq2_xs: ComputePipelineState,
    matvec_bf16: ComputePipelineState,
    matvec_f16: ComputePipelineState,
    rmsnorm_sumsq: ComputePipelineState,
    rmsnorm_normalize: ComputePipelineState,
    softmax_exp: ComputePipelineState,
    softmax_normalize: ComputePipelineState,
    rope_f32: ComputePipelineState,
    silu_f32: ComputePipelineState,
    elementwise_mul: ComputePipelineState,
    flash_attention_f32: ComputePipelineState,
    fused_swiglu_f32: ComputePipelineState,
    fused_residual_rmsnorm_sumsq: ComputePipelineState,
}

unsafe impl Send for MetalGpu {}
unsafe impl Sync for MetalGpu {}

impl MetalGpu {
    /// Create a new Metal GPU context with compiled shader pipelines
    pub fn new() -> Option<Self> {
        #[cfg(target_os = "macos")]
        {
            let device = Device::system_default()?;
            info!("Metal GPU: {}", device.name());

            let queue = device.new_command_queue();

            // Compile shaders
            let shader_src = include_str!("shaders/kernels.metal");
            let library = device
                .new_library_with_source(shader_src, &metal::CompileOptions::new())
                .map_err(|e| {
                    warn!("Failed to compile Metal shaders: {}", e);
                    e
                })
                .ok()?;

            let pipelines = Self::build_pipelines(&device, &library)?;
            info!("Metal GPU pipelines compiled successfully");

            Some(Self {
                device,
                queue,
                pipelines,
                available: true,
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            warn!("Metal GPU not available on this platform");
            None
        }
    }

    #[cfg(target_os = "macos")]
    fn build_pipelines(device: &Device, library: &Library) -> Option<GpuPipelines> {
        let make_pipeline = |name: &str| -> Option<ComputePipelineState> {
            let func = library.get_function(name, None).ok()?;
            device.new_compute_pipeline_state_with_function(&func).ok()
        };

        Some(GpuPipelines {
            matvec_f32: make_pipeline("matvec_f32")?,
            matvec_q4_0: make_pipeline("matvec_q4_0")?,
            matvec_q3_k: make_pipeline("matvec_q3_k")?,
            matvec_q4_k: make_pipeline("matvec_q4_k")?,
            matvec_q5_k: make_pipeline("matvec_q5_k")?,
            matvec_q6_k: make_pipeline("matvec_q6_k")?,
            matvec_q8_0: make_pipeline("matvec_q8_0")?,
            matvec_q2_k: make_pipeline("matvec_q2_k")?,
            matvec_q8_k: make_pipeline("matvec_q8_k")?,
            matvec_iq4_nl: make_pipeline("matvec_iq4_nl")?,
            matvec_iq4_xs: make_pipeline("matvec_iq4_xs")?,
            matvec_iq3_xxs: make_pipeline("matvec_iq3_xxs")?,
            matvec_iq3_s: make_pipeline("matvec_iq3_s")?,
            matvec_iq2_xxs: make_pipeline("matvec_iq2_xxs")?,
            matvec_iq2_xs: make_pipeline("matvec_iq2_xs")?,
            matvec_bf16: make_pipeline("matvec_bf16")?,
            matvec_f16: make_pipeline("matvec_f16")?,
            rmsnorm_sumsq: make_pipeline("rmsnorm_sumsq")?,
            rmsnorm_normalize: make_pipeline("rmsnorm_normalize")?,
            softmax_exp: make_pipeline("softmax_exp")?,
            softmax_normalize: make_pipeline("softmax_normalize")?,
            rope_f32: make_pipeline("rope_f32")?,
            silu_f32: make_pipeline("silu_f32")?,
            elementwise_mul: make_pipeline("elementwise_mul_f32")?,
            flash_attention_f32: make_pipeline("flash_attention_f32")?,
            fused_swiglu_f32: make_pipeline("fused_swiglu_f32")?,
            fused_residual_rmsnorm_sumsq: make_pipeline("fused_residual_rmsnorm_sumsq")?,
        })
    }

    /// Whether Metal GPU is available
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Whether GPU dispatch is available and beneficial for this size
    pub fn should_dispatch(&self, elements: usize) -> bool {
        self.available && elements >= MIN_GPU_ELEMENTS
    }

    /// GPU matrix-vector multiply: y = W × x
    /// W: (out_dim × in_dim), x: (in_dim,), y: (out_dim,)
    #[cfg(target_os = "macos")]
    pub fn matvec_f32(&self, w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let w_buf = self.create_buffer_with_data(w);
        let x_buf = self.create_buffer_with_data(x);
        let y_buf = self.create_buffer::<f32>(out_dim);
        let out_dim_buf = self.create_buffer_with_data(&[out_dim as u32]);
        let in_dim_buf = self.create_buffer_with_data(&[in_dim as u32]);

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.matvec_f32);
        encoder.set_buffer(0, Some(&w_buf), 0);
        encoder.set_buffer(1, Some(&x_buf), 0);
        encoder.set_buffer(2, Some(&y_buf), 0);
        encoder.set_buffer(3, Some(&out_dim_buf), 0);
        encoder.set_buffer(4, Some(&in_dim_buf), 0);

        let thread_count = MTLSize::new(out_dim as u64, 1, 1);
        let threadgroup_size = MTLSize::new(
            (self
                .pipelines
                .matvec_f32
                .max_total_threads_per_threadgroup())
            .min(out_dim as u64),
            1,
            1,
        );
        encoder.dispatch_threads(thread_count, threadgroup_size);
        encoder.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        self.read_buffer::<f32>(&y_buf, out_dim)
    }

    /// GPU RMS normalization in-place
    #[cfg(target_os = "macos")]
    pub fn rmsnorm_f32(&self, x: &mut [f32], weight: &[f32], eps: f32) {
        let n = x.len();

        // Phase 1: compute sum of squares on GPU
        let x_buf = self.create_buffer_with_data(x);
        let n_threads = 256usize;
        let partial_buf = self.create_buffer::<f32>(n_threads);
        let n_buf = self.create_buffer_with_data(&[n as u32]);

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines.rmsnorm_sumsq);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&partial_buf), 0);
        encoder.set_buffer(2, Some(&n_buf), 0);
        let tg = MTLSize::new(n_threads as u64, 1, 1);
        encoder.dispatch_threads(tg, tg);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // CPU reduction of partial sums
        let partials = self.read_buffer::<f32>(&partial_buf, n_threads);
        let sum_sq: f32 = partials.iter().sum();
        let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

        // Phase 2: normalize on GPU
        let weight_buf = self.create_buffer_with_data(weight);
        let inv_rms_buf = self.create_buffer_with_data(&[inv_rms]);

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines.rmsnorm_normalize);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&weight_buf), 0);
        encoder.set_buffer(2, Some(&inv_rms_buf), 0);
        encoder.set_buffer(3, Some(&n_buf), 0);
        let threads = MTLSize::new(n as u64, 1, 1);
        let tg = MTLSize::new(
            (self
                .pipelines
                .rmsnorm_normalize
                .max_total_threads_per_threadgroup())
            .min(n as u64),
            1,
            1,
        );
        encoder.dispatch_threads(threads, tg);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let result = self.read_buffer::<f32>(&x_buf, n);
        x.copy_from_slice(&result);
    }

    /// GPU softmax in-place
    #[cfg(target_os = "macos")]
    pub fn softmax_f32(&self, x: &mut [f32]) {
        let n = x.len();
        if n == 0 {
            return;
        }

        // CPU: find max (small reduction)
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let x_buf = self.create_buffer_with_data(x);
        let max_buf = self.create_buffer_with_data(&[max_val]);
        let n_threads = 256usize.min(n);
        let partial_buf = self.create_buffer::<f32>(n_threads);
        let n_buf = self.create_buffer_with_data(&[n as u32]);

        // Phase 1: exp and partial sums
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines.softmax_exp);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&max_buf), 0);
        encoder.set_buffer(2, Some(&partial_buf), 0);
        encoder.set_buffer(3, Some(&n_buf), 0);
        let tg = MTLSize::new(n_threads as u64, 1, 1);
        encoder.dispatch_threads(tg, tg);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // CPU reduction
        let partials = self.read_buffer::<f32>(&partial_buf, n_threads);
        let sum: f32 = partials.iter().sum();
        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };

        // Phase 2: normalize
        let inv_sum_buf = self.create_buffer_with_data(&[inv_sum]);
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines.softmax_normalize);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&inv_sum_buf), 0);
        encoder.set_buffer(2, Some(&n_buf), 0);
        let threads = MTLSize::new(n as u64, 1, 1);
        let tg = MTLSize::new(
            (self
                .pipelines
                .softmax_normalize
                .max_total_threads_per_threadgroup())
            .min(n as u64),
            1,
            1,
        );
        encoder.dispatch_threads(threads, tg);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let result = self.read_buffer::<f32>(&x_buf, n);
        x.copy_from_slice(&result);
    }

    /// GPU RoPE in-place
    #[cfg(target_os = "macos")]
    pub fn rope_f32(
        &self,
        x: &mut [f32],
        head_dim: usize,
        n_heads: usize,
        position: usize,
        theta_base: f32,
    ) {
        let x_buf = self.create_buffer_with_data(x);
        let head_dim_buf = self.create_buffer_with_data(&[head_dim as u32]);
        let n_heads_buf = self.create_buffer_with_data(&[n_heads as u32]);
        let pos_buf = self.create_buffer_with_data(&[position as u32]);
        let theta_buf = self.create_buffer_with_data(&[theta_base]);

        let total_pairs = (n_heads * head_dim / 2) as u64;

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines.rope_f32);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&head_dim_buf), 0);
        encoder.set_buffer(2, Some(&n_heads_buf), 0);
        encoder.set_buffer(3, Some(&pos_buf), 0);
        encoder.set_buffer(4, Some(&theta_buf), 0);
        let threads = MTLSize::new(total_pairs, 1, 1);
        let tg = MTLSize::new(
            (self.pipelines.rope_f32.max_total_threads_per_threadgroup()).min(total_pairs),
            1,
            1,
        );
        encoder.dispatch_threads(threads, tg);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let result = self.read_buffer::<f32>(&x_buf, x.len());
        x.copy_from_slice(&result);
    }

    /// GPU SiLU in-place
    #[cfg(target_os = "macos")]
    pub fn silu_f32(&self, x: &mut [f32]) {
        let n = x.len();
        let x_buf = self.create_buffer_with_data(x);
        let n_buf = self.create_buffer_with_data(&[n as u32]);

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines.silu_f32);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&n_buf), 0);
        let threads = MTLSize::new(n as u64, 1, 1);
        let tg = MTLSize::new(
            (self.pipelines.silu_f32.max_total_threads_per_threadgroup()).min(n as u64),
            1,
            1,
        );
        encoder.dispatch_threads(threads, tg);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let result = self.read_buffer::<f32>(&x_buf, n);
        x.copy_from_slice(&result);
    }

    /// GPU quantized matvec dispatch for Q4_0
    #[cfg(target_os = "macos")]
    pub fn matvec_q4_0(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_q4_0, w, x, out_dim, in_dim)
    }

    /// GPU quantized matvec dispatch for Q3_K
    #[cfg(target_os = "macos")]
    pub fn matvec_q3_k(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_q3_k, w, x, out_dim, in_dim)
    }

    /// GPU quantized matvec dispatch for Q4_K
    #[cfg(target_os = "macos")]
    pub fn matvec_q4_k(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_q4_k, w, x, out_dim, in_dim)
    }

    /// GPU quantized matvec dispatch for Q5_K
    #[cfg(target_os = "macos")]
    pub fn matvec_q5_k(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_q5_k, w, x, out_dim, in_dim)
    }

    /// GPU quantized matvec dispatch for Q6_K
    #[cfg(target_os = "macos")]
    pub fn matvec_q6_k(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_q6_k, w, x, out_dim, in_dim)
    }

    /// GPU quantized matvec dispatch for Q8_0
    #[cfg(target_os = "macos")]
    pub fn matvec_q8_0(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_q8_0, w, x, out_dim, in_dim)
    }

    /// GPU quantized matvec dispatch for Q2_K
    #[cfg(target_os = "macos")]
    pub fn matvec_q2_k(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_q2_k, w, x, out_dim, in_dim)
    }

    /// GPU quantized matvec dispatch for Q8_K
    #[cfg(target_os = "macos")]
    pub fn matvec_q8_k(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_q8_k, w, x, out_dim, in_dim)
    }

    /// GPU quantized matvec dispatch for IQ4_NL
    #[cfg(target_os = "macos")]
    pub fn matvec_iq4_nl(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_iq4_nl, w, x, out_dim, in_dim)
    }

    /// GPU quantized matvec dispatch for IQ4_XS
    #[cfg(target_os = "macos")]
    pub fn matvec_iq4_xs(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_iq4_xs, w, x, out_dim, in_dim)
    }

    /// GPU quantized matvec dispatch for IQ3_XXS
    #[cfg(target_os = "macos")]
    pub fn matvec_iq3_xxs(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_iq3_xxs, w, x, out_dim, in_dim)
    }

    /// GPU quantized matvec dispatch for IQ3_S
    #[cfg(target_os = "macos")]
    pub fn matvec_iq3_s(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_iq3_s, w, x, out_dim, in_dim)
    }

    /// GPU quantized matvec dispatch for IQ2_XXS
    #[cfg(target_os = "macos")]
    pub fn matvec_iq2_xxs(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_iq2_xxs, w, x, out_dim, in_dim)
    }

    /// GPU quantized matvec dispatch for IQ2_XS
    #[cfg(target_os = "macos")]
    pub fn matvec_iq2_xs(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_iq2_xs, w, x, out_dim, in_dim)
    }

    /// GPU matvec dispatch for BF16 (brain float 16)
    #[cfg(target_os = "macos")]
    pub fn matvec_bf16(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_bf16, w, x, out_dim, in_dim)
    }

    /// GPU matvec dispatch for F16 (IEEE 754 half)
    #[cfg(target_os = "macos")]
    pub fn matvec_f16(&self, w: &[u8], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.dispatch_quantized_matvec(&self.pipelines.matvec_f16, w, x, out_dim, in_dim)
    }

    /// Generic dispatch for quantized matvec kernels
    #[cfg(target_os = "macos")]
    fn dispatch_quantized_matvec(
        &self,
        pipeline: &ComputePipelineState,
        w: &[u8],
        x: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> Vec<f32> {
        let w_buf = self.device.new_buffer_with_data(
            w.as_ptr() as *const std::ffi::c_void,
            w.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let x_buf = self.create_buffer_with_data(x);
        let y_buf = self.create_buffer::<f32>(out_dim);
        let out_dim_buf = self.create_buffer_with_data(&[out_dim as u32]);
        let in_dim_buf = self.create_buffer_with_data(&[in_dim as u32]);

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&w_buf), 0);
        encoder.set_buffer(1, Some(&x_buf), 0);
        encoder.set_buffer(2, Some(&y_buf), 0);
        encoder.set_buffer(3, Some(&out_dim_buf), 0);
        encoder.set_buffer(4, Some(&in_dim_buf), 0);

        let thread_count = MTLSize::new(out_dim as u64, 1, 1);
        let threadgroup_size = MTLSize::new(
            pipeline
                .max_total_threads_per_threadgroup()
                .min(out_dim as u64),
            1,
            1,
        );
        encoder.dispatch_threads(thread_count, threadgroup_size);
        encoder.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        self.read_buffer::<f32>(&y_buf, out_dim)
    }

    /// GPU Flash Attention: fused QK scoring + online softmax + V accumulation
    /// Returns attention output [n_head × head_dim] for all heads in a single dispatch.
    ///
    /// Arguments:
    ///   q_heads:  [n_head, head_dim] — pre-projected, RoPE'd query vectors
    ///   k_cache:  [seq_len, n_head_kv, head_dim] — flattened key cache
    ///   v_cache:  [seq_len, n_head_kv, head_dim] — flattened value cache
    ///   n_head, n_head_kv, head_dim, seq_len: dimensions
    ///   window_start, window_end: attention window bounds
    #[cfg(target_os = "macos")]
    pub fn flash_attention_f32(
        &self,
        q_heads: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        seq_len: usize,
        window_start: usize,
        window_end: usize,
    ) -> Vec<f32> {
        let q_buf = self.create_buffer_with_data(q_heads);
        let k_buf = self.create_buffer_with_data(k_cache);
        let v_buf = self.create_buffer_with_data(v_cache);
        let out_buf = self.create_buffer::<f32>(n_head * head_dim);
        let params: [u32; 6] = [
            n_head as u32,
            n_head_kv as u32,
            head_dim as u32,
            seq_len as u32,
            window_start as u32,
            window_end as u32,
        ];
        let params_buf = self.create_buffer_with_data(&params);

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.flash_attention_f32);
        encoder.set_buffer(0, Some(&q_buf), 0);
        encoder.set_buffer(1, Some(&k_buf), 0);
        encoder.set_buffer(2, Some(&v_buf), 0);
        encoder.set_buffer(3, Some(&out_buf), 0);
        encoder.set_buffer(4, Some(&params_buf), 0);

        let thread_count = MTLSize::new(n_head as u64, 1, 1);
        let threadgroup_size = MTLSize::new(
            self.pipelines
                .flash_attention_f32
                .max_total_threads_per_threadgroup()
                .min(n_head as u64),
            1,
            1,
        );
        encoder.dispatch_threads(thread_count, threadgroup_size);
        encoder.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        self.read_buffer::<f32>(&out_buf, n_head * head_dim)
    }

    // --- Buffer helpers ---

    #[cfg(target_os = "macos")]
    fn create_buffer_with_data<T: Copy>(&self, data: &[T]) -> Buffer {
        let size = std::mem::size_of_val(data) as u64;
        self.device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            size,
            MTLResourceOptions::StorageModeShared,
        )
    }

    #[cfg(target_os = "macos")]
    fn create_buffer<T>(&self, count: usize) -> Buffer {
        let size = (count * std::mem::size_of::<T>()) as u64;
        self.device
            .new_buffer(size, MTLResourceOptions::StorageModeShared)
    }

    /// Fused SwiGLU feed-forward on GPU: gate_proj + silu + up_proj + mul + down_proj
    /// Returns the FFN output vector of size n_embd.
    #[cfg(target_os = "macos")]
    pub fn feed_forward_f32(
        &self,
        x: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        n_embd: usize,
    ) -> Vec<f32> {
        let n_ff = w_gate.len() / n_embd;
        if n_ff == 0 {
            return vec![0.0f32; n_embd];
        }

        // Phase 1: Fused SwiGLU — gate + silu + up + mul in one dispatch
        let w_gate_buf = self.create_buffer_with_data(w_gate);
        let w_up_buf = self.create_buffer_with_data(w_up);
        let x_buf = self.create_buffer_with_data(x);
        let intermediate_buf = self.create_buffer::<f32>(n_ff);
        let n_ff_buf = self.create_buffer_with_data(&[n_ff as u32]);
        let n_embd_buf = self.create_buffer_with_data(&[n_embd as u32]);

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines.fused_swiglu_f32);
        encoder.set_buffer(0, Some(&w_gate_buf), 0);
        encoder.set_buffer(1, Some(&w_up_buf), 0);
        encoder.set_buffer(2, Some(&x_buf), 0);
        encoder.set_buffer(3, Some(&intermediate_buf), 0);
        encoder.set_buffer(4, Some(&n_ff_buf), 0);
        encoder.set_buffer(5, Some(&n_embd_buf), 0);

        let threads = MTLSize::new(n_ff as u64, 1, 1);
        let tg = MTLSize::new(
            (self
                .pipelines
                .fused_swiglu_f32
                .max_total_threads_per_threadgroup())
            .min(n_ff as u64),
            1,
            1,
        );
        encoder.dispatch_threads(threads, tg);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Phase 2: down_proj matvec — reuse existing GPU matvec
        let intermediate = self.read_buffer::<f32>(&intermediate_buf, n_ff);
        self.matvec_f32(w_down, &intermediate, n_embd, n_ff)
    }

    /// Fused residual add + RMS normalization in-place on GPU.
    /// Computes: x[i] += residual[i], then x[i] = (x[i] / rms) * weight[i]
    /// Saves one full memory pass compared to separate residual add + rmsnorm.
    #[cfg(target_os = "macos")]
    pub fn fused_residual_rmsnorm_f32(
        &self,
        x: &mut [f32],
        residual: &[f32],
        weight: &[f32],
        eps: f32,
    ) {
        let n = x.len();
        debug_assert_eq!(n, residual.len());
        debug_assert_eq!(n, weight.len());

        // Phase 1: fused residual add + sum of squares
        let x_buf = self.create_buffer_with_data(x);
        let res_buf = self.create_buffer_with_data(residual);
        let n_threads = 256usize;
        let partial_buf = self.create_buffer::<f32>(n_threads);
        let n_buf = self.create_buffer_with_data(&[n as u32]);

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines.fused_residual_rmsnorm_sumsq);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&res_buf), 0);
        encoder.set_buffer(2, Some(&partial_buf), 0);
        encoder.set_buffer(3, Some(&n_buf), 0);
        let tg = MTLSize::new(n_threads as u64, 1, 1);
        encoder.dispatch_threads(tg, tg);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // CPU reduction of partial sums
        let partials = self.read_buffer::<f32>(&partial_buf, n_threads);
        let sum_sq: f32 = partials.iter().sum();
        let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

        // Phase 2: normalize (reuse existing rmsnorm_normalize pipeline)
        let weight_buf = self.create_buffer_with_data(weight);
        let inv_rms_buf = self.create_buffer_with_data(&[inv_rms]);

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines.rmsnorm_normalize);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&weight_buf), 0);
        encoder.set_buffer(2, Some(&inv_rms_buf), 0);
        encoder.set_buffer(3, Some(&n_buf), 0);
        let threads = MTLSize::new(n as u64, 1, 1);
        let tg = MTLSize::new(
            (self
                .pipelines
                .rmsnorm_normalize
                .max_total_threads_per_threadgroup())
            .min(n as u64),
            1,
            1,
        );
        encoder.dispatch_threads(threads, tg);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let result = self.read_buffer::<f32>(&x_buf, n);
        x.copy_from_slice(&result);
    }

    #[cfg(target_os = "macos")]
    fn read_buffer<T: Copy>(&self, buffer: &Buffer, count: usize) -> Vec<T> {
        let ptr = buffer.contents() as *const T;
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        slice.to_vec()
    }
}
