//! Metal GPU dispatch via metal-rs
//!
//! v0.4: Real GPU compute pipeline using the metal crate.
//! Dispatches matmul, RMSNorm, softmax, RoPE, and SiLU to the GPU
//! for tensors above the dispatch threshold.

#[cfg(target_os = "macos")]
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize};

use tracing::{debug, info, warn};

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
    matvec_q8_0: ComputePipelineState,
    rmsnorm_sumsq: ComputePipelineState,
    rmsnorm_normalize: ComputePipelineState,
    softmax_exp: ComputePipelineState,
    softmax_normalize: ComputePipelineState,
    rope_f32: ComputePipelineState,
    silu_f32: ComputePipelineState,
    elementwise_mul: ComputePipelineState,
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
            let library = device.new_library_with_source(shader_src, &metal::CompileOptions::new())
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
            matvec_q8_0: make_pipeline("matvec_q8_0")?,
            rmsnorm_sumsq: make_pipeline("rmsnorm_sumsq")?,
            rmsnorm_normalize: make_pipeline("rmsnorm_normalize")?,
            softmax_exp: make_pipeline("softmax_exp")?,
            softmax_normalize: make_pipeline("softmax_normalize")?,
            rope_f32: make_pipeline("rope_f32")?,
            silu_f32: make_pipeline("silu_f32")?,
            elementwise_mul: make_pipeline("elementwise_mul_f32")?,
        })
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
            (self.pipelines.matvec_f32.max_total_threads_per_threadgroup() as u64).min(out_dim as u64),
            1, 1,
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
            (self.pipelines.rmsnorm_normalize.max_total_threads_per_threadgroup() as u64).min(n as u64),
            1, 1,
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
        if n == 0 { return; }

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
            (self.pipelines.softmax_normalize.max_total_threads_per_threadgroup() as u64).min(n as u64),
            1, 1,
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
    pub fn rope_f32(&self, x: &mut [f32], head_dim: usize, n_heads: usize, position: usize, theta_base: f32) {
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
            (self.pipelines.rope_f32.max_total_threads_per_threadgroup() as u64).min(total_pairs),
            1, 1,
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
            (self.pipelines.silu_f32.max_total_threads_per_threadgroup() as u64).min(n as u64),
            1, 1,
        );
        encoder.dispatch_threads(threads, tg);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let result = self.read_buffer::<f32>(&x_buf, n);
        x.copy_from_slice(&result);
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
        self.device.new_buffer(size, MTLResourceOptions::StorageModeShared)
    }

    #[cfg(target_os = "macos")]
    fn read_buffer<T: Copy>(&self, buffer: &Buffer, count: usize) -> Vec<T> {
        let ptr = buffer.contents() as *const T;
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        slice.to_vec()
    }
}
