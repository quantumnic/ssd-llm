//! Lazy mmap-based model loader with memory-mapped file access

use anyhow::{Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// A memory-mapped model file providing zero-copy access to tensor data
pub struct MmapLoader {
    mmap: Mmap,
    data_offset: u64,
}

impl MmapLoader {
    pub fn new(path: &Path, data_offset: u64) -> Result<Self> {
        let file = File::open(path).context("Failed to open model file")?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Advise the OS that we'll access this sequentially initially
        #[cfg(unix)]
        {
            unsafe {
                libc::madvise(
                    mmap.as_ptr() as *mut libc::c_void,
                    mmap.len(),
                    libc::MADV_SEQUENTIAL,
                );
            }
        }

        Ok(Self { mmap, data_offset })
    }

    /// Get a slice of tensor data at the given offset and size
    pub fn get_tensor_data(&self, offset: u64, size: u64) -> Result<&[u8]> {
        let abs_offset = self.data_offset + offset;
        let end = abs_offset + size;
        if end as usize > self.mmap.len() {
            anyhow::bail!(
                "Tensor data out of bounds: offset={}, size={}, file_len={}",
                abs_offset, size, self.mmap.len()
            );
        }
        Ok(&self.mmap[abs_offset as usize..end as usize])
    }

    /// Prefetch tensor data into memory using madvise WILLNEED
    pub fn prefetch(&self, offset: u64, size: u64) {
        let abs_offset = self.data_offset + offset;
        #[cfg(unix)]
        {
            let ptr = unsafe { self.mmap.as_ptr().add(abs_offset as usize) };
            unsafe {
                libc::madvise(
                    ptr as *mut libc::c_void,
                    size as usize,
                    libc::MADV_WILLNEED,
                );
            }
        }
    }

    /// Release (evict) tensor data from memory using madvise DONTNEED
    pub fn evict(&self, offset: u64, size: u64) {
        let abs_offset = self.data_offset + offset;
        #[cfg(unix)]
        {
            let ptr = unsafe { self.mmap.as_ptr().add(abs_offset as usize) };
            unsafe {
                libc::madvise(
                    ptr as *mut libc::c_void,
                    size as usize,
                    libc::MADV_DONTNEED,
                );
            }
        }
    }

    /// Total file size
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }
}
