//! Shared-memory (`ivshmem-plain`) RPC protocol between the guest UEFI
//! `VgpuGate` and the host Linux `stela-vgpu-daemon`.
//!
//! Layout inside the 2 MiB `ivshmem-plain` BAR 2 aperture:
//! - `0x0000..0x1000`: [`VgpuShmHeader`] (atomic request/response + `CallerMemory`
//!   callback channel + lock-free `signaled`/`awaits` flags).
//! - `0x1000..0x200000`: payload buffer (up to ~2 MiB for `import` ranges and
//!   `CallerMemory::read`/`write` transfers).

use core::sync::atomic::{AtomicU32, AtomicU64};

/// `"STELVGPU"` in ASCII (little-endian).
pub const VGPU_MAGIC: u64 = 0x5550_4756_4C45_5453;
pub const VGPU_VERSION: u32 = 1;

/// PCI Vendor/Device ID for QEMU `ivshmem-plain`.
pub const IVSHMEM_VENDOR_ID: u16 = 0x1af4;
pub const IVSHMEM_DEVICE_ID: u16 = 0x1110;

/// Total size of the `ivshmem-plain` shared memory region (2 MiB).
pub const VGPU_SHM_SIZE: usize = 2 * 1024 * 1024;
/// Byte offset where the payload buffer begins.
pub const VGPU_PAYLOAD_OFFSET: usize = 4096;
/// Maximum payload bytes per transfer chunk.
pub const VGPU_PAYLOAD_MAX: usize = VGPU_SHM_SIZE - VGPU_PAYLOAD_OFFSET;

// Primary RPC operations (Guest -> Host)
pub const OP_CONNECT: u32 = 1;
pub const OP_DISCONNECT: u32 = 2;
pub const OP_SERVE: u32 = 3;
pub const OP_IMPORT: u32 = 4;
pub const OP_RETIRED: u32 = 5;
pub const OP_QUIESCED: u32 = 6;
pub const OP_BUNDLE: u32 = 7;
pub const OP_DROPPED: u32 = 8;

// Status codes
pub const STATUS_OK: u32 = 0;
pub const STATUS_ERR: u32 = 1;
pub const STATUS_FAULT_DENIED: u32 = 2;
pub const STATUS_FAULT_NOT_RAM: u32 = 3;
pub const STATUS_FAULT_UNSUPPORTED: u32 = 4;

// CallerMemory callback operations (Host -> Guest, nested inside OP_SERVE)
pub const CB_READ: u32 = 1;
pub const CB_WRITE: u32 = 2;

/// One physical memory run serialized in the payload for `OP_IMPORT`.
/// `host_pa` is the QEMU guest-physical (host-RAM file) offset of the run.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct WireRange {
    pub address: u64,
    pub bytes: u64,
}

/// Shared-memory header mapped at offset 0 of `ivshmem-plain` BAR 2.
#[repr(C, align(64))]
pub struct VgpuShmHeader {
    pub magic: AtomicU64,
    pub version: AtomicU32,
    pub host_ready: AtomicU32,

    // Primary RPC channel (Guest writes args & increments guest_seq;
    // Host executes and sets host_seq = guest_seq).
    pub guest_seq: AtomicU64,
    pub host_seq: AtomicU64,
    pub op: AtomicU32,
    pub status: AtomicU32,
    pub arg0: AtomicU64,
    pub arg1: AtomicU64,
    pub arg2: AtomicU64,
    pub arg3: AtomicU64,
    pub ret0: AtomicU64,
    pub ret1: AtomicU64,

    // Nested CallerMemory callback channel (active while OP_SERVE is in flight).
    pub cb_req: AtomicU64,
    pub cb_ack: AtomicU64,
    pub cb_op: AtomicU32,
    pub cb_status: AtomicU32,
    pub cb_addr: AtomicU64,
    pub cb_len: AtomicU64,

    // Lock-free completion flags updated by Host, read/swapped by Guest.
    pub signaled_flag: AtomicU32,
    pub awaits_flag: AtomicU32,
}
