//! Guest-side UEFI `GateService` proxy over QEMU `ivshmem-plain` shared memory.
//!
//! When QEMU is launched with `-device ivshmem-plain` and the host Linux
//! `stela-vgpu-daemon` is running, this gate forwards all `StelaGPU` Metal
//! calls to the host daemon (which runs Stela's real `Service<NativeBackend,
//! AirCompiler>` over host Vulkan / Lavapipe with zero-copy access to guest
//! RAM).

use super::super::host;
use crate::{native_mmio::Window, output};
use alloc::{format, sync::Arc, vec::Vec};
use core::{
    arch::asm,
    num::NonZeroU64,
    ptr,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};
use stela_boot_core::page_window::Cache;
use stela_contracts::{
    gate::{BundleReport, CallerFault, CallerMemory, GateService, MemoryRange, window},
    vgpu::*,
};
use stela_loader::{apple_device_tree::DeviceTree, stela_gpu};

const PAGE: u64 = 4096;

unsafe fn pci_read_u32(bus: u8, dev: u8, func: u8, offset: u8) -> u32 {
    let addr: u32 = 0x8000_0000
        | (u32::from(bus) << 16)
        | (u32::from(dev) << 11)
        | (u32::from(func) << 8)
        | u32::from(offset & 0xFC);
    let val: u32;
    // SAFETY: Standard x86 PCI Configuration Space I/O ports 0xCF8/0xCFC.
    unsafe {
        asm!(
            "out dx, eax",
            in("dx") 0x0CF8u16,
            in("eax") addr,
            options(nomem, nostack, preserves_flags)
        );
        asm!(
            "in eax, dx",
            in("dx") 0x0CFCu16,
            out("eax") val,
            options(nomem, nostack, preserves_flags)
        );
    }
    val
}

unsafe fn pci_write_u32(bus: u8, dev: u8, func: u8, offset: u8, value: u32) {
    let addr: u32 = 0x8000_0000
        | (u32::from(bus) << 16)
        | (u32::from(dev) << 11)
        | (u32::from(func) << 8)
        | u32::from(offset & 0xFC);
    // SAFETY: Standard x86 PCI Configuration Space I/O ports 0xCF8/0xCFC.
    unsafe {
        asm!(
            "out dx, eax",
            in("dx") 0x0CF8u16,
            in("eax") addr,
            options(nomem, nostack, preserves_flags)
        );
        asm!(
            "out dx, eax",
            in("dx") 0x0CFCu16,
            in("eax") value,
            options(nomem, nostack, preserves_flags)
        );
    }
}

/// Scans PCI bus 0 for QEMU `ivshmem-plain` (`1af4:1110`) and returns BAR 2 physical address.
fn find_ivshmem_bar2() -> Option<u64> {
    for dev in 0..32u8 {
        let id = unsafe { pci_read_u32(0, dev, 0, 0x00) };
        if id == 0xFFFF_FFFF {
            continue;
        }
        let vendor = (id & 0xFFFF) as u16;
        let device = (id >> 16) as u16;
        if vendor == IVSHMEM_VENDOR_ID && device == IVSHMEM_DEVICE_ID {
            // Ensure Memory Space Enable (bit 1) is set in PCI Command register.
            let cmd = unsafe { pci_read_u32(0, dev, 0, 0x04) };
            if (cmd & 0x2) == 0 {
                unsafe { pci_write_u32(0, dev, 0, 0x04, cmd | 0x2) };
            }
            let bar2_lo = unsafe { pci_read_u32(0, dev, 0, 0x18) };
            if (bar2_lo & 0x1) != 0 {
                continue; // I/O BAR, unexpected
            }
            let is_64 = (bar2_lo & 0x6) == 0x4;
            let bar2_hi = if is_64 {
                unsafe { pci_read_u32(0, dev, 0, 0x1C) }
            } else {
                0
            };
            let pa = ((u64::from(bar2_hi)) << 32) | u64::from(bar2_lo & !0xF);
            if pa != 0 {
                return Some(pa);
            }
        }
    }
    None
}

struct VgpuGate {
    base: usize,
    rpc_lock: AtomicBool,
    seq: AtomicU64,
}

// SAFETY: `base` points to the permanently mapped 2 MiB ivshmem BAR 2 region,
// and all RPC transactions are serialized by `rpc_lock`.
unsafe impl Send for VgpuGate {}
unsafe impl Sync for VgpuGate {}

impl VgpuGate {
    #[inline]
    fn header(&self) -> &VgpuShmHeader {
        // SAFETY: `base` is page-aligned and maps `VGPU_SHM_SIZE` bytes.
        unsafe { &*(self.base as *const VgpuShmHeader) }
    }

    #[inline]
    fn payload_ptr(&self) -> *mut u8 {
        (self.base + VGPU_PAYLOAD_OFFSET) as *mut u8
    }

    fn lock_rpc(&self) {
        while self
            .rpc_lock
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
    }

    fn unlock_rpc(&self) {
        self.rpc_lock.store(false, Ordering::Release);
    }

    fn call_simple(&self, op: u32, arg0: u64, arg1: u64, arg2: u64, arg3: u64) -> (u32, u64, u64) {
        self.lock_rpc();
        let h = self.header();
        h.op.store(op, Ordering::Relaxed);
        h.arg0.store(arg0, Ordering::Relaxed);
        h.arg1.store(arg1, Ordering::Relaxed);
        h.arg2.store(arg2, Ordering::Relaxed);
        h.arg3.store(arg3, Ordering::Relaxed);
        let next = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        h.guest_seq.store(next, Ordering::Release);

        while h.host_seq.load(Ordering::Acquire) != next {
            core::hint::spin_loop();
        }
        let status = h.status.load(Ordering::Relaxed);
        let ret0 = h.ret0.load(Ordering::Relaxed);
        let ret1 = h.ret1.load(Ordering::Relaxed);
        self.unlock_rpc();
        (status, ret0, ret1)
    }
}

impl GateService for VgpuGate {
    fn connect(&self) -> Result<NonZeroU64, &'static str> {
        let (status, ret0, _) = self.call_simple(OP_CONNECT, 0, 0, 0, 0);
        if status == STATUS_OK {
            NonZeroU64::new(ret0).ok_or("vgpu connect returned zero handle")
        } else {
            Err("vgpu connect refused by host daemon")
        }
    }

    fn disconnect(&self, connection: NonZeroU64) {
        let _ = self.call_simple(OP_DISCONNECT, connection.get(), 0, 0, 0);
    }

    fn serve(
        &self,
        connection: NonZeroU64,
        record: u64,
        caller: &mut dyn CallerMemory,
    ) -> Result<(), CallerFault> {
        self.lock_rpc();
        let h = self.header();
        h.op.store(OP_SERVE, Ordering::Relaxed);
        h.arg0.store(connection.get(), Ordering::Relaxed);
        h.arg1.store(record, Ordering::Relaxed);
        h.arg2.store(0, Ordering::Relaxed);
        h.arg3.store(0, Ordering::Relaxed);
        let next = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        h.guest_seq.store(next, Ordering::Release);

        let mut last_cb = h.cb_ack.load(Ordering::Relaxed);
        while h.host_seq.load(Ordering::Acquire) != next {
            let req = h.cb_req.load(Ordering::Acquire);
            if req != last_cb {
                let cb_op = h.cb_op.load(Ordering::Relaxed);
                let addr = h.cb_addr.load(Ordering::Relaxed);
                let len = (h.cb_len.load(Ordering::Relaxed) as usize).min(VGPU_PAYLOAD_MAX);
                let buf = unsafe { core::slice::from_raw_parts_mut(self.payload_ptr(), len) };
                let res = match cb_op {
                    CB_READ => caller.read(addr, buf),
                    CB_WRITE => caller.write(addr, buf),
                    _ => Err(CallerFault::Unsupported("unknown vgpu callback op")),
                };
                let cb_status = match res {
                    Ok(()) => STATUS_OK,
                    Err(CallerFault::Denied { .. }) => STATUS_FAULT_DENIED,
                    Err(CallerFault::NotRam { .. }) => STATUS_FAULT_NOT_RAM,
                    Err(CallerFault::Unsupported(_)) => STATUS_FAULT_UNSUPPORTED,
                };
                h.cb_status.store(cb_status, Ordering::Relaxed);
                last_cb = req;
                h.cb_ack.store(req, Ordering::Release);
            } else {
                core::hint::spin_loop();
            }
        }

        let status = h.status.load(Ordering::Relaxed);
        let ret0 = h.ret0.load(Ordering::Relaxed);
        self.unlock_rpc();

        match status {
            STATUS_OK => Ok(()),
            STATUS_FAULT_DENIED => Err(CallerFault::Denied { address: ret0 }),
            STATUS_FAULT_NOT_RAM => Err(CallerFault::NotRam { address: ret0 }),
            _ => Err(CallerFault::Unsupported("vgpu host serve fault")),
        }
    }

    fn import(
        &self,
        connection: NonZeroU64,
        ranges: &[MemoryRange],
        token: NonZeroU64,
    ) -> Result<NonZeroU64, &'static str> {
        let mut wire_runs: Vec<WireRange> = Vec::with_capacity(ranges.len());
        for r in ranges {
            let ok = host::guest_runs(r.address, r.bytes, |host_pa, len| {
                wire_runs.push(WireRange {
                    address: host_pa,
                    bytes: len,
                });
            });
            if !ok {
                return Err("vgpu import: guest range not in host RAM");
            }
        }
        let max_runs = VGPU_PAYLOAD_MAX / core::mem::size_of::<WireRange>();
        if wire_runs.len() > max_runs {
            return Err("vgpu import: too many physical runs");
        }

        self.lock_rpc();
        let h = self.header();
        unsafe {
            ptr::copy_nonoverlapping(
                wire_runs.as_ptr() as *const u8,
                self.payload_ptr(),
                wire_runs.len() * core::mem::size_of::<WireRange>(),
            );
        }
        h.op.store(OP_IMPORT, Ordering::Relaxed);
        h.arg0.store(connection.get(), Ordering::Relaxed);
        h.arg1.store(token.get(), Ordering::Relaxed);
        h.arg2.store(wire_runs.len() as u64, Ordering::Relaxed);
        h.arg3.store(0, Ordering::Relaxed);
        let next = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        h.guest_seq.store(next, Ordering::Release);

        while h.host_seq.load(Ordering::Acquire) != next {
            core::hint::spin_loop();
        }
        let status = h.status.load(Ordering::Relaxed);
        let ret0 = h.ret0.load(Ordering::Relaxed);
        self.unlock_rpc();

        if status == STATUS_OK {
            NonZeroU64::new(ret0).ok_or("vgpu import returned zero handle")
        } else {
            Err("vgpu import refused by host daemon")
        }
    }

    fn retired(&self) -> Option<NonZeroU64> {
        let (status, ret0, _) = self.call_simple(OP_RETIRED, 0, 0, 0, 0);
        if status == STATUS_OK {
            NonZeroU64::new(ret0)
        } else {
            None
        }
    }

    fn quiesced(&self) -> bool {
        let (status, ret0, _) = self.call_simple(OP_QUIESCED, 0, 0, 0, 0);
        status == STATUS_OK && ret0 != 0
    }

    fn signaled(&self) -> bool {
        self.header().signaled_flag.swap(0, Ordering::AcqRel) != 0
    }

    fn awaits_completion(&self) -> bool {
        self.header().awaits_flag.load(Ordering::Acquire) != 0
    }

    fn bundle(&self, report: BundleReport) {
        let raw = match report {
            BundleReport::Current => 0u64,
            BundleReport::Updated => 1u64,
            BundleReport::Missing => 2u64,
            BundleReport::Failed(_) => 3u64,
        };
        let _ = self.call_simple(OP_BUNDLE, raw, 0, 0, 0);
    }

    fn dropped(&self, _reason: &'static str) {
        let _ = self.call_simple(OP_DROPPED, 0, 0, 0, 0);
    }
}

/// Attempts to attach to the host `stela-vgpu-daemon` via QEMU `ivshmem-plain`.
/// Returns `None` if no `ivshmem-plain` device or live daemon is present so
/// the caller can fall back to `SoftGate`.
pub(super) fn try_place(
    tree: &mut DeviceTree,
    window: Option<&'static Window>,
) -> Option<Vec<(stela_gpu::Placement, Arc<dyn GateService>)>> {
    let window = window?;
    let bar2_pa = find_ivshmem_bar2()?;
    let pages = VGPU_SHM_SIZE / (PAGE as usize);
    let va = window.reserve(pages)?;
    for p in 0..pages as u64 {
        if !window.install(va + p * PAGE, bar2_pa + p * PAGE, Cache::WriteBack) {
            if window.remove(va, pages) {
                window.release(va, pages);
            }
            return None;
        }
    }
    let header = unsafe { &*(va as *const VgpuShmHeader) };
    let magic = header.magic.load(Ordering::Acquire);
    let ready = header.host_ready.load(Ordering::Acquire);
    if magic != VGPU_MAGIC || ready == 0 {
        output::arguments(format_args!(
            "GPU vgpu: ivshmem BAR2 at pa={bar2_pa:#x} found, but host daemon not ready (magic={magic:#x} ready={ready}); falling back to SoftGPU\n"
        ));
        return None;
    }

    let initial_seq = header.guest_seq.load(Ordering::Acquire);
    let gate: Arc<dyn GateService> = Arc::new(VgpuGate {
        base: va as usize,
        rpc_lock: AtomicBool::new(false),
        seq: AtomicU64::new(initial_seq),
    });

    match stela_gpu::place_gpu_gate(tree, 0, window::BYTES) {
        Ok(placement) => {
            output::arguments(format_args!(
                "GPU vgpu: connected to host daemon via ivshmem BAR2 pa={bar2_pa:#x} va={va:#x}; StelaGPU gate placed at pa={:#x} irq={}\n",
                placement.physical, placement.irq
            ));
            Some(alloc::vec![(placement, gate)])
        }
        Err(error) => {
            crate::status::warn(&format!("GPU vgpu gate has no place: {error}"));
            None
        }
    }
}
