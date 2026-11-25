//! Pure-Rust software Metal call gate (`SoftBackend` + `SoftCompiler`).
//!
//! Mounts a `StelaGPU` call gate into the guest's DeviceTree when `soft-gpu`
//! is enabled without `nvk`, giving macOS `WindowServer` a software Metal
//! device that draws clears, copies, and blits directly into guest `IOSurface`
//! RAM (which Stela's DCP compositor presents onto QEMU's GOP framebuffer).

use super::super::host;
use crate::{layout, output};
use alloc::{format, sync::Arc, vec::Vec};
use core::{
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};
use stela_contracts::gate::{
    BundleReport, CallerFault, CallerMemory, GateService, MemoryRange, window,
};
use stela_drivers::gpu::metal::{
    device::Device,
    service::{Served, Service, ServiceHost, SharedService, identity::DeviceEpoch},
    software::{SoftBackend, SoftCompiler},
};
use stela_loader::{apple_device_tree::DeviceTree, stela_gpu};

const SOFT_VRAM_BYTES: u64 = 512 << 20;

static SERVED_COUNT: AtomicU64 = AtomicU64::new(0);

fn guest_runs_host(guest_pa: u64, bytes: u64, visit: &mut dyn FnMut(*mut u8, usize)) -> bool {
    host::guest_runs(guest_pa, bytes, |host_pa, len| {
        let ptr = layout::physical(host_pa) as *mut u8;
        visit(ptr, len as usize);
    })
}

fn relax() {
    core::hint::spin_loop();
}

fn relax_within() -> bool {
    core::hint::spin_loop();
    true
}

fn run(_stack: u64, work: &mut dyn FnMut()) {
    work();
}

fn report(message: &str) {
    output::arguments(format_args!("GPU soft refused: {message}\n"));
}

fn note(message: &str) {
    output::arguments(format_args!("GPU soft: {message}\n"));
}

fn served(info: Served) {
    let n = SERVED_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if let Some((submitted, completed)) = info.progress
        && (n <= 20 || n.is_multiple_of(200))
    {
        output::arguments(format_args!(
            "GPU soft served #{n}: op={} status={:?} submitted={submitted} completed={completed}\n",
            info.operation, info.status
        ));
    }
}

fn wake() {}

fn bundle(report: BundleReport) {
    output::arguments(format_args!("GPU soft bundle: {report:?}\n"));
}

fn job_waiting() {}

/// Wraps [`SharedService`] so synchronous software submissions immediately
/// notify the completion timeline and raise the `StelaGPU` gate interrupt.
struct SoftGate(Arc<SharedService<SoftBackend, SoftCompiler>>);

impl GateService for SoftGate {
    fn connect(&self) -> Result<NonZeroU64, &'static str> {
        self.0.connect()
    }

    fn disconnect(&self, connection: NonZeroU64) {
        self.0.disconnect(connection);
    }

    fn serve(
        &self,
        connection: NonZeroU64,
        record: u64,
        caller: &mut dyn CallerMemory,
    ) -> Result<(), CallerFault> {
        let result = self.0.serve(connection, record, caller);
        if self.0.awaits_completion() {
            self.0.interrupted();
        }
        result
    }

    fn import(
        &self,
        connection: NonZeroU64,
        ranges: &[MemoryRange],
        token: NonZeroU64,
    ) -> Result<NonZeroU64, &'static str> {
        self.0.import(connection, ranges, token)
    }

    fn retired(&self) -> Option<NonZeroU64> {
        self.0.retired()
    }

    fn quiesced(&self) -> bool {
        self.0.quiesced()
    }

    fn signaled(&self) -> bool {
        if self.0.awaits_completion() {
            self.0.interrupted();
        }
        self.0.signaled()
    }

    fn awaits_completion(&self) -> bool {
        self.0.awaits_completion()
    }

    fn bundle(&self, report: BundleReport) {
        self.0.bundle(report);
    }

    fn dropped(&self, reason: &'static str) {
        self.0.dropped(reason);
    }
}

/// Create the software Metal `SharedService` and place its `StelaGPU` gate
/// node in `tree`.
pub(in crate::engine) fn place(
    tree: &mut DeviceTree,
    mmio_window: Option<&'static crate::native_mmio::Window>,
) -> Vec<(stela_gpu::Placement, Arc<dyn GateService>)> {
    if let Some(gates) = super::vgpu_gate::try_place(tree, mmio_window) {
        return gates;
    }
    let backend = SoftBackend::new(guest_runs_host, SOFT_VRAM_BYTES).with_log(note);
    let platform = SoftBackend::platform();
    let device = match Device::new(backend, SoftCompiler::default(), &platform) {
        Ok(device) => device,
        Err(error) => {
            output::arguments(format_args!("GPU soft: Device::new failed: {error:?}\n"));
            return Vec::new();
        }
    };
    let service = Service::new(device, DeviceEpoch::from_registered(NonZeroU64::MIN));
    let shared = Arc::new(SharedService::new(
        service,
        ServiceHost {
            relax,
            relax_within,
            stack: 0,
            run,
            report,
            note,
            ticks: host::rdtsc,
            served,
            wake,
            bundle,
            job_waiting,
        },
    ));
    let gate: Arc<dyn GateService> = Arc::new(SoftGate(shared));
    match stela_gpu::place_gpu_gate(tree, 0, window::BYTES) {
        Ok(placement) => {
            output::arguments(format_args!(
                "GPU soft: StelaGPU software Metal gate placed at pa={:#x} irq={} (vram={} MiB)\n",
                placement.physical,
                placement.irq,
                SOFT_VRAM_BYTES >> 20
            ));
            alloc::vec![(placement, gate)]
        }
        Err(error) => {
            crate::status::warn(&format!("GPU soft gate has no place: {error}"));
            Vec::new()
        }
    }
}
