//! Host-side Linux vGPU daemon for Stela (`stela-vgpu-daemon`).
//!
//! Runs Stela's real `Service<NativeBackend, AirCompiler>` over Mesa 26.2.5
//! Lavapipe (Vulkan 1.4.354 + LLVM 21) with zero-copy `VK_EXT_external_memory_host`
//! access directly into QEMU's shared guest RAM (`/dev/shm/stela-guest-ram`),
//! serving the guest UEFI `VgpuGate` over `ivshmem-plain` (`/dev/shm/stela-vgpu-shm`).

use std::{
    ffi::{CString, c_char, c_void},
    fs::OpenOptions,
    num::NonZeroU64,
    os::fd::{AsRawFd, RawFd},
    ptr::{self, NonNull},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use stela_contracts::{
    gate::{BundleReport, CallerFault, CallerMemory, GateService, MemoryRange},
    vgpu::*,
};
use stela_driver_gpu::metal::{
    compiler::{self, AirCompiler, NIR_STAGES, ffi},
    device::Device,
    error::Error as MetalError,
    native::{ExportedMemory, Loader, MemoryExporter, NativeBackend, vk},
    open,
    service::{Served, Service, ServiceHost, SharedService, identity::DeviceEpoch},
    software::SoftBackend,
    vulkan::{backend::Backend, physical::PciBusInfo},
};

type VisitFunction = unsafe extern "C" fn(*mut c_void, *const c_char, usize, u32) -> bool;
type VisitBytes = unsafe extern "C" fn(*mut c_void, *const c_void, usize);

const SHM_PATH: &str = "/dev/shm/stela-vgpu-shm";
const DEFAULT_RAM_PATH: &str = "/home/lucas/.gemini/antigravity/scratch/stela-mod/.stela-guest-ram";
const GUEST_RAM_SIZE: usize = 4 * 1024 * 1024 * 1024;

unsafe extern "system" {
    fn vk_icdGetInstanceProcAddr(
        instance: vk::Instance,
        name: *const c_char,
    ) -> vk::PFN_vkVoidFunction;
}

unsafe extern "C" {
    fn stela_lvp_nir_options(vk_physical_device: *mut c_void, out: *mut *const c_void);
    fn stela_generic_nir_options(out: *mut *const c_void);
    fn stela_metal_compile_cached(
        cache_dir: *const c_char,
        air: *const c_void,
        air_size: usize,
        entry: *const c_char,
        target: *const ffi::Target,
        spec: *const ffi::Specialization,
        reflected: *const ffi::Reflection,
        error: *mut ffi::AirError,
        out_hit: *mut libc::c_int,
    ) -> *mut c_void;

    fn stela_metal_reflect(
        air: *const c_void,
        air_size: usize,
        entry: *const c_char,
        constants: *const ffi::ConstantValue,
        constant_count: u32,
        out: *mut ffi::Reflection,
        error: *mut ffi::AirError,
    ) -> bool;
    fn stela_metal_reflect_bindings(
        air: *const c_void,
        air_size: usize,
        entry: *const c_char,
        constants: *const ffi::ConstantValue,
        constant_count: u32,
        out: *mut ffi::Bindings,
        error: *mut ffi::AirError,
    ) -> bool;
    fn stela_metal_sampler_states(
        air: *const c_void,
        air_size: usize,
        out: *mut u64,
        count: *mut u32,
        error: *mut ffi::AirError,
    ) -> bool;
    fn stela_metal_functions(
        air: *const c_void,
        air_size: usize,
        visit: VisitFunction,
        ctx: *mut c_void,
        error: *mut ffi::AirError,
    ) -> bool;
    fn stela_metal_exports(
        air: *const c_void,
        air_size: usize,
        visit: VisitFunction,
        ctx: *mut c_void,
        error: *mut ffi::AirError,
    ) -> bool;
    fn stela_metal_stitch(
        sources: *const ffi::StitchSource,
        source_count: u32,
        graphs: *const ffi::StitchGraph,
        graph_count: u32,
        visit: VisitBytes,
        ctx: *mut c_void,
        error: *mut ffi::AirError,
    ) -> bool;
    fn stela_metal_tessellation_stages(
        target: *const ffi::Target,
        tess: *const ffi::Tessellation,
        out: *mut *mut c_void,
    );
    fn stela_metal_accumulate_kernel(nir_options: *const c_void) -> *mut c_void;
    fn stela_metal_select_kernel(nir_options: *const c_void) -> *mut c_void;
    fn stela_metal_resolve_draws_kernel(nir_options: *const c_void) -> *mut c_void;
    fn stela_metal_resolve_dispatches_kernel(nir_options: *const c_void) -> *mut c_void;
    fn stela_metal_store_texels_kernel(nir_options: *const c_void) -> *mut c_void;
    fn stela_metal_load_texels_kernel(nir_options: *const c_void) -> *mut c_void;
    fn stela_metal_average_samples_kernel(nir_options: *const c_void) -> *mut c_void;
    fn stela_metal_decode_etc_kernel(nir_options: *const c_void) -> *mut c_void;
    fn stela_metal_decode_pvrtc_kernel(nir_options: *const c_void) -> *mut c_void;
    fn stela_metal_reorder_blocks_kernel(nir_options: *const c_void) -> *mut c_void;
    fn stela_metal_decode_astc_kernel(nir_options: *const c_void) -> *mut c_void;
    fn stela_metal_compose_kernel(nir_options: *const c_void) -> *mut c_void;
    fn stela_metal_release(shader: *mut c_void);
    fn stela_metal_buffer_reads(shader: *const c_void) -> u32;
}

static GUEST_RAM_FD: OnceLock<RawFd> = OnceLock::new();
static START_TIME: OnceLock<Instant> = OnceLock::new();
static SERVED_COUNT: AtomicU64 = AtomicU64::new(0);
static LAST_SUBMITTED: AtomicU64 = AtomicU64::new(0);
static USING_EXTERNAL_VK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static NIR_CACHE_DIR: OnceLock<Option<CString>> = OnceLock::new();

fn ensure_guest_ram_fd() -> Result<RawFd, MetalError> {
    if let Some(&fd) = GUEST_RAM_FD.get() {
        return Ok(fd);
    }
    let ram_path =
        std::env::var("STELA_GUEST_RAM_PATH").unwrap_or_else(|_| DEFAULT_RAM_PATH.to_string());
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&ram_path)
        .map_err(|_| MetalError::Invalid("failed to open QEMU guest RAM backing file"))?;
    let fd = file.as_raw_fd();
    std::mem::forget(file); // keep fd open for daemon lifetime
    let _ = GUEST_RAM_FD.set(fd);
    eprintln!("[vgpu-daemon] Opened QEMU guest RAM backing file ({ram_path}, fd={fd})");
    Ok(*GUEST_RAM_FD.get().unwrap())
}

/// On QEMU Q35 with 4 GiB RAM, RAM sits at `[0..2 GiB)` and `[4 GiB..6 GiB)`.
fn qemu_q35_ram_offset(pa: u64) -> Option<i64> {
    const LOW_RAM_END: u64 = 0x8000_0000;
    const HIGH_RAM_BASE: u64 = 0x1_0000_0000;
    if pa < LOW_RAM_END {
        Some(pa as i64)
    } else if pa >= HIGH_RAM_BASE {
        let off = (pa - HIGH_RAM_BASE) + LOW_RAM_END;
        if (off as usize) < GUEST_RAM_SIZE {
            Some(off as i64)
        } else {
            None
        }
    } else {
        None
    }
}

static LARGE_EXPORTS: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

/// Zero-copy `MemoryExporter` that maps scatter-gather guest RAM pages from
/// `/dev/shm/stela-guest-ram` into a single contiguous host virtual range via
/// `mmap(MAP_SHARED | MAP_FIXED)`, returning `fd = -1` so `NativeBackend`
/// imports it into Lavapipe via `VK_EXT_external_memory_host`.
struct HostPtrExporter;

impl MemoryExporter for HostPtrExporter {
    fn export(&self, ranges: &[MemoryRange]) -> Result<ExportedMemory, MetalError> {
        let fd = ensure_guest_ram_fd()?;
        let total: u64 = ranges.iter().map(|r| r.bytes).sum();
        if total == 0 {
            return Err(MetalError::Invalid("empty memory export"));
        }
        let map_len = total as usize;
        // Reserve a contiguous virtual address range.
        let base = unsafe {
            libc::mmap(
                ptr::null_mut(),
                map_len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(MetalError::OutOfMemory);
        }
        let mut cursor = 0usize;
        for r in ranges {
            let Some(file_off) = qemu_q35_ram_offset(r.address) else {
                unsafe { libc::munmap(base, map_len) };
                return Err(MetalError::Invalid("guest PA outside QEMU RAM"));
            };
            let run_len = r.bytes as usize;
            let mapped = unsafe {
                libc::mmap(
                    (base as *mut u8).add(cursor) as *mut libc::c_void,
                    run_len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    fd,
                    file_off,
                )
            };
            if mapped == libc::MAP_FAILED {
                unsafe { libc::munmap(base, map_len) };
                return Err(MetalError::OutOfMemory);
            }
            cursor += run_len;
        }
        let pointer = NonNull::new(base as *mut u8).ok_or(MetalError::OutOfMemory)?;
        if map_len >= 1024 * 1024 {
            if let Ok(mut list) = LARGE_EXPORTS.lock() {
                list.push((base as usize, map_len));
            }
        }
        Ok(ExportedMemory {
            fd: -1,
            pointer,
            bytes: total,
        })
    }

    fn close(&self, exported: ExportedMemory) {
        self.unmap(exported);
    }

    fn unmap(&self, exported: ExportedMemory) {
        if let Ok(mut list) = LARGE_EXPORTS.lock() {
            let addr = exported.pointer.as_ptr() as usize;
            list.retain(|&(p, _)| p != addr);
        }
        unsafe {
            libc::munmap(exported.pointer.as_ptr() as *mut libc::c_void, exported.bytes as usize);
        }
    }
}

fn load_lavapipe_proc_addr() -> vk::PFN_vkGetInstanceProcAddr {
    eprintln!("[vgpu-daemon] Using statically linked Lavapipe vk_icdGetInstanceProcAddr");
    vk_icdGetInstanceProcAddr
}

fn load_external_vulkan_proc_addr(lib_path: &str) -> Result<vk::PFN_vkGetInstanceProcAddr, String> {
    let c_path = CString::new(lib_path).map_err(|_| "invalid library path".to_string())?;
    let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        let err = unsafe {
            let p = libc::dlerror();
            if p.is_null() {
                "unknown dlopen error".to_string()
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        return Err(format!("dlopen({lib_path}) failed: {err}"));
    }
    for sym in [c"vk_icdGetInstanceProcAddr", c"vkGetInstanceProcAddr"] {
        let fp = unsafe { libc::dlsym(handle, sym.as_ptr()) };
        if !fp.is_null() {
            eprintln!(
                "[vgpu-daemon] Loaded external Vulkan symbol {:?} from {lib_path}",
                sym
            );
            let proc_addr: vk::PFN_vkGetInstanceProcAddr = unsafe { std::mem::transmute(fp) };
            return Ok(proc_addr);
        }
    }
    Err(format!(
        "neither vk_icdGetInstanceProcAddr nor vkGetInstanceProcAddr found in {lib_path}"
    ))
}

/// Probes an external Vulkan loader/ICD for non-CPU (hardware) physical devices
/// before handing it to Stela's `open::stack` (which compiles Mesa 26.2.5 NIR).
fn probe_external_hardware_gpu(
    proc_addr: vk::PFN_vkGetInstanceProcAddr,
) -> Result<(String, u32), String> {
    unsafe {
        let entry = ash::Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr: proc_addr,
        });
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let info = vk::InstanceCreateInfo::default().application_info(&app);
        let instance = entry
            .create_instance(&info, None)
            .map_err(|e| format!("vkCreateInstance failed: {e:?}"))?;
        let devices = instance
            .enumerate_physical_devices()
            .map_err(|e| {
                instance.destroy_instance(None);
                format!("vkEnumeratePhysicalDevices failed: {e:?}")
            })?;
        let mut found_hw: Option<(String, u32)> = None;
        let mut seen = Vec::new();
        for pdev in devices {
            let props = instance.get_physical_device_properties(pdev);
            let name = std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned();
            let maj = vk::api_version_major(props.api_version);
            let min = vk::api_version_minor(props.api_version);
            let kind = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => "DiscreteGPU",
                vk::PhysicalDeviceType::INTEGRATED_GPU => "IntegratedGPU",
                vk::PhysicalDeviceType::VIRTUAL_GPU => "VirtualGPU",
                vk::PhysicalDeviceType::CPU => "CPU",
                _ => "Other",
            };
            seen.push(format!("{name} ({kind}, Vulkan {maj}.{min})"));
            if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                || props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
            {
                if found_hw.is_none() {
                    found_hw = Some((name, props.api_version));
                }
            }
        }
        instance.destroy_instance(None);
        found_hw.ok_or_else(|| {
            format!(
                "no hardware GPU exposed by Vulkan loader (only found: {})",
                if seen.is_empty() {
                    "none".to_string()
                } else {
                    seen.join(", ")
                }
            )
        })
    }
}

fn nir_cache_dir_ptr() -> *const c_char {
    NIR_CACHE_DIR
        .get_or_init(|| {
            std::env::var("STELA_NIR_CACHE_DIR")
                .ok()
                .filter(|s| !s.is_empty())
                .and_then(|s| {
                    let _ = std::fs::create_dir_all(&s);
                    CString::new(s).ok()
                })
        })
        .as_ref()
        .map_or(ptr::null(), |c| c.as_ptr())
}

unsafe extern "C" {
    fn malloc_trim(pad: usize) -> libc::c_int;
}

unsafe extern "C" fn compile_wrapper(
    air: *const c_void,
    air_size: usize,
    entry: *const c_char,
    target: *const ffi::Target,
    spec: *const ffi::Specialization,
    reflected: *const ffi::Reflection,
    error: *mut ffi::AirError,
) -> *mut c_void {
    let mut hit: libc::c_int = 0;
    let res = unsafe {
        stela_metal_compile_cached(
            nir_cache_dir_ptr(),
            air,
            air_size,
            entry,
            target,
            spec,
            reflected,
            error,
            &mut hit,
        )
    };
    static COMPILED_COUNT: AtomicU64 = AtomicU64::new(0);
    static CACHE_HITS: AtomicU64 = AtomicU64::new(0);
    let n = COMPILED_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let hits = if hit != 0 {
        CACHE_HITS.fetch_add(1, Ordering::Relaxed) + 1
    } else {
        CACHE_HITS.load(Ordering::Relaxed)
    };
    if n <= 10 || n % 25 == 0 {
        let entry_str = if entry.is_null() {
            "<null>"
        } else {
            unsafe { std::ffi::CStr::from_ptr(entry).to_str().unwrap_or("<utf8>") }
        };
        let tag = if hit != 0 { "CACHE HIT" } else { "compiled" };
        eprintln!(
            "[vgpu-daemon] {tag} shader #{n} (hits={hits}): {entry_str} ({air_size}B) -> {:p}",
            res
        );
    }
    if hit == 0 && n % 10 == 0 {
        unsafe { malloc_trim(0) };
    }
    res
}

unsafe extern "C" fn crash_handler(sig: libc::c_int) {
    let msg = b"\n[vgpu-daemon] FATAL SIGNAL CAUGHT -- backtrace:\n";
    unsafe {
        libc::write(2, msg.as_ptr() as *const c_void, msg.len());
        let mut addrs: [*mut c_void; 64] = [ptr::null_mut(); 64];
        let n = libc::backtrace(addrs.as_mut_ptr(), addrs.len() as libc::c_int);
        libc::backtrace_symbols_fd(addrs.as_ptr(), n, 2);
        libc::_exit(128 + sig);
    }
}

fn build_compiler_service(physical: vk::PhysicalDevice) -> compiler::Service {
    use stela_driver_gpu::metal::native::vk::Handle;
    let external = USING_EXTERNAL_VK.load(Ordering::Relaxed);
    eprintln!(
        "[vgpu-daemon] build_compiler_service physical={:#x} external={external}",
        physical.as_raw()
    );
    let mut raw_opts = [ptr::null::<c_void>(); NIR_STAGES];
    unsafe {
        if external {
            stela_generic_nir_options(raw_opts.as_mut_ptr());
        } else {
            stela_lvp_nir_options(physical.as_raw() as *mut c_void, raw_opts.as_mut_ptr());
        }
    }
    eprintln!("[vgpu-daemon] got nir_options[0]={:p} [5]={:p}", raw_opts[0], raw_opts[5]);
    let nir_options: [NonNull<c_void>; NIR_STAGES] = raw_opts.map(|p| {
        NonNull::new(p as *mut c_void).expect("Vulkan driver returned null nir_shader_compiler_options")
    });
    compiler::Service {
        reflect: stela_metal_reflect,
        reflect_bindings: stela_metal_reflect_bindings,
        sampler_states: stela_metal_sampler_states,
        functions: stela_metal_functions,
        exports: stela_metal_exports,
        stitch: stela_metal_stitch,
        compile: compile_wrapper,
        tessellation_stages: stela_metal_tessellation_stages,
        accumulate_kernel: stela_metal_accumulate_kernel,
        select_kernel: stela_metal_select_kernel,
        resolve_draws_kernel: stela_metal_resolve_draws_kernel,
        resolve_dispatches_kernel: stela_metal_resolve_dispatches_kernel,
        store_texels_kernel: stela_metal_store_texels_kernel,
        load_texels_kernel: stela_metal_load_texels_kernel,
        average_samples_kernel: stela_metal_average_samples_kernel,
        decode_etc_kernel: stela_metal_decode_etc_kernel,
        decode_pvrtc_kernel: stela_metal_decode_pvrtc_kernel,
        reorder_blocks_kernel: stela_metal_reorder_blocks_kernel,
        decode_astc_kernel: stela_metal_decode_astc_kernel,
        compose_kernel: stela_metal_compose_kernel,
        release: stela_metal_release,
        buffer_reads: stela_metal_buffer_reads,
        nir_options,
    }
}

struct RemoteCallerMemory<'a> {
    header: &'a VgpuShmHeader,
    payload: *mut u8,
    cb_seq: u64,
}

impl<'a> RemoteCallerMemory<'a> {
    fn new(header: &'a VgpuShmHeader, payload: *mut u8) -> Self {
        let cb_seq = header.cb_req.load(Ordering::Acquire);
        Self {
            header,
            payload,
            cb_seq,
        }
    }

    fn call_cb(&mut self, op: u32, addr: u64, len: usize) -> Result<(), CallerFault> {
        self.header.cb_op.store(op, Ordering::Relaxed);
        self.header.cb_addr.store(addr, Ordering::Relaxed);
        self.header.cb_len.store(len as u64, Ordering::Relaxed);
        self.cb_seq += 1;
        let seq = self.cb_seq;
        self.header.cb_req.store(seq, Ordering::Release);

        while self.header.cb_ack.load(Ordering::Acquire) != seq {
            std::hint::spin_loop();
        }
        match self.header.cb_status.load(Ordering::Relaxed) {
            STATUS_OK => Ok(()),
            STATUS_FAULT_DENIED => Err(CallerFault::Denied { address: addr }),
            STATUS_FAULT_NOT_RAM => Err(CallerFault::NotRam { address: addr }),
            _ => Err(CallerFault::Unsupported("guest caller memory fault")),
        }
    }
}

impl CallerMemory for RemoteCallerMemory<'_> {
    fn read(&mut self, mut address: u64, mut buffer: &mut [u8]) -> Result<(), CallerFault> {
        while !buffer.is_empty() {
            let chunk = buffer.len().min(VGPU_PAYLOAD_MAX);
            self.call_cb(CB_READ, address, chunk)?;
            unsafe {
                ptr::copy_nonoverlapping(self.payload as *const u8, buffer.as_mut_ptr(), chunk);
            }
            address = address.saturating_add(chunk as u64);
            buffer = &mut buffer[chunk..];
        }
        Ok(())
    }

    fn write(&mut self, mut address: u64, mut data: &[u8]) -> Result<(), CallerFault> {
        while !data.is_empty() {
            let chunk = data.len().min(VGPU_PAYLOAD_MAX);
            unsafe {
                ptr::copy_nonoverlapping(data.as_ptr(), self.payload, chunk);
            }
            self.call_cb(CB_WRITE, address, chunk)?;
            address = address.saturating_add(chunk as u64);
            data = &data[chunk..];
        }
        Ok(())
    }
}

fn ticks() -> u64 {
    START_TIME
        .get_or_init(Instant::now)
        .elapsed()
        .as_nanos() as u64
}

fn relax() {
    std::hint::spin_loop();
}

fn relax_within() -> bool {
    std::hint::spin_loop();
    true
}

fn run(_stack: u64, work: &mut dyn FnMut()) {
    work();
}

fn report(msg: &str) {
    if msg.contains("GPU family") || msg.contains("calibrated timestamps") {
        return;
    }
    eprintln!("[vgpu-daemon] refused: {msg}");
}

fn note(msg: &str) {
    if msg.starts_with("PIPELINE ") || msg.starts_with("no function of that name") {
        return;
    }
    eprintln!("[vgpu-daemon] {msg}");
}

fn served(info: Served) {
    if let Some((submitted, _completed)) = info.progress {
        LAST_SUBMITTED.store(submitted, Ordering::Relaxed);
    }
    let n = SERVED_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 15 || n % 500 == 0 {
        eprintln!(
            "[vgpu-daemon] served #{n}: op={} status={:?} progress={:?}",
            info.operation, info.status, info.progress
        );
    }
}

fn wake() {}
fn bundle(report: BundleReport) {
    eprintln!("[vgpu-daemon] bundle: {report:?}");
}
fn job_waiting() {}

fn main() {
    START_TIME.get_or_init(Instant::now);

    unsafe {
        libc::signal(libc::SIGSEGV, crash_handler as *const () as usize);
        libc::signal(libc::SIGABRT, crash_handler as *const () as usize);
    }
    let vk_mode = std::env::var("STELA_VK_BACKEND").unwrap_or_else(|_| "auto".to_string());
    let vk_icd = std::env::var("STELA_VK_ICD").unwrap_or_else(|_| "libvulkan.so.1".to_string());
    let any_pci = PciBusInfo {
        domain: 0,
        bus: 0,
        device: 0,
        function: 0,
    };

    let open_with = |proc_addr: vk::PFN_vkGetInstanceProcAddr, external: bool| {
        USING_EXTERNAL_VK.store(external, Ordering::Relaxed);
        let loader = Loader {
            get_instance_proc_addr: proc_addr,
            log: note,
            submit_thread: None,
            refusals: None,
            keep_lost: false,
            relax,
        };
        unsafe {
            open::stack(
                loader,
                any_pci,
                Box::new(HostPtrExporter),
                build_compiler_service,
            )
        }
    };

    let (backend, compiler): (NativeBackend, AirCompiler) = match vk_mode.as_str() {
        "hardware" => {
            eprintln!("[vgpu-daemon] STELA_VK_BACKEND=hardware: loading {vk_icd} (no fallback)...");
            let proc_addr = load_external_vulkan_proc_addr(&vk_icd)
                .unwrap_or_else(|e| panic!("hardware Vulkan load failed: {e}"));
            let (gpu_name, api) = probe_external_hardware_gpu(proc_addr)
                .unwrap_or_else(|e| panic!("hardware Vulkan probe failed: {e}"));
            eprintln!(
                "[vgpu-daemon] Found hardware GPU: {gpu_name} (Vulkan {}.{})",
                vk::api_version_major(api),
                vk::api_version_minor(api)
            );
            open_with(proc_addr, true)
                .unwrap_or_else(|e| panic!("hardware Vulkan init failed on {gpu_name}: {e:?}"))
        }
        "lavapipe" => {
            eprintln!("[vgpu-daemon] Loading Mesa 26.2.5 Lavapipe + Stela AIR-to-NIR compiler...");
            open_with(load_lavapipe_proc_addr(), false)
                .expect("failed to initialize NativeBackend + AirCompiler on Lavapipe")
        }
        _ => {
            // auto: probe external Vulkan loader for a real hardware GPU first;
            // fall back to built-in Mesa 26.2.5 Lavapipe if only CPU/software is available.
            let hw_attempt = load_external_vulkan_proc_addr(&vk_icd).and_then(|proc_addr| {
                let (gpu_name, api) = probe_external_hardware_gpu(proc_addr)?;
                eprintln!(
                    "[vgpu-daemon] Found hardware GPU: {gpu_name} (Vulkan {}.{})",
                    vk::api_version_major(api),
                    vk::api_version_minor(api)
                );
                open_with(proc_addr, true).map_err(|e| format!("{gpu_name}: {e:?}"))
            });
            match hw_attempt {
                Ok((b, c)) => {
                    eprintln!("[vgpu-daemon] Hardware Vulkan GPU accepted!");
                    (b, c)
                }
                Err(e) => {
                    eprintln!(
                        "[vgpu-daemon] External hardware Vulkan unavailable ({e}); using built-in Mesa 26.2.5 Lavapipe..."
                    );
                    open_with(load_lavapipe_proc_addr(), false)
                        .expect("failed to initialize NativeBackend + AirCompiler on Lavapipe")
                }
            }
        }
    };

    eprintln!(
        "[vgpu-daemon] Vulkan ready! Physical device: {} (type={:?})",
        backend.physical_device().identity.device_name,
        backend.physical_device().identity.device_type
    );

    let platform = SoftBackend::platform();
    let device = Device::new(backend, compiler, &platform)
        .expect("failed to initialize Stela Metal Device over Lavapipe");
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
            ticks,
            served,
            wake,
            bundle,
            job_waiting,
        },
    ));

    let shm_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(SHM_PATH)
        .expect("failed to open /dev/shm/stela-vgpu-shm");
    shm_file
        .set_len(VGPU_SHM_SIZE as u64)
        .expect("failed to size /dev/shm/stela-vgpu-shm");

    let shm_ptr = unsafe {
        libc::mmap(
            ptr::null_mut(),
            VGPU_SHM_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            shm_file.as_raw_fd(),
            0,
        )
    };
    assert!(shm_ptr != libc::MAP_FAILED, "mmap {SHM_PATH} failed");

    unsafe {
        ptr::write_bytes(shm_ptr as *mut u8, 0, VGPU_PAYLOAD_OFFSET);
    }

    let header = unsafe { &*(shm_ptr as *const VgpuShmHeader) };
    let payload = unsafe { (shm_ptr as *mut u8).add(VGPU_PAYLOAD_OFFSET) };

    header.version.store(VGPU_VERSION, Ordering::Relaxed);
    header.magic.store(VGPU_MAGIC, Ordering::Release);
    header.host_ready.store(1, Ordering::Release);

    eprintln!(
        "[vgpu-daemon] REAL METAL vGPU READY on {SHM_PATH} (2 MiB). Waiting for QEMU Stela UEFI..."
    );

    let mut last_seq = 0u64;
    let mut idle_spins = 0u32;

    loop {
        // Poll async Vulkan timeline completion even when idle.
        if shared.awaits_completion() {
            shared.interrupted();
        }
        if shared.signaled() {
            header.signaled_flag.store(1, Ordering::Release);
        }
        header
            .awaits_flag
            .store(u32::from(shared.awaits_completion()), Ordering::Release);

        let req_seq = header.guest_seq.load(Ordering::Acquire);
        if req_seq == last_seq {
            idle_spins = idle_spins.saturating_add(1);
            if idle_spins < 5_000 {
                std::hint::spin_loop();
            } else if last_seq == 0 || idle_spins > 10_000 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            } else {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            continue;
        }
        idle_spins = 0;

        let op = header.op.load(Ordering::Relaxed);
        let arg0 = header.arg0.load(Ordering::Relaxed);
        let arg1 = header.arg1.load(Ordering::Relaxed);
        let arg2 = header.arg2.load(Ordering::Relaxed);

        let submitted_before = LAST_SUBMITTED.load(Ordering::Relaxed);
        let mut status = STATUS_OK;
        let mut ret0 = 0u64;
        let ret1 = 0u64;

        match op {
            OP_CONNECT => match shared.connect() {
                Ok(conn) => {
                    eprintln!("[vgpu-daemon] macOS client connected -> conn={}", conn.get());
                    ret0 = conn.get();
                }
                Err(e) => {
                    eprintln!("[vgpu-daemon] connect error: {e}");
                    status = STATUS_ERR;
                }
            },
            OP_DISCONNECT => {
                if let Some(conn) = NonZeroU64::new(arg0) {
                    shared.disconnect(conn);
                }
            }
            OP_SERVE => {
                if let Some(conn) = NonZeroU64::new(arg0) {
                    let mut caller = RemoteCallerMemory::new(header, payload);
                    match shared.serve(conn, arg1, &mut caller) {
                        Ok(()) => {}
                        Err(CallerFault::Denied { address }) => {
                            status = STATUS_FAULT_DENIED;
                            ret0 = address;
                        }
                        Err(CallerFault::NotRam { address }) => {
                            status = STATUS_FAULT_NOT_RAM;
                            ret0 = address;
                        }
                        Err(CallerFault::Unsupported(_)) => {
                            status = STATUS_FAULT_UNSUPPORTED;
                        }
                    }
                } else {
                    status = STATUS_ERR;
                }
            }
            OP_IMPORT => {
                let count = (arg2 as usize).min(VGPU_PAYLOAD_MAX / std::mem::size_of::<WireRange>());
                let wire_slice =
                    unsafe { std::slice::from_raw_parts(payload as *const WireRange, count) };
                let ranges: Vec<MemoryRange> = wire_slice
                    .iter()
                    .map(|r| MemoryRange {
                        address: r.address,
                        bytes: r.bytes,
                    })
                    .collect();
                match (NonZeroU64::new(arg0), NonZeroU64::new(arg1)) {
                    (Some(conn), Some(token)) => match shared.import(conn, &ranges, token) {
                        Ok(handle) => ret0 = handle.get(),
                        Err(e) => {
                            eprintln!("[vgpu-daemon] import error: {e}");
                            status = STATUS_ERR;
                        }
                    },
                    _ => status = STATUS_ERR,
                }
            }
            OP_RETIRED => {
                ret0 = shared.retired().map_or(0, NonZeroU64::get);
            }
            OP_QUIESCED => {
                ret0 = u64::from(shared.quiesced());
            }
            OP_BUNDLE => {
                let rep = match arg0 {
                    0 => BundleReport::Current,
                    1 => BundleReport::Updated,
                    2 => BundleReport::Missing,
                    _ => BundleReport::Missing,
                };
                shared.bundle(rep);
            }
            OP_DROPPED => {
                shared.dropped("vgpu guest dropped");
            }
            _ => {
                status = STATUS_ERR;
            }
        }

        let submitted_after = LAST_SUBMITTED.load(Ordering::Relaxed);
        let just_submitted = op == OP_SERVE && submitted_after > submitted_before;
        if just_submitted && shared.awaits_completion() {
            // Wait up to 1200ms for Lavapipe to finish rendering + aligned_copy
            // writeback so every committed frame completes cleanly (no gray/blank
            // or vanishing windows!). Non-submit ops never block here.
            let max_wait = std::time::Duration::from_millis(1200);
            let wait_start = Instant::now();
            while shared.awaits_completion() {
                shared.interrupted();
                if !shared.awaits_completion() {
                    break;
                }
                if wait_start.elapsed() > max_wait {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(150));
            }
            static COMPLETED_LOG: AtomicU64 = AtomicU64::new(0);
            let c = COMPLETED_LOG.fetch_add(1, Ordering::Relaxed) + 1;
            let sig = shared.signaled();
            if sig {
                header.signaled_flag.store(1, Ordering::Release);
            }
            if c <= 10 || c % 200 == 0 {
                eprintln!(
                    "[vgpu-daemon] GPU submit #{c} on seq={req_seq} in {}ms (signaled={sig})",
                    wait_start.elapsed().as_millis()
                );
            }
            if c % 50 == 0 {
                unsafe { malloc_trim(0) };
            }
        } else {
            if shared.awaits_completion() {
                shared.interrupted();
            }
            if shared.signaled() {
                header.signaled_flag.store(1, Ordering::Release);
            }
        }
        header
            .awaits_flag
            .store(u32::from(shared.awaits_completion()), Ordering::Release);

        header.status.store(status, Ordering::Relaxed);
        header.ret0.store(ret0, Ordering::Relaxed);
        header.ret1.store(ret1, Ordering::Relaxed);

        last_seq = req_seq;
        header.host_seq.store(req_seq, Ordering::Release);
    }
}
