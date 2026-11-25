#!/usr/bin/env bash
set -euo pipefail

VM_DIR="/mnt/c/Users/lucas/Documents/Virtual Machines/macOS 26"
BOOT_VMDK="$VM_DIR/stela-boot.vmdk"
MACOS_VMDK="$VM_DIR/macOS 26.vmdk"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OVMF_VARS="$SCRIPT_DIR/OVMF_VARS.fd"

if [ -d "$VM_DIR/stela-boot.vmdk.lck" ] || [ -d "$VM_DIR/macOS 26.vmdk.lck" ]; then
    echo "[ERRO] A VM 'macOS 26' ainda esta ligada no VMware! Desligue no VMware primeiro."
    exit 1
fi

# Auto-detect OVMF CODE and VARS
OVMF_CODE=""
for p in /usr/share/OVMF/OVMF_CODE_4M.fd /usr/share/OVMF/OVMF_CODE.fd /usr/share/ovmf/OVMF.fd; do
    if [ -f "$p" ]; then OVMF_CODE="$p"; break; fi
done
if [ -z "$OVMF_CODE" ]; then
    echo "[ERRO] OVMF nao encontrado. Rode: sudo apt update && sudo apt install -y qemu-system-x86 ovmf"
    exit 1
fi

if [ ! -f "$OVMF_VARS" ]; then
    for v in /usr/share/OVMF/OVMF_VARS_4M.fd /usr/share/OVMF/OVMF_VARS.fd; do
        if [ -f "$v" ]; then cp "$v" "$OVMF_VARS"; break; fi
    done
fi

# Pure 2D virtio-vga (default 1024x768 — standard VESA/GOP mode verified with OVMF).
# Override with STELA_XRES=800 STELA_YRES=600 ./run-qemu-stela.sh if desired.
XRES="${STELA_XRES:-1024}"
YRES="${STELA_YRES:-768}"
GPU_ARGS=(-device "virtio-vga,xres=${XRES},yres=${YRES}")
DISPLAY_ARGS=(-display gtk,grab-on-hover=on)

# Ensure /dev/shm is large enough for 4 GiB guest RAM + 2 MiB vGPU SHM
# (avoids both the default WSL2 /dev/shm cap and ext4 disk writeback stutters).
SHM_MB=$(df -Pm /dev/shm | awk 'NR==2 {print $2}')
if [ "${SHM_MB:-0}" -lt 4200 ]; then
    echo "[vGPU] Aumentando /dev/shm de ${SHM_MB}M para 4500M (evita travamentos no disco)..."
    sudo mount -o remount,size=4500M /dev/shm
fi

VGPU_SHM="/dev/shm/stela-vgpu-shm"
GUEST_RAM="/dev/shm/stela-guest-ram"
VGPU_BIN="$SCRIPT_DIR/crates/stela-vgpu-daemon/target/release/stela-vgpu-daemon"
VGPU_LOG="$SCRIPT_DIR/vgpu-daemon.log"

rm -f "$VGPU_SHM" "$GUEST_RAM"
truncate -s 2M "$VGPU_SHM"

VGPU_PID=""
cleanup() {
    if [ -n "$VGPU_PID" ] && kill -0 "$VGPU_PID" 2>/dev/null; then
        kill -9 "$VGPU_PID" 2>/dev/null || true
    fi
    rm -f "$GUEST_RAM" "$VGPU_SHM"
}
trap cleanup EXIT

SHADER_CACHE="$SCRIPT_DIR/.mesa-shader-cache"
NIR_CACHE="$SCRIPT_DIR/.stela-nir-cache"
mkdir -p "$SHADER_CACHE" "$NIR_CACHE"

VGPU_TASKSET=()
QEMU_TASKSET=()
if [ "$(nproc)" -ge 4 ] && command -v taskset >/dev/null 2>&1; then
    QEMU_TASKSET=(taskset -c 0,1)
    VGPU_TASKSET=(taskset -c 2,3)
fi

VK_BACKEND="${STELA_VK_BACKEND:-auto}"

if [ -x "$VGPU_BIN" ]; then
    echo "[vGPU] Iniciando stela-vgpu-daemon (STELA_VK_BACKEND=$VK_BACKEND + NIR disk cache)..."
    STELA_VK_BACKEND="$VK_BACKEND" \
    STELA_NIR_CACHE_DIR="$NIR_CACHE" \
    MESA_SHADER_CACHE_DISABLE=false \
    MESA_DISK_CACHE_SINGLE_FILE=true \
    MESA_SHADER_CACHE_DIR="$SHADER_CACHE" \
    GALLIVM_PERF=nopt,brilinear,rho_approx \
    LP_NUM_THREADS=2 \
    STELA_GUEST_RAM_PATH="$GUEST_RAM" \
    "${VGPU_TASKSET[@]}" nice -n 15 "$VGPU_BIN" >"$VGPU_LOG" 2>&1 &
    VGPU_PID=$!
    # Wait up to 8s for daemon to publish magic on /dev/shm/stela-vgpu-shm
    for _ in $(seq 1 80); do
        if grep -q "REAL METAL vGPU READY" "$VGPU_LOG" 2>/dev/null; then
            echo "[vGPU] Pronto! (log em $VGPU_LOG)"
            break
        fi
        if ! kill -0 "$VGPU_PID" 2>/dev/null; then
            echo "[vGPU] ERRO ao iniciar stela-vgpu-daemon! Veja $VGPU_LOG:"
            cat "$VGPU_LOG"
            exit 1
        fi
        sleep 0.1
    done
fi

"${QEMU_TASKSET[@]}" nice -n 10 qemu-system-x86_64 \
    -machine q35,accel=kvm \
    -cpu host \
    -smp 2,cores=2,threads=1 \
    -m 4096 \
    -object memory-backend-file,id=mem1,size=4096M,mem-path="$GUEST_RAM",share=on \
    -machine memory-backend=mem1 \
    -object memory-backend-file,id=vgpu_shm,size=2M,mem-path="$VGPU_SHM",share=on \
    -device ivshmem-plain,memdev=vgpu_shm \
    -drive if=pflash,format=raw,readonly=on,file="$OVMF_CODE" \
    -drive if=pflash,format=raw,file="$OVMF_VARS" \
    -device ich9-ahci,id=sata \
    -drive if=none,id=stela_disk,format=vmdk,file="$BOOT_VMDK",cache=writeback \
    -device ide-hd,drive=stela_disk,bus=sata.0,serial=00000000000000000001,bootindex=1 \
    -drive if=none,id=macos_disk,format=vmdk,file="$MACOS_VMDK",cache=writeback \
    -device ide-hd,drive=macos_disk,bus=sata.1,serial=01000000000000000001 \
    -net none \
    "${GPU_ARGS[@]}" \
    "${DISPLAY_ARGS[@]}" \
    -device qemu-xhci \
    -chardev stdio,id=ser0,mux=on,logfile="$SCRIPT_DIR/qemu-serial.log",logappend=off \
    -serial chardev:ser0 \
    -mon chardev=ser0
