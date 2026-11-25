#!/usr/bin/env bash
set -euo pipefail

# GoldenOSX-OS macOS 27.2 VM Launcher (Bare-Metal Linux + KVM + Metal vGPU)
BOOT_VMDK="${1:?Usage: goldenosx-launch.sh <stela-boot.vmdk> <macOS.vmdk>}"
MACOS_VMDK="${2:?Usage: goldenosx-launch.sh <stela-boot.vmdk> <macOS.vmdk>}"

VM_DIR="$(dirname "$MACOS_VMDK")"
CACHE_BASE="$VM_DIR/.goldenosx-cache"
mkdir -p "$CACHE_BASE/nir" "$CACHE_BASE/mesa" "$CACHE_BASE/logs"

OVMF_CODE=""
for p in /usr/share/OVMF/OVMF_CODE_4M.fd /usr/share/OVMF/OVMF_CODE.fd /usr/share/ovmf/OVMF.fd; do
    if [ -f "$p" ]; then OVMF_CODE="$p"; break; fi
done
if [ -z "$OVMF_CODE" ]; then
    echo "[ERRO] OVMF nao encontrado!"
    exit 1
fi

OVMF_VARS="$CACHE_BASE/OVMF_VARS.fd"
if [ ! -f "$OVMF_VARS" ]; then
    for v in /usr/share/OVMF/OVMF_VARS_4M.fd /usr/share/OVMF/OVMF_VARS.fd; do
        if [ -f "$v" ]; then cp "$v" "$OVMF_VARS"; break; fi
    done
fi

# Auto-size guest RAM: total RAM - 2048 MB (min 4096 MB, max 16384 MB)
TOTAL_MB=$(awk '/MemTotal:/ {printf "%d", $2/1024}' /proc/meminfo)
DEFAULT_GUEST_MB=$(( TOTAL_MB - 2048 ))
if [ "$DEFAULT_GUEST_MB" -lt 4096 ]; then DEFAULT_GUEST_MB=4096; fi
if [ "$DEFAULT_GUEST_MB" -gt 16384 ]; then DEFAULT_GUEST_MB=16384; fi
GUEST_MB="${STELA_GUEST_RAM_MB:-$DEFAULT_GUEST_MB}"

# Ensure /dev/shm fits guest RAM + 512 MB overhead
NEED_SHM_MB=$(( GUEST_MB + 512 ))
CUR_SHM_MB=$(df -Pm /dev/shm | awk 'NR==2 {print $2}')
if [ "${CUR_SHM_MB:-0}" -lt "$NEED_SHM_MB" ]; then
    mount -o "remount,size=${NEED_SHM_MB}M" /dev/shm
fi

VGPU_SHM="/dev/shm/stela-vgpu-shm"
GUEST_RAM="/dev/shm/stela-guest-ram"
VGPU_BIN="/opt/goldenosx/bin/stela-vgpu-daemon"
VGPU_LOG="$CACHE_BASE/logs/vgpu.log"
SERIAL_LOG="$CACHE_BASE/logs/serial.log"
ln -sf "$VGPU_LOG" /var/log/goldenosx-vgpu.log
ln -sf "$SERIAL_LOG" /var/log/goldenosx-serial.log

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

XRES="${STELA_XRES:-1024}"
YRES="${STELA_YRES:-768}"
VK_BACKEND="${STELA_VK_BACKEND:-auto}"

NPROC=$(nproc)
if [ "$NPROC" -ge 4 ]; then
    QEMU_SMP="4,cores=2,threads=2"
else
    QEMU_SMP="${NPROC},cores=${NPROC},threads=1"
fi
LP_THREADS="$NPROC"
[ "$LP_THREADS" -gt 8 ] && LP_THREADS=8

export LD_LIBRARY_PATH="/opt/goldenosx/lib:${LD_LIBRARY_PATH:-}"

echo "[GoldenOSX] Iniciando stela-vgpu-daemon (RAM=${GUEST_MB}M, Vulkan=${VK_BACKEND}, LP_THREADS=${LP_THREADS})..."
STELA_GUEST_RAM_MB="$GUEST_MB" \
STELA_GUEST_RAM_PATH="$GUEST_RAM" \
STELA_VK_BACKEND="$VK_BACKEND" \
STELA_NIR_CACHE_DIR="$CACHE_BASE/nir" \
MESA_SHADER_CACHE_DISABLE=false \
MESA_DISK_CACHE_SINGLE_FILE=true \
MESA_SHADER_CACHE_DIR="$CACHE_BASE/mesa" \
GALLIVM_PERF=nopt,brilinear,rho_approx,no_quad_lod \
LP_NUM_THREADS="$LP_THREADS" \
"$VGPU_BIN" >"$VGPU_LOG" 2>&1 &
VGPU_PID=$!

for _ in $(seq 1 80); do
    if grep -q "REAL METAL vGPU READY" "$VGPU_LOG" 2>/dev/null; then
        echo "[GoldenOSX] vGPU pronta!"
        break
    fi
    if ! kill -0 "$VGPU_PID" 2>/dev/null; then
        echo "[GoldenOSX] ERRO ao iniciar stela-vgpu-daemon! Log:"
        cat "$VGPU_LOG"
        read -rp "Pressione Enter para voltar ao menu..."
        exit 1
    fi
    sleep 0.1
done

QEMU_CMD=(
    qemu-system-x86_64
    -machine q35,accel=kvm
    -cpu host
    -smp "${QEMU_SMP}"
    -m "${GUEST_MB}"
    -object "memory-backend-file,id=mem1,size=${GUEST_MB}M,mem-path=${GUEST_RAM},share=on"
    -machine memory-backend=mem1
    -object "memory-backend-file,id=vgpu_shm,size=2M,mem-path=${VGPU_SHM},share=on"
    -device ivshmem-plain,memdev=vgpu_shm
    -drive "if=pflash,format=raw,readonly=on,file=${OVMF_CODE}"
    -drive "if=pflash,format=raw,file=${OVMF_VARS}"
    -device ich9-ahci,id=sata
    -drive "if=none,id=stela_disk,format=vmdk,file=${BOOT_VMDK},cache=writeback"
    -device ide-hd,drive=stela_disk,bus=sata.0,serial=00000000000000000001,bootindex=1
    -drive "if=none,id=macos_disk,format=vmdk,file=${MACOS_VMDK},cache=writeback"
    -device ide-hd,drive=macos_disk,bus=sata.1,serial=01000000000000000001
    -net none
    -device "virtio-vga,xres=${XRES},yres=${YRES}"
    -device qemu-xhci
    -chardev "file,id=ser0,path=${SERIAL_LOG}"
    -serial chardev:ser0
)

# Run fullscreen via Cage (Wayland kiosk directly on DRM/KMS) when on Linux console
QEMU_LOG="/var/log/goldenosx-qemu.log"
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ] && command -v cage >/dev/null 2>&1; then
    systemctl start seatd 2>/dev/null || true
    export LIBSEAT_BACKEND=seatd
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/0}"
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"
    export WLR_LIBINPUT_NO_DEVICES=1
    if ! cage -m last -s -- "${QEMU_CMD[@]}" -display gtk,full-screen=on,grab-on-hover=on,show-cursor=off 2>"$QEMU_LOG"; then
        echo "[GoldenOSX] cage -m last falhou, log:"
        cat "$QEMU_LOG"
    fi
else
    if ! "${QEMU_CMD[@]}" -display gtk,full-screen=on,grab-on-hover=on,show-cursor=off 2>"$QEMU_LOG"; then
        echo "[GoldenOSX] QEMU saiu com erro! Log ($QEMU_LOG):"
        cat "$QEMU_LOG"
    fi
fi
