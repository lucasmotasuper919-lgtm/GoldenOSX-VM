#!/usr/bin/env bash
set -euo pipefail

# GoldenOSX-OS ISO Builder (Debian 13 Trixie Minimal + KVM + QEMU + Stela Metal vGPU)
# Must be run as root (sudo ./goldenosx-os/build-iso.sh)

if [ "$(id -u)" -ne 0 ]; then
    echo "[ERRO] Rode com sudo: sudo $0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="$REPO_ROOT/target/goldenosx-os-build"
ROOTFS="$WORK_DIR/rootfs"
ISO_DIR="$WORK_DIR/iso"
OUT_ISO="$REPO_ROOT/GoldenOSX-OS.iso"
VGPU_BIN="$REPO_ROOT/crates/stela-vgpu-daemon/target/release/stela-vgpu-daemon"

for tool in debootstrap mksquashfs xorriso grub-mkstandalone mkfs.fat mcopy; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "[ERRO] Ferramenta '$tool' nao encontrada!"
        echo "Instale com: sudo apt-get update && sudo apt-get install -y debootstrap squashfs-tools xorriso grub-efi-amd64-bin mtools dosfstools"
        exit 1
    fi
done

if [ ! -x "$VGPU_BIN" ]; then
    echo "[ERRO] Binario stela-vgpu-daemon nao encontrado em $VGPU_BIN"
    exit 1
fi

echo "=== [1/6] Preparando diretorio de build em $WORK_DIR ==="
# Unmount any leftover bind mounts safely
for mp in dev/pts dev proc sys; do
    mountpoint -q "$ROOTFS/$mp" && umount -lf "$ROOTFS/$mp" || true
done
rm -rf "$WORK_DIR"
mkdir -p "$ROOTFS" "$ISO_DIR/live" "$ISO_DIR/boot/grub" "$ISO_DIR/EFI/BOOT"

echo "=== [2/6] Baixando Debian 13 (Trixie) minimal com debootstrap ==="
debootstrap --arch=amd64 --variant=minbase trixie "$ROOTFS" http://deb.debian.org/debian

echo "=== [3/6] Instalando Kernel, KVM, QEMU, OVMF, Cage e drivers Vulkan ==="
mount --bind /dev "$ROOTFS/dev"
mount --bind /dev/pts "$ROOTFS/dev/pts"
mount -t proc proc "$ROOTFS/proc"
mount -t sysfs sysfs "$ROOTFS/sys"

cleanup_mounts() {
    for mp in dev/pts dev proc sys; do
        mountpoint -q "$ROOTFS/$mp" && umount -lf "$ROOTFS/$mp" || true
    done
}
trap cleanup_mounts EXIT

cat > "$ROOTFS/etc/apt/sources.list" <<'EOF'
deb http://deb.debian.org/debian trixie main contrib non-free non-free-firmware
EOF

chroot "$ROOTFS" /usr/bin/env DEBIAN_FRONTEND=noninteractive bash -c '
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        linux-image-amd64 live-boot systemd-sysv udev kmod \
        qemu-system-x86 qemu-system-gui ovmf \
        cage seatd xwayland \
        libvulkan1 mesa-vulkan-drivers vulkan-tools \
        ntfs-3g pciutils usbutils util-linux kbd \
        grub-efi-amd64 efibootmgr e2fsprogs rsync \
        libdrm2 libudev1 zlib1g libzstd1 libstdc++6 libgcc-s1 libelf1 libedit2 libtinfo6 libxml2 libffi8 libbsd0 libmd0 \
        firmware-linux-free firmware-misc-nonfree intel-microcode
    apt-get clean
    rm -rf /var/lib/apt/lists/* /usr/share/doc/* /usr/share/man/*
'

echo "=== [4/6] Instalando GoldenOSX-OS (stela-vgpu-daemon, libs LLVM-21, menu TUI) ==="
mkdir -p "$ROOTFS/opt/goldenosx/bin" "$ROOTFS/opt/goldenosx/lib"
install -m 755 "$VGPU_BIN" "$ROOTFS/opt/goldenosx/bin/stela-vgpu-daemon"
install -m 755 "$SCRIPT_DIR/goldenosx-launch.sh" "$ROOTFS/opt/goldenosx/bin/goldenosx-launch.sh"
install -m 755 "$SCRIPT_DIR/goldenosx-menu.sh" "$ROOTFS/opt/goldenosx/bin/goldenosx-menu.sh"
install -m 755 "$SCRIPT_DIR/goldenosx-install.sh" "$ROOTFS/opt/goldenosx/bin/goldenosx-install.sh"
mkdir -p "$ROOTFS/opt/goldenosx/boot"
cp "$ROOTFS"/boot/vmlinuz-* "$ROOTFS/opt/goldenosx/boot/vmlinuz"
cp "$ROOTFS"/boot/initrd.img-* "$ROOTFS/opt/goldenosx/boot/initrd"

# Bundle libLLVM.so.21.1 and libxml2.so.16 so stela-vgpu-daemon runs identically on Debian 13
for soname in libLLVM.so.21.1 libxml2.so.16; do
    src=$(ldd "$VGPU_BIN" | awk -v s="$soname" '$1 == s {print $3; exit}')
    if [ -n "$src" ] && [ -e "$src" ]; then
        cp -L "$src" "$ROOTFS/opt/goldenosx/lib/$soname"
    fi
done

# Hostname, root password (empty for local console), auto-login on tty1
echo "goldenosx-os" > "$ROOTFS/etc/hostname"
chroot "$ROOTFS" passwd -d root
chroot "$ROOTFS" systemctl enable seatd || true

mkdir -p "$ROOTFS/etc/systemd/system/getty@tty1.service.d"
cat > "$ROOTFS/etc/systemd/system/getty@tty1.service.d/autologin.conf" <<'EOF'
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin root --noclear %I $TERM
EOF

cat > "$ROOTFS/root/.bash_profile" <<'EOF'
if [ "$(tty)" = "/dev/tty1" ]; then
    exec /opt/goldenosx/bin/goldenosx-menu.sh
fi
EOF

# Load KVM & GPU modules at boot
cat > "$ROOTFS/etc/modules-load.d/goldenosx.conf" <<'EOF'
kvm
kvm_intel
kvm_amd
ntfs3
EOF

# Copy kernel & initrd to ISO
cp "$ROOTFS"/boot/vmlinuz-* "$ISO_DIR/live/vmlinuz"
cp "$ROOTFS"/boot/initrd.img-* "$ISO_DIR/live/initrd"

cleanup_mounts
trap - EXIT

echo "=== [5/6] Compactando filesystem.squashfs (zstd) ==="
mksquashfs "$ROOTFS" "$ISO_DIR/live/filesystem.squashfs" \
    -comp zstd -Xcompression-level 9 -noappend -e boot

echo "=== [6/6] Gerando UEFI ISO bootavel: $OUT_ISO ==="
cat > "$ISO_DIR/boot/grub/grub.cfg" <<'EOF'
set default=0
set timeout=3

insmod part_gpt
insmod part_msdos
insmod iso9660
insmod fat
insmod search
insmod search_fs_file
insmod search_label

search --no-floppy --set=root --file /live/vmlinuz

menuentry "GoldenOSX-OS (macOS 27.2 - Monitor Externo LG)" {
    linux /live/vmlinuz boot=live toram quiet loglevel=3
    initrd /live/initrd
}

menuentry "GoldenOSX-OS (Desligar Tela Interna eDP-1)" {
    linux /live/vmlinuz boot=live toram quiet loglevel=3 video=eDP-1:d
    initrd /live/initrd
}

menuentry "GoldenOSX-OS (Verbose Boot / Debug)" {
    linux /live/vmlinuz boot=live toram
    initrd /live/initrd
}
EOF

# Create standalone UEFI GRUB binary and FAT12/16 EFI System Partition image
grub-mkstandalone \
    --format=x86_64-efi \
    --output="$ISO_DIR/EFI/BOOT/BOOTX64.EFI" \
    --locales="" \
    --fonts="" \
    "boot/grub/grub.cfg=$ISO_DIR/boot/grub/grub.cfg"

EFI_IMG="$WORK_DIR/efiboot.img"
dd if=/dev/zero of="$EFI_IMG" bs=1M count=8 status=none
mkfs.fat -F 12 "$EFI_IMG" >/dev/null
mmd -i "$EFI_IMG" ::/EFI ::/EFI/BOOT
mcopy -i "$EFI_IMG" "$ISO_DIR/EFI/BOOT/BOOTX64.EFI" ::/EFI/BOOT/BOOTX64.EFI
cp "$EFI_IMG" "$ISO_DIR/boot/grub/efiboot.img"

xorriso -as mkisofs \
    -iso-level 3 \
    -full-iso9660-filenames \
    -volid "GOLDENOSX_OS" \
    -eltorito-alt-boot \
    -e boot/grub/efiboot.img \
    -no-emul-boot \
    -isohybrid-gpt-basdat \
    -output "$OUT_ISO" \
    "$ISO_DIR"

echo ""
echo "================================================================"
echo "  SUCESSO! ISO gerada em: $OUT_ISO"
ls -lh "$OUT_ISO"
echo "================================================================"
echo "  Grave no pendrive de 8 GB com Rufus (modo DD) ou copie pro Ventoy!"
