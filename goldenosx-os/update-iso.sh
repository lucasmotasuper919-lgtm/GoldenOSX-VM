#!/usr/bin/env bash
set -euo pipefail

# Fast repack of GoldenOSX-OS.iso after editing scripts/daemon (~25 seconds, no debootstrap!)
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

if [ ! -d "$ROOTFS/opt/goldenosx/bin" ]; then
    echo "[ERRO] Build base nao encontrado em $ROOTFS. Rode sudo ./goldenosx-os/build-iso.sh primeiro."
    exit 1
fi

echo "=== [1/3] Atualizando scripts, pacotes do instalador e binario no rootfs ==="
if [ ! -e "$ROOTFS/usr/lib/x86_64-linux-gnu/qemu/ui-gtk.so" ] || [ ! -x "$ROOTFS/usr/bin/rsync" ] || [ ! -x "$ROOTFS/usr/sbin/grub-install" ]; then
    echo "  -> Instalando qemu-system-gui + grub-efi-amd64 + efibootmgr + e2fsprogs + rsync no Debian 13 rootfs..."
    mount --bind /dev "$ROOTFS/dev"
    mount --bind /dev/pts "$ROOTFS/dev/pts"
    mount -t proc proc "$ROOTFS/proc"
    mount -t sysfs sysfs "$ROOTFS/sys"
    chroot "$ROOTFS" /usr/bin/env DEBIAN_FRONTEND=noninteractive bash -c '
        apt-get update -qq && apt-get install -y --no-install-recommends qemu-system-gui grub-efi-amd64 efibootmgr e2fsprogs rsync && apt-get clean && rm -rf /var/lib/apt/lists/*
    '
    for mp in dev/pts dev proc sys; do
        mountpoint -q "$ROOTFS/$mp" && umount -lf "$ROOTFS/$mp" || true
    done
fi
if ! chroot "$ROOTFS" qemu-system-x86_64 -display help 2>&1 | grep -q "gtk"; then
    echo "[ERRO] QEMU ainda nao tem suporte a display gtk!"
    exit 1
fi
echo "  -> QEMU display gtk + ferramentas do instalador verificadas OK!"

install -m 755 "$VGPU_BIN" "$ROOTFS/opt/goldenosx/bin/stela-vgpu-daemon"
install -m 755 "$SCRIPT_DIR/goldenosx-launch.sh" "$ROOTFS/opt/goldenosx/bin/goldenosx-launch.sh"
install -m 755 "$SCRIPT_DIR/goldenosx-menu.sh" "$ROOTFS/opt/goldenosx/bin/goldenosx-menu.sh"
install -m 755 "$SCRIPT_DIR/goldenosx-install.sh" "$ROOTFS/opt/goldenosx/bin/goldenosx-install.sh"
mkdir -p "$ROOTFS/opt/goldenosx/boot"
cp "$ISO_DIR/live/vmlinuz" "$ROOTFS/opt/goldenosx/boot/vmlinuz"
cp "$ISO_DIR/live/initrd" "$ROOTFS/opt/goldenosx/boot/initrd"

echo "=== [1.5/3] Self-test do stela-vgpu-daemon DENTRO do Debian 13 chroot ==="
mkdir -p "$ROOTFS/dev/shm"
SELF_OUT=$(chroot "$ROOTFS" /usr/bin/env LD_LIBRARY_PATH=/opt/goldenosx/lib STELA_VK_BACKEND=lavapipe \
    timeout 3 /opt/goldenosx/bin/stela-vgpu-daemon 2>&1 || true)
if echo "$SELF_OUT" | grep -q "REAL METAL vGPU READY"; then
    echo "  -> SELF-TEST OK: stela-vgpu-daemon iniciou 100% limpo no Debian 13!"
else
    echo "  -> [ERRO] Self-test falhou! Saida:"
    echo "$SELF_OUT"
    exit 1
fi
rm -f "$ROOTFS/dev/shm/stela-vgpu-shm"

echo "=== [2/3] Recompactando filesystem.squashfs (zstd) ==="
mksquashfs "$ROOTFS" "$ISO_DIR/live/filesystem.squashfs" \
    -comp zstd -Xcompression-level 9 -noappend -e boot

echo "=== [3/3] Gerando UEFI ISO + copiando para Downloads ==="
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

if [ -d /mnt/c/Users/lucas/Downloads ]; then
    cp "$OUT_ISO" /mnt/c/Users/lucas/Downloads/GoldenOSX-OS.iso
fi

echo "PRONTO! ISO atualizada em /mnt/c/Users/lucas/Downloads/GoldenOSX-OS.iso"
ls -lh "$OUT_ISO"
