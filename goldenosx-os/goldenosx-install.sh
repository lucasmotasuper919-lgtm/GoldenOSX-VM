#!/usr/bin/env bash
set -euo pipefail

# GoldenOSX-OS Safe Dual-Boot Installer (UEFI + ext4 + 4 GB swapfile + GRUB)
# NEVER touches Windows NTFS partitions; installs ONLY into a user-confirmed Linux/empty partition.

if [ "$(id -u)" -ne 0 ]; then
    echo "[ERRO] Precisa rodar como root."
    exit 1
fi

clear
echo "================================================================"
echo "       GoldenOSX-OS — Instalador Dual-Boot Seguro (UEFI)"
echo "================================================================"
echo ""
echo "Este instalador copia o GoldenOSX-OS para UMA particao vazia do"
echo "seu SSD (recomendado: 10 GB+), cria 4 GB de swap (adeus travamentos"
echo "de RAM!) e adiciona o GoldenOSX-OS ao boot UEFI junto com o Windows."
echo ""
echo "Particoes detectadas no seu PC:"
echo "----------------------------------------------------------------"
lsblk -o NAME,SIZE,FSTYPE,LABEL,PARTTYPENAME,MOUNTPOINT
echo "----------------------------------------------------------------"
echo ""
echo "DICA: Crie uma particao vazia de ~10-15 GB no Gerenciamento de Disco"
echo "do Windows (ou via cfdisk/parted) ANTES de escolher abaixo."
echo ""
read -rp "Digite o dispositivo da particao de destino (ex: /dev/nvme0n1p5 ou /dev/sda5, ou 'q' para sair): " TARGET_PART

if [ "$TARGET_PART" = "q" ] || [ -z "$TARGET_PART" ]; then
    echo "Cancelado."
    exit 0
fi

if [ ! -b "$TARGET_PART" ]; then
    echo "[ERRO] '$TARGET_PART' nao e um dispositivo de bloco valido!"
    exit 1
fi

# Safety check: refuse Windows NTFS / EFI System Partitions!
FS_CUR=$(lsblk -no FSTYPE "$TARGET_PART" | head -n 1 || true)
PART_TYPE=$(lsblk -no PARTTYPENAME "$TARGET_PART" | head -n 1 || true)
PART_SIZE_BYTES=$(lsblk -bno SIZE "$TARGET_PART" | head -n 1 || echo 0)
PART_SIZE_GB=$(( PART_SIZE_BYTES / 1024 / 1024 / 1024 ))

if [ "$FS_CUR" = "ntfs" ]; then
    echo "[BLOQUEADO] '$TARGET_PART' e uma particao NTFS (Windows)!"
    echo "Por seguranca, o instalador NAO formata particoes NTFS."
    echo "Diminua o C: no Windows e crie uma particao vazia/FAT/ext4 de 10+ GB."
    exit 1
fi

if echo "$PART_TYPE" | grep -qi "EFI System"; then
    echo "[BLOQUEADO] '$TARGET_PART' e a particao EFI do sistema!"
    exit 1
fi

if [ "$PART_SIZE_GB" -lt 8 ]; then
    echo "[ERRO] A particao precisa ter pelo menos 8 GB (recomendado 10+ GB). Atual: ${PART_SIZE_GB} GB."
    exit 1
fi

# Detect EFI System Partition automatically
EFI_PART=$(lsblk -rn -o PATH,FSTYPE,PARTTYPE | awk 'tolower($3) ~ /c12a7328-f81f-11d2-ba4b-00a0c93ec93b/ {print $1; exit}')
if [ -z "$EFI_PART" ]; then
    EFI_PART=$(lsblk -rn -o PATH,FSTYPE | awk '$2 == "vfat" {print $1; exit}')
fi
if [ -z "$EFI_PART" ]; then
    echo "[ERRO] Particao EFI (FAT32) nao encontrada!"
    exit 1
fi

echo ""
echo "================================================================"
echo "  RESUMO DA INSTALACAO:"
echo "  - Particao raiz (sera formatada em ext4): $TARGET_PART (${PART_SIZE_GB} GB, atual: ${FS_CUR:-vazia})"
echo "  - Particao EFI (NAO sera formatada!):     $EFI_PART"
echo "  - Swapfile automatico no SSD:             4 GB (/swapfile)"
echo "================================================================"
read -rp "Tem certeza que deseja formatar $TARGET_PART em ext4? Digite SIM: " CONFIRM
if [ "$CONFIRM" != "SIM" ]; then
    echo "Cancelado."
    exit 0
fi

umount -lf "$TARGET_PART" 2>/dev/null || true

echo ""
echo "=== [1/5] Formatando $TARGET_PART em ext4 (GOLDENOSX_ROOT) ==="
mkfs.ext4 -F -L GOLDENOSX_ROOT "$TARGET_PART"

MNT="/mnt/goldenosx-install"
mkdir -p "$MNT"
mount "$TARGET_PART" "$MNT"

cleanup() {
    for mp in dev/pts dev proc sys boot/efi; do
        mountpoint -q "$MNT/$mp" && umount -lf "$MNT/$mp" || true
    done
    mountpoint -q "$MNT" && umount -lf "$MNT" || true
}
trap cleanup EXIT

echo "=== [2/5] Copiando sistema GoldenOSX-OS para o SSD ==="
rsync -aAX --info=progress2 \
    --exclude='/dev/*' \
    --exclude='/proc/*' \
    --exclude='/sys/*' \
    --exclude='/tmp/*' \
    --exclude='/run/*' \
    --exclude='/mnt/*' \
    --exclude='/media/*' \
    --exclude='/lost+found' \
    --exclude='/swapfile' \
    / "$MNT/"

mkdir -p "$MNT"/{dev,proc,sys,tmp,run,mnt,media,boot/efi}
chmod 1777 "$MNT/tmp"

# Copy kernel & initrd into /boot on the installed SSD if not present
if ! ls "$MNT"/boot/vmlinuz-* >/dev/null 2>&1; then
    if [ -f /opt/goldenosx/boot/vmlinuz ] && [ -f /opt/goldenosx/boot/initrd ]; then
        cp /opt/goldenosx/boot/vmlinuz "$MNT/boot/vmlinuz-goldenosx"
        cp /opt/goldenosx/boot/initrd "$MNT/boot/initrd.img-goldenosx"
    elif [ -f /run/live/medium/live/vmlinuz ]; then
        cp /run/live/medium/live/vmlinuz "$MNT/boot/vmlinuz-goldenosx"
        cp /run/live/medium/live/initrd "$MNT/boot/initrd.img-goldenosx"
    else
        echo "[ERRO] Kernel vmlinuz/initrd nao encontrado!"
        exit 1
    fi
fi

echo "=== [3/5] Criando /swapfile de 4 GB no SSD (evita travamentos de RAM!) ==="
fallocate -l 4G "$MNT/swapfile" || dd if=/dev/zero of="$MNT/swapfile" bs=1M count=4096 status=progress
chmod 600 "$MNT/swapfile"
mkswap "$MNT/swapfile"

ROOT_UUID=$(blkid -s UUID -o value "$TARGET_PART")
EFI_UUID=$(blkid -s UUID -o value "$EFI_PART")

cat > "$MNT/etc/fstab" <<EOF
UUID=$ROOT_UUID  /          ext4  defaults,noatime  0 1
UUID=$EFI_UUID   /boot/efi  vfat  umask=0077        0 2
/swapfile        none       swap  sw                0 0
EOF

# Tune swappiness so macOS RAM stays fast while preventing OOM freezes
mkdir -p "$MNT/etc/sysctl.d"
cat > "$MNT/etc/sysctl.d/99-goldenosx.conf" <<'EOF'
vm.swappiness=20
vm.vfs_cache_pressure=50
EOF

echo "=== [4/5] Instalando GRUB UEFI (Dual-Boot com Windows) ==="
mount "$EFI_PART" "$MNT/boot/efi"
mount --bind /dev "$MNT/dev"
mount --bind /dev/pts "$MNT/dev/pts"
mount -t proc proc "$MNT/proc"
mount -t sysfs sysfs "$MNT/sys"

chroot "$MNT" grub-install --target=x86_64-efi --efi-directory=/boot/efi --bootloader-id=GoldenOSX-OS --recheck

VMLINUZ=$(basename "$(ls "$MNT"/boot/vmlinuz-* | head -n 1)")
INITRD=$(basename "$(ls "$MNT"/boot/initrd.img-* | head -n 1)")

mkdir -p "$MNT/boot/grub"
cat > "$MNT/boot/grub/grub.cfg" <<EOF
set default=0
set timeout=5

insmod part_gpt
insmod ext2
insmod fat
insmod chain
insmod search_fs_uuid

menuentry "GoldenOSX-OS (macOS 27.2 KVM + Metal vGPU)" {
    search --no-floppy --fs-uuid --set=root $ROOT_UUID
    linux /boot/$VMLINUZ root=UUID=$ROOT_UUID rw quiet loglevel=3
    initrd /boot/$INITRD
}

menuentry "GoldenOSX-OS (Verbose Boot / Debug)" {
    search --no-floppy --fs-uuid --set=root $ROOT_UUID
    linux /boot/$VMLINUZ root=UUID=$ROOT_UUID rw
    initrd /boot/$INITRD
}

menuentry "Windows Boot Manager" {
    search --no-floppy --fs-uuid --set=root $EFI_UUID
    chainloader /EFI/Microsoft/Boot/bootmgfw.efi
}
EOF

# Write full menu directly to EFI/GoldenOSX-OS/grub.cfg too so GRUB always shows the visual menu
mkdir -p "$MNT/boot/efi/EFI/GoldenOSX-OS"
cp "$MNT/boot/grub/grub.cfg" "$MNT/boot/efi/EFI/GoldenOSX-OS/grub.cfg"

echo "=== [5/5] Sincronizando discos ==="
sync
cleanup
trap - EXIT

echo ""
echo "================================================================"
echo "  INSTALACAO CONCLUIDA COM SUCESSO!"
echo "  - GoldenOSX-OS instalado em $TARGET_PART (ext4 + 4 GB swap)"
echo "  - Entrada UEFI 'GoldenOSX-OS' criada + Windows Boot Manager!"
echo "  - Agora pode remover o pendrive e reiniciar!"
echo "================================================================"
