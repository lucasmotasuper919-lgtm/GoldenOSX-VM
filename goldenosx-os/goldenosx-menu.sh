#!/usr/bin/env bash
set -u

# GoldenOSX-OS Interactive Console Menu (tty1)
BOOT_VMDK=""
MACOS_VMDK=""
STELA_VK_BACKEND="${STELA_VK_BACKEND:-lavapipe}"
STELA_XRES="${STELA_XRES:-1024}"
STELA_YRES="${STELA_YRES:-768}"

TOTAL_MB=$(awk '/MemTotal:/ {printf "%d", $2/1024}' /proc/meminfo)
if [ "$TOTAL_MB" -le 9000 ]; then
    DEFAULT_RAM=5120
else
    DEFAULT_RAM=$(( TOTAL_MB - 3072 ))
    [ "$DEFAULT_RAM" -gt 16384 ] && DEFAULT_RAM=16384
fi
STELA_GUEST_RAM_MB="${STELA_GUEST_RAM_MB:-$DEFAULT_RAM}"

scan_disks() {
    mkdir -p /mnt/disks
    BOOT_VMDK=""
    MACOS_VMDK=""
    echo "[GoldenOSX] Procurando discos macOS (NTFS / ext4)..."
    while read -r dev fstype; do
        [ -z "$dev" ] && continue
        case "$fstype" in
            ntfs|ext4|ext3|btrfs|exfat) ;;
            *) continue ;;
        esac
        name=$(basename "$dev")
        mp="/mnt/disks/$name"
        mkdir -p "$mp"
        if ! mountpoint -q "$mp"; then
            if [ "$fstype" = "ntfs" ]; then
                mount -t ntfs3 -o rw "$dev" "$mp" 2>/dev/null \
                    || mount -t ntfs3 -o rw,force "$dev" "$mp" 2>/dev/null \
                    || mount -t ntfs-3g -o rw "$dev" "$mp" 2>/dev/null \
                    || mount -t ntfs3 -o ro "$dev" "$mp" 2>/dev/null \
                    || continue
            else
                mount "$dev" "$mp" 2>/dev/null || continue
            fi
        fi
        echo "  -> Montado $dev ($fstype) em $mp"
        # Fast direct check in Windows Virtual Machines folders first (<0.1s)
        for dir in "$mp"/Users/*/Documents/"Virtual Machines"/* "$mp"/"Virtual Machines"/* "$mp"/macOS* "$mp"; do
            [ -d "$dir" ] || continue
            b=$(find "$dir" -maxdepth 2 -iname "stela-boot.vmdk" 2>/dev/null | head -n 1 || true)
            m=$(find "$dir" -maxdepth 2 -iname "macOS*.vmdk" ! -iname "*stela*" ! -iname "*-s[0-9]*.vmdk" ! -iname "*-flat.vmdk" ! -iname "*-f[0-9]*.vmdk" 2>/dev/null | head -n 1 || true)
            if [ -n "$b" ] && [ -n "$m" ]; then
                BOOT_VMDK="$b"
                MACOS_VMDK="$m"
                return 0
            fi
        done
        # Deeper fallback search (up to depth 7)
        found_boot=$(find "$mp" -maxdepth 7 -iname "stela-boot.vmdk" 2>/dev/null | head -n 1 || true)
        found_macos=$(find "$mp" -maxdepth 7 -iname "macOS*.vmdk" ! -iname "*stela*" ! -iname "*-s[0-9]*.vmdk" ! -iname "*-flat.vmdk" ! -iname "*-f[0-9]*.vmdk" 2>/dev/null | head -n 1 || true)
        if [ -n "$found_boot" ] && [ -n "$found_macos" ]; then
            BOOT_VMDK="$found_boot"
            MACOS_VMDK="$found_macos"
            return 0
        fi
    done < <(lsblk -rn -o PATH,FSTYPE | awk '$2 != ""')
    return 1
}

scan_disks || true

while true; do
    clear
    CPU_MODEL=$(awk -F: '/model name/ {gsub(/^ +/, "", $2); print $2; exit}' /proc/cpuinfo)
    KVM_OK="NAO"; [ -e /dev/kvm ] && KVM_OK="SIM (/dev/kvm)"
    DRM_OK="NAO"; [ -e /dev/dri/renderD128 ] && DRM_OK="SIM (/dev/dri/renderD128)"

    echo "================================================================"
    echo "              GoldenOSX-OS  (Debian 13 Minimal)"
    echo "         Apple Silicon macOS 27.2 + Stela Metal vGPU"
    echo "================================================================"
    echo "  CPU:       $CPU_MODEL ($(nproc) threads)"
    echo "  RAM Total: ${TOTAL_MB} MB  |  RAM macOS: ${STELA_GUEST_RAM_MB} MB"
    echo "  KVM:       $KVM_OK  |  GPU DRM: $DRM_OK"
    echo "  Vulkan:    $STELA_VK_BACKEND  |  Resolucao: ${STELA_XRES}x${STELA_YRES}"
    echo "  Boot VMDK: ${BOOT_VMDK:-<nao encontrado>}"
    echo "  macOS VMDK:${MACOS_VMDK:-<nao encontrado>}"
    echo "================================================================"
    echo ""
    echo "  [1] Iniciar macOS 27.2 (Tela Cheia)"
    echo "  [2] Alterar RAM do macOS (atual: ${STELA_GUEST_RAM_MB} MB)"
    echo "  [3] Alterar Resolucao (atual: ${STELA_XRES}x${STELA_YRES})"
    echo "  [4] Alterar Backend Vulkan (atual: ${STELA_VK_BACKEND})"
    echo "  [5] Re-escanear Discos (NTFS / ext4)"
    echo "  [6] Ver Logs (vGPU + Serial)"
    echo "  [7] Abrir Shell (bash)"
    echo "  [8] Reiniciar"
    echo "  [9] Desligar"
    echo "  [0] Instalar GoldenOSX-OS no SSD (Dual-Boot com Windows)"
    echo ""
    read -rp "Escolha uma opcao [0-9]: " opt
    case "$opt" in
        0)
            /opt/goldenosx/bin/goldenosx-install.sh || true
            read -rp "Pressione Enter para voltar ao menu..."
            ;;
        1)
            if [ -z "$BOOT_VMDK" ] || [ -z "$MACOS_VMDK" ]; then
                echo "[ERRO] Discos stela-boot.vmdk / macOS.vmdk nao encontrados!"
                echo "Dica: desative a Inicializacao Rapida no Windows se o NTFS estiver bloqueado."
                read -rp "Pressione Enter..."
            else
                export STELA_GUEST_RAM_MB STELA_XRES STELA_YRES STELA_VK_BACKEND
                /opt/goldenosx/bin/goldenosx-launch.sh "$BOOT_VMDK" "$MACOS_VMDK"
                echo ""
                echo "VM encerrada."
                read -rp "Pressione Enter para voltar ao menu..."
            fi
            ;;
        2)
            read -rp "RAM para o macOS em MB (ex: 4096, 5120, 6144): " new_ram
            if [[ "$new_ram" =~ ^[0-9]+$ ]] && [ "$new_ram" -ge 4096 ]; then
                STELA_GUEST_RAM_MB="$new_ram"
            fi
            ;;
        3)
            echo "  1) 800x600   2) 1024x768   3) 1280x800"
            read -rp "Escolha [1-3]: " r
            case "$r" in
                1) STELA_XRES=800; STELA_YRES=600 ;;
                2) STELA_XRES=1024; STELA_YRES=768 ;;
                3) STELA_XRES=1280; STELA_YRES=800 ;;
            esac
            ;;
        4)
            echo "  1) auto (GPU fisica se compativel, senao Lavapipe)"
            echo "  2) hardware (forcar GPU fisica via libvulkan.so.1)"
            echo "  3) lavapipe (Vulkan 1.4 CPU)"
            read -rp "Escolha [1-3]: " b
            case "$b" in
                1) STELA_VK_BACKEND=auto ;;
                2) STELA_VK_BACKEND=hardware ;;
                3) STELA_VK_BACKEND=lavapipe ;;
            esac
            ;;
        5)
            scan_disks || true
            read -rp "Scan concluido. Pressione Enter..."
            ;;
        6)
            echo "=== /var/log/goldenosx-vgpu.log (ultimas 25 linhas) ==="
            tail -n 25 /var/log/goldenosx-vgpu.log 2>/dev/null || echo "(vazio)"
            echo "=== /var/log/goldenosx-serial.log (GUEST SESSION) ==="
            grep "GUEST SESSION" /var/log/goldenosx-serial.log 2>/dev/null | tail -n 5 || echo "(vazio)"
            read -rp "Pressione Enter..."
            ;;
        7)
            bash
            ;;
        8)
            sync
            umount /mnt/disks/* 2>/dev/null || true
            reboot -f
            ;;
        9)
            sync
            umount /mnt/disks/* 2>/dev/null || true
            poweroff -f
            ;;
    esac
done
