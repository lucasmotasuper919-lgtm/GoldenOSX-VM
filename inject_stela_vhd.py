#!/usr/bin/env python3
"""Inject a newly built stela.efi into /EFI/BOOT/BOOTX64.EFI inside stela-boot.vhd."""
import os
import struct
import sys

VHD_PATH = "/mnt/c/Users/lucas/Documents/Virtual Machines/macOS 26/stela-boot.vhd"
EFI_PATH = "/home/lucas/.gemini/antigravity/scratch/stela-mod/crates/stela-boot/target/x86_64-unknown-uefi/release/stela.efi"
BACKUP_PATH = "/home/lucas/.gemini/antigravity/scratch/stela-mod/BOOTX64.v050-patched.efi.bak"


def find_fat32_partition(f):
    f.seek(0)
    mbr = f.read(512)
    # Check GPT protective MBR (type 0xEE) or MBR FAT32 partition
    for i in range(4):
        entry = mbr[446 + i * 16 : 462 + i * 16]
        ptype = entry[4]
        lba = struct.unpack_from("<I", entry, 8)[0]
        if ptype == 0xEE:
            # Parse GPT header at LBA 1
            f.seek(512)
            hdr = f.read(512)
            assert hdr[:8] == b"EFI PART", "Invalid GPT header"
            part_lba = struct.unpack_from("<Q", hdr, 72)[0]
            num_parts = struct.unpack_from("<I", hdr, 80)[0]
            part_size = struct.unpack_from("<I", hdr, 84)[0]
            f.seek(part_lba * 512)
            for _ in range(num_parts):
                p = f.read(part_size)
                if any(p[:16]):
                    first_lba = struct.unpack_from("<Q", p, 32)[0]
                    f.seek(first_lba * 512)
                    bpb = f.read(512)
                    if bpb[82:90] == b"FAT32   ":
                        return first_lba * 512
        elif ptype in (0x0B, 0x0C, 0xEF) and lba > 0:
            return lba * 512
    raise RuntimeError("FAT32 EFI partition not found in VHD")


def read_fat(f, fat_off, cluster):
    f.seek(fat_off + cluster * 4)
    return struct.unpack("<I", f.read(4))[0] & 0x0FFFFFFF


def write_fat(f, fat_offsets, cluster, value):
    for fo in fat_offsets:
        f.seek(fo + cluster * 4)
        old = struct.unpack("<I", f.read(4))[0]
        new_val = (old & 0xF0000000) | (value & 0x0FFFFFFF)
        f.seek(fo + cluster * 4)
        f.write(struct.pack("<I", new_val))


def cluster_chain(f, fat_off, start_cluster):
    chain = []
    cur = start_cluster
    while 2 <= cur < 0x0FFFFFF8:
        chain.append(cur)
        cur = read_fat(f, fat_off, cur)
    return chain


def find_dirent(f, fat_off, data_off, clus_bytes, dir_cluster, short_name):
    for clus in cluster_chain(f, fat_off, dir_cluster):
        off = data_off + (clus - 2) * clus_bytes
        f.seek(off)
        data = f.read(clus_bytes)
        for pos in range(0, len(data), 32):
            ent = data[pos : pos + 32]
            if ent[0] == 0x00:
                return None
            if ent[0] == 0xE5 or ent[11] == 0x0F or (ent[11] & 0x08):
                continue
            if ent[:11] == short_name:
                hi = struct.unpack_from("<H", ent, 20)[0]
                lo = struct.unpack_from("<H", ent, 26)[0]
                size = struct.unpack_from("<I", ent, 28)[0]
                return (off + pos, (hi << 16) | lo, size)
    return None


def main():
    efi_path = sys.argv[1] if len(sys.argv) > 1 else EFI_PATH
    with open(efi_path, "rb") as ef:
        new_efi = ef.read()
    assert new_efi[:2] == b"MZ", f"{efi_path} is not a PE/EFI binary"
    print(f"New stela.efi size: {len(new_efi):,} bytes ({len(new_efi)/1024/1024:.2f} MiB)")

    with open(VHD_PATH, "r+b") as f:
        part_off = find_fat32_partition(f)
        f.seek(part_off)
        bpb = f.read(512)
        bps = struct.unpack_from("<H", bpb, 11)[0]
        spc = bpb[13]
        rsvd = struct.unpack_from("<H", bpb, 14)[0]
        nfats = bpb[16]
        spf = struct.unpack_from("<I", bpb, 36)[0]
        root_clus = struct.unpack_from("<I", bpb, 44)[0]
        clus_bytes = bps * spc
        fat_offsets = [part_off + (rsvd + i * spf) * bps for i in range(nfats)]
        data_off = part_off + (rsvd + nfats * spf) * bps

        efi_dir = find_dirent(f, fat_offsets[0], data_off, clus_bytes, root_clus, b"EFI        ")
        assert efi_dir is not None, "/EFI not found"
        boot_dir = find_dirent(f, fat_offsets[0], data_off, clus_bytes, efi_dir[1], b"BOOT       ")
        assert boot_dir is not None, "/EFI/BOOT not found"
        target = find_dirent(f, fat_offsets[0], data_off, clus_bytes, boot_dir[1], b"BOOTX64 EFI")
        assert target is not None, "/EFI/BOOT/BOOTX64.EFI not found"

        dirent_off, start_clus, old_size = target
        chain = cluster_chain(f, fat_offsets[0], start_clus)
        print(f"Existing BOOTX64.EFI: {old_size:,} bytes across {len(chain)} clusters (cluster={clus_bytes}B)")

        # Backup existing BOOTX64.EFI once
        if not os.path.exists(BACKUP_PATH):
            old_bytes = bytearray()
            for c in chain:
                f.seek(data_off + (c - 2) * clus_bytes)
                old_bytes.extend(f.read(clus_bytes))
            with open(BACKUP_PATH, "wb") as bf:
                bf.write(old_bytes[:old_size])
            print(f"Backed up previous BOOTX64.EFI to {BACKUP_PATH}")

        needed = (len(new_efi) + clus_bytes - 1) // clus_bytes
        if needed > len(chain):
            extra = needed - len(chain)
            # Allocate free clusters starting right after chain[-1]
            cand = chain[-1] + 1
            max_clus = (spf * bps) // 4
            added = []
            while len(added) < extra and cand < max_clus:
                if read_fat(f, fat_offsets[0], cand) == 0:
                    added.append(cand)
                cand += 1
            if len(added) < extra:
                raise RuntimeError(f"Not enough free clusters: needed {extra}, found {len(added)}")
            prev = chain[-1]
            for c in added:
                write_fat(f, fat_offsets, prev, c)
                prev = c
            chain.extend(added)
            print(f"Extended BOOTX64.EFI chain by {extra} free clusters")

        # Write new EFI into the first `needed` clusters (coalescing contiguous runs for fast 9P I/O)
        idx = 0
        while idx < needed:
            run_start = idx
            while idx + 1 < needed and chain[idx + 1] == chain[idx] + 1:
                idx += 1
            run_len = idx - run_start + 1
            buf = new_efi[run_start * clus_bytes : (idx + 1) * clus_bytes]
            pad = run_len * clus_bytes - len(buf)
            if pad > 0:
                buf = buf + b"\x00" * pad
            f.seek(data_off + (chain[run_start] - 2) * clus_bytes)
            f.write(buf)
            idx += 1

        # Terminate FAT chain at cluster needed - 1 and free trailing clusters (batched per FAT)
        write_fat(f, fat_offsets, chain[needed - 1], 0x0FFFFFFF)
        trailing = chain[needed:]
        if trailing:
            # Coalesce contiguous trailing clusters into bulk zero writes
            t_idx = 0
            while t_idx < len(trailing):
                t_start = t_idx
                while t_idx + 1 < len(trailing) and trailing[t_idx + 1] == trailing[t_idx] + 1:
                    t_idx += 1
                count = t_idx - t_start + 1
                c0 = trailing[t_start]
                zeros = b"\x00" * (count * 4)
                for fo in fat_offsets:
                    f.seek(fo + c0 * 4)
                    f.write(zeros)
                t_idx += 1

        # Update file size in directory entry
        f.seek(dirent_off + 28)
        f.write(struct.pack("<I", len(new_efi)))
        f.flush()
        os.fsync(f.fileno())
        print(f"SUCCESS: Injected {len(new_efi):,} bytes ({needed} clusters) into BOOTX64.EFI!")


if __name__ == "__main__":
    main()
