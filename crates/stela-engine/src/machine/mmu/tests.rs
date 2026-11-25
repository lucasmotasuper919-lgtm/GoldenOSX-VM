use super::*;

#[test]
fn par_reports_the_output_attributes_or_the_fault() {
    // PAR_EL1.PA holds the chip's 48-bit physical address.
    test_translation_chip();
    let translation = Translation {
        physical: 0x8_1234_5678,
        memory: MemoryType::Normal,
        attributes: 0xff,
        shareability: 0b11,
        readable: true,
        writable: true,
        executable: false,
        pan_sensitive: false,
        guarded: false,
        global: true,
        asid: 0,
        size: 1 << 14,
        alignment: Alignment::default(),
    };
    assert_eq!(
        par(&Outcome::Mapped(translation)),
        Some(0xff00_0008_1234_5000 | 1 << 11 | 1 << 9 | 0b11 << 7)
    );
    let device = Translation {
        attributes: 0x04,
        shareability: 0b10,
        memory: MemoryType::Device,
        ..translation
    };
    assert_eq!(
        par(&Outcome::Mapped(device)),
        Some(0x0400_0008_1234_5000 | 1 << 11 | 1 << 9 | 0b10 << 7)
    );
    let fault = Fault::stage1(FaultKind::Permission, 3);
    assert_eq!(
        par(&Outcome::Fault(fault)),
        Some(1 << 11 | 0b00_1111 << 1 | 1)
    );
    let second_stage = |walk| Fault {
        kind: FaultKind::Translation,
        level: 2,
        second_stage: Some(SecondStage {
            ipa: 0x8000_4000,
            walk,
        }),
    };
    assert_eq!(
        par(&Outcome::Fault(second_stage(false))),
        Some(1 << 11 | 1 << 9 | 0b00_0110 << 1 | 1)
    );
    assert_eq!(
        par(&Outcome::Fault(second_stage(true))),
        Some(1 << 11 | 1 << 9 | 1 << 8 | 0b00_0110 << 1 | 1)
    );
    assert_eq!(par(&Outcome::Unsupported(Unsupported::Stage2)), None);
}

#[test]
fn ctrr_permission_faults_preserve_allowed_reads_at_and_physical_boundaries() {
    let protected = (0x8000_4000, 0x8000_4fff);
    let mut controls = Controls {
        ctrr: Ctrr {
            privileged: [
                Some(CtrrRegion {
                    first: protected.0,
                    last: protected.1,
                    write_protect: true,
                    execute_inside_only: true,
                }),
                None,
            ],
            memory: [None; 4],
        },
        ..Controls::default()
    };
    let mut tables = Tables::default();
    let permission = Outcome::Fault(Fault::stage1(FaultKind::Permission, 3));
    assert_eq!(
        translate(&controls, protected.0, Access::Write, KERNEL, &mut tables),
        permission
    );
    assert_eq!(
        translate(
            &controls,
            protected.1 + 1,
            Access::Execute,
            KERNEL,
            &mut tables
        ),
        permission
    );
    let Outcome::Mapped(read) =
        translate(&controls, protected.0, Access::Read, KERNEL, &mut tables)
    else {
        panic!("CTRR must permit reads");
    };
    assert!(!read.writable && read.executable);
    for (address, access) in [
        (protected.0 - 1, Access::Write),
        (protected.1 + 1, Access::Write),
        (protected.0, Access::Execute),
    ] {
        assert!(matches!(
            translate(&controls, address, access, KERNEL, &mut tables),
            Outcome::Mapped(_)
        ));
    }
    // AT reports the translation tables, not the subsequent physical
    // core protection. Core CTRR does not restrict an EL0 requester.
    assert!(matches!(
        translate_tables(&controls, protected.0, Access::Write, KERNEL, &mut tables),
        Outcome::Mapped(_)
    ));
    let user = Requester { el: 0, ..KERNEL };
    assert!(matches!(
        translate(&controls, protected.0, Access::Write, user, &mut tables),
        Outcome::Mapped(_)
    ));
    // Memory-controller locks bind writes from EL0 as well.
    controls.ctrr.privileged = [None; 2];
    controls.ctrr.memory[0] = Some(protected);
    assert_eq!(
        translate(&controls, protected.0, Access::Write, user, &mut tables),
        permission
    );
    assert!(matches!(
        translate(&controls, protected.0, Access::Read, user, &mut tables),
        Outcome::Mapped(_)
    ));
}

#[test]
fn ctrr_checks_the_final_pa_and_does_not_report_a_stage_two_walk_fault() {
    let (mut controls, mut tables) = two_stages(0x4000_0000);
    controls.ctrr.memory[0] = Some((0x8000_4000, 0x8000_4fff));
    let address = 0x4008;
    let Outcome::Fault(fault) = translate(&controls, address, Access::Write, KERNEL, &mut tables)
    else {
        panic!("the final physical CTRR region must deny the store");
    };
    assert_eq!((fault.kind, fault.level), (FaultKind::Permission, 3));
    assert_eq!(fault.data_syndrome(true, false), 0x4f);
    assert_eq!(fault.instruction_syndrome(), 0x0f);
    assert!(!fault.to_el2());
    assert_eq!(fault.hpfar(), None);
    assert!(matches!(
        translate_tables(&controls, address, Access::Write, KERNEL, &mut tables),
        Outcome::Mapped(Translation {
            physical: 0x8000_4008,
            ..
        })
    ));
}

#[derive(Default)]
struct Tables(alloc::collections::BTreeMap<u64, u64>);

impl TableMemory for Tables {
    fn read_descriptor(&mut self, physical: u64) -> Option<u64> {
        (physical < 0x1_0000_0000).then(|| self.0.get(&physical).copied().unwrap_or(0))
    }
    fn update_descriptor(&mut self, _: u64, _: u64, _: u64) -> Option<bool> {
        unreachable!("the profile has no hardware descriptor updates")
    }
}

const KERNEL: Requester = Requester {
    el: 1,
    pan: false,
    unprivileged: false,
    guarded: false,
};
/// Stage 2: 4 KiB, 40-bit IPA from two concatenated level-1 tables at
/// PA 0x10_0000, mapping IPA 1-2 GiB to PA 2-3 GiB as a Normal
/// Write-Back, Inner Shareable, read/write block.
const VTCR: u64 = 24 | 1 << 6 | 5 << 16 | 1 << 31;
const STAGE2_BLOCK: u64 = 0x8000_0000 | 1 | 1 << 10 | 0b11 << 6 | 0b1111 << 2 | 0b11 << 8;
/// Stage 1: 4 KiB, 39-bit VA from level 1 (EPD1), 48-bit IPS.
const TCR: u64 = 25 | 1 << 23 | 5 << 32;

/// Stage-1 table at IPA 0x4000_0000 (PA 0x8000_0000) whose first entry
/// maps VA 0-1 GiB as a block at `output`.
fn two_stages(output: u64) -> (Controls, Tables) {
    let mut tables = Tables::default();
    tables.0.insert(0x10_0008, STAGE2_BLOCK);
    tables
        .0
        .insert(0x8000_0000, output | 1 | 1 << 10 | 0b11 << 8);
    let controls = Controls {
        sctlr: SCTLR_M | alignment::SCTLR_C,
        tcr: TCR,
        ttbr0: 0x4000_0000,
        mair: 0xff,
        stage2: true,
        second_stage: Some(Stage2 {
            hcr: HCR_VM,
            vtcr: VTCR,
            vttbr: 0x10_0000,
            xs: test_xs(),
        }),
        xs: test_xs(),
        ..Controls::default()
    };
    (controls, tables)
}

#[test]
fn stage_one_tables_and_output_translate_through_stage_two() {
    let (controls, mut tables) = two_stages(0x4000_0000);
    let Outcome::Mapped(full) = translate(&controls, 0x1234, Access::Write, KERNEL, &mut tables)
    else {
        panic!("full translation")
    };
    assert_eq!(full.physical, 0x8000_1234);
    assert_eq!(
        (full.attributes, full.shareability, full.size),
        (0xff, 0b11, 1 << 30)
    );
    let Outcome::Mapped(first) =
        translate_stage1(&controls, 0x1234, Access::Read, KERNEL, &mut tables)
    else {
        panic!("stage 1 alone")
    };
    assert_eq!(first.physical, 0x4000_1234);

    // An output IPA stage 2 does not map: a stage-2 Translation fault on
    // the access, reported with its IPA.
    let (controls, mut tables) = two_stages(0xc000_0000);
    let Outcome::Fault(fault) = translate(&controls, 0x1234, Access::Read, KERNEL, &mut tables)
    else {
        panic!("stage-2 fault")
    };
    assert_eq!((fault.kind, fault.level), (FaultKind::Translation, 1));
    assert_eq!(
        fault.second_stage,
        Some(SecondStage {
            ipa: 0xc000_1234,
            walk: false
        })
    );
    assert_eq!(fault.hpfar(), Some(0x000c_0001 << 4));

    // A stage-1 table at an IPA stage 2 does not map: a stage-2 fault
    // on the walk, for AT S1E* as well.
    let (mut controls, mut tables) = two_stages(0x4000_0000);
    controls.ttbr0 = 0;
    let walk = Fault {
        kind: FaultKind::Translation,
        level: 1,
        second_stage: Some(SecondStage { ipa: 0, walk: true }),
    };
    assert_eq!(
        translate(&controls, 0x1234, Access::Read, KERNEL, &mut tables),
        Outcome::Fault(walk)
    );
    assert_eq!(
        translate_stage1(&controls, 0x1234, Access::Read, KERNEL, &mut tables),
        Outcome::Fault(walk)
    );
    assert_eq!(
        par(&Outcome::Fault(walk)),
        Some(1 << 11 | 1 << 9 | 1 << 8 | 0b00_0101 << 1 | 1)
    );
}

#[test]
fn pan_shadow_classification_includes_stage_two_read_permissions() {
    let (controls, mut tables) = two_stages(0x8000_0000);
    // The table walk still uses the original read/write block. The data
    // access uses the next block, writable but unreadable at stage 2.
    tables.0.insert(0x10_0010, STAGE2_BLOCK & !(1 << 6));
    let Outcome::Mapped(mapped) = translate(&controls, 0x1234, Access::Write, KERNEL, &mut tables)
    else {
        panic!("stage 2 must permit the write");
    };
    assert!(!mapped.readable && mapped.writable && !mapped.pan_sensitive);
    let pan = Requester {
        pan: true,
        ..KERNEL
    };
    let Outcome::Fault(fault) = translate(&controls, 0x1234, Access::Read, pan, &mut tables) else {
        panic!("stage 2 must deny the read");
    };
    assert_eq!(fault.kind, FaultKind::Permission);
    assert!(fault.second_stage.is_some());
}

#[test]
fn device_tlb_uses_stage_two_permissions_and_rewalks_after_maintenance() {
    let (controls, mut tables) = two_stages(0x8000_0000);
    let slot = 0x10_0010;
    let address = 0x1234;
    let mut tlb = translated::Tlb::default();
    // The first-stage walk remains readable. The data output is write-only.
    tables.0.insert(slot, STAGE2_BLOCK & !(1 << 6));
    let Outcome::Mapped(mapped) = translate(&controls, address, Access::Write, KERNEL, &mut tables)
    else {
        panic!("stage 2 must permit the write");
    };
    tlb.record_device(address, KERNEL, mapped);
    assert_eq!(
        tlb.device_physical(address, KERNEL, true),
        Some(mapped.physical)
    );
    assert_eq!(tlb.device_physical(address, KERNEL, false), None);
    assert!(matches!(
        translate(&controls, address, Access::Read, KERNEL, &mut tables),
        Outcome::Fault(Fault {
            kind: FaultKind::Permission,
            second_stage: Some(_),
            ..
        })
    ));

    // Clearing AF after caching requires maintenance. The next translation
    // reports the fault instead of writing AF back or reusing the old entry.
    tables.0.insert(slot, STAGE2_BLOCK & !(1 << 10));
    tlb.clear();
    assert_eq!(tlb.device_physical(address, KERNEL, true), None);
    assert!(matches!(
        translate(&controls, address, Access::Write, KERNEL, &mut tables),
        Outcome::Fault(Fault {
            kind: FaultKind::AccessFlag,
            second_stage: Some(_),
            ..
        })
    ));

    tables.0.insert(slot, STAGE2_BLOCK);
    *tables.0.get_mut(&0x8000_0000).unwrap() &= !(1 << 10);
    assert!(matches!(
        translate(&controls, address, Access::Write, KERNEL, &mut tables),
        Outcome::Fault(Fault {
            kind: FaultKind::AccessFlag,
            second_stage: None,
            ..
        })
    ));
}

#[test]
fn device_tlb_cannot_admit_writes_to_a_ctrr_locked_page() {
    let (mut controls, mut tables) = two_stages(0x4000_0000);
    controls.ctrr.memory[0] = Some((0x8000_4000, 0x8000_4fff));
    let address = 0x4008;
    let Outcome::Mapped(mapped) = translate(&controls, address, Access::Read, KERNEL, &mut tables)
    else {
        panic!("CTRR must permit the read");
    };
    let mut tlb = translated::Tlb::default();
    tlb.record_device(address, KERNEL, mapped);
    assert_eq!(
        tlb.device_physical(address, KERNEL, false),
        Some(mapped.physical)
    );
    assert_eq!(tlb.device_physical(address, KERNEL, true), None);
    assert!(matches!(
        translate(&controls, address, Access::Write, KERNEL, &mut tables),
        Outcome::Fault(Fault {
            kind: FaultKind::Permission,
            second_stage: None,
            ..
        })
    ));
}

#[test]
fn stage_two_applies_under_vm_or_dc_without_tge_and_is_never_skipped() {
    let (mut controls, mut tables) = two_stages(0x4000_0000);
    // HCR_EL2.DC alone: stage 1 behaves as disabled with Normal
    // Write-Back Non-shareable output, and stage 2 applies.
    controls.second_stage = Some(Stage2 {
        hcr: HCR_DC,
        ..controls.second_stage.unwrap()
    });
    let Outcome::Mapped(default) =
        translate(&controls, 0x4000_1234, Access::Read, KERNEL, &mut tables)
    else {
        panic!("HCR_EL2.DC")
    };
    assert_eq!(default.physical, 0x8000_1234);
    assert_eq!((default.attributes, default.shareability), (0xff, 0b11));
    // HCR_EL2.DCT makes that default Tagged where the chip has
    // FEAT_MTE2 (AArch64.S1DisabledOutput), through stage 2's
    // Write-Back memory.
    controls.second_stage = Some(Stage2 {
        hcr: HCR_DC | HCR_DCT,
        ..controls.second_stage.unwrap()
    });
    for tagged in [false, true] {
        controls.tagged = tagged;
        let Outcome::Mapped(default) =
            translate(&controls, 0x4000_1234, Access::Read, KERNEL, &mut tables)
        else {
            panic!("HCR_EL2.DCT")
        };
        assert_eq!(
            default.attributes,
            if tagged { attributes::TAGGED } else { 0xff }
        );
    }
    // A descriptor selecting MAIR's 0xf0 is Tagged through stage 2's
    // Write-Back block; stage-2 Non-cacheable memory is Untagged.
    controls.mair = u64::from(attributes::TAGGED);
    for (block, expected) in [
        (STAGE2_BLOCK, attributes::TAGGED),
        (
            STAGE2_BLOCK & !(0b1111 << 2) | 0b0101 << 2,
            attributes::non_cacheable(attributes::TAGGED, test_xs()),
        ),
    ] {
        tables.0.insert(0x10_0008, block);
        controls.second_stage = Some(Stage2 {
            hcr: HCR_VM,
            ..controls.second_stage.unwrap()
        });
        let Outcome::Mapped(tagged) =
            translate(&controls, 0x1234, Access::Read, KERNEL, &mut tables)
        else {
            panic!("Tagged stage 1")
        };
        assert_eq!(tagged.attributes, expected);
    }
    tables.0.insert(0x10_0008, STAGE2_BLOCK);
    controls.mair = 0xff;
    controls.tagged = false;
    controls.second_stage = Some(Stage2 {
        hcr: HCR_DC,
        ..controls.second_stage.unwrap()
    });
    // TGE set: the EL1&0 regime has no stage 2, so TTBR0 is a PA.
    tables.0.insert(0x4000_0000, tables.0[&0x8000_0000]);
    controls.second_stage = Some(Stage2 {
        hcr: HCR_VM | HCR_TGE,
        ..controls.second_stage.unwrap()
    });
    let Outcome::Mapped(stage1) = translate(&controls, 0x1234, Access::Read, KERNEL, &mut tables)
    else {
        panic!("TGE")
    };
    assert_eq!(stage1.physical, 0x4000_1234);
    // Stage 2 required without its controls refuses.
    controls.second_stage = None;
    assert_eq!(
        translate(&controls, 0x1234, Access::Read, KERNEL, &mut tables),
        Outcome::Unsupported(Unsupported::Stage2)
    );
    assert_eq!(
        Stage2 {
            vttbr: 0xab << 48,
            ..Stage2::default()
        }
        .vmid(),
        0xab
    );
}

#[test]
fn alignment_is_checked_before_translation_and_ahead_of_permissions() {
    use alignment::{Kind, Shape};
    let plain = Shape {
        kind: Kind::Plain,
        bytes: 8,
        pair: false,
    };
    let atomic = Shape {
        kind: Kind::Atomic,
        ..plain
    };
    let alignment_fault = |outcome: Outcome| match outcome {
        Outcome::Fault(fault) => Some((fault.kind, fault.second_stage.is_some())),
        _ => None,
    };
    let (mut controls, mut tables) = two_stages(0x4000_0000);
    controls.stage2 = false;
    controls.second_stage = None;
    let checked = |controls: &Controls, address, shape, requester, tables: &mut Tables| {
        translate_checked(controls, address, Access::Read, requester, shape, tables)
    };
    // Stage 1 maps VA 0-1 GiB at PA 0x4000_0000 (no stage 2), Normal
    // Write-Back Inner Shareable, EL1 only.
    tables
        .0
        .insert(0x4000_0000, 0x4000_0000 | 1 | 1 << 10 | 0b11 << 8);
    controls.ttbr0 = 0x4000_0000;
    let user = Requester { el: 0, ..KERNEL };
    assert!(matches!(
        checked(&controls, 0x1004, plain, KERNEL, &mut tables),
        Outcome::Mapped(_)
    ));
    // SCTLR.A faults before the walk, even where nothing is mapped.
    controls.sctlr |= alignment::SCTLR_A;
    assert_eq!(
        alignment_fault(checked(
            &controls,
            0x7fff_ffff_0004,
            plain,
            KERNEL,
            &mut tables
        )),
        Some((FaultKind::Alignment, false))
    );
    controls.sctlr &= !(alignment::SCTLR_A | alignment::SCTLR_C);
    // An unaligned atomic inside 16 bytes needs no Write-Back memory or
    // SCTLR.C (Arm's choice for Unpredictable_LSE2_ALIGNMENT_FAULT).
    for sctlr in [controls.sctlr, controls.sctlr | alignment::SCTLR_C] {
        controls.sctlr = sctlr;
        assert!(matches!(
            checked(&controls, 0x1004, atomic, KERNEL, &mut tables),
            Outcome::Mapped(_)
        ));
    }
    // Device memory (AttrIndx 1, MAIR byte 0) faults an unaligned access
    // ahead of the Permission fault EL0 would take.
    controls.mair = 0x00ff;
    tables
        .0
        .insert(0x4000_0000, 0x4000_0000 | 1 | 1 << 10 | 1 << 2);
    assert_eq!(
        alignment_fault(checked(&controls, 0x1004, plain, user, &mut tables)),
        Some((FaultKind::Alignment, false))
    );
    assert_eq!(
        alignment_fault(checked(&controls, 0x1008, plain, user, &mut tables)),
        Some((FaultKind::Permission, false))
    );

    // Stage 2 Device memory under Normal stage 1: a stage-2 Alignment
    // fault, taken to EL2 without HPFAR_EL2.
    let (controls, mut tables) = two_stages(0x4000_0000);
    let device = (STAGE2_BLOCK & !(0xf << 2)) | 0b0001 << 2;
    tables.0.insert(0x10_0008, device);
    let Outcome::Fault(fault) = checked(&controls, 0x1004, plain, KERNEL, &mut tables) else {
        panic!("stage-2 alignment")
    };
    assert_eq!(fault.kind, FaultKind::Alignment);
    assert!(fault.to_el2() && fault.hpfar().is_none());
    assert_eq!(fault.data_syndrome(false, false), 0b10_0001);
    let Outcome::Mapped(aligned) = checked(&controls, 0x1008, plain, KERNEL, &mut tables) else {
        panic!("aligned Device access")
    };
    assert_eq!(aligned.alignment, Alignment { device: true });
}

#[test]
fn fault_syndromes_and_hpfar_follow_the_stage_and_the_walk() {
    // HPFAR_EL2.FIPA holds IPA[47:12] of the chip's 48-bit IPA space.
    test_translation_chip();
    let stage1 = Fault::stage1(FaultKind::Translation, 3);
    assert_eq!(stage1.data_syndrome(true, false), 0b100_0111);
    assert_eq!(stage1.data_syndrome(false, true), 1 << 8 | 1 << 6 | 0b111);
    assert!(!stage1.to_el2() && stage1.hpfar().is_none());
    let fault = |kind, walk| Fault {
        kind,
        level: 1,
        second_stage: Some(SecondStage {
            ipa: 0x8_1234_5678,
            walk,
        }),
    };
    let walk = fault(FaultKind::Permission, true);
    assert!(walk.to_el2());
    assert_eq!(walk.data_syndrome(false, false), 1 << 7 | 0b00_1101);
    assert_eq!(walk.instruction_syndrome(), 1 << 7 | 0b00_1101);
    assert_eq!(walk.hpfar(), Some(0x0081_2345 << 4));
    // A stage-2 Permission fault on the access itself leaves HPFAR_EL2
    // UNKNOWN (no FEAT_EAESR); Translation, Access flag and Address size
    // faults report it.
    assert_eq!(fault(FaultKind::Permission, false).hpfar(), None);
    assert_eq!(
        fault(FaultKind::AccessFlag, false).hpfar(),
        Some(0x0081_2345 << 4)
    );
    assert_eq!(fault(FaultKind::ExternalOnWalk, false).hpfar(), None);
}

/// A deterministic generator for the walk-table tests (xorshift64*).
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
}

/// Stage-2 tables at this PA map IPA 0-4 GiB to the same PAs in 1 GiB
/// Normal Write-Back read/write blocks.
const IDENTITY_VTTBR: u64 = 0xf000_0000;

/// A random leaf of `size` bytes: a page (`page`) or a block descriptor,
/// with random access flag, permissions, attributes, nG, PXN, UXN and GP,
/// its output mostly in the first 4 GiB.
fn random_leaf(random: &mut Random, size: u64, page: bool) -> u64 {
    let output = if random.chance(95) {
        random.below(0x1_0000_0000 / size) * size
    } else {
        (1 << 45) | (random.below(1 << 20) * size)
    };
    let mut descriptor = output | if page { 0b11 } else { 0b01 };
    for (bit, percent) in [(10, 90), (11, 50), (53, 30), (54, 30), (50, 10)] {
        if random.chance(percent) {
            descriptor |= 1 << bit;
        }
    }
    descriptor | random.below(4) << 6 | random.below(4) << 8 | random.below(8) << 2
}

/// A random table descriptor to `child`, with random hierarchical limits.
fn random_table(random: &mut Random, child: u64) -> u64 {
    let mut descriptor = child | 0b11;
    for bit in 59..=62 {
        if random.chance(25) {
            descriptor |= 1 << bit;
        }
    }
    descriptor
}

/// The granule and input bits of the half of `controls` that `address`
/// selects.
fn walk_shape(controls: &Controls, address: u64) -> (u32, u32) {
    let upper = controls.regime != Regime::El2 && address & (1 << 55) != 0;
    let (tnsz_shift, tg_shift) = if upper { (16, 30) } else { (0, 14) };
    let granule_bits = match (controls.tcr >> tg_shift & 3, upper) {
        (0, false) | (2, true) => 12,
        _ => 14,
    };
    let input_bits = 64 - geometry::forced_tnsz((controls.tcr >> tnsz_shift & 63) as u32);
    (granule_bits, input_bits)
}

/// Random stage-1 tables on the walk of `address` under `controls`: a path
/// of random table descriptors to a table at level 2 or 3 whose entries
/// are all random (invalid, pages, blocks, reserved encodings and, above
/// level 3, tables of random pages). Returns that table's level.
fn random_tables(
    random: &mut Random,
    controls: &Controls,
    address: u64,
    tables: &mut Tables,
) -> i8 {
    let upper = controls.regime != Regime::El2 && address & (1 << 55) != 0;
    let ttbr = if upper {
        controls.ttbr1
    } else {
        controls.ttbr0
    };
    let (granule_bits, input_bits) = walk_shape(controls, address);
    let geometry = geometry::Geometry::stage1(granule_bits, input_bits);
    let granule = 1u64 << granule_bits;
    let mut next = 0x100_0000 + u64::from(upper) * 0x800_0000;
    let mut allocate = || {
        let table = next;
        next += granule;
        table
    };
    let last = (if random.chance(30) { 2 } else { 3 }).max(geometry.start_level);
    let mut table = ttbr & 0x0000_ffff_ffff_f000 & !(geometry.start_table_bytes().max(64) - 1);
    let mut level = geometry.start_level;
    while level < last {
        let child = allocate();
        let slot = table + geometry.index(address, level) * 8;
        tables.0.insert(slot, random_table(random, child));
        table = child;
        level += 1;
    }
    let size = 1u64 << geometry.shift(level);
    for index in 0..1u64 << geometry.index_bits(level) {
        let descriptor = match random.below(100) {
            0..15 => 0,
            15..20 => random.next() & !1,
            _ if level == 3 => {
                let page = random.chance(90);
                random_leaf(random, size, page)
            }
            20..60 => random_leaf(random, size, false),
            _ => {
                let child = allocate();
                let page = size >> (granule_bits - 3);
                for _ in 0..32 {
                    let slot = child + random.below(granule / 8) * 8;
                    let leaf = random_leaf(random, page, true);
                    tables.0.insert(slot, leaf);
                }
                random_table(random, child)
            }
        };
        if descriptor != 0 {
            tables.0.insert(table + index * 8, descriptor);
        }
    }
    level
}

/// Random translation controls: any regime, either granule, region sizes,
/// HPD, TBI, TBID, E0PD, EPD, WXN, EPAN, MAIR, SPRR and, in EL1&0, a stage
/// 2 that maps the first 4 GiB as they are.
fn random_controls(random: &mut Random) -> Controls {
    let regime = random.pick(&[Regime::El20, Regime::El20, Regime::El10, Regime::El2]);
    let sixteen = random.chance(70);
    let mut tcr = if regime == Regime::El2 {
        random.pick(&[17u64, 25, 28]) | if sixteen { 2 << 14 } else { 0 } | 5 << 16
    } else {
        random.pick(&[17u64, 25, 28])
            | random.pick(&[17u64, 25, 28]) << 16
            | if sixteen { 2 << 14 | 1 << 30 } else { 2 << 30 }
            | 5 << 32
    };
    let bits: &[u32] = if regime == Regime::El2 {
        &[20, 24, 29]
    } else {
        &[7, 23, 36, 37, 38, 41, 42, 51, 52, 55, 56]
    };
    for &bit in bits {
        let percent = match bit {
            7 | 23 | 55 | 56 => 4,
            _ => 30,
        };
        if random.chance(percent) {
            tcr |= 1 << bit;
        }
    }
    let mut sctlr = SCTLR_M | alignment::SCTLR_C | 1 << 12;
    if random.chance(10) {
        sctlr |= SCTLR_WXN;
    }
    if random.chance(30) {
        sctlr |= 1 << 57;
    }
    let mair = (0..8).fold(0u64, |mair, index| {
        mair | u64::from(random.pick(&[0xffu8, 0x44, 0x00, 0x04, 0xbb, 0xf0])) << (8 * index)
    });
    let sprr = if random.chance(20) {
        Sprr {
            config: SPRR_CONFIG_EN,
            el1: random.next(),
            el0: random.next(),
            ..Sprr::default()
        }
    } else {
        Sprr::default()
    };
    let stage2 = regime == Regime::El10 && random.chance(25);
    Controls {
        sctlr,
        tcr,
        ttbr0: 0x10_0000 | random.below(256) << 48,
        ttbr1: 0x20_0000 | random.below(256) << 48,
        mair,
        xs: test_xs(),
        sprr,
        regime,
        stage2,
        second_stage: stage2.then_some(Stage2 {
            hcr: HCR_VM,
            vtcr: VTCR,
            vttbr: IDENTITY_VTTBR,
            xs: test_xs(),
        }),
        ..Controls::default()
    }
}

/// A walk that starts at a table an earlier walk noted gives what a full
/// walk gives, for every entry of that table, every access and requester,
/// any regime, granule, stage 2, SPRR and hierarchical limits; an address
/// the table does not hold walks in full.
#[test]
fn walks_from_a_noted_table_translate_as_full_walks() {
    test_translation_chip();
    let mut random = Random(0x5eed_0f7a_b1e5);
    let requesters = [
        KERNEL,
        Requester {
            pan: true,
            ..KERNEL
        },
        Requester { el: 0, ..KERNEL },
        Requester {
            unprivileged: true,
            ..KERNEL
        },
        Requester {
            guarded: true,
            ..KERNEL
        },
    ];
    let accesses = [Access::Read, Access::Write, Access::Execute];
    let (mut compared, mut held, mut outside, mut mapped) = (0u64, 0u64, 0u64, 0u64);
    for _ in 0..300 {
        let controls = random_controls(&mut random);
        let mut tables = Tables::default();
        for block in 0..4u64 {
            tables.0.insert(
                IDENTITY_VTTBR + block * 8,
                block << 30 | 1 | 1 << 10 | 0b11 << 6 | 0b1111 << 2 | 0b11 << 8,
            );
        }
        let upper = controls.regime != Regime::El2 && random.chance(50);
        let low = random.below(1 << 36);
        let mut address = if upper {
            low | 0xffff_fff0_0000_0000
        } else {
            low
        };
        if random.chance(20) {
            address = address & !(0xff << 56) | random.below(256) << 56;
        }
        let level = random_tables(&mut random, &controls, address, &mut tables);
        let requester = random.pick(&requesters);
        let access = random.pick(&accesses);
        let (outcome, noted) = translate_noting(&controls, address, access, requester, &mut tables);
        assert_eq!(
            outcome,
            translate(&controls, address, access, requester, &mut tables)
        );
        let Some(noted) = noted else {
            continue;
        };
        let (granule_bits, _) = walk_shape(&controls, address);
        let stride = granule_bits - 3;
        let entry_bits = granule_bits + stride * (3 - level) as u32;
        let span_bits = entry_bits + stride;
        let mut targets: alloc::vec::Vec<u64> = (0..64)
            .map(|_| {
                let entry = random.below(1 << stride);
                let offset = random.below(1 << entry_bits);
                address & !((1u64 << span_bits) - 1) | entry << entry_bits | offset
            })
            .collect();
        for distance in 1..=4u64 {
            targets.push(address.wrapping_add(distance << entry_bits));
            targets.push(address.wrapping_sub(distance << entry_bits));
        }
        // Outside the table: other indices above it, the other half,
        // another top byte.
        targets.extend([
            address ^ 1 << 33,
            address ^ 1 << 55,
            address ^ 1 << 60,
            address.wrapping_add(1 << span_bits),
        ]);
        for target in targets {
            if noted.holds(target) {
                held += 1;
            } else {
                outside += 1;
            }
            for requester in requesters {
                for access in accesses {
                    compared += 1;
                    let full = translate(&controls, target, access, requester, &mut tables);
                    mapped += u64::from(matches!(full, Outcome::Mapped(_)));
                    assert_eq!(
                        translate_from(&controls, &noted, target, access, requester, &mut tables),
                        full,
                        "{target:#x} {access:?} {requester:?} from {address:#x} {controls:#x?}"
                    );
                }
            }
        }
    }
    assert!(held > 10_000 && outside > 500, "{held} {outside}");
    assert!(compared > 200_000 && mapped > 20_000, "{compared} {mapped}");
}
