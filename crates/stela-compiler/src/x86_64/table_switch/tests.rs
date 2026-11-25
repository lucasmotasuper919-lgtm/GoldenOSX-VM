use super::*;
use crate::{Anchor, GuestState, x86_64::register_read::Cell};

const PC: GuestAddress = GuestAddress(0x4000);
const RT: u8 = 3;
const OLD: u64 = 1 << 48 | 0x8000;
const NEW: u64 = 2 << 48 | 0xc000;
const UPPER: u64 = 0x77;
const BASE: u64 = 0x10_0000;

/// Everything a switch thunk reads and writes, at the Mirrors anchor.
#[repr(C)]
struct World {
    cell: Cell,
    register: u64,
    table: Table,
    contexts: [u64; 2],
    salts: [u32; 2],
    contexts_generation: u64,
    closes: [u64; 2],
    cr3: u64,
    translating: u32,
    pending: u8,
    sets: Sets,
}

#[derive(Debug, PartialEq, Eq)]
#[repr(C)]
struct Exit {
    pc: u64,
    kind: u64,
}

const NEXT: Exit = Exit {
    pc: PC.0 + 4,
    kind: ExitKind::Fallthrough as u64,
};
const MISS: Exit = Exit {
    pc: PC.0,
    kind: ExitKind::System as u64,
};

fn at(field: usize) -> Place {
    Place::new(Anchor::Mirrors, field)
}

fn places(sets: bool, world: &World) -> Places {
    Places {
        table: at(offset_of!(World, table)),
        contexts: at(offset_of!(World, contexts)),
        salts: at(offset_of!(World, salts)),
        contexts_generation: at(offset_of!(World, contexts_generation)),
        closes: alloc::vec![
            at(offset_of!(World, closes)),
            at(offset_of!(World, closes) + 8)
        ],
        cr3: sets.then_some(Cr3::Word(&world.cr3 as *const u64 as u64)),
        isolate: false,
    }
}

fn root(set: u64, view: u64) -> u64 {
    BASE + (set * SET_ROOTS as u64 + view) * 4096
}

/// A world whose register holds OLD, with an entry for NEW to set 5 and
/// the current CR3 set 2's view 3.
fn world(sets: bool) -> alloc::boxed::Box<World> {
    let mut world = alloc::boxed::Box::new(World {
        cell: Cell::default(),
        register: OLD,
        table: Table::default(),
        contexts: [0x11, UPPER],
        salts: [0x21, 0x22],
        contexts_generation: 9,
        closes: [1, 1],
        cr3: root(2, 3) | (FIRST_CONTEXT + 2 * 8 + 3),
        translating: 0x100,
        pending: 0,
        sets: Sets::default(),
    });
    world.cell = Cell {
        readable: 1,
        value: &world.register as *const u64 as u64,
    };
    world.table.translating = &world.translating as *const u32 as u64;
    world.table.pending = &world.pending as *const u8 as u64;
    if sets {
        world.table.sets = &world.sets as *const Sets as u64;
        world.sets.open = 1;
        world.sets.base = BASE;
        world.sets.tagged = 1;
        world.sets.classes[2] = 7;
        world.sets.classes[5] = 7;
        world.sets.generations[5] = 4;
        world.sets.stale[5 * 8 + 3] = 0;
    }
    world.table.keep(Entry {
        generation: 0,
        value: NEW,
        upper: UPPER,
        lower: 0x55,
        salt: 0x66,
        translating: 0x200,
        set: 5,
        set_generation: 4,
    });
    world
}

fn run(thunk: &Thunk, world: &mut World, x: u64) -> Exit {
    use core::ffi::c_void;
    unsafe extern "C" {
        fn mmap(a: *mut c_void, b: usize, p: i32, f: i32, fd: i32, o: i64) -> *mut c_void;
        fn mprotect(a: *mut c_void, b: usize, p: i32) -> i32;
        fn munmap(a: *mut c_void, b: usize) -> i32;
    }
    #[cfg(target_os = "macos")]
    const MAP_ANONYMOUS: i32 = 0x1000;
    #[cfg(target_os = "linux")]
    const MAP_ANONYMOUS: i32 = 0x20;
    const BYTES: usize = 8192;
    let mut state = GuestState::default();
    state.x[RT as usize] = x;
    state.anchors[Anchor::Mirrors] = world as *mut World as u64;
    let code = crate::x86_64::pinned::entered(&thunk.code);
    assert!(code.len() <= BYTES);
    // SAFETY: private anonymous memory, RX after the copy, unmapped after
    // the call; the thunk touches only `state` and `world`.
    unsafe {
        let page = mmap(core::ptr::null_mut(), BYTES, 3, 2 | MAP_ANONYMOUS, -1, 0);
        assert_ne!(page as usize, usize::MAX);
        core::ptr::copy_nonoverlapping(code.as_ptr(), page.cast(), code.len());
        assert_eq!(mprotect(page, BYTES, 5), 0);
        let exit = crate::x86_64::native::call::<Exit>(page as *const u8, &mut state, 0, 0);
        assert_eq!(munmap(page, BYTES), 0);
        exit
    }
}

fn thunk(sets: bool, world: &World) -> Thunk {
    let all = Written {
        keep: u64::MAX,
        set: 0,
    };
    switch_thunk(
        PC,
        RT,
        at(offset_of!(World, cell)),
        all,
        &places(sets, world),
    )
}

/// What a switch leaves: the register, contexts, salts, contexts
/// generation, closed cells, translating word, switches and CR3.
type Seen = (u64, [u64; 2], [u32; 2], u64, [u64; 2], u32, u64, u64);

/// What a switch leaves, as a tuple to compare.
fn seen(world: &World) -> Seen {
    (
        world.register,
        world.contexts,
        world.salts,
        world.contexts_generation,
        world.closes,
        world.translating,
        world.table.switches,
        world.cr3,
    )
}

#[test]
fn an_unchanged_value_completes_and_changes_nothing() {
    for sets in [false, true] {
        let mut world = world(sets);
        let before = seen(&world);
        assert_eq!(run(&thunk(sets, &world), &mut world, OLD), NEXT);
        assert_eq!(seen(&world), before);
    }
}

#[test]
fn a_value_without_an_entry_exits_and_changes_nothing() {
    for sets in [false, true] {
        let mut world = world(sets);
        let before = seen(&world);
        assert_eq!(run(&thunk(sets, &world), &mut world, 3 << 48), MISS);
        assert_eq!(seen(&world), before);
    }
}

#[test]
fn an_entry_switches_the_register_contexts_cells_and_publication() {
    let mut world = world(false);
    assert_eq!(run(&thunk(false, &world), &mut world, NEW), NEXT);
    assert_eq!(
        seen(&world),
        (
            NEW,
            [0x55, UPPER],
            [0x66, 0x22],
            10,
            [0, 0],
            0x200,
            1,
            world.cr3
        )
    );
    // And back is a miss until the engine keeps an entry for OLD.
    assert_eq!(run(&thunk(false, &world), &mut world, OLD), MISS);
}

#[test]
fn stale_entries_and_other_states_exit() {
    // An entry of an older generation.
    let mut old = world(false);
    old.table.forget();
    let before = seen(&old);
    assert_eq!(run(&thunk(false, &old), &mut old, NEW), MISS);
    assert_eq!(seen(&old), before);
    // Another state's upper context.
    let mut other = world(false);
    other.contexts[1] = UPPER + 1;
    let before = seen(&other);
    assert_eq!(run(&thunk(false, &other), &mut other, NEW), MISS);
    assert_eq!(seen(&other), before);
    // A closed cell.
    let mut closed = world(false);
    closed.cell.readable = 0;
    let before = seen(&closed);
    assert_eq!(run(&thunk(false, &closed), &mut closed, NEW), MISS);
    assert_eq!(seen(&closed), before);
    // A table naming sets where the thunk was made without CR3.
    let mut sets = world(true);
    let before = seen(&sets);
    assert_eq!(run(&thunk(false, &sets), &mut sets, NEW), MISS);
    assert_eq!(seen(&sets), before);
}

#[test]
fn pending_mail_after_a_publication_exits_before_the_switch() {
    let mut world = world(false);
    world.pending = 1;
    assert_eq!(run(&thunk(false, &world), &mut world, NEW), MISS);
    // Published, but nothing else moved: the engine reads the mail and
    // completes the write.
    assert_eq!(world.translating, 0x200);
    assert_eq!(
        (world.register, world.contexts[0], world.table.switches),
        (OLD, 0x11, 0)
    );
    // An unchanged publication does not look at the mail.
    let mut same = self::world(false);
    same.pending = 1;
    same.translating = 0x200;
    assert_eq!(run(&thunk(false, &same), &mut same, NEW), NEXT);
    assert_eq!(same.register, NEW);
}

#[test]
fn a_set_loads_the_same_view_of_its_root_in_its_context() {
    let mut world = world(true);
    assert_eq!(run(&thunk(true, &world), &mut world, NEW), NEXT);
    let index = 5 * 8 + 3;
    assert_eq!(
        world.cr3,
        root(5, 3) | (FIRST_CONTEXT + index) | KEEP_ENTRIES
    );
    assert_eq!((world.sets.switched, world.sets.current), (6, root(5, 0)));
    assert_eq!(world.register, NEW);
    assert_eq!(world.sets.flushes, 0);
    // A stale view flushes as it loads, counted, and is fresh after.
    let mut stale = self::world(true);
    stale.sets.stale[index as usize] = 1;
    assert_eq!(run(&thunk(true, &stale), &mut stale, NEW), NEXT);
    assert_eq!(stale.cr3, root(5, 3) | (FIRST_CONTEXT + index));
    assert_eq!(stale.sets.stale[index as usize], 0);
    assert_eq!(stale.sets.flushes, 1);
    // Untagged roots load bare.
    let mut untagged = self::world(true);
    untagged.sets.tagged = 0;
    untagged.cr3 = root(2, 3);
    assert_eq!(run(&thunk(true, &untagged), &mut untagged, NEW), NEXT);
    assert_eq!(untagged.cr3, root(5, 3));
}

#[test]
fn sets_of_another_class_generation_or_closed_exit() {
    let cases: [fn(&mut World); 5] = [
        |world| world.sets.classes[5] = 8,
        |world| world.sets.generations[5] = 5,
        |world| world.sets.open = 0,
        // CR3 outside the sets: the host's own root.
        |world| world.cr3 = BASE - 4096,
        |world| world.table.entries[0].set = SETS as u64,
    ];
    for (case, change) in cases.into_iter().enumerate() {
        let mut world = world(true);
        change(&mut world);
        let before = seen(&world);
        assert_eq!(
            run(&thunk(true, &world), &mut world, NEW),
            MISS,
            "case {case}"
        );
        assert_eq!(seen(&world), before, "case {case}");
        assert_eq!(world.sets.switched, 0, "case {case}");
    }
}

#[test]
fn every_entry_slot_is_found() {
    for index in 0..ENTRIES as u64 {
        let mut world = world(false);
        for other in 0..index {
            world.table.keep(Entry {
                value: 0x9000 + other,
                upper: UPPER,
                ..Entry::default()
            });
        }
        world.table.keep(Entry {
            value: 0xa000,
            upper: UPPER,
            lower: 0x99,
            ..Entry::default()
        });
        assert_eq!(
            run(&thunk(false, &world), &mut world, 0xa000),
            NEXT,
            "slot {index}"
        );
        assert_eq!(world.contexts[0], 0x99);
    }
}
