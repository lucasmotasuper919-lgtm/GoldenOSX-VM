//! A shown output connected again, perhaps to another monitor.
use super::outputs::{at, two_monitors};
use super::*;

#[test]
fn a_returning_monitor_runs_a_mode_from_its_new_list() {
    let mut card = Card::new(vec![at(2, 2, PREFERRED, 60)]);
    let front;
    {
        let mut display = Display::take(&mut card, (2, 2)).unwrap();
        front = display.buffers[0].framebuffer;
        display.drm.connected = false;
        let lost = Watched {
            lost: 1,
            ..Watched::default()
        };
        assert_eq!(display.watch(), lost);
        // Another monitor: larger, and the display's size only at 75 Hz.
        display.drm.modes = vec![at(4, 2, PREFERRED, 60), at(2, 2, 0, 75)];
        display.drm.connected = true;
        let restored = Watched {
            restored: 1,
            ..Watched::default()
        };
        assert_eq!(display.watch(), restored);
    }
    assert_eq!(card.set, Some((at(2, 2, 0, 75), front)));
}

#[test]
fn a_returning_monitor_without_the_size_is_left_dark_and_counted_once() {
    let mut card = Card::new(vec![at(2, 2, PREFERRED, 60)]);
    let mut display = Display::take(&mut card, (2, 2)).unwrap();
    display.drm.connected = false;
    display.watch();
    display.drm.modes = vec![at(4, 2, PREFERRED, 60)];
    display.drm.connected = true;
    let sets = display.drm.mode_sets;
    // Probed at each of the tries, in case its EDID comes late; counted at
    // the last; the controller is never touched.
    assert_eq!(display.watch(), Watched::default());
    assert_eq!(display.watch(), Watched::default());
    let modeless = Watched {
        modeless: 1,
        ..Watched::default()
    };
    assert_eq!(display.watch(), modeless);
    let probes = display.drm.probes;
    assert_eq!(display.watch(), Watched::default());
    assert_eq!(display.drm.probes, probes);
    assert_eq!(display.drm.mode_sets, sets);
    // Connected again with the size: set.
    display.drm.connected = false;
    assert_eq!(display.watch(), Watched::default());
    display.drm.modes = vec![at(4, 2, PREFERRED, 60), at(2, 2, 0, 60)];
    display.drm.connected = true;
    let restored = Watched {
        restored: 1,
        ..Watched::default()
    };
    assert_eq!(display.watch(), restored);
    assert_eq!(display.drm.set.map(|(mode, _)| mode), Some(at(2, 2, 0, 60)));
}

#[test]
fn an_output_that_cannot_be_set_again_does_not_stop_the_next() {
    let mut card = two_monitors();
    let front;
    {
        let mut display = Display::take(&mut card, (6, 4)).unwrap();
        assert_eq!(display.outputs(), 2);
        front = display.buffers[0].framebuffer;
        display.drm.first_connected = false;
        display.drm.connected = false;
        let lost = Watched {
            lost: 2,
            ..Watched::default()
        };
        assert_eq!(display.watch(), lost);
        display.drm.set = None;
        display.drm.first_connected = true;
        display.drm.connected = true;
        // The first output's controller refuses every request.
        display.drm.refused_crtc = Some(40);
        let refusal = Some(Error::Request(-22));
        let restored = Watched {
            restored: 1,
            error: refusal,
            ..Watched::default()
        };
        assert_eq!(display.watch(), restored);
        let tried = Watched {
            error: refusal,
            ..Watched::default()
        };
        assert_eq!(display.watch(), tried);
        let refused = Watched {
            refused: 1,
            error: refusal,
            ..Watched::default()
        };
        assert_eq!(display.watch(), refused);
        assert_eq!(display.watch(), Watched::default());
    }
    assert_eq!(card.set, Some((at(6, 4, PREFERRED, 60), front)));
}

/// The first output's controller is off after a mode set the driver
/// refused, and Linux answers every flip on it as busy: the next output
/// paces the flips, where the frame stayed queued and every other output
/// kept its old picture.
#[test]
fn another_output_paces_the_flips_while_the_firsts_controller_is_off() {
    let mut card = two_monitors();
    let (spare, next);
    {
        let mut display = Display::take(&mut card, (6, 4)).unwrap();
        assert_eq!(display.heads[0].crtc, 40);
        display.drm.first_connected = false;
        assert_eq!(display.watch().lost, 1);
        // Back, but its mode set is refused after its controller went off.
        display.drm.first_connected = true;
        display.drm.refused_light = Some(40);
        assert_eq!(display.watch().error, Some(Error::Request(-22)));
        assert_eq!(display.drm.set_40, None);
        spare = display.buffers[2].framebuffer;
        display.present(&[7; 96]).unwrap();
        assert!(!display.queued());
        assert_eq!(display.buffers[0].framebuffer, spare);
        assert_eq!(display.drm.flips, [spare]);
        assert!(display.drm.flips_40.is_empty());
        // Set at a later look, it shows the newest frame and paces again.
        display.drm.refused_light = None;
        assert_eq!(display.watch().restored, 1);
        assert_eq!(display.drm.set_40.map(|(_, shown)| shown), Some(spare));
        next = display.buffers[2].framebuffer;
        display.present(&[9; 96]).unwrap();
        assert!(!display.queued());
    }
    assert_eq!(card.flips_40, [next]);
    assert_eq!(card.flips, [spare, next]);

    // No output is on: the frame stays queued for the one set next.
    let mut alone = Card::new(vec![at(2, 2, PREFERRED, 60)]);
    let mut display = Display::take(&mut alone, (2, 2)).unwrap();
    display.drm.connected = false;
    display.watch();
    display.drm.connected = true;
    display.drm.refused_light = Some(41);
    display.watch();
    let hidden = display.buffers[2].framebuffer;
    display.present(&[7; 16]).unwrap();
    assert!(display.queued() && display.drm.flips.is_empty());
    display.drm.refused_light = None;
    assert_eq!(display.watch().restored, 1);
    display.flush().unwrap();
    assert!(!display.queued());
    assert_eq!(display.drm.flips, [hidden]);
}

/// The first output's monitor is disconnected while the outputs sleep, so
/// the wake leaves its controller off: the next output paces the flips,
/// where the first, whose every flip Linux answered as busy, kept the
/// frame queued and the other monitor kept its old picture.
#[test]
fn another_output_paces_the_flips_after_a_wake_without_the_first_monitor() {
    let mut card = two_monitors();
    let (spare, next);
    {
        let mut display = Display::take(&mut card, (6, 4)).unwrap();
        display.set_power(false).unwrap();
        display.drm.first_connected = false;
        assert_eq!(display.watch().lost, 1);
        display.set_power(true).unwrap();
        assert_eq!(display.drm.set_40, None);
        spare = display.buffers[2].framebuffer;
        display.present(&[7; 96]).unwrap();
        assert!(!display.queued());
        assert!(display.drm.flips_40.is_empty());
        // Connected again, it is set on the newest frame and paces again.
        display.drm.first_connected = true;
        assert_eq!(display.watch().restored, 1);
        assert_eq!(display.drm.set_40.map(|(_, shown)| shown), Some(spare));
        next = display.buffers[2].framebuffer;
        display.present(&[9; 96]).unwrap();
        assert!(!display.queued());
    }
    assert_eq!(card.flips, [spare, next]);
    assert_eq!(card.flips_40, [next]);
}
