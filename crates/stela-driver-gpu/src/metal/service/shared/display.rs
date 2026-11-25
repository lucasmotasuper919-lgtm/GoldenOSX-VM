//! Display compositions the platform asks for, outside the guest's calls.
//!
//! The display model offers each composed swap to the host's display on
//! the processor that presented it. A display whose scanout buffers are
//! shared with the service's device has the GPU compose the frame
//! ([`crate::metal::device::display`]): a turn within the wait bound
//! commits the composition, and the platform flips the buffer once the
//! published completion, or a free turn's poll, shows it complete. A
//! device lost or replaced since the buffers were shared refuses them.
use super::SharedService;
use crate::metal::{
    device::display::{DisplayFrame, ScanoutBuffer},
    error::{Error, Result},
    service::Service,
    shader::Compiler,
    vulkan::backend::Backend,
};
use alloc::vec::Vec;

/// A display's scanout buffers the service's device imported, and the
/// device's generation then.
#[derive(Debug)]
pub struct DisplayTarget {
    epoch: u64,
    scanouts: Vec<ScanoutBuffer>,
}

impl DisplayTarget {
    pub fn len(&self) -> usize {
        self.scanouts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scanouts.is_empty()
    }
}

impl<B: Backend, C: Compiler + 'static> SharedService<B, C> {
    /// Imports a display's scanout buffers, each a dma-buf descriptor and
    /// its bytes. Every descriptor is the driver's once this succeeds.
    /// After an error the driver may have taken some, which it closes with
    /// their memory: the caller leaves them all open.
    pub fn share_scanouts(&self, buffers: &[(i32, u64)]) -> Result<DisplayTarget> {
        let epoch = self.read_backs.generation.current();
        self.in_turn(|service| {
            let device = service.device();
            let mut scanouts = Vec::new();
            for &(fd, bytes) in buffers {
                match device.import_scanout(fd, bytes) {
                    Ok(scanout) => scanouts.push(scanout),
                    Err(error) => {
                        for scanout in scanouts {
                            device.release_scanout(scanout);
                        }
                        return Err(error);
                    }
                }
            }
            Ok(DisplayTarget { epoch, scanouts })
        })?
    }

    /// Frees `target`'s buffers once no composition writes them; those of a
    /// device since replaced went with it.
    pub fn release_display(&self, target: DisplayTarget) {
        if target.epoch != self.read_backs.generation.current() {
            return;
        }
        self.call(|service| {
            for scanout in target.scanouts {
                service.device().release_scanout(scanout);
            }
        });
    }

    /// Commits the composition of `frame` into `target`'s buffer `index`,
    /// answering the command buffer whose completion makes it the
    /// display's, and how many of its layers are composed from a device
    /// copy of their planes.
    pub fn compose_display(
        &self,
        target: &DisplayTarget,
        index: usize,
        frame: &DisplayFrame,
    ) -> Result<(u64, u64)> {
        let scanout = target.scanouts.get(index).ok_or(Error::Invalid(
            "a display composition names a shared buffer",
        ))?;
        self.in_turn(|service| {
            if target.epoch != self.read_backs.generation.current() {
                return Err(Error::DeviceLost);
            }
            let device = service.device();
            let serial = device.compose_display(scanout, frame)?;
            Ok((serial, device.take_display_copies()))
        })?
    }

    /// Whether composition `serial` into `target` completed; `None` once
    /// the device that shares it was lost or replaced.
    pub fn display_composed(&self, target: &DisplayTarget, serial: u64) -> Option<bool> {
        self.serial_done(target.epoch, serial)
    }

    /// Runs `body` on the service in a turn, unless the wait bound passes
    /// first ([`Self::set_wait_bound`]).
    fn in_turn<R>(&self, body: impl FnOnce(&mut Service<B, C>) -> R) -> Result<R> {
        let mut turn = self
            .turn_within((self.host.ticks)())
            .ok_or(Error::Unsupported("a turn at the busy GPU service"))?;
        turn.catch_up();
        let (mut body, mut result) = (Some(body), None);
        (self.host.run)(self.host.stack, &mut || {
            if let Some(body) = body.take() {
                result = Some(body(&mut turn));
            }
        });
        Ok(result.expect("the platform runs what it is given"))
    }
}
