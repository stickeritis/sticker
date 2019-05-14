use std::io::{self, Read, Seek, SeekFrom};

use indicatif::{ProgressBar, ProgressStyle};

pub struct ReadProgress<R> {
    inner: R,
    progress: ProgressBar,
}

/// A progress bar that implements the `Read` and `Seek` traits.
///
/// This wrapper of `indicatif`'s `ProgressBar` updates progress based on the
/// current offset within the file.
impl<R> ReadProgress<R>
where
    R: Seek,
{
    pub fn new(mut read: R) -> io::Result<Self> {
        let len = read.seek(SeekFrom::End(0))? + 1;
        let progress = ProgressBar::new(len);
        progress.set_style(ProgressStyle::default_bar().template("{bar} {bytes}/{total_bytes}"));

        Ok(ReadProgress {
            inner: read,
            progress,
        })
    }

    pub fn set_style(&self, style: ProgressStyle) {
        self.progress.set_style(style)
    }
}

impl<R> Read for ReadProgress<R>
where
    R: Read + Seek,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n_read = self.inner.read(buf)?;
        let pos = self.inner.seek(SeekFrom::Current(0))?;
        self.progress.set_position(pos);
        Ok(n_read)
    }
}

impl<R> Seek for ReadProgress<R>
where
    R: Seek,
{
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let pos = self.inner.seek(pos)?;
        self.progress.set_position(pos);
        Ok(pos)
    }
}

impl<R> Drop for ReadProgress<R> {
    fn drop(&mut self) {
        self.progress.finish();
    }
}
