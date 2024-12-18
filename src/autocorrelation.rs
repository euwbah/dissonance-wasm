// "pitch" - Licensed under the Boost Software License (see /LICENSE)

//! Originally from https://gitlab.com/plopgrizzly/audio/pitch/-/
//!
//! Modified to work with BPM detection based on note on events instead of audio samples.

// BCF constants:
const SPS: u32 = 48_000; // Sample Hz
const MAX_FREQ: f32 = 10_000.0; // Stupidly high note
const MIN_PERIOD: f32 = (SPS as f32) / MAX_FREQ; // Minumum Period Samples

const NBITS: usize = ::std::mem::size_of::<usize>() * 8;

struct ZeroCross(bool);

impl ZeroCross {
	fn new() -> Self {
		ZeroCross(false)
	}

	fn get(&mut self, s: f32, t: f32) -> bool {
		if s < -t {
			self.0 = false;
		} else if s > t {
			self.0 = true;
		}

		self.0
	}
}

struct BitStream {
	bits: Vec<usize>,
	len: usize,
}

impl BitStream {
	fn get(&self, index: usize, shift: usize) -> usize {
		let v = self.bits[index];
		if shift > 0 {
			v >> shift | self.bits[index + 1] << (NBITS - shift)
		} else {
			v
		}
	}

	fn autocorrelate(&mut self, start_pos: usize, f: &mut FnMut(usize, u32))
	{
		let mid_array = (self.bits.len() / 2) - 1;
		let mid_pos = self.len / 2;
		let mut index = start_pos / NBITS;
		let mut shift = start_pos % NBITS;

		// get autocorrelation values for the first half of the sample.
		for pos in start_pos..mid_pos {
			let mut count = 0;
			for i in 0..mid_array {
				count += (self.get(i, 0)
						^ self.get(i + index, shift))
					 .count_ones();
			}
			shift += 1;
			if shift == NBITS {
				shift = 0;
				index += 1;
			}
			f(pos, count);
		}
	}
}

fn bcf(samples: &[f32]) -> Option<(f32, f32)> {
	// Get The Amplitude (Volume).
	let mut volume = 0.0f32;
	for i in samples.iter() {
		volume = volume.max(i.abs());
	}

	// Convert Into a Bitstream of Zero-Crossings
	let mut zc = ZeroCross::new();
	let t = volume * 0.00001;
	let mut bin = BitStream {
		bits: vec![],
		len: samples.len(),
	};

	let mut i = 0;
	'a: loop {
		let mut register = 0usize;
		for shift in 0..NBITS {
			let setv = zc.get(samples[i], t);
			register ^= (if setv { ::std::usize::MAX } else { 0 }
				^ register) & (1 << shift);
			i += 1;
			if i == samples.len() {
				bin.bits.push(register);
				break 'a;
			}
		}
		bin.bits.push(register);
	}

	// Binary Autocorrelation
	let mut min_count = ::std::u32::MAX;
	let mut est_index = 0usize;

	bin.autocorrelate(MIN_PERIOD as usize, &mut |pos, count| {
		if count < min_count {
			min_count = count;
			est_index = pos;
		}
	});

	// Estimate the pitch:
	// - Get the start edge
	let mut prev = 0.0f32;
	let mut esam = samples.iter().enumerate();
	let start_edge = loop {
		let (i, start_edge2) = esam.next()?;
		if *start_edge2 > 0.0 {
			break (i, start_edge2);
		}
		prev = *start_edge2;
	};

	let mut dy = start_edge.1 - prev;
	let dx1 = -prev / dy;

	// - Get the next edge
	let mut nsam = esam.skip(est_index - start_edge.0 - 1);
	let next_edge = loop {
		let (i, next_edge2) = nsam.next()?;
		if *next_edge2 > 0.0 {
			break (i, next_edge2);
		}
		prev = *next_edge2;
	};

	dy = next_edge.1 - prev;
	let dx2 = -prev / dy;

	let n_samples: f32 = (next_edge.0 - start_edge.0) as f32 + (dx2 - dx1);

	println!("{} {} {} {} {} {}", est_index, n_samples, next_edge.0, start_edge.0, dx2, dx1);

	// The frequency
	Some(((SPS as f32) / n_samples, volume))
}

/// Do the BCF calculation on raw samples.  Returns `(hz, amplitude[0-1])`.
pub fn detect(samples: &[f32]) -> (f32, f32) {
	bcf(samples).unwrap_or((0.0, 0.0))
}
