//! IMPORTANT: When changing the algorithm for dyad roughness or tonicity calculation, remember to
//! update the codegen by running the tests `calculate_roughness` and `calculate_tonicity`
//! respectively.
//!
//! Otherwise, the changes made to the algorithm would not be applied.

use compute::predict::PolynomialRegressor;
use std::sync::OnceLock;

#[cfg(not(target_family = "wasm"))]
use rayon::prelude::*;

use crate::{roughness_codegen::*, sethares_with_harms, tonicity_codegen::*, utils::cents_to_hz};

/// How many harmonics to use in the precalculated (normalized) sethares roughness score for
/// tonicity dyad lookup.
const NUM_HARMS_TONALITY: u32 = 19;

/// How many harmonics to use in the precalculated sethares roughness score for dissonance measure.
const NUM_HARMS_ROUGHNESS: u32 = 31;

/// When calculating dyadic roughness, how many values to sample per cent to average out jitter.
///
/// Without average sub-cent values, certain JI intervals like 17/12, which is 603.000408635 cents,
/// is very near the 1-cent resolution point of 603. These intervals will cause a sudden drop in
/// roughness because of how accurate the 1-cent resolution lookup is. (Creds to @hyperbolekillsme)
///
/// Increasing this value increases the precalculation time of the dyad lookup table.
///
/// **⚠️ This value must be a multiple of 2!**
const NUM_SUB_CENT_SAMPLES: usize = 20;

const HARMS_ADD_OCTAVES_TONALITY: u32 = 5;

const AMP_MULT_TONALITY: f64 = 0.95;

/// Roughness of the first octave for sethares_mult (subsequent octaves will have lower roughness).
const OCT_ROUGHNESS: f64 = 1.1;

pub struct DyadLookup {
    /// Sethares score of interval per cent. Index = cents.
    /// 9 octaves worth (+ 10th octave unison)
    sethares_roughness: [f64; 10801],

    /// Minimum of unnormalized roughness
    roughness_min: f64,
    /// Max of unnormalized roughness
    roughness_max: f64,

    /// Linearly offset such that the minimum roughness is 1, and the roughness of one octave is OCT_ROUGHNESS.
    ///
    /// Use this for multiplicatively applying dissonance.
    sethares_mult: [f64; 10801],

    /// Linearly offset such that the minimum roughness is 0, and the max roughness is 1.
    ///
    /// Use for an absolute dissonance measure between 0 and 1.
    sethares_add: [f64; 10801],

    /// Similar to `sethares_roughness` but normalized by degree 9 polynomial regressions such that
    /// all 10 octaves have almost identical dissonance scores and the peak roughness around each
    /// octave is about equal (though there is still a slight falloff).
    ///
    /// Normalized roughness is between 1 and 2.
    ///
    /// Used for calculating dyadic tonicity.
    sethares_normalized_roughness: [f64; 10801],
}

/// Global instance of [DyadLookup]
static DYAD_LOOKUP: OnceLock<Box<DyadLookup>> = OnceLock::new();

pub enum RoughnessType {
    /// Normalized between 1 and 2 such that every octave has approximately the same roughness.
    ///
    /// Used for TonicityLookup calculatiion and the dyadic tonicity heuristic.
    TonicityNormalized,

    /// Linearly scaled such that the minimum roughness is 1 and the roughness of one octave is OCT_ROUGHNESS.
    Multiplicative,

    /// Linearly scaled such that the minimum roughness is 0 and the max roughness is 1.
    Additive,
}

/// Generates a Gaussian kernel with `num_bins` bins and standard deviation `sigma`.
///
/// Kernel normalized such that sum of all bins is 1.
///
/// Returns a vector of length `num_bins`, symmetric around the center.
fn gaussian_kernel(num_bins: usize, sigma: f64) -> Vec<f64> {
    let mut kernel = vec![0f64; num_bins];
    let half = num_bins as isize / 2;
    let mut sum = 0.0;
    for i in 0..num_bins {
        let x = (i as isize - half) as f64;
        kernel[i] = (-0.5 * (x * x) / (sigma * sigma)).exp();
        sum += kernel[i];
    }
    for i in 0..num_bins {
        kernel[i] /= sum;
    }
    kernel
}

impl DyadLookup {
    pub fn get() -> &'static DyadLookup {
        DYAD_LOOKUP.get_or_init(|| {
            Box::new(DyadLookup {
                sethares_roughness: SETHARES_ROUGHNESS,
                roughness_min: ROUGHNESS_MIN,
                roughness_max: ROUGHNESS_MAX,
                sethares_mult: SETHARES_MULT,
                sethares_add: SETHARES_ADD,
                sethares_normalized_roughness: SETHARES_NORMALIZED_ROUGHNESS,
            })
        })
    }

    /// Recomputes the dyad roughness table from scratch (for updating codegen cache).
    ///
    /// Only run this when dyad roughness algo changes.
    ///
    /// ⚠️ Do not call from WASM.
    #[cfg(not(target_family = "wasm"))]
    pub fn recompute() -> &'static DyadLookup {
        assert!(
            NUM_SUB_CENT_SAMPLES % 2 == 0,
            "NUM_SUB_CENT_SAMPLES must be a multiple of 2"
        );
        DYAD_LOOKUP.get_or_init(|| {
            use rayon::iter::IntoParallelIterator;

            let octave_cents = [
                0.0, 1200.0, 2400.0, 3600.0, 4800.0, 6000.0, 7200.0, 8400.0, 9600.0, 10800.0,
            ];
            // 1 cent resolution sethares roughness values for 9 octaves.
            let mut sethares_roughness = [0f64; 10801];

            // Used for normalized roughness for dyadic tonicity calculation.
            let mut sethares_roughness_tonicity = [0f64; 10801];

            let mut min_rough = f64::MAX;
            let mut max_rough = f64::MIN;

            let mut peak_roughness_around_octave = [0f64; 10];

            // Using parallel iterator to speed up computation.
            // Collects a vec of (roughness, roughness_tonicity) pairs for each sub-cent value.
            let hi_res_roughness_tonicity: Vec<(f64, f64)> = (0..=(10800 * NUM_SUB_CENT_SAMPLES))
                .into_par_iter()
                .map(|partial_cents| {
                    // We assume all dyads are computed w.r.t. A4 as the lower note.
                    //
                    // The lower interval limit effect is added later in the polyadic algorithm.
                    let freq = cents_to_hz(
                        440.0,
                        (partial_cents as f64) / (NUM_SUB_CENT_SAMPLES as f64),
                    );
                    let roughness = sethares_with_harms(&[440.0, freq], NUM_HARMS_ROUGHNESS, 1, 0.86, true);

                    let roughness_tonicity = sethares_with_harms(
                        &[440.0, freq],
                        NUM_HARMS_TONALITY,
                        HARMS_ADD_OCTAVES_TONALITY,
                        AMP_MULT_TONALITY,
                        true,
                    );
                    (roughness, roughness_tonicity)
                })
                .collect();

            // First round of smoothing: uniform average every 1/2 cent bucket.

            println!("Computed {} sub-cent raw roughness values", hi_res_roughness_tonicity.len());

            let mut sethares_roughness_half_cent_res = [0f64; 10800 * 2 + 1];
            let mut sethares_roughness_tonicity_half_cent_res = [0f64; 10800 * 2 + 1];

            for half_cents in 0..=(10800 * 2) {
                // make sure the half-cent bucket is centered at 0 and 0.5.
                let start =
                    ((half_cents * NUM_SUB_CENT_SAMPLES / 2) as isize - 5isize).max(0) as usize;
                let end = (start + 9 - 5).min(10800 * NUM_SUB_CENT_SAMPLES);
                let mut sum = 0.0;
                let mut sum_tonicity = 0.0;
                for i in start..=end {
                    let (roughness, roughness_tonicity) = hi_res_roughness_tonicity[i];
                    sum += roughness;
                    sum_tonicity += roughness_tonicity;
                }
                sethares_roughness_half_cent_res[half_cents] = sum / (end - start + 1) as f64;
                sethares_roughness_tonicity_half_cent_res[half_cents] =
                    sum_tonicity / (end - start + 1) as f64;
            }

            // Second round of smoothing: model human tolerance to detunings & model increased
            // tolerances for low-complexity intervals.
            //
            // Average each 1/2-cent roughness weighted by
            //
            //   (+/- 20c gaussian kernel with sigma = 15 cents) * exp(-roughness * 4)
            //
            // These values are computed every 1 cent to form the final 1-cent resolution
            // sethares_roughness and sethares_roughness_tonicity tables.

            let gaussian_41_9 = gaussian_kernel(41, 15.0);
            let half_gaussian: isize = gaussian_41_9.len() as isize / 2; // index of center of kernel
            for cents in 0isize..=10800 {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;
                let mut sum_tonicity = 0.0;
                let mut weight_sum_tonicity = 0.0;

                let idx_half_cents_start = (cents * 2 - half_gaussian).max(0);
                let idx_half_cents_end = (cents * 2 + half_gaussian).min(10800 * 2);
                for idx_half_cents in idx_half_cents_start..=idx_half_cents_end {
                    let idx_half_cents = idx_half_cents as usize;
                    let roughness = sethares_roughness_half_cent_res[idx_half_cents as usize];
                    let gaussian_weight = gaussian_41_9
                        [(idx_half_cents as isize - cents * 2 + half_gaussian) as usize];
                    let weight = gaussian_weight * (-roughness * 4.0).exp();
                    sum += sethares_roughness_half_cent_res[idx_half_cents] * weight;
                    weight_sum += weight;

                    let roughness_tonicity =
                        sethares_roughness_tonicity_half_cent_res[idx_half_cents as usize];
                    let weight_tonicity = gaussian_weight * (-roughness_tonicity * 2.0).exp();
                    sum_tonicity +=
                        sethares_roughness_tonicity_half_cent_res[idx_half_cents] * weight_tonicity;
                    weight_sum_tonicity += weight_tonicity;
                }
                sethares_roughness[cents as usize] = sum / weight_sum;
                sethares_roughness_tonicity[cents as usize] = sum_tonicity / weight_sum_tonicity;

                let near_octave_above = cents % 1200 <= 200;
                let near_octave_below = cents % 1200 >= 1000;
                let octave = cents as usize / 1200 + if near_octave_below { 1 } else { 0 };

                if near_octave_above || near_octave_below {
                    if peak_roughness_around_octave[octave] < sethares_roughness_tonicity[cents as usize] {
                        peak_roughness_around_octave[octave] = sethares_roughness_tonicity[cents as usize];
                    }
                }

                if sethares_roughness[cents as usize] < min_rough {
                    min_rough = sethares_roughness[cents as usize];
                }
                if sethares_roughness[cents as usize] > max_rough {
                    max_rough = sethares_roughness[cents as usize];
                }
            }

            let mut sethares_normalized_roughness = [0f64; 10801];

            // regressor for multiples of 1200 cents, lower bound of roughness.
            let mut octave_regressor = PolynomialRegressor::new(9);
            octave_regressor.fit(
                &octave_cents,
                &[
                    sethares_roughness_tonicity[0],
                    sethares_roughness_tonicity[1200],
                    sethares_roughness_tonicity[2400],
                    sethares_roughness_tonicity[3600],
                    sethares_roughness_tonicity[4800],
                    sethares_roughness_tonicity[6000],
                    sethares_roughness_tonicity[7200],
                    sethares_roughness_tonicity[8400],
                    sethares_roughness_tonicity[9600],
                    sethares_roughness_tonicity[10800],
                ],
            );

            // regressor for peak roughness around each octave, upper bound of roughness.
            let mut upper_bound_regression = PolynomialRegressor::new(9);
            upper_bound_regression.fit(&octave_cents, &peak_roughness_around_octave);

            let cents_list = Vec::from_iter((0..=10800).map(|x| x as f64));
            let lower_bound_regression = octave_regressor.predict(&cents_list);
            let upper_bound_regression = upper_bound_regression.predict(&cents_list);

            for cents in 0..=10800 {
                let mut width = upper_bound_regression[cents] - lower_bound_regression[cents];
                if width < 0.0001 {
                    width = 0.0001; // tonicity detection above 7 octaves interval has some wonky asymptotes.
                }
                sethares_normalized_roughness[cents] =
                    (sethares_roughness_tonicity[cents] - lower_bound_regression[cents]) / width
                        + 1.0;
            }

            let mut sethares_mult = [0f64; 10801];
            // multiplier required to reach desired 1st octave roughness after fitting min to 1.
            let octave_roughness_multiplier =
                (OCT_ROUGHNESS + min_rough - 1.0) / (sethares_roughness[1200]);

            for cents in 0..=10800 {
                sethares_mult[cents] =
                    sethares_roughness[cents] * octave_roughness_multiplier + (1.0 - min_rough);
            }

            let mut sethares_add = [0f64; 10801];
            for cents in 0..=10800 {
                sethares_add[cents] = (sethares_roughness[cents] - min_rough) / (max_rough - min_rough);
            }

            Box::new(DyadLookup {
                sethares_roughness,
                sethares_normalized_roughness,
                sethares_mult,
                sethares_add,
                roughness_max: max_rough,
                roughness_min: min_rough,
            })
        })
    }

    /// Obtains the linearly interpolated sethares roughness measure from an interval of `cents`.
    pub fn get_roughness(cents: f64, rough_type: RoughnessType) -> f64 {
        let roughness = match rough_type {
            RoughnessType::TonicityNormalized => DyadLookup::get().sethares_normalized_roughness,
            RoughnessType::Multiplicative => DyadLookup::get().sethares_mult,
            RoughnessType::Additive => DyadLookup::get().sethares_add,
        };
        if cents > 10800.0 {
            return 0.0;
        }

        if cents == 10800.0 {
            return roughness[10800];
        }

        let cents_floor = cents.floor() as usize;
        let remainder = cents - cents_floor as f64;
        let sethares_lower = roughness[cents_floor];
        let sethares_higher = roughness[cents_floor + 1];

        sethares_lower + (sethares_higher - sethares_lower) * remainder
    }
}

/// The dyad tonicity score models the probability of hearing the lower note of an interval as the
/// tonic, as compared to the higher note.
///
/// It is modelled by comparing the dissonance measure score of the original interval
/// ("otonal"-esque) with the octave inverted version ("utonal"), while keeping the same octave
/// displacement (modelling changes in roughness by harmonic content). We assume that the lower the
/// roughness of the otonal/utonal interpretation, the more likely that interpretation is preferred
/// as the tonic.
///
/// E.g., a fifth is compared with a fourth, a major third is compared with a minor sixth, 11th
/// compared with 12th.
///
/// Sum of tonicity score of each note of the dyad is 1. Tonicity of the higher note can be obtained
/// with 1 - lower note tonicity.
pub struct TonicityLookup {
    /// Tonicity score of interval per half-cents. cents = 1/2 * index.
    ///
    /// Tonicity is relative to the lower note.
    raw_tonicity: [f64; 9601],

    /// The per-octave normalized tonicity. Since tonicity calculation is done by inverting
    /// intervals about the octave, there is a bias where intervals in the upper half of that octave
    /// will have higher tonicity than the lower half, since the wider the interval, the lower its
    /// roughness.
    ///
    /// This undoes that by fitting a 5th order polynomial to model the bias, and subtracting the
    /// bias from the raw tonicity, and re-centering it at 0.5.
    ///
    /// Also, higher octaves have higher variance in tonicity, so we divide by some function of the
    /// number of octaves to normalize that.
    normalized_tonicity: [f64; 9601],

    /// This is `normalized_tonicity`` smoothed with a 21-bin (+/- 10 cents) with sigma = 5.
    ///
    /// The raw normalized tonicity has very strict tolerances for detunings, e.g., the tonicity score between
    /// 700 cents 12edo fifth is 0.5499, while the 701 cents just fifth is 0.554, which is a relatively large
    /// jump for a 1 cent difference. This smoothed tonicity lookup can be used to reduce that jitter.
    smooth_tonicity: [f64; 9601],
}

fn compute_smooth_tonicity(normalized: &[f64; 9601]) -> [f64; 9601] {
    const KERNEL_BINS: usize = 41;
    const KERNEL_SIGMA: f64 = 20.0;
    let kernel = gaussian_kernel(KERNEL_BINS, KERNEL_SIGMA);
    let kernel_half = KERNEL_BINS as isize / 2;
    let mut smooth_tonicity = [0f64; 9601];
    for i in 0..=9600 {
        let mut sum = 0.0;
        for k in 0..KERNEL_BINS {
            let index = ((i + k) as isize - kernel_half).clamp(0, 9600isize) as usize;
            sum += normalized[index] * kernel[k];
        }
        smooth_tonicity[i] = sum;
    }

    smooth_tonicity
}

/// Global instance of [TonicityLookup]
static TONICITY_LOOKUP: OnceLock<Box<TonicityLookup>> = OnceLock::new();

impl TonicityLookup {
    /// Obtain global tonicity lookup table from codegen.
    pub fn get() -> &'static TonicityLookup {
        TONICITY_LOOKUP.get_or_init(|| {
            Box::new(TonicityLookup {
                raw_tonicity: RAW_TONICITY,
                normalized_tonicity: NORMALIZED_TONICITY,
                smooth_tonicity: compute_smooth_tonicity(&NORMALIZED_TONICITY),
            })
        })
    }
    /// Recomputes the tonicity lookup table from scratch (for updating codegen cache).
    ///
    /// Only run this when Tonicity lookup algo changes.
    ///
    /// This function depends on DYAD_LOOKUP, but does not recompute it!
    ///
    /// If a change is made to the dyad roughness calculation, both codegens must be updated.
    pub fn recompute() -> &'static TonicityLookup {
        TONICITY_LOOKUP.get_or_init(|| {
            // Arbitrary scale, should have value 0.5 at octaves denoting neutral tonicity.
            // Each index corresponds to half a cent.
            let mut raw_tonicity = [0f64; 9601];

            for half_cents in 0..=9600 {
                let cents = half_cents as f64 * 0.5;
                let octave_reduced_cents = cents % 1200.0;
                let inverted_cents = cents - octave_reduced_cents + 1200.0 - octave_reduced_cents;
                let upward_dissonance =
                    DyadLookup::get_roughness(cents, RoughnessType::TonicityNormalized);
                let inverted_dissonance =
                    DyadLookup::get_roughness(inverted_cents, RoughnessType::TonicityNormalized);

                // High inverted dissonance relative to low upward dissonance = high tonicity for lower note.
                raw_tonicity[half_cents] =
                    inverted_dissonance / (upward_dissonance + inverted_dissonance);
            }

            let mut normalized_tonicity = [0f64; 9601];

            // Used to be degree 5. Check which is better.
            let mut octave_regressor = PolynomialRegressor::new(5);

            /* OLD NORMALIZED TONICITY
            // Now we normalize each octave over the indices [0, 2400], [2400, 4800], [4800, 7200], [7200, 9600] inclusive.


            for octave in 0..=3 {
                let start = octave * 2400;
                let end = start + 2400;
                octave_regressor.fit(
                    &Vec::from_iter((start..=end).map(|x| x as f64)),
                    &raw_tonicity[start..=end],
                );
                let fit = octave_regressor.predict(&Vec::from_iter((start..=end).map(|x| x as f64)));
                for i in start..=end {
                    let nt = (raw_tonicity[i] - fit[i - start]) * 2.0 / (octave as f64 + 1.0); // this is about 0
                    let nt = nt / 0.05; // this scales it so that min and max is close to -1 and 1.

                    // make it more opinionated, (skew away from 0)
                    if nt >= 0.0 {
                        normalized_tonicity[i] = (nt * 10.0 + 1.0).log(10.0);
                    } else {
                        normalized_tonicity[i] = -(- nt * 10.0 + 1.0).log(10.0);
                    }
                    normalized_tonicity[i] = normalized_tonicity[i] / 5.0 + 0.5;
                }
            }
            */

            // NEW NORMALIZED TONICITY

            // Every octave has a sawtooth-like upwards linear bias, and each octave has its own
            // mean and variance. These discrepancies are caused by roughness scores having
            // different distributions in each octave, which is expected.
            //
            // Here's how I want to normalize it: First, apply polynomial regression to find
            // a best-fit to match the bias. The bias can be fitted to a degree 4 polynomial
            // and subtracted.
            //
            // Next, obtain the standard deviation of the adjusted tonicity in that octave, and
            // scale so that each octave has equal variance (and preferably, min/max is within 0.4-0.6)

            for octave in 0..=3 {
                let start = octave * 2400;
                let end = start + 2400;

                octave_regressor.fit(
                    &Vec::from_iter((start..=end).map(|x| x as f64)),
                    &raw_tonicity[start..=end],
                );
                let fit =
                    octave_regressor.predict(&Vec::from_iter((start..=end).map(|x| x as f64)));

                let mut sum_tonicity_octave = 0.0; // for calculating mean & variance

                // First, subtract bias
                for half_cents in start..=end {
                    normalized_tonicity[half_cents] =
                        raw_tonicity[half_cents] - fit[half_cents - start];
                    sum_tonicity_octave += normalized_tonicity[half_cents];
                }

                let mean_tonicity_octave = sum_tonicity_octave / 2401.0;

                let mean_squared_sum = normalized_tonicity[start..=end]
                    .iter()
                    .map(|x| (x - mean_tonicity_octave).powi(2))
                    .sum::<f64>();

                let tonicity_octave_variance = mean_squared_sum / 2401.0;

                // Normalize to variance 0.0001 and mean 0.5
                let scale = (0.0001 / tonicity_octave_variance).sqrt();
                for half_cents in start..=end {
                    normalized_tonicity[half_cents] =
                        (normalized_tonicity[half_cents] - mean_tonicity_octave) * scale + 0.5;
                }
            }

            Box::new(TonicityLookup {
                raw_tonicity,
                normalized_tonicity,
                smooth_tonicity: compute_smooth_tonicity(&normalized_tonicity),
            })
        })
    }

    /// The tonicity score of a dyad.
    ///
    /// The dyad tonicity score models the probability of hearing the lower note of an interval
    /// given by `cents` as the tonic over the other (without context).
    ///
    /// It is modelled by comparing the dissonance measure score with the octave inverted version,
    /// while keeping the same octave displacement. It is assumed to otonal structures have
    /// lower dissonance than octave inverted ones, after applying octave equalization, so this model
    /// can also be seen as attributing tonicity to the likelihood of otonality perception.
    ///
    /// E.g., a fifth becomes a fourth, a major third becomes a minor sixth, 11th becomes a 12th.
    ///
    /// Sum of tonicity score of each note of the dyad is 1 (so the tonicity of the higher note is 1
    /// - dyad_tonicity)
    ///
    /// IMPORTANT: The tonicity of very wide dyads (> 4 octaves) is extremely wonky and doesn't help much,
    /// any higher than 4 octaves will be treated as the 3rd octave.
    pub fn dyad_tonicity(cents: f64) -> f64 {
        let tonicity = TonicityLookup::get().normalized_tonicity;
        let cents = if cents > 4800.0 {
            cents - (cents / 1200.0 - 3.0).floor() * 1200.0
        } else {
            cents
        };

        let half_cents = cents * 2.0; // double resolution lookup.
                                      // linearly interpolate values
        let half_cents_floor = half_cents.floor();
        let remainder = half_cents - half_cents_floor;
        let tonicity_lower = tonicity[half_cents_floor as usize];
        let tonicity_higher = if half_cents_floor <= 9599.0 {
            tonicity[half_cents_floor as usize + 1]
        } else {
            0.5
        };

        tonicity_lower + (tonicity_higher - tonicity_lower) * remainder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::prelude::*;

    /// Recomputes and exports the sethares roughness codegen & CSV.
    ///
    /// Originally this took around 5 minutes, but with rayon parallel iterator it is now reduced to
    /// 30 seconds :)
    #[test]
    fn calculate_roughness() {
        let dyad_lookup = DyadLookup::recompute();

        let mut csv_file = File::create(format!(
            "sethares_roughness_{NUM_HARMS_ROUGHNESS}_{HARMS_ADD_OCTAVES_TONALITY}_{AMP_MULT_TONALITY}.csv"
        ))
        .unwrap();

        csv_file
            .write_all(b"cents,roughness,scaled mult,scaled add,normalized\n")
            .unwrap();
        for cents in 0..=10800 {
            let roughness = dyad_lookup.sethares_roughness[cents];
            let scaled_mult = dyad_lookup.sethares_mult[cents];
            let scaled_add = dyad_lookup.sethares_add[cents];
            let normalized = dyad_lookup.sethares_normalized_roughness[cents];
            let line = format!(
                "{},{},{},{},{}\n",
                cents, roughness, scaled_mult, scaled_add, normalized
            );
            csv_file.write_all(line.as_bytes()).unwrap();
        }

        let mut codegen_file = File::create("src/roughness_codegen.rs").unwrap();

        let size = dyad_lookup.sethares_roughness.len();
        codegen_file
            .write_all(
                format!(
                    "pub const SETHARES_ROUGHNESS: [f64; {size}] = {:?};\n\
                     pub const ROUGHNESS_MIN: f64 = {};\n\
                     pub const ROUGHNESS_MAX: f64 = {};\n\
                     pub const SETHARES_MULT: [f64; {size}] = {:?};\n\
                     pub const SETHARES_ADD: [f64; {size}] = {:?};\n\
                     pub const SETHARES_NORMALIZED_ROUGHNESS: [f64; {size}] = {:?};\n",
                    dyad_lookup.sethares_roughness,
                    dyad_lookup.roughness_min,
                    dyad_lookup.roughness_max,
                    dyad_lookup.sethares_mult,
                    dyad_lookup.sethares_add,
                    dyad_lookup.sethares_normalized_roughness,
                )
                .as_bytes(),
            )
            .unwrap();
    }

    #[test]
    fn test_sethares_interpolation() {
        let last_cents = DyadLookup::get().sethares_roughness.len() - 1;
        assert!(
            DyadLookup::get_roughness(last_cents as f64, RoughnessType::Multiplicative)
                == DyadLookup::get().sethares_roughness[last_cents]
        );
        assert!(
            DyadLookup::get_roughness(701.239, RoughnessType::Multiplicative)
                < DyadLookup::get_roughness(387.1, RoughnessType::Multiplicative)
        )
    }

    /// Recomputes and exports the tonicity codegen & CSV.
    ///
    /// Less than 1 second.
    #[test]
    fn calculate_tonicity() {
        let lookup = TonicityLookup::recompute();
        let mut csv_file = File::create(format!(
            "dyad_tonicity_{NUM_HARMS_TONALITY}_{HARMS_ADD_OCTAVES_TONALITY}_{AMP_MULT_TONALITY}.csv"
        ))
        .unwrap();
        csv_file
            .write_all(b"cents,tonicity,normalized,smooth\n")
            .unwrap();
        for half_cents in 0..=9600 {
            let line = format!(
                "{:.1},{},{},{}\n",
                half_cents as f64 / 2.0,
                lookup.raw_tonicity[half_cents],
                lookup.normalized_tonicity[half_cents],
                lookup.smooth_tonicity[half_cents]
            );
            csv_file.write_all(line.as_bytes()).unwrap();
        }

        let size = lookup.raw_tonicity.len();

        // Codegen only computes raw and normalized, smooth tonicity is fast and easy to compute.
        let mut codegen_file = File::create("src/tonicity_codegen.rs").unwrap();
        codegen_file
            .write_all(
                format!(
                    "pub const RAW_TONICITY: [f64; {size}] = {:?};\n\
                     pub const NORMALIZED_TONICITY: [f64; {size}] = {:?};\n",
                    lookup.raw_tonicity, lookup.normalized_tonicity,
                )
                .as_bytes(),
            )
            .unwrap();
    }
}
