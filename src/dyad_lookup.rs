use std::sync::OnceLock;

use compute::predict::PolynomialRegressor;

use crate::{sethares_with_harms, utils::cents_to_hz};

/// How many harmonics to use in the precalculated sethares dissonance score of dyad lookup.
/// 32 will do just fine. 128 has a huge performance hit but yet the graphs look almost the same.
const NUM_HARMS_TONALITY: u32 = 19;

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
    /// Final dissonance calculation should use this so that values are easily scalable.
    sethares_add: [f64; 10801],

    /// Similar to `sethares_roughness` but normalized by degree 9 polynomial regressions such that
    /// all 10 octaves have the same dissonance score and the peak roughness around each octave is
    /// approximately equal.
    ///
    /// Normalized roughness is between 1 and 2.
    ///
    /// Used for tonality discernment.
    sethares_normalized_roughness: [f64; 10801],
}

/// Global instance of [DyadLookup]
static DYAD_LOOKUP: OnceLock<Box<DyadLookup>> = OnceLock::new();

pub enum RoughnessType {
    /// Normalized between 1 and 2 such that every octave has approximately the same roughness.
    ///
    /// Used for TonicityLookup calculatiion and the dyadic tonicity heuristic.
    TonalityNormalized,

    /// Linearly scaled such that the minimum roughness is 1 and the roughness of one octave is OCT_ROUGHNESS.
    Multiplicative,

    /// Linearly scaled such that the minimum roughness is 0 and the max roughness is 1.
    Additive
}

impl DyadLookup {
    pub fn get() -> &'static DyadLookup {
        DYAD_LOOKUP.get_or_init(|| {
            let octave_cents = [
                0.0, 1200.0, 2400.0, 3600.0, 4800.0, 6000.0, 7200.0, 8400.0, 9600.0, 10800.0,
            ];
            let mut sethares_roughness = [0f64; 10801];
            let mut sethares_roughness_tonality = [0f64; 10801]; // for normalized roughness for tonality calculation.
            let mut freqs = [440.0, 440.0];
            let mut min = f64::MAX;
            let mut max = f64::MIN;

            let mut peak_roughness_around_octave = [0f64; 10];

            for cents in 0..=10800 {
                freqs[1] = cents_to_hz(440.0, cents as f64);
                sethares_roughness[cents] =
                    sethares_with_harms(&freqs, 31, 1, 0.86, true);
                sethares_roughness_tonality[cents] =
                    sethares_with_harms(&freqs, NUM_HARMS_TONALITY, HARMS_ADD_OCTAVES_TONALITY, AMP_MULT_TONALITY, true);

                let near_octave_above = cents % 1200 <= 550;
                let near_octave_below = cents % 1200 >= 650;
                let octave = cents / 1200 + if near_octave_below { 1 } else { 0 };

                if near_octave_above || near_octave_below {
                    if peak_roughness_around_octave[octave] < sethares_roughness_tonality[cents] {
                        peak_roughness_around_octave[octave] = sethares_roughness_tonality[cents];
                    }
                }

                if sethares_roughness[cents] < min {
                    min = sethares_roughness[cents];
                }
                if sethares_roughness[cents] > max {
                    max = sethares_roughness[cents];
                }
            }

            let mut sethares_normalized_roughness = [0f64; 10801];

            // regressor for multiples of 1200 cents, lower bound of roughness.
            let mut octave_regressor = PolynomialRegressor::new(9);
            octave_regressor.fit(
                &octave_cents,
                &[
                    sethares_roughness_tonality[0],
                    sethares_roughness_tonality[1200],
                    sethares_roughness_tonality[2400],
                    sethares_roughness_tonality[3600],
                    sethares_roughness_tonality[4800],
                    sethares_roughness_tonality[6000],
                    sethares_roughness_tonality[7200],
                    sethares_roughness_tonality[8400],
                    sethares_roughness_tonality[9600],
                    sethares_roughness_tonality[10800],
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
                    width = 0.0001; // tonality detection above 7 octaves interval has some wonky asymptotes.
                }
                sethares_normalized_roughness[cents] =
                    (sethares_roughness_tonality[cents] - lower_bound_regression[cents]) / width + 1.0;
            }

            let mut sethares_mult = [0f64; 10801];
            // multiplier required to reach desired 1st octave roughness after fitting min to 1.
            let octave_roughness_multiplier = (OCT_ROUGHNESS + min - 1.0) / (sethares_roughness[1200]);

            for cents in 0..=10800 {
                sethares_mult[cents] = sethares_roughness[cents] * octave_roughness_multiplier + (1.0 - min);
            }

            let mut sethares_add = [0f64; 10801];
            for cents in 0..=10800 {
                sethares_add[cents] = (sethares_roughness[cents] - min) / (max - min);
            }

            Box::new(DyadLookup {
                sethares_roughness,
                sethares_normalized_roughness,
                sethares_mult,
                sethares_add,
                roughness_max: max,
                roughness_min: min
            })
        })
    }

    /// Obtains the linearly interpolated sethares roughness measure from an interval of `cents`.
    ///
    /// If using for dyad tonality calculation, set `for_tonality` to true.
    ///
    /// It will used the normalized roughness score.
    pub fn get_roughness(cents: f64, rough_type: RoughnessType) -> f64 {
        let roughness = match rough_type {
            RoughnessType::TonalityNormalized => DyadLookup::get().sethares_normalized_roughness,
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

    /// The per-octave normalized tonicity. Since tonality inversion is done about the octave, there
    /// is a bias where intervals in the upper half of that octave will have higher tonicity than
    /// the lower half, by virtue that the wider the interval, the lower its roughness.
    ///
    /// This undoes that by fitting a 5th order polynomial to model the bias, and subtracting the
    /// bias from the raw tonicity, and re-centering it at 0.5.
    ///
    /// Also, higher octaves have higher variance in tonicity, so we divide by some function of the
    /// number of octaves to normalize that.
    normalized_tonicity: [f64; 9601]
}

/// Global instance of [TonicityLookup]
static TONICITY_LOOKUP: OnceLock<Box<TonicityLookup>> = OnceLock::new();

impl TonicityLookup {
    pub fn get() -> &'static TonicityLookup {
        TONICITY_LOOKUP.get_or_init(|| {
            let mut raw_tonicity = [0f64; 9601];

            for half_cents in 0..=9600 {
                let cents = half_cents as f64 * 0.5;
                let octave_reduced_cents = cents % 1200.0;
                let inverted_cents = cents - octave_reduced_cents + 1200.0 - octave_reduced_cents;
                let upward_dissonance = DyadLookup::get_roughness(cents, RoughnessType::TonalityNormalized);
                let inverted_dissonance = DyadLookup::get_roughness(inverted_cents, RoughnessType::TonalityNormalized);

                // High inverted dissonance relative to low upward dissonance = high tonicity for lower note.
                raw_tonicity[half_cents] = inverted_dissonance / (upward_dissonance + inverted_dissonance);
            }

            // Now we normalize each octave over the indices [0, 2400], [2400, 4800], [4800, 7200], [7200, 9600] inclusive.

            let mut normalized_tonicity = [0f64; 9601];

            let mut octave_regressor = PolynomialRegressor::new(5);

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

            Box::new(TonicityLookup {
                raw_tonicity,
                normalized_tonicity
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
        let tonicity_higher = if half_cents_floor <= 9599.0 {tonicity[half_cents_floor as usize + 1]} else {0.5};

        tonicity_lower + (tonicity_higher - tonicity_lower) * remainder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::prelude::*;
    #[test]
    fn export_roughness_to_csv() {

        let dyad_lookup = DyadLookup::get();

        let mut file = File::create(format!(
            "sethares_roughness_{NUM_HARMS_TONALITY}_{HARMS_ADD_OCTAVES_TONALITY}_{AMP_MULT_TONALITY}.csv"
        ))
        .unwrap();
        file.write_all(b"cents,roughness,scaled mult,scaled add,normalized\n").unwrap();
        for cents in 0..=10800 {
            let roughness = dyad_lookup.sethares_roughness[cents];
            let scaled_mult = dyad_lookup.sethares_mult[cents];
            let scaled_add = dyad_lookup.sethares_add[cents];
            let normalized = dyad_lookup.sethares_normalized_roughness[cents];
            let line = format!("{},{},{},{},{}\n", cents, roughness, scaled_mult, scaled_add, normalized);
            file.write_all(line.as_bytes()).unwrap();
        }
    }

    #[test]
    fn test_sethares_interpolation() {
        let last_cents = DyadLookup::get().sethares_roughness.len() - 1;
        assert!(
            DyadLookup::get_roughness(last_cents as f64, RoughnessType::Multiplicative)
                == DyadLookup::get().sethares_roughness[last_cents]
        );
        assert!(DyadLookup::get_roughness(701.239, RoughnessType::Multiplicative) < DyadLookup::get_roughness(387.1, RoughnessType::Multiplicative))
    }

    #[test]
    fn export_dyad_tonicity_to_csv() {
        let lookup = TonicityLookup::get();
        let mut file = File::create(format!(
            "dyad_tonicity_{NUM_HARMS_TONALITY}_{HARMS_ADD_OCTAVES_TONALITY}_{AMP_MULT_TONALITY}.csv"
        ))
        .unwrap();
        file.write_all(b"cents,tonicity,normalized\n").unwrap();
        for half_cents in 0..=9600 {
            let line = format!("{:.1},{},{}\n", half_cents as f64 / 2.0, lookup.raw_tonicity[half_cents], lookup.normalized_tonicity[half_cents]);
            file.write_all(line.as_bytes()).unwrap();
        }
    }
}
