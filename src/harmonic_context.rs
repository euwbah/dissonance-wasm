//! Implements methods for real-time analysis using [polyadic::graph_dissonance]

use core::f64;

use wasm_bindgen::prelude::*;

use crate::polyadic::graph_dissonance;

/// TODO: SMOOTHING should not be hardcoded, but set based on rhythmic entrainment. On strong beats
/// this value should decrease and on weak beats it should increase.
const SMOOTHING: f64 = 0.9;

/// Maximum number of random trees to sample when evaluating dissonance.
///
/// Setting this value too high can lead to long computation times. Setting too low can lead to
/// deterioration for larger chords.
///
/// Number of interpretation trees for N notes:
///
/// 2. 2
/// 3. 9
/// 4. 64
/// 5. 400
/// 6. 1,842
/// 7. 6,972
/// 8. 23,104
const MAX_TREES: usize = 800;

/// Maximum number of trees to compute amongst all candidates.
///
/// The max_trees computed per candidate will be scaled by the number of candidates, so more
/// candidates = more deterioration.
const MAX_TREES_CANDIDATES: usize = 4_000;

/// Epsilon value to determine if two dissonance values are "close enough" to fall back to
/// tonicity comparison.
const DISS_EPSILON: f64 = 0.001;

/// Evaluate tonicity and dissonaance of the current state of notes.
///
///  * `freqs`: List of frequencies in pitch memory.
///  * `context`: Current tonicity context (should be same length as freqs)
///  * `elapsed_seconds`: Time elapsed since last update, in seconds.
///
/// Returns a list of floats `[tonicity_1, tonicity_2, ..., dissonance]` where:
///
///  * `tonicity_i` is the relative tonicity of the i-th note. (Tonicities sum to 1)
///  * `dissonance` is the final evaluated dissonance of the chord.
#[wasm_bindgen(js_name=updateTonicity)]
pub fn update_tonicity(freqs: &[f64], context: &[f64], elapsed_seconds: f64) -> Vec<f64> {
    let diss = &graph_dissonance(
        freqs,
        &[],
        context,
        SMOOTHING,
        elapsed_seconds,
        MAX_TREES,
        None,
    )[0];

    let mut res = vec![];

    res.extend_from_slice(&diss.tonicity_context);

    res.push(diss.dissonance);

    res
}

/// Given the existing `freqs` in pitch memery, the number of candidates per root, and a flattened
/// array of the ratios for each candidate for each root in `freqs`, select the best candidate root
/// and candidate ratio.
///
/// - `freqs`: List of frequencies in pitch memory.
///
/// - `num_cands_per_freq`: How many candidate ratios per frequency in `freqs` short term memory
///
/// - `candidate_ratios`: Flattened array of frequency multiples of frequencies in `freqs`.
///
/// - `context`: Current tonicity context (should be same length as freqs)
///
/// - `elapsed_seconds`: Time elapsed since last update, in seconds.
///
/// - `top_n_roots`: How many notes in `freqs` should be considered as relative roots for candidate
///   frequencies. Only the top n highest tonicity notes in `freqs` will be considered.
///
/// - `method`: 0 to select candidate with minimum dissonance, 1 to select candidate with maximum
///   tonicity.
///
/// E.g., if freqs = [200, 300] and cands_per_freq = [1, 2], and candidate_ratios = [1.5, 2, 1.5]
///
/// then the candidates are:
///
/// - 1.5 * 200hz = 300hz seen relative to 200hz,
/// - 2 * 300hz = 600hz seen relative to 300hz,
/// - 1.5 * 300hz = 450hz seen relative to 300hz.
///
/// ## Returns
///
/// (cand_idx, cand_ratio_idx, tonicity_1, tonicity_2, ..., tonicity_cand)
///
/// - `cand_idx`: Index of which note in short term memory the candidate note is heard relative to.
///
/// - `cand_ratio_idx`: Index of which candidate ratio should be used (as per the unflattened array)
///
/// - `tonicity_1, ..., tonicity_cand`: the tonicities of the existing notes in the same order as
///   freqs, together with the tonicity of the newly added candidate note at the end, rescaled so
///   that tonicities sum to 1 after including the candidate.
///
/// of the index of frequency in `freqs` and the index of the candidate ratio (relative to the
/// unflattened map, not the flattened one).
///
/// E.g. using the above example candidates, a return value of (1, 0, t_1, t_2, t_3) means the
/// algorithm has selected the second freq 300hz and the first candidate ratio of the second freqs
/// which is 2, so the selected candidate is 2/1 of 300hz. The tonicities t_1, t_2 correspond to the
/// updated tonicities of freqs[0] and freqs[1], after including the new candidate frequency, and
/// t_3 is the tonicity of the new candidate frequency.
///
#[wasm_bindgen(js_name=selectCandidate)]
pub fn select_candidate(
    freqs: &[f64],
    num_cands_per_freq: &[usize],
    candidate_ratios: &[f64],
    context: &[f64],
    elapsed_seconds: f64,
    top_n_roots: usize,
    method: usize,
) -> Vec<f64> {
    let mut candidate_frequencies_per_freq: Vec<&[f64]> = vec![];
    let mut idx = 0;
    for num_cands in num_cands_per_freq {
        candidate_frequencies_per_freq.push(&candidate_ratios[idx..(idx + num_cands)]);
        idx += num_cands;
    }

    let method_min_diss = method == 0;

    let mut best_diss = f64::MAX;
    let mut best_cand_tonicity = -1.0;
    let mut best_cand_freq_idx = 0; // idx of freqs that yields the best candidate
    let mut best_cand_ratio_idx = 0; // idx of candidate ratio that yields the best candidate
    let mut best_cand_tonicities: Vec<f64> = vec![]; // tonicity context when using best candidate

    let max_trees = MAX_TREES_CANDIDATES / candidate_frequencies_per_freq.len();

    let top_n_freq_tonicity_indices = {
        let mut tonicity_indices: Vec<(usize, f64)> =
            context.iter().enumerate().map(|(i, t)| (i, *t)).collect();
        tonicity_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        tonicity_indices
            .iter()
            .take(top_n_roots)
            .map(|(i, _)| *i)
            .collect::<Vec<usize>>()
    };

    for &freq_idx in &top_n_freq_tonicity_indices {
        let candidate_freqs = &candidate_frequencies_per_freq[freq_idx];
        let results = graph_dissonance(
            freqs,
            candidate_freqs,
            context,
            SMOOTHING,
            elapsed_seconds,
            max_trees,
            None,
        );

        for (cand_idx, diss) in results.iter().enumerate() {
            if method_min_diss {
                let lower_diss = diss.dissonance <= best_diss - DISS_EPSILON;
                let same_diss_but_more_tonic_relative_note = diss.dissonance - best_diss
                    <= DISS_EPSILON
                    && context[freq_idx] > context[best_cand_freq_idx];
                if lower_diss || same_diss_but_more_tonic_relative_note {
                    best_cand_freq_idx = freq_idx;
                    best_cand_ratio_idx = cand_idx;
                    best_cand_tonicities = diss.tonicity_context.clone();
                    best_diss = diss.dissonance;
                    best_cand_tonicity = context[freq_idx];
                }
            } else {
                // max_tonicity
                let higher_tonicity = diss.tonicity_context[freq_idx] > best_cand_tonicity;
                if higher_tonicity {
                    best_cand_freq_idx = freq_idx;
                    best_cand_ratio_idx = cand_idx;
                    best_cand_tonicities = diss.tonicity_context.clone();
                    best_diss = diss.dissonance;
                    best_cand_tonicity = diss.tonicity_context[freq_idx];
                }
            }
        }
    }

    let mut output = vec![best_cand_freq_idx as f64, best_cand_ratio_idx as f64];

    output.extend_from_slice(&best_cand_tonicities);

    output
}
