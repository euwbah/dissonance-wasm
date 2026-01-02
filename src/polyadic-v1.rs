//! Polyadic dissonance & tonicity of notes.
//!
//! This module implements the first flawed algorithm as described in the article from the beginning
//! until the section before "The big problem: Duality is still hiding".

use std::collections::{BTreeSet, HashMap};

use compute::prelude::{max, softmax};

use crate::{
    dyad_lookup::{DyadLookup, RoughnessType, TonicityLookup},
    tree_gen::{ST, TREES},
    utils::hz_to_cents,
};

const PRINT_GRAPH: bool = false;

/// How opinionated the tonicity heuristic should be when computing the initial dyadic tonicities.
///
/// Lower temperature = more opinionated.
const HEURISTIC_DYAD_TONICITY_TEMP: f64 = 0.8;

/// How opinionated local tonicity score (compared between only children of a subtree's root) is.
///
/// Lower temperature = more opinionated.
const LOCAL_TONICITY_TEMP: f64 = 0.7;

/// What fraction of the heuristic tonicity computed from `dyadic_tonicity_heur()` should be
/// assigned to a new candidate note.
///
/// Using a smaller value than the heuristic models the idea that newer notes are always less
/// accepted/certain than notes already present in the pitch memory.
const NEW_CANDIDATE_TONICITY_RATIO: f64 = 0.2;

/// The tonicity vector is a list of heuristic probabilities of each note being perceived as tonic
/// relative to the entire chord.
///
/// This should sum to 1.
pub type Tonicities = Vec<f64>;

/// The result of a dissonance calculation.
#[derive(Debug, Clone)]
pub struct Dissonance {
    /// dissonance score
    pub dissonance: f64,

    /// the target tonicity of the notes in the chord.
    pub tonicity_target: Tonicities,

    /// The new tonicity context applying smoothing towards the target.
    pub tonicity_context: Tonicities,
}

/// Helper for statistical computing of [Dissonance] object from aggregated per-root complexities.
///
/// - `complexities_per_root`: Roots are arranged in order of original `freqs` (not ascending). If
///   candidate is in the computation, the complexity scores where the candidate is the root is the
///   last vec in this vec.
///
/// - `target_tonicity_temperature`: How opinionated the target tonicity should be (lower = more
///   opinionated)
///
/// - `tonicity_context`: Existing tonicities of notes in original order. If candidate is in the
///   computation, the candidate tonicity should be at the last index of this array.
///
/// - `smoothing`: how quickly the tonicities converge to target tonicities.
///
/// - `elapsed_seconds`: seconds between now and previous update.
fn compute_interpretation_trees(
    complexities_per_root: &[&[f64]],
    target_tonicity_temperature: f64,
    tonicity_context: &[f64],
    smoothing: f64,
    elapsed_seconds: f64,
) -> Dissonance {
    // Aggregate complexity scores per-root.
    //
    // This is following the results of test_tonicity_update in
    // paper/triad_sts_computation_example.py
    //
    // For tonicity calculation, use exponentially weighted average to prioritize higher
    // complexity scores more (modelling ear choosing to choose roots that do not admit
    // "unreasonably complex" interpretations).
    //
    // For dissonance calculation, use inverse exponentially weighted average to prioritize lower
    // complexity scores (modelling ear choosing to perceive interpretation tree that admits a
    // low-complexity interpretation).

    // Aggregates each root indexed in original `freqs` order.
    let num_notes = complexities_per_root.len();
    let mut agg_exp_weight_comp_per_root = vec![0.0; num_notes];
    let mut agg_inv_exp_weight_comp_per_root = vec![0.0; num_notes];
    for root_idx in 0..num_notes {
        let comps = &complexities_per_root[root_idx];
        let exp_weights: Vec<f64> = comps.iter().map(|c| c.exp()).collect();
        let exp_weight_sum: f64 = exp_weights.iter().sum();
        let inv_exp_weights: Vec<f64> = comps.iter().map(|c| (1.0 - c).exp()).collect();
        let inv_exp_weight_sum: f64 = inv_exp_weights.iter().sum();

        // Pair of (exp_weighted_avg, inv_exp_weighted_avg)
        let weighted_avgs: (f64, f64) = comps
            .iter()
            .zip(exp_weights.iter())
            .zip(inv_exp_weights.iter())
            .map(|((c, exp_w), inv_exp_w)| (c * exp_w, c * inv_exp_w))
            .fold((0.0, 0.0), |acc, (w_c, inv_w_c)| {
                (acc.0 + w_c, acc.1 + inv_w_c)
            });
        let exp_weighted_avg = weighted_avgs.0 / exp_weight_sum;
        let inv_exp_weighted_avg = weighted_avgs.1 / inv_exp_weight_sum;
        agg_exp_weight_comp_per_root[root_idx] = exp_weighted_avg;
        agg_inv_exp_weight_comp_per_root[root_idx] = inv_exp_weighted_avg;
    }

    // Compute target tonicity scores using softmax of negative aggregated complexities.

    let target_tonicities = softmax(
        &agg_exp_weight_comp_per_root
            .iter()
            .map(|c| (1.0 - c) / target_tonicity_temperature)
            .collect::<Vec<f64>>(),
    );

    // The final dissonance is computed from aggregated inverse exp weighted complexities
    // per root, multiplied by root tonicity of the current context.

    let complexity: f64 = agg_inv_exp_weight_comp_per_root
        .iter()
        .enumerate()
        .map(|(root_idx, c)| c * tonicity_context[root_idx])
        .sum();

    let updated_tonicity_context = tonicity_smoothing(
        &target_tonicities,
        tonicity_context,
        smoothing,
        elapsed_seconds,
    );

    Dissonance {
        dissonance: complexity,
        tonicity_target: target_tonicities,
        tonicity_context: updated_tonicity_context,
    }
}

/// Represents a tree-dissonance pair.
///
/// Ord is implemented with respect to increasing dissonance.
#[derive(Clone, Debug)]
pub struct TreeDissEntry {
    pub tree: &'static ST,
    pub diss: f64,
}

impl PartialEq for TreeDissEntry {
    fn eq(&self, other: &Self) -> bool {
        self.diss == other.diss
    }
}

impl Eq for TreeDissEntry {}

impl PartialOrd for TreeDissEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.diss.partial_cmp(&other.diss)
    }
}

impl Ord for TreeDissEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.diss.partial_cmp(&other.diss).unwrap()
    }
}

/// Pass an empty `&mut` [GraphDissDebug] into [graph_dissonance] to obtain
/// debug & test data.
///
/// DO NOT USE THIS IN PROD. SLOW ⚠️
pub struct GraphDissDebug {
    /// What `n` is in the variable names below.
    N: usize,

    /// For each root, what are the N lowest complexity trees.
    ///
    /// NOTE: This debug value is only available if there are no candidate notes provided.
    ///
    /// The purpose of this value is to check whether the most consonant trees correspond to intuitive
    /// interpretations of the chord.
    n_lowest_comp_trees_per_root: Vec<BTreeSet<TreeDissEntry>>,
}

impl GraphDissDebug {
    /// Create a new [GraphDissDebug].
    ///
    /// - `n`: how many trees to collect (see variable names)
    /// - `num_notes`: how many notes in this chord tree
    pub fn new(n: usize, num_notes: usize) -> Self {
        let n_lowest_comp_trees_per_root = vec![BTreeSet::new(); num_notes];
        GraphDissDebug {
            N: n,
            n_lowest_comp_trees_per_root: n_lowest_comp_trees_per_root,
        }
    }
}

/// Polyadic dissonance by iterating over all spanning trees and root
///
/// ### Parameters
///
/// - `freqs`: list of frequencies of the notes in the chord.
///
/// - `candidate_freqs`: list of possible frequency interpretations of a note that will be added to
/// the chord (by detempering). If empty, only `freqs` will be computed. However, if provided,
/// dissonances will be calculated for each choice of frequency in `candidate_freqs`.
///
/// - `tonicity_context`: corresponds to existing tonicity scores of corresponding freqs.
///   `tonicity_context` will be normalized to sum to 1.
///
/// - `smoothing`: How quickly the updated tonicity context should approach the target tonicity. 0.0
///   = no smoothing, 1.0 = no movement at all.
///
/// - `elapsed_seconds`: time elapsed used to scale smoothing - higher elapsed time = less
///   smoothing.
///
/// - `target_tonicity_temperature`: softmax temperature which controls opinionatedness of target
///   tonicity (lower = more opinionated). 0.5 is recommended.
///
/// If a new loud note is played, the tonicities can be jerked by setting smoothing lower or
/// elapsed_seconds higher than usual. Rhythmic entrainment can be implemented by setting smoothing
/// lower at regular time intervals and higher at others.
///
/// ### Returns
///
/// A vector of [Dissonance] objects. If no `candidate_freqs` are provided, the returned vector will
/// have one element, which is the dissonance of the chord given by `freqs`.
///
/// Otherwise, the index of the returned vector corresponds to the index of the choice of added note
/// in the `candidate_freqs` list.
pub fn graph_dissonance(
    freqs: &[f64],
    candidate_freqs: &[f64],
    tonicity_context: &[f64],
    smoothing: f64,
    elapsed_seconds: f64,
    target_tonicity_temperature: f64,
    mut debug: Option<&mut GraphDissDebug>,
) -> Vec<Dissonance> {
    assert!(
        tonicity_context.len() == freqs.len(),
        "tonicity_context must correspond to notes in freqs"
    );

    let has_candidate_notes = candidate_freqs.len() != 0;
    let num_candidates = candidate_freqs.len();
    let num_notes = freqs.len() + if has_candidate_notes { 1 } else { 0 };

    if num_notes < 2 {
        return vec![Dissonance {
            dissonance: 1.0,
            tonicity_target: tonicity_context.to_vec(),
            tonicity_context: tonicity_context.to_vec(),
        }];
    }

    let cents: Vec<f64> = freqs.iter().map(|x| hz_to_cents(440.0, *x)).collect();

    let candidate_cents: Vec<f64> = candidate_freqs
        .iter()
        .map(|x: &f64| hz_to_cents(440.0, *x))
        .collect();

    // Create lookup table for all dyads.
    // The lookup tables are indexed in same order as provided in `freqs`
    let heuristic_tonicities = dyadic_tonicity_heur(
        freqs[0],
        &cents,
        &candidate_cents,
        HEURISTIC_DYAD_TONICITY_TEMP,
    );
    let dyad_roughs = heuristic_tonicities.add_roughness_map; // use additive roughness.
    let cand_roughness_map = heuristic_tonicities.cand_add_roughness_map;

    let mut results = vec![];

    if num_candidates == 0 {
        // First, we need an index mapping of the frequencies in low-to-high pitch order.
        //
        // Note that we have to compute a different sorted idx mapping for different candidate
        // notes, as different candidate notes may result in different pitch order.

        // Input key: ascending pitch order idx, Output: original freqs idx.
        let asc_to_og_idxs = {
            let mut pairs: Vec<(usize, f64)> = cents.iter().cloned().enumerate().collect();
            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            pairs.iter().map(|(idx, _)| *idx).collect::<Vec<usize>>()
        };

        // Input: original freqs idx, Output: ascending pitch order.
        let og_to_asc_idx: Vec<usize> = {
            let mut inv = vec![0; asc_to_og_idxs.len()];
            for (sorted_idx, &orig_idx) in asc_to_og_idxs.iter().enumerate() {
                inv[orig_idx] = sorted_idx;
            }
            inv
        };

        // `tonicity_context` sorted in low-to-high pitch order.
        let sorted_tonicity_ctx: Vec<f64> = asc_to_og_idxs
            .iter()
            .map(|&idx| tonicity_context[idx])
            .collect();

        // No new candidate note: don't use heuristic tonicities, instead use only provided
        // tonicity_context.

        let st_trees = TREES
            .get(num_notes)
            .expect("No precomputed trees for this number of notes");

        // Function to obtain precomputed dyad complexity of pitches indexed in ASCENDING PITCH
        // order.
        let dyad_comp = |from: usize, to: usize| {
            let idx_from = asc_to_og_idxs[from];
            let idx_to = asc_to_og_idxs[to];
            let bitmask = (1 << idx_from) | (1 << idx_to);
            *dyad_roughs
                .get(&bitmask)
                .expect("Missing edge in precomputed dyadic roughness lookup!")
        };

        // Each item in this vec corresponds to complexity scores computed for each root
        //
        // The root indices are ordered in ascending pitch order. To convert back to original
        // order in freqs, use inverse_idx_map.
        let mut complexities_per_root_lo_to_hi = vec![vec![]; num_notes];

        for tree in st_trees.iter() {
            let comp = dfs_st_comp_tonicity(tree, &sorted_tonicity_ctx, dyad_comp);
            complexities_per_root_lo_to_hi[tree.root].push(comp);
            if let Some(d) = &mut debug {
                let entry = TreeDissEntry {
                    tree: &tree,
                    diss: comp,
                };

                let root_btree = &mut d.n_lowest_comp_trees_per_root[tree.root];
                root_btree.insert(entry);

                // Keep only the N lowest complexity trees
                while root_btree.len() > d.N {
                    root_btree.pop_last();
                }
            }
        }

        let complexities_per_root_og_order = (0..num_notes)
            .map(|i| complexities_per_root_lo_to_hi[og_to_asc_idx[i]].as_slice())
            .collect::<Vec<_>>();

        results.push(compute_interpretation_trees(
            &complexities_per_root_og_order,
            target_tonicity_temperature,
            tonicity_context,
            smoothing,
            elapsed_seconds,
        ));

        return results;
    }

    // If candidates are supplied, we have to repeat the above process with each candidate
    // as the last indexed note provided freqs.

    for candidate_idx in 0..num_candidates {
        let curr_candidate_cents = candidate_cents[candidate_idx];
        let asc_to_og_idxs = {
            let mut pairs: Vec<(usize, f64)> = cents
                .iter()
                .cloned()
                .enumerate()
                .collect::<Vec<(usize, f64)>>();
            pairs.push((freqs.len(), curr_candidate_cents));
            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            pairs.iter().map(|(idx, _)| *idx).collect::<Vec<usize>>()
        };

        let og_to_asc_idxs: Vec<usize> = {
            let mut inv = vec![0; asc_to_og_idxs.len()];
            for (sorted_idx, &orig_idx) in asc_to_og_idxs.iter().enumerate() {
                inv[orig_idx] = sorted_idx;
            }
            inv
        };

        // Since we are adding candidate notes, we have to use the dyadic tonicity heuristic to
        // find what initial tonicity score to use for the candidate note and scale accordingly.

        // To model the intuition that "new notes are less accepted", I will only use a fraction of
        // the heuristic tonicity score, and scale down the rest of the existing notes' tonicities
        // linearly. This scaling is arbitrary and can be adjusted later.
        //
        // Other initialization options: preset 0.001 tonicity, 100% heuristic tonicity, etc...

        let curr_candidate_tonicity = heuristic_tonicities.tonicities[candidate_idx][num_notes - 1]
            * NEW_CANDIDATE_TONICITY_RATIO;

        // The new tonicity context including the candidate note, in original order of `freqs`
        // where the candidate note is at the last index.
        let new_tonicity_ctx: Vec<f64> = {
            let scale = 1.0 - curr_candidate_tonicity;
            tonicity_context
                .iter()
                .map(|x| x * scale)
                .chain(std::iter::once(curr_candidate_tonicity))
                .collect()
        };

        let st_trees = TREES
            .get(num_notes)
            .expect("No precomputed trees for this number of notes");

        // Computes dyad complexities for (from, to) indices based on ascending pitch order.
        let dyad_comp = |from: usize, to: usize| {
            let idx_from = asc_to_og_idxs[from];
            let idx_to = asc_to_og_idxs[to];

            let bitmask = (1 << idx_from) | (1 << idx_to);

            if let Some(roughness) = dyad_roughs.get(&bitmask) {
                // both notes are non-candidate notes, the complexity is found in the dyad_roughs lookup.
                return *roughness;
            }
            // implicitly, if idx_from == freqs.len() || idx_to == freqs.len() {

            // "from" node is candidate note
            cand_roughness_map[idx_from.min(idx_to)] // index of existing pitch
                .get(candidate_idx) // which candidate to use
                .cloned()
                .expect("Missing edge in candidate dyadic roughness lookup!")
        };

        let mut complexities_per_root_lo_to_hi: Vec<Vec<f64>> = vec![vec![]; num_notes];

        for tree in st_trees.iter() {
            let complexity = dfs_st_comp_tonicity(tree, tonicity_context, dyad_comp);
            complexities_per_root_lo_to_hi[tree.root].push(complexity);
        }

        let complexities_per_root = (0..num_notes)
            .map(|i| complexities_per_root_lo_to_hi[og_to_asc_idxs[i]].as_slice())
            .collect::<Vec<_>>();

        results.push(compute_interpretation_trees(
            &complexities_per_root,
            target_tonicity_temperature,
            &new_tonicity_ctx,
            smoothing,
            elapsed_seconds,
        ));
    }

    results
}

/// Evaluate spanning tree complexity using DFS.
///
/// IMPORTANT ⚠️⚠️⚠️:
///
///     This function requires input notes to be indexed in ASCENDING PITCH order. The
///     freqs provided from the visualizer are in no particular order, so an sorted index mapping
///     must be obtained first.
///
///     The search space of precomputed trees is pruned so that pre-order inversions are capped
///     at `max_inversions`, which models the natural low-to-high precendence of pitch perception.
///
/// See `paper/article.md` for details.
///
/// Same algo as `dfs()` in `paper/triad_sts_computation_example.py`
///
/// - `tree`: the spanning tree to evaluate
///
/// - `tonicity_ctx`: tonicity context per node. ⚠️ IMPORTANT ⚠️ this must be indexed in order of ASCENDING
///   PITCH
///
/// - `dyad_comp`: function that returns the dyadic/edge additive complexity (range 0-1) of (from,
///   to) edge.
///
///    ⚠️ IMPORTANT ⚠️ the indices of from/to nodes must be in order of ASCENDING PITCH.
///
/// ## Returns
///
/// The subjective complexity of this interpretation tree.
///
fn dfs_st_comp_tonicity<F>(tree: &ST, tonicity_ctx: &[f64], dyad_comp: F) -> f64
where
    F: Fn(usize, usize) -> f64,
{
    // stack contains (node index, children visited?)
    //
    // For post-order traversal.
    let mut stack: Vec<(usize, bool)> = vec![(tree.root, false)];

    // store computed (tonicity, complexity) per node
    //
    // Indexed by same order as tree nodes.
    let mut results: Vec<Option<(f64, Option<f64>)>> = vec![None; tonicity_ctx.len()];

    while let Some((node, visited)) = stack.pop() {
        if !visited {
            // `node` children not yet visited: visit the children of node first.
            stack.push((node, true));
            for &ch in tree.children[node].iter() {
                stack.push((ch, false));
            }
        } else {
            let children = &tree.children[node];
            if children.is_empty() {
                results[node] = Some((tonicity_ctx[node], None));
                continue;
            }

            let mut child_tonicities = Vec::with_capacity(children.len());
            let mut child_complexities = Vec::with_capacity(children.len());

            for &ch in children {
                let (child_ton, child_comp) =
                    results[ch].expect("Child result missing. Check if tree is valid");
                let edge_comp = dyad_comp(node, ch);
                let unweighted = match child_comp {
                    None => edge_comp,
                    Some(cc) => (edge_comp + cc) * 0.5 * (1.0 + 0.5 * (edge_comp - cc)),
                };
                child_tonicities.push(child_ton);
                child_complexities.push(unweighted);
            }

            let total_tonicity: f64 = child_tonicities.iter().sum();
            let normalized: Vec<f64> = child_tonicities
                .iter()
                .map(|ct| ct / total_tonicity)
                .collect();
            let softmax_local_tonicities = softmax(
                &normalized
                    .iter()
                    .map(|t| t / LOCAL_TONICITY_TEMP)
                    .collect::<Vec<f64>>(),
            );

            let total_complexity: f64 = child_complexities
                .iter()
                .zip(softmax_local_tonicities.iter())
                .map(|(c, w)| c * w)
                .sum();

            results[node] = Some((total_tonicity, Some(total_complexity)));
        }
    }

    results[tree.root]
        .expect("Root result missing. Check tree indices")
        .1
        .expect("No complexity computed! Tree only has one node?")
}

#[derive(Debug, Clone)]
pub struct TonicityHeuristic {
    /// One list per note in `candidate_cents`, which contains the relative tonicities of all the existing
    /// notes and the particular new note.
    ///
    /// E.g., if `cents` is [a, b, c] and `candidate_cents` is [d, e], then the return value will be
    ///
    /// [tonicities of [a, b, c, d], tonicities of [a, b, c, e]]
    pub tonicities: Vec<Tonicities>,

    /// `tonicity_map` is a lookup of dyad tonicities of the existing notes in `cents`. The key is a
    /// bitstring containing two 1s, where the bit position (from the LSB) corresponds to the
    /// index of each note in the dyad in `cents`.
    ///
    /// The tonicity in the lookup is with respect to the note with lower index (not lower pitch),
    /// e.g. if the bitstring is 0b1001, then the tonicity is with respect to hearing the note at
    /// index 0 as tonic.
    pub tonicity_map: HashMap<u32, f64>,

    /// `mult_roughness_map` is a lookup of multiplicative roughnesses of the existing notes in
    /// `cents`.
    ///
    /// The i-th bit from the LSB is the index of each note in the dyad in `cents`. E.g., the value
    /// at 0b1001 gives the roughness between the 1st and 4th notes in `cents`.
    pub mult_roughness_map: HashMap<u32, f64>,

    /// `add_roughness_map` is a lookup of additive roughnesses of the existing notes in `cents`.
    pub add_roughness_map: HashMap<u32, f64>,

    /// `cand_tonicity_map` is a lookup of dyad tonicities is indexed by the existing note, followed
    /// by the choice of new candidate note. Its tonicity is with respect to the existing note
    /// against the candidate note.
    ///
    /// E.g., `cand_tonicity_map[i][j] == 0.7` means that i-th existing note in `cents` is 70% tonic
    /// relative to the j-th candidate note in `candidate_cents`.
    ///
    /// This convention agrees with the tonicities in `tonicity_map` being with respect to the lower
    /// index note, and that the candidate note is at the last index.
    pub cand_tonicity_map: Vec<Vec<f64>>,

    /// `cand_mult_roughness_map` is indexed by the existing note, followed by the index of
    /// candidate note.
    ///
    /// E.g., `cand_mult_roughness_map[i][j] = 1.5` means that the roughness of the dyad between the
    /// i-th note in `cents` and the j-th candidate note in `candidate_cents` is 1.5.
    pub cand_mult_roughness_map: Vec<Vec<f64>>,

    /// Indexed by existing note, followed by index of candidate note.
    ///
    /// E.g., `cand_add_roughness_map[i][j] = 0.5` means that the roughness of the dyad between the
    /// i-th note in `cents` and the j-th candidate note in `candidate_cents` is 0.5.
    ///
    /// Same as `cand_mult_roughness_map`, but for additive roughness.
    pub cand_add_roughness_map: Vec<Vec<f64>>,
}

/// Obtain a score from -0.05 to 1 that adds a roughness penalty for intervals that are in a low
/// octave.
///
/// This function should return 0 if the centroid frequency is 400 Hz, and approach 1 as the
/// frequency goes down to 20Hz.
///
/// - `centroid_freq`: the average frequency of the two notes in the dyad in Hz.
fn lower_interval_limit_penalty(centroid_freq: f64) -> f64 {
    return (1.0 / centroid_freq.sqrt() - 0.05).min(1.0);
}

/// Compute additive roughness based on interval in cents and LIL penalty computed from
/// `lower_interval_limit_penalty()`.
///
/// The LIL penalty for additive roughness is computed by taking roots (reciprocal powers). At max
/// penalty, we raise to (1 / 1.5), at min penalty, we raise to (1 / 0.975).
fn get_add_roughness(interval_cents: f64, lil_penalty: f64) -> f64 {
    DyadLookup::get_roughness(interval_cents, RoughnessType::Additive)
        .powf(1.0 / (1.0 + 0.5 * lil_penalty))
}

/// Compute multiplicative roughness based on interval in cents and LIL penalty computed from
/// `lower_interval_limit_penalty()`.
///
/// The LIL penalty for multiplicative roughness is computed by taking the power. At max penalty
/// (1.0), the roughness is raised to the power of 1.1, at min penalty (-0.05), the roughness is
/// raised to the power of 0.995.
///
/// This increases the max roughness for lower intervals, while min roughness is kept at 1.
///
/// NOTE: multiplicative roughness is currently not being used anywhere in this algo. TODO: if
/// unnecessary, remove this computation.
fn get_mult_roughness(interval_cents: f64, lil_penalty: f64) -> f64 {
    DyadLookup::get_roughness(interval_cents, RoughnessType::Multiplicative)
        .powf(1.0 + lil_penalty / 10.0)
}

/// Heuristic initial tonicity of notes in a chord using only dyadic tonicities, where a new note
/// with several possible interpretations is added.
///
/// The heuristic tonicity is a probability distribution of each note being perceived as tonic
/// relative to the other notes.
///
/// ### Parameters
///
/// - `reference_freq`: Reference frequency of the first note in `cents`. This is used to compute a
///   lower interval limit heuristic scaling for roughness scores.
/// - `cents`: the list of already existing notes in the chord.
/// - `candidate_cents`: list of possible interpretations of the new note, relative to the same
/// offset as `cents`.
/// - `tonicity_temperature`: softmax temperature used when normalizing the tonicity scores into a
/// probability distribution. Lower values will make the distribution more "opinionated".
///
/// ### Returns
///
/// See [TonicityHeuristic].
///
/// If `cand_cents` is not provided, the return value will have a single value in `tonicities`,
/// which is the tonicity of the notes in `cents` amongst themselves only.
///
pub fn dyadic_tonicity_heur(
    reference_freq: f64,
    cents: &[f64],
    candidate_cents: &[f64],
    tonicity_temperature: f64,
) -> TonicityHeuristic {
    /// Function that scales the perception of tonicity by otonal/utonal dyadic tonicity weighted by
    /// how likely it is to choose to hear that interval relation between two note (modelled using
    /// roughness).
    ///
    /// Returns a tuple: (tonicity of lower note, tonicity of higher note).
    ///
    /// TODO: figure out the best function for this.
    fn heuristic_function(dyad_tonicity: f64, roughness: f64) -> (f64, f64) {
        // (
        //     dyad_tonicity.powf(roughness / 2.0),
        //     (1.0 - dyad_tonicity).powf(roughness / 2.0),
        // )
        (
            dyad_tonicity / roughness.powf(0.333),
            (1.0 - dyad_tonicity) / roughness.powf(0.333),
        )
    }

    let mut tonicity_map = HashMap::new();
    let mut mult_roughness_map = HashMap::new();
    let mut add_roughness_map = HashMap::new();

    // This is an unnormalized vector of tonicities of the notes in `cents`, amongst themselves.
    let mut existing_tonicities: Tonicities = vec![0.0; cents.len()];
    for i in 0..cents.len() {
        for j in (i + 1)..cents.len() {
            let i_is_higher = cents[i] >= cents[j];
            let hi_idx = if i_is_higher { i } else { j };
            let lo_idx = if i_is_higher { j } else { i };
            let dyad_tonicity = TonicityLookup::dyad_tonicity(cents[hi_idx] - cents[lo_idx]);

            let avg_freq = reference_freq * 2f64.powf((cents[hi_idx] + cents[lo_idx]) / 2400.0);
            let lil_penalty = lower_interval_limit_penalty(avg_freq); // (-0.05, 1.0)

            let mult_roughness = get_mult_roughness(cents[hi_idx] - cents[lo_idx], lil_penalty);

            let add_roughness = get_add_roughness(cents[hi_idx] - cents[lo_idx], lil_penalty);

            let (lo_contrib, hi_contrib) = heuristic_function(dyad_tonicity, mult_roughness);
            existing_tonicities[lo_idx] += lo_contrib;
            existing_tonicities[hi_idx] += hi_contrib;
            let bitmask = (1 << hi_idx) | (1 << lo_idx);
            // If i is higher, then dyad_tonicity is the probability of hearing the lower note j as tonic.
            // We want the probability of i.
            let tonicity_lower_index = if i_is_higher {
                1.0 - dyad_tonicity
            } else {
                dyad_tonicity
            };
            tonicity_map.insert(bitmask, tonicity_lower_index);
            mult_roughness_map.insert(bitmask, mult_roughness);
            add_roughness_map.insert(bitmask, add_roughness);
        }
    }

    let existing_sum = existing_tonicities.iter().sum::<f64>();

    if candidate_cents.len() == 0 {
        // Apply softmax on existing_tonicities.
        //
        // Normalize sum to 1 first.
        existing_tonicities = existing_tonicities
            .iter()
            .map(|x| x / existing_sum)
            .collect();

        let max_tonicity = max(existing_tonicities.as_slice());

        let softmax_tonicities = softmax(
            &existing_tonicities
                .iter()
                .map(|x| (x - max_tonicity) / tonicity_temperature)
                .collect::<Vec<f64>>(),
        );

        return TonicityHeuristic {
            tonicities: vec![softmax_tonicities],
            tonicity_map,
            mult_roughness_map,
            add_roughness_map,
            cand_tonicity_map: vec![],
            cand_mult_roughness_map: vec![],
            cand_add_roughness_map: vec![],
        };
    }

    let mut cand_tonicity_map = vec![vec![]; cents.len()];
    let mut cand_mult_roughness_map = vec![vec![]; cents.len()];
    let mut cand_add_roughness_map = vec![vec![]; cents.len()];

    let mut tonicities_per_candidate: Vec<Tonicities> =
        vec![[existing_tonicities.as_slice(), &[0f64]].concat(); candidate_cents.len()];

    for (cand_idx, cand_cents) in candidate_cents.iter().enumerate() {
        for (existing_idx, existing_cents) in cents.iter().enumerate() {
            let existing_higher = existing_cents >= cand_cents;
            let (hi_idx, lo_idx) = if existing_higher {
                (existing_idx, cents.len())
            } else {
                (cents.len(), existing_idx)
            };
            let (hi_cents, lo_cents) = if existing_higher {
                (existing_cents, cand_cents)
            } else {
                (cand_cents, existing_cents)
            };
            let avg_freq = reference_freq * 2f64.powf((hi_cents + lo_cents) / 2400.0);
            let lil_penalty = lower_interval_limit_penalty(avg_freq); // (-0.05, 1.0)
            let dyad_tonicity = TonicityLookup::dyad_tonicity(hi_cents - lo_cents);
            let mult_roughness = get_mult_roughness(hi_cents - lo_cents, lil_penalty);
            let add_roughness = get_add_roughness(hi_cents - lo_cents, lil_penalty);

            let (lo_contrib, hi_contrib) = heuristic_function(dyad_tonicity, mult_roughness);
            tonicities_per_candidate[cand_idx][lo_idx] += lo_contrib;
            tonicities_per_candidate[cand_idx][hi_idx] += hi_contrib;
            cand_tonicity_map[existing_idx].push(if existing_higher {
                1.0 - dyad_tonicity
            } else {
                dyad_tonicity
            });
            cand_mult_roughness_map[existing_idx].push(mult_roughness);
            cand_add_roughness_map[existing_idx].push(add_roughness);
        }

        // Version 2: normalizing by softmax.

        let cand_tonicities_sum = tonicities_per_candidate[cand_idx].iter().sum::<f64>();
        // Normalize sum to 1 first for consistent opinionatedness/confidence.
        tonicities_per_candidate[cand_idx] = tonicities_per_candidate[cand_idx]
            .iter()
            .map(|x| x / cand_tonicities_sum)
            .collect::<Vec<f64>>();

        // subtract max raw tonicity from raw tonicity scores to improve numerical stability

        let max_tonicity = max(tonicities_per_candidate[cand_idx].as_slice());

        tonicities_per_candidate[cand_idx] = softmax(
            &tonicities_per_candidate[cand_idx]
                .iter()
                .map(|x| (x - max_tonicity) / tonicity_temperature)
                .collect::<Vec<f64>>(),
        );
    }

    TonicityHeuristic {
        tonicities: tonicities_per_candidate,
        tonicity_map,
        mult_roughness_map,
        add_roughness_map,
        cand_tonicity_map,
        cand_mult_roughness_map,
        cand_add_roughness_map,
    }
}

fn tonicity_smoothing(
    target: &[f64],
    current: &[f64],
    smoothing: f64,
    elapsed_seconds: f64,
) -> Vec<f64> {
    assert!(target.len() == current.len());
    assert!(smoothing <= 1.0 && smoothing >= 0.0);
    let smoothing = smoothing.powf(13.0 * elapsed_seconds);
    let smoothed: Vec<f64> = current
        .iter()
        .enumerate()
        .map(|(idx, x)| *x * smoothing + target[idx] * (1.0 - smoothing))
        .collect();
    let sum: f64 = smoothed.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / smoothed.len() as f64; smoothed.len()];
    }
    smoothed.iter().map(|x| x / sum).collect()
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::utils::cents_to_hz;
    use std::usize;

    #[test]
    fn test_dyadic_tonicity() {
        dyadic_tonicity_single(&[0.0, 700.0], "C5");
        dyadic_tonicity_single(&[0.0, 1000.0, 1600.0, 2100.0, 2500.0], "C13b9");
        dyadic_tonicity_single(&[0.0, 300.0, 700.0], "Cmin");
        dyadic_tonicity_single(&[0.0, 400.0, 700.0], "Cmaj");
        dyadic_tonicity_single(&[700.0, 400.0, 0.0], "Cmaj opp order");
        dyadic_tonicity_single(&[0.0, 500.0, 700.0], "Csus4");
        dyadic_tonicity_single(&[0.0, 200.0, 700.0], "Csus2");
        dyadic_tonicity_single(&[400.0, 700.0, 1200.0], "C/E close");
        dyadic_tonicity_single(&[300.0, 700.0, 1200.0], "Cm/Eb close");
        dyadic_tonicity_single(&[400.0, 1200.0, 1900.0], "C/E spread");
        dyadic_tonicity_single(&[300.0, 1200.0, 1900.0], "Cm/Eb spread");
        dyadic_tonicity_single(&[0.0, 400.0, 700.0, 1100.0], "Cmaj7");
        dyadic_tonicity_single(&[-100.0, 0.0, 400.0, 700.0], "Cmaj7/B");
        dyadic_tonicity_single(&[0.0, 350.0, 700.0], "C neutral");
        dyadic_tonicity_single(&[0.0, 500.0, 900.0], "F/C");
        dyadic_tonicity_single(&[0.0, 500.0, 800.0], "Fm/C");
    }

    fn dyadic_tonicity_single(cents: &[f64], name: &str) {
        let tonicities =
            &dyadic_tonicity_heur(261.63, cents, &[], HEURISTIC_DYAD_TONICITY_TEMP).tonicities[0];
        println!("Dyadic tonicity for {}:", name);
        for (idx, t) in tonicities.iter().enumerate() {
            println!("{:>9.4}c: {}", cents[idx], t);
        }
    }

    #[test]
    fn test_tonicity_candidates() {
        tonicity_candidates(
            &[0.0, 701.9],
            &[386.314, 400.0, 407.82, 435.084],
            "Choosing a major third (5/4, 4\\12, 81/64, 9/7)",
        );
    }

    fn tonicity_candidates(cents: &[f64], cand_cents: &[f64], name: &str) {
        let out = dyadic_tonicity_heur(261.63, cents, cand_cents, HEURISTIC_DYAD_TONICITY_TEMP);
        println!("\n{}:", name);
        for (add_idx, tonicities) in out.tonicities.iter().enumerate() {
            println!("");
            for (idx, t) in tonicities.iter().enumerate() {
                println!(
                    "{:>9.4}c: {}",
                    if idx != cents.len() {
                        cents[idx]
                    } else {
                        cand_cents[add_idx]
                    },
                    t
                );
            }
        }
    }

    #[test]
    fn test_tonicity_smoothing() {
        let target = vec![0.1, 0.5, 0.4];
        let mut current = vec![0.2, 0.3, 0.5];
        for i in 0..100 {
            current = tonicity_smoothing(&target, &current, 0.9, 0.1);
            println!("{i}: {:?}", current);
        }
    }

    #[test]
    fn temp_test_graph_diss() {
        let time = 1.0;
        let iters = 1;

        graph_diss(&[0.0, 500.0], "P4", time, iters);
        graph_diss(&[200.0, 500.0, 800.0, 1200.0], "Dm7b5", time, iters);
        graph_diss(&[500.0, 800.0, 1200.0, 1400.0], "Fm6", time, iters);
    }

    #[test]
    fn test_graph_diss() {
        let time = 2.5;
        let iters = 1;
        graph_diss(&[0.0, 500.0], "P4", time, iters);
        graph_diss(&[0.0, 600.0], "Tritone", time, iters);
        graph_diss(&[0.0, 700.0], "P5", time, iters);

        graph_diss(&[0.0, 400.0, 700.0], "C maj", time, iters);
        graph_diss(&[0.0, 300.0, 700.0], "C min", time, iters);
        graph_diss(&[0.0, -800.0, 700.0], "C/E spread", time, iters);
        graph_diss(&[0.0, -900.0, 700.0], "Cm/Eb spread", time, iters);
        graph_diss(&[0.0, 500.0, 900.0], "F/C", time, iters);
        graph_diss(&[0.0, 500.0, 800.0], "Fm/C", time, iters);
        graph_diss(&[0.0, 300.0, 800.0], "Ab/C", time, iters);
        graph_diss(&[0.0, 400.0, 900.0], "Am/C", time, iters);

        graph_diss(
            &[0.0, 700.0, 1400.0, 1600.0],
            "C maj add9 spread",
            time,
            iters,
        );
        graph_diss(&[0.0, 700.0, 200.0, 400.0], "C2 closed", time, iters);
        graph_diss(&[0.0, 400.0, 700.0, 900.0], "C6", time, iters);
        graph_diss(&[0.0, 400.0, 700.0, 1100.0], "C maj7", time, iters);
        graph_diss(&[0.0, 300.0, 700.0, 1000.0], "C min7", time, iters);
        graph_diss(&[0.0, 300.0, 700.0, 900.0], "C min6", time, iters);
        graph_diss(&[0.0, 400.0, 700.0, 1000.0], "C7", time, iters);
        graph_diss(&[0.0, 386.0, 702.0, 970.0], "C7 harmonic ish", time, iters);
        graph_diss(&[0.0, 300.0, 600.0, 900.0], "C dim7", time, iters);

        graph_diss(&[0.0, 1000.0, 1600.0, 2100.0, 2500.0], "C13b9", time, iters);
        graph_diss(
            &[0.0, 900.0, 1300.0, 1600.0, 2200.0],
            "C13b9 terrible voicing",
            time,
            iters,
        );
    }

    #[test]
    fn test_scaling_number_of_notes() {
        let time = 2.5;
        let iters = 1;

        println!("===================== OCTAVES =======================");
        graph_diss(&[0.0, 1200.0], "2 notes", time, iters);
        graph_diss(&[0.0, 1200.0, 2400.0], "3 notes", time, iters);
        graph_diss(&[0.0, 1200.0, 2400.0, 3600.0], "4 notes", time, iters);
        graph_diss(
            &[0.0, 1200.0, 2400.0, 3600.0, 4800.0],
            "5 notes",
            time,
            iters,
        );
        graph_diss(
            &[0.0, 1200.0, 2400.0, 3600.0, 4800.0, 6000.0],
            "6 notes",
            time,
            iters,
        );

        println!("===================== FIFTHS ========================");
        graph_diss(&[0.0, 700.0], "2 notes", time, iters);
        graph_diss(&[0.0, 700.0, 1400.0], "3 notes", time, iters);
        graph_diss(&[0.0, 700.0, 1400.0, 2100.0], "4 notes", time, iters);
        graph_diss(
            &[0.0, 700.0, 1400.0, 2100.0, 2800.0],
            "5 notes",
            time,
            iters,
        );
        graph_diss(
            &[0.0, 700.0, 1400.0, 2100.0, 2800.0, 3500.0],
            "6 notes",
            time,
            iters,
        );

        println!("===================== FOURTHS =======================");
        graph_diss(&[0.0, 500.0], "2 notes", time, iters);
        graph_diss(&[0.0, 500.0, 1000.0], "3 notes", time, iters);
        graph_diss(&[0.0, 500.0, 1000.0, 1500.0], "4 notes", time, iters);
        graph_diss(
            &[0.0, 500.0, 1000.0, 1500.0, 2000.0],
            "5 notes",
            time,
            iters,
        );
        graph_diss(
            &[0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0],
            "6 notes",
            time,
            iters,
        );
    }

    fn graph_diss(cents: &[f64], name: &str, elapsed_seconds: f64, iters: usize) {
        const SHOW_LOWEST_COMPLEXITY_TREES: usize = 3; // if non-zero, show this many lowest complexity trees per root

        const INIT_WITH_DYADIC_HEURISTIC: bool = true; // if false, inits with uniform tonicity context.
        let freqs = cents
            .iter()
            .map(|x| cents_to_hz(440.0, *x))
            .collect::<Vec<f64>>();
        let mut context = if INIT_WITH_DYADIC_HEURISTIC {
            dyadic_tonicity_heur(440.0, cents, &[], HEURISTIC_DYAD_TONICITY_TEMP).tonicities[0]
                .clone()
        } else {
            vec![1.0 / cents.len() as f64; cents.len()]
        };

        println!(
            "\n============  Graph diss: {}  =====================\n\nVoicing:",
            name
        );
        for cents in cents.iter() {
            println!("    {:>7.2}c", cents);
        }
        for i in 0..iters {
            let mut debug = if SHOW_LOWEST_COMPLEXITY_TREES == 0 {
                None
            } else {
                Some(GraphDissDebug::new(SHOW_LOWEST_COMPLEXITY_TREES, cents.len()))
            };
            let diss = graph_dissonance(
                &freqs,
                &[],
                &context,
                0.9,
                elapsed_seconds,
                0.5,
                debug.as_mut(),
            );
            println!("Diss: {:.4}", diss[0].dissonance);
            println!("{}s: {:#?}", (i + 1) as f64 * elapsed_seconds, diss);

            // Print the lowest complexity trees per root

            if let Some(debug) = debug {
                for (root_idx, root) in cents.iter().enumerate() {
                    println!(
                        "Lowest {} complexity trees for root {:>7.2}c:",
                        debug.N, root
                    );
                    for tree in debug.n_lowest_comp_trees_per_root[root_idx].iter() {
                        println!(" -> comp {:.4}:", tree.diss);
                        tree.tree.print();
                    }
                }
            }
            context = diss[0].tonicity_context.clone();
        }
    }

    #[test]
    fn test_candidate_graph_diss() {
        candidate(
            &[0.0, 701.9],
            &[386.314, 400.0, 407.82, 435.084],
            "Choosing a major third (5/4, 4\\12, 81/64, 9/7) for P5 dyad context",
        );
        candidate(
            &[0.0, 386.314, 701.9],
            &[884.359, 905.865],
            "Choosing a major sixth (5/3, 27/16) for a major triad",
        );
        candidate(
            &[0.0, 386.314, 701.9],
            &[182.404, 203.910, 231.174],
            "Choosing a second (10/9, 9/8, 8/7) for a major triad",
        );
        candidate(
            &[0.0, 386.314, 701.9, 1088.269, 1403.910],
            &[1751.318, 1782.512, 1790.224, 1811.730],
            "Choosing a #11th (11/4, 14/5, 45/16, 729/256) for a maj9 JI chord",
        );
    }

    /// Test contextual candidates. Starting with `cents` for 1 second, then choosing a candidate
    /// with that context.
    fn candidate(cents: &[f64], candidate_cents: &[f64], name: &str) {
        println!("\n\nContextual diss: {}", name);

        let freqs = cents
            .iter()
            .map(|x| cents_to_hz(440.0, *x))
            .collect::<Vec<f64>>();
        let candidate_freqs = candidate_cents
            .iter()
            .map(|x| cents_to_hz(440.0, *x))
            .collect::<Vec<f64>>();
        let diss =
            graph_dissonance(&freqs, &[], &vec![0f64; cents.len()], 0.9, 1.0, 0.5, None)[0].clone();
        println!("Starting context");
        println!("{:#?}\n", diss);

        println!("Obtained tonicity context {:?}.\n", diss.tonicity_context);

        let cands = graph_dissonance(
            &freqs,
            &candidate_freqs,
            &diss.tonicity_context,
            0.9,
            1.0,
            0.5,
            None,
        );

        for (idx, cand) in cands.iter().enumerate() {
            println!("\nCandidate {}c: {:#?}", candidate_cents[idx], cand);
        }
    }

    #[test]
    /// This tests the expectation that having [a, b, c, d] as existing notes, and having one
    /// candidate [d] to add to [a, b, c] should both return almost the same results.
    fn test_single_candidate_graph_diss_same_treatment() {
        let cents = [0.0, 700.0, 1400.0, 1600.0];
        let freqs = cents
            .iter()
            .map(|x| cents_to_hz(440.0, *x))
            .collect::<Vec<f64>>();

        let diss_existing =
            graph_dissonance(&freqs, &[], &vec![0f64; freqs.len()], 0.9, 0.1, 0.5, None)[0].clone();

        let diss_candidate = graph_dissonance(
            &freqs[..(freqs.len() - 1)],
            &[*freqs.last().unwrap()],
            &vec![0f64; freqs.len() - 1],
            0.9,
            0.1,
            0.5,
            None,
        )[0]
        .clone();

        println!("\nExisting: {:#?}", diss_existing);
        println!("\nCandidate: {:#?}", diss_candidate);
        println!(
            "\nDissonance difference: {}",
            diss_existing.dissonance - diss_candidate.dissonance
        );
    }

    /// This tests the expectation that having [a, b, c, d] as existing notes, and having one
    /// candidate [d] to add to [a, b, c] should both return almost the same results.
    #[test]
    fn test_single_candidate_tonicity_heur_same_treatment() {
        let cents = [0.0, 700.0, 1400.0, 1600.0];
        let freqs = cents
            .iter()
            .map(|x| cents_to_hz(440.0, *x))
            .collect::<Vec<f64>>();

        let heur_existing = dyadic_tonicity_heur(261.63, &freqs, &[], HEURISTIC_DYAD_TONICITY_TEMP);

        let heur_candidate = dyadic_tonicity_heur(
            261.63,
            &freqs[..(freqs.len() - 1)],
            &[*freqs.last().unwrap()],
            HEURISTIC_DYAD_TONICITY_TEMP,
        );

        println!("\nExisting: {:#?}", heur_existing);
        println!("\nCandidate: {:#?}", heur_candidate);
    }

    /// Ensures that the tonicity heuristic should return the same result no matter the order of the
    /// input notes.
    #[test]
    fn test_tonicity_heur_note_order_invariant() {
        let cents = [0.0, 700.0, 1400.0, 1600.0];
        let freqs = cents
            .iter()
            .map(|x| cents_to_hz(440.0, *x))
            .collect::<Vec<f64>>();

        let forwards = dyadic_tonicity_heur(261.63, &freqs, &[], HEURISTIC_DYAD_TONICITY_TEMP);
        let backwards = dyadic_tonicity_heur(
            261.63,
            &freqs.iter().rev().copied().collect::<Vec<f64>>(),
            &[],
            HEURISTIC_DYAD_TONICITY_TEMP,
        );

        println!("\nForwards: {:#?}", forwards.tonicities);
        println!(
            "\nBackwards: {:#?}",
            backwards.tonicities.iter().rev().collect::<Vec<_>>()
        );
    }

    /// Count the number of inversions between two lists (elements needn't be the same)
    ///
    /// Here, an inversion is defined as a pair of elements in the list `a` such that `a[i] > a[j]` but `b[i] < b[j]`.
    fn count_inversions<T: PartialOrd>(a: &[T], b: &[T]) -> usize {
        // We're only going to have lists of up to 8 numbers maximum, so there's no point
        // implementing the more efficient merge sort method for O(n log n).
        let mut inversions = 0;
        for i in 0..a.len() {
            for j in (i + 1)..a.len() {
                if a[i] > a[j] && b[i] < b[j] {
                    inversions += 1;
                }
            }
        }
        inversions
    }
}
