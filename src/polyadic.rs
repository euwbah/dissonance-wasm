//! Polyadic dissonance & tonicity of notes.

use std::{
    collections::{BTreeSet, HashMap},
    ops::Deref,
};

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

/// A parameter k in [-1, 1] describing how much more precedence edge complexity (of the edge
/// between parent and child subtree) has over child subtree complexity when aggregating child
/// complexity contributions.
///
/// This is the parameter k in https://www.desmos.com/calculator/jho2nihmwl
const EDGE_OVER_SUBTREE_COMPLEXITY_BIAS: f64 = 1.0;

/// When computing the relative subtree weights of each child node, what fraction of the final
/// weight should be from the sum of global tonicities of the subtree's nodes (compared to using
/// subtree sizes/weighting each each node equally)
const TONICITY_BIAS: f64 = 0.6;

/// How much more likely is a subtree with complexity 0.0 over a subtree with complexity 1.0?
///
/// This scale is exponential, so a value of 2.0 means a complexity = 0 subtree has 4 times the
/// likelihood of a complexity = 1 subtree.
const COMPLEXITY_LIKELIHOOD_BIAS: f64 = 1.5;

/// How much should the complexity of a child's subtree affect its likelihood contribution.
///
/// This value is on a scale of 0 to 2.0. A value of 0 means that complexity of the subtree should
/// not affect the likelihood contribution of the subtree, and a value of 2 means that a maximally
/// complex (comp = 1) subtree should not be allowed to affect the likelihood of the rest of the
/// interpretation tree, whereas a complexity = 0 subtree has "two times" (multiplicative scale) the
/// average weight of likelihood contribution.
///
/// If this value goes beyond 2.0, very complex subtrees can contribute inverse likelihoods, where
/// less likely (< 1.0) subtrees that are very complex end up increaasing overall likelihoods (>
/// 1.0).
const COMPLEXITY_LIKELIHOOD_SCALING: f64 = 1.0;

/// How much should the likelihood contribution of a subtree be decreased by for each level deeper
/// it is nested in the interpretation tree.
///
/// This is exponentially scaled by 2^PENALTY, such that a subtree rooted at depth-1 contributes
/// 2^PENALTY more likelihood than the same subtree rooted at depth-2.
///
/// HOW TO TUNE: Increasing this value increases the major-minor triad dissonance gap, but will also
/// increase the tonicity confidence/variance of chords with >= 3 notes. Dyadic confidences will not
/// be affected.
const DEEP_TREE_LIKELIHOOD_PENALTY: f64 = 2.2;

/// How much more likely is a subtree if the parent of the subtree has global tonicity 1.0 and
/// subtree root has global tonicity 0.0, than if the parent had global tonicity 0.0 and
/// subtree root had global tonicity 1.0?
///
/// A value of 1.0 means that global tonicity context does not affect likelihood.
///
/// A value equal to [DYADIC_TONICITY_LIKELIHOOD_SCALING] means that global tonicity context
/// matters about as much as dyadic tonicity alignment.
///
/// HOW TO TUNE: Increasing this value will increase the major-minor triad dissonance gap, but will
/// make tonicity scores more confident. Additionally (unlike [DYADIC_TONICITY_LIKELIHOOD_SCALING]),
/// increasing this will increase the stubbornness of the global tonicity context, so you may want to
/// decrease global tonicity context smoothing.
const GLOBAL_TONICITY_LIKELIHOOD_SCALING: f64 = (2u128 << 12) as f64;

/// How much more likely is a subtree if the parent-child dyad connecting the subtree to its parent
/// has parent dyad tonicity 1.0, than if the parent dyad tonicity was 0.0? (Relative to the
/// parent-child dyad being heard in a vacuum)
///
/// NOTE: dyadic tonicity lookups are mostly conservatively in the range [0.44, 0.56], so this value
/// probably has to be set MUCH HIGHER than other scaling parameters to have any effect.
///
/// E.g., in the maximal case of dyadic tonicity (P5), the root has a dyadic tonicity of nearly
/// 0.56.
///
/// The formula used to calculate the dyadic tonicity component of likelihood is
///
/// ```
/// DYADIC_TONICITY_LIKELIHOOD_SCALING.powf(dyad_tonicity - 0.5)
/// ```
///
/// The maximum amount of increased likelihood if this value is set to 30 is only 1.2x.
///
/// HOW TO TUNE: Increasing this value will increase the major-minor triad dissonance gap, but will
/// also make tonicity scores more confident. Balance this by increasing
/// [TONICITY_CONTEXT_TEMPERATURE_TARGET] to make global tonicity less opinionated, or by increasing
/// global tonicity context smoothing to slow down the change in tonicity, or by increasing
/// [GLOBAL_TONICITY_LIKELIHOOD_SCALING] to make global tonicity context more persistent.
const DYADIC_TONICITY_LIKELIHOOD_SCALING: f64 = (2u128 << 38) as f64;

/// How much more likely is a subtree if the parent node it connects to is 1 octave lower it is.
///
/// Scales multiplicatively per octave (e.g. 2 octaves = this value squared).
const LOW_NOTE_ROOT_LIKELIHOOD_SCALING: f64 = 1.04;

/// Softmax temperature for normalizing likelihoods of all tree interpretations for the purpose of
/// target tonicity calculation.
///
/// Lower temperature = more opinionated.
const TONICITY_CONTEXT_TEMPERATURE_TARGET: f64 = 0.5;

/// Softmax temperature applied to global tonicity context. The softmax of global tonicities weights
/// the final dissonance contribution for each root.
///
/// Note that the target global tonicity context is already a softmax scaled by
/// [LIKELIHOOD_SOFTMAX_TEMPERATURE_TONICITY], so this value should be appropriately scaled
/// according to the above parameter.
///
/// HOW TO TUNE: Decreasing this value will increase the major-minor triad dissonance gap, but will
/// decrease the dissonance contribution from less likely roots according to current global tonicity
/// context. If the subjective root perception of the listener does not match the root perception of
/// this algorithm, the dissonance ratings may be off.
const TONICITY_CONTEXT_TEMPERATURE_DISS: f64 = 0.1;

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
/// - `comp_like_per_root`: For each root, contains the (complexity, exp likelihood) values of each
///   interpretation tree having that root. Roots are arranged in order of original `freqs` (not
///   ascending pitch).
///
///   ⚠️ Likelihood must be exponentially scaled.
///
/// - `likelihood_exp_sum`: Sum of exp(likelihood / LIKELIHOOD_SOFTMAX_TEMPERATURE) over all trees.
///
/// - `tonicity_context`: Existing tonicities of notes in original order. If candidate is in the
///   computation, the candidate tonicity should be at the last index of this array.
///
/// - `smoothing`: how quickly the tonicities converge to target tonicities.
///
/// - `elapsed_seconds`: seconds between now and previous update.
///
/// - `debug`: optional debug object to collect per-root complexities and likelihoods.
fn compute_interpretation_trees(
    comp_like_per_root: &[&[(f64, f64)]],
    likelihood_exp_sum: f64,
    tonicity_context: &[f64],
    smoothing: f64,
    elapsed_seconds: f64,
    debug: Option<&mut GraphDissDebug>,
) -> Dissonance {
    // Aggregate complexity scores per-root.
    //
    // For tonicity calculation, for each root, we sum the softmax of likelihoods of each tree
    // interpretation.
    //
    //   TODO: Is it necessary to average softmax per-root likelihood by softmax negative aggregate
    //         per-root complexity? E.g., if we want to model the idea that root choices with lower
    //         root complexity should be more likely? Or is this already accounted for in the model?
    //
    // For dissonance calculation, we weight each tree's complexity by the softmax-normalized
    // likelihood of the interpretation.
    //
    //   TODO: If there are many interpretation trees, taking the softmax of many values may be
    //         computationally expensive. In this case, find a better way to normalize likelihoods.
    //
    // For the sake of debug values, the dissonance scores for each root will be stored

    // Aggregates each root indexed in original `freqs` order, candidate at last idx if present.
    let num_notes = comp_like_per_root.len();
    let mut complexity_per_root = vec![0.0; num_notes];
    let mut target_tonicities = vec![0.0; num_notes];

    // Sum of exp likelihoods per root, used for normalizing per-root dissonance contributions
    let mut exp_likelihood_sum_per_root = vec![0.0; num_notes];
    for root_idx in 0..num_notes {
        let comp_likes = &comp_like_per_root[root_idx];

        let (unnormalized_root_complexity_contribution, root_exp_like_sum) = comp_likes
            .iter()
            // Here, the complexity is weighted by exp likelihood, later will be divided by
            // likelihood_exp_sum so effectively what's going on is that the complexity is weighted
            // by softmax-normalized likelihood for each tree.
            .map(|(comp, like)| (comp * like, like))
            .fold((0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1));

        // Target tonicities is softmax of per-root likelihoods.
        target_tonicities[root_idx] = root_exp_like_sum / likelihood_exp_sum;

        exp_likelihood_sum_per_root[root_idx] = root_exp_like_sum;

        // This is the complexity of all trees with this root weighted by softmax-normalized
        // likelihood.
        complexity_per_root[root_idx] =
            unnormalized_root_complexity_contribution / likelihood_exp_sum;
    }

    let updated_tonicity_context = tonicity_smoothing(
        &target_tonicities,
        tonicity_context,
        smoothing,
        elapsed_seconds,
    );

    // The final dissonance is computed from aggregated inverse exp weighted complexities
    // per root, multiplied by root tonicity of the updated context.

    // In the old algo, complexity contribution of each root is weighted by tonicity of each root,
    // because we incorrectly equated the tonicity of the root with the probability that the entire
    // interpretation tree is a model of the listener's perception.
    //
    // But now we just should rely on the likelihood score of each tree.
    //
    // let complexity: f64 = complexity_per_root .iter() .enumerate() .map(|(root_idx, c)| c *
    //     updated_tonicity_context[root_idx]) .sum();

    let softmax_global_tonicities_diss_calc = softmax(
        &updated_tonicity_context
            .iter()
            .map(|x| x / TONICITY_CONTEXT_TEMPERATURE_DISS)
            .collect::<Vec<_>>(),
    );

    let complexity_per_root_weighted_global = complexity_per_root
        .iter()
        .zip(&softmax_global_tonicities_diss_calc)
        .zip(&exp_likelihood_sum_per_root)
        .map(|((c, t), lsum)| c * t / lsum * likelihood_exp_sum)
        .collect::<Vec<_>>();

    let complexity: f64 = complexity_per_root_weighted_global.iter().sum::<f64>();

    if let Some(d) = debug {
        d.per_root_agg_complexities
            .extend_from_slice(&complexity_per_root);

        d.per_root_agg_complexities_weighted_global
            .extend_from_slice(&complexity_per_root_weighted_global);
    }

    Dissonance {
        dissonance: complexity,
        tonicity_target: target_tonicities,
        tonicity_context: updated_tonicity_context,
    }
}

/// Result of computing one interpretation tree.
///
/// Used only for debugging/testing purposes.
///
/// Ord is implemented with respect to increasing complexity.
#[derive(Clone, Debug)]
pub struct TreeResult {
    pub tree: &'static ST,
    pub complexity: f64,
    pub likelihood: f64,
}

/// Wrapper for sorting [TreeResult] by ascending complexity
#[derive(Clone, Debug)]
pub struct TreeResultCompAsc(TreeResult);
impl Deref for TreeResultCompAsc {
    type Target = TreeResult;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl PartialEq for TreeResultCompAsc {
    fn eq(&self, other: &Self) -> bool {
        self.complexity == other.complexity
    }
}
impl Eq for TreeResultCompAsc {}
impl PartialOrd for TreeResultCompAsc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.complexity.partial_cmp(&other.complexity)
    }
}
impl Ord for TreeResultCompAsc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.complexity.partial_cmp(&other.complexity).unwrap()
    }
}

/// Wrapper for sorting [TreeResult] by ascending likelihood
#[derive(Clone, Debug)]
pub struct TreeResultLikeAsc(TreeResult);
impl Deref for TreeResultLikeAsc {
    type Target = TreeResult;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl PartialEq for TreeResultLikeAsc {
    fn eq(&self, other: &Self) -> bool {
        self.likelihood == other.likelihood
    }
}
impl Eq for TreeResultLikeAsc {}
impl PartialOrd for TreeResultLikeAsc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.likelihood.partial_cmp(&other.likelihood)
    }
}
impl Ord for TreeResultLikeAsc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.likelihood.partial_cmp(&other.likelihood).unwrap()
    }
}

/// Wrapper for sorting [TreeResult] by ascending dissonance contribution complexity *
/// exp(likelihood)
#[derive(Clone, Debug)]
pub struct TreeResultContribAsc(TreeResult);
impl Deref for TreeResultContribAsc {
    type Target = TreeResult;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl PartialEq for TreeResultContribAsc {
    fn eq(&self, other: &Self) -> bool {
        self.complexity * self.likelihood.exp() == other.complexity * other.likelihood.exp()
    }
}
impl Eq for TreeResultContribAsc {}
impl PartialOrd for TreeResultContribAsc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (self.complexity * self.likelihood.exp())
            .partial_cmp(&(other.complexity * other.likelihood.exp()))
    }
}
impl Ord for TreeResultContribAsc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.complexity * self.likelihood.exp())
            .partial_cmp(&(other.complexity * other.likelihood.exp()))
            .unwrap()
    }
}

/// Pass an empty `&mut` [GraphDissDebug] into [graph_dissonance] to obtain debug & test data.
///
/// NOTE: per-root likelihoods is equivalent to target tonicity, no need to collect extra data.
///
/// DO NOT USE THIS IN PROD. SLOW ⚠️
#[derive(Clone, Debug)]
pub struct GraphDissDebug {
    /// What `n` is in the variable names below.
    N: usize,

    /// For each root, what are the N lowest complexity trees (front of the BTree)
    ///
    /// NOTE: This debug value is only available if there are no candidate notes provided.
    ///
    /// The purpose of this value is to check whether the most consonant trees correspond to
    /// intuitive interpretations of the chord.
    n_lowest_comp_trees_per_root: Vec<BTreeSet<TreeResultCompAsc>>,

    /// For each root, what are the N highest likelihood trees (back of the BTree)
    ///
    /// NOTE: This debug value is only available if there are no candidate notes provided.
    ///
    /// The purpose of this value is to check whether the most consonant trees correspond to
    /// intuitive interpretations of the chord.
    n_highest_like_trees_per_root: Vec<BTreeSet<TreeResultLikeAsc>>,

    /// How much complexity did each root contribute to the final dissonance, weighted by the
    /// likelihoods of each tree.
    per_root_agg_complexities: Vec<f64>,

    /// How much complexity did each root contribute to the final dissonance weighted by the
    /// likelihoods of each tree and the softmax of the updated global tonicity context.
    per_root_agg_complexities_weighted_global: Vec<f64>,

    /// All trees sorted by increasing contribution to dissonance: complexity * exp(likelihood)
    trees_sorted_asc_contrib: BTreeSet<TreeResultContribAsc>,
}

impl GraphDissDebug {
    /// Create a new [GraphDissDebug].
    ///
    /// - `n`: how many trees to collect (see variable names)
    /// - `num_notes`: how many notes in this chord tree
    pub fn new(n: usize, num_notes: usize) -> Self {
        GraphDissDebug {
            N: n,
            n_lowest_comp_trees_per_root: vec![BTreeSet::new(); num_notes],
            n_highest_like_trees_per_root: vec![BTreeSet::new(); num_notes],
            per_root_agg_complexities: vec![],
            per_root_agg_complexities_weighted_global: vec![],
            trees_sorted_asc_contrib: BTreeSet::new(),
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
    let cand_roughs = heuristic_tonicities.cand_add_roughness_map;
    let dyad_tonics = heuristic_tonicities.tonicity_map;
    let cand_tonics = heuristic_tonicities.cand_tonicity_map;

    let mut results = vec![];

    if num_candidates == 0 {
        // First, we need an index mapping of the frequencies in low-to-high pitch order.
        //
        // Note that we have to compute a different sorted idx mapping for different candidate
        // notes, as different candidate notes may result in different pitch order.

        let cents_asc_order = {
            let mut sorted = cents.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted
        };

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

        let dyad_tonicity = |from: usize, to: usize| {
            let idx_from = asc_to_og_idxs[from];
            let idx_to = asc_to_og_idxs[to];
            let bitmask = (1 << idx_from) | (1 << idx_to);
            let lower_idx_tonic = *dyad_tonics
                .get(&bitmask)
                .expect("Missing edge in precomputed dyadic tonicity lookup!");

            if idx_from < idx_to {
                lower_idx_tonic
            } else {
                1.0 - lower_idx_tonic
            }
        };

        // Each item in this vec corresponds to complexity scores computed for each root
        //
        // The root indices are ordered in ascending pitch order. To convert back to original
        // order in freqs, use inverse_idx_map.
        let mut comp_like_per_root_lo_to_hi = vec![vec![]; num_notes];

        let mut likelihood_exp_sum = 0.0;

        for tree in st_trees.iter() {
            let (comp, like) = dfs_st_comp_likelihood(
                &cents_asc_order,
                tree,
                &sorted_tonicity_ctx,
                dyad_comp,
                dyad_tonicity,
            );
            let exp_likelihood = (like / TONICITY_CONTEXT_TEMPERATURE_TARGET).exp();
            comp_like_per_root_lo_to_hi[tree.root].push((comp, exp_likelihood));
            likelihood_exp_sum += exp_likelihood;

            if let Some(d) = &mut debug {
                let entry = TreeResult {
                    tree: &tree,
                    complexity: comp,
                    likelihood: like,
                };

                let lowest_comp_trees = &mut d.n_lowest_comp_trees_per_root[tree.root];
                lowest_comp_trees.insert(TreeResultCompAsc(entry.clone()));

                // Keep only the N lowest complexity trees
                while lowest_comp_trees.len() > d.N {
                    lowest_comp_trees.pop_last();
                }

                let highest_like_trees = &mut d.n_highest_like_trees_per_root[tree.root];
                highest_like_trees.insert(TreeResultLikeAsc(entry.clone()));

                while highest_like_trees.len() > d.N {
                    highest_like_trees.pop_first();
                }

                d.trees_sorted_asc_contrib
                    .insert(TreeResultContribAsc(entry));
            }
        }

        let comp_like_per_root_og_order = (0..num_notes)
            .map(|i| comp_like_per_root_lo_to_hi[og_to_asc_idx[i]].as_slice())
            .collect::<Vec<_>>();

        results.push(compute_interpretation_trees(
            &comp_like_per_root_og_order,
            likelihood_exp_sum,
            tonicity_context,
            smoothing,
            elapsed_seconds,
            debug,
        ));

        return results;
    }

    // If candidates are supplied, we have to repeat the above process with each candidate
    // as the last indexed note provided freqs.

    for candidate_idx in 0..num_candidates {
        let curr_candidate_cents = candidate_cents[candidate_idx];
        let cents_asc_order = {
            let mut combined_cents = cents.clone();
            combined_cents.push(curr_candidate_cents);
            combined_cents.sort_by(|a, b| a.partial_cmp(b).unwrap());
            combined_cents
        };
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

            let existing_pitch_idx = idx_from.min(idx_to);
            cand_roughs[existing_pitch_idx]
                .get(candidate_idx) // which candidate to use
                .cloned()
                .expect("Missing edge in candidate dyadic roughness lookup!")
        };

        let dyad_tonicity = |from: usize, to: usize| {
            let idx_from = asc_to_og_idxs[from];
            let idx_to = asc_to_og_idxs[to];

            let bitmask = (1 << idx_from) | (1 << idx_to);

            if let Some(lower_idx_tonic) = dyad_tonics.get(&bitmask) {
                // both notes are non-candidate notes, the tonicity is found in the dyad_tonics lookup.
                if idx_from < idx_to {
                    return *lower_idx_tonic;
                } else {
                    return 1.0 - *lower_idx_tonic;
                }
            }
            // implicitly, if idx_from == freqs.len() || idx_to == freqs.len() {

            let from_is_candidate = idx_from == freqs.len();

            let tonicity_of_existing_note = cand_tonics[idx_to]
                .get(candidate_idx)
                .cloned()
                .expect("Missing edge in candidate dyadic tonicity lookup!");

            // this function should return the tonicity of the `from` node.
            if from_is_candidate {
                1.0 - tonicity_of_existing_note
            } else {
                tonicity_of_existing_note
            }
        };

        let mut comp_like_per_root_lo_to_hi: Vec<Vec<(f64, f64)>> = vec![vec![]; num_notes];
        let mut likelihood_exp_sum = 0.0;

        for tree in st_trees.iter() {
            let (complexity, likelihood) = dfs_st_comp_likelihood(
                &cents_asc_order,
                tree,
                tonicity_context,
                dyad_comp,
                dyad_tonicity,
            );

            let exp_likelihood = (likelihood / TONICITY_CONTEXT_TEMPERATURE_TARGET).exp();
            comp_like_per_root_lo_to_hi[tree.root].push((complexity, exp_likelihood));
            likelihood_exp_sum += exp_likelihood;
        }

        let comp_like_per_root_og_order = (0..num_notes)
            .map(|i| comp_like_per_root_lo_to_hi[og_to_asc_idxs[i]].as_slice())
            .collect::<Vec<_>>();

        results.push(compute_interpretation_trees(
            &comp_like_per_root_og_order,
            likelihood_exp_sum,
            &new_tonicity_ctx,
            smoothing,
            elapsed_seconds,
            None,
        ));
    }

    results
}

#[derive(Clone, Debug)]
struct DFSResult {
    /// Sum of tonicities of all nodes in subtree, including root itself.
    ///
    /// If subtree is a single leaf node, this is the global tonicity of that node.
    subtree_tonicity: f64,

    /// Subtree complexity computed as the weighted sum of child complexity contributions.
    ///
    /// Subtree complexity is in the range [0, 1], absolute scale.
    ///
    /// If subtree is a single leaf node, this is 0.0.
    subtree_complexity: f64,

    /// How many notes in this subtree, including root.
    ///
    /// Number of edges = `subtree_size - 1`
    subtree_size: usize,

    /// Likelihood of this subtree interpretation, computed as some aggregate of child likelihood contributions.
    ///
    /// Likelihood is in the range [0, infty), and has no units/relative scale. The likelihoods will be normalized
    /// after all trees are evaluated.
    ///
    /// If subtree is a single leaf node, this is 1.0.
    subtree_likelihood: f64,
}

/// Computes the absolute contribution of a non-leaf child & its connecting parent-child edge to the
/// overall complexity of the parent subtree.
///
/// The sum of both leaf & subtree child contributions should be within the range [0, 1].
///
/// - `subtree_weight`: normalized local weight of this child node (sum of weights of children =
///   1.0)
/// - `dyadic_complexity`: complexity of the dyad between child and parent from roughness lookup
///   table
/// - `subtree_complexity`: complexity of the child's subtree
///
/// Link to graph: https://www.desmos.com/calculator/jho2nihmwl
fn compute_child_subtree_complexity_contribution(
    subtree_weight: f64,
    dyadic_complexity: f64,
    subtree_complexity: f64,
) -> f64 {
    let unweighted = 0.5
        * (dyadic_complexity + subtree_complexity)
        * (1.0
            + 0.5 * EDGE_OVER_SUBTREE_COMPLEXITY_BIAS * (dyadic_complexity - subtree_complexity));

    unweighted * subtree_weight
}

/// Computes the absolute contribution of a leaf child & its connecting parent-child edge to the
/// overall complexity of the parent subtree.
///
/// The sum of both leaf & subtree child contributions should be within the range [0, 1].
fn compute_child_leaf_complexity_contribution(subtree_weight: f64, dyadic_complexity: f64) -> f64 {
    dyadic_complexity * subtree_weight
}

/// Computes subtree weights for all children (leaf & non-leaf) of a parent node.
///
/// - `unnormalized_child_subtree_tonicities`: The sum of global tonicities of the nodes in each
///   child's subtree. If the child is a leaf, this is the tonicity of the child node itself.
/// - `child_subtree_sizes`: number of nodes in each child's subtree, including the child node itself.
///
/// Returns normalized weights summing to 1 in input node order.
fn compute_subtree_weights(
    unnormalized_child_subtree_tonicities: &[f64],
    child_subtree_sizes: &[usize],
) -> Vec<f64> {
    let sum_subtree_sizes = child_subtree_sizes.iter().sum::<usize>() as f64;
    let uniform_weighted_nodes: Vec<f64> = child_subtree_sizes
        .iter()
        .map(|s| *s as f64 / sum_subtree_sizes)
        .collect();

    let total_tonicity: f64 = unnormalized_child_subtree_tonicities.iter().sum();
    let normalized: Vec<f64> = unnormalized_child_subtree_tonicities
        .iter()
        .map(|ct| ct / total_tonicity)
        .collect();
    let softmax_local_tonicities = softmax(
        &normalized
            .iter()
            .map(|t| t / LOCAL_TONICITY_TEMP)
            .collect::<Vec<f64>>(),
    );

    let weights: Vec<f64> = softmax_local_tonicities
        .iter()
        .zip(uniform_weighted_nodes.iter())
        .map(|(tonic_w, size_w)| TONICITY_BIAS * tonic_w + (1.0 - TONICITY_BIAS) * size_w)
        .collect();

    weights
}

/// Computes the absolute likelihood contribution of a child's subtree.
///
/// See the article at
/// [/paper/article.md#computing-values-in-revised-model](/paper/article.md#computing-values-in-revised-model)
///
/// - `parent_child_interval_cents`: the interval from parent to child in cents (e.g., 1200.0 means
///   child is exactly one octave above parent).
/// - `parent_global_tonicity`: tonicity of the parent node in current global tonicity context.
/// - `child_global_tonicity`: tonicity of the child node in current global tonicity context.
/// - `dyadic_tonicity`: tonicity of the parent node according to parent-child dyadic tonicity
/// - `subtree_likelihood`: likelihood of the child's subtree. If the child is a leaf, this should
///   be 1.0.
/// - `subtree_complexity`: complexity of the child's subtree. If the child is a leaf, this should
///   be 0.0.
fn compute_child_likelihood_contribution(
    parent_child_interval_cents: f64,
    parent_global_tonicity: f64,
    child_global_tonicity: f64,
    dyadic_tonicity: f64,
    subtree_likelihood: f64,
    subtree_complexity: f64,
) -> f64 {
    // Strategy: multiplicative unitless model (1.0 is neutral, <1.0 reduces likelihood, >1.0 increases likelihood)
    // Use geometric mean to combine factors.

    let global_tonicity_alignment_ratio = if parent_global_tonicity + child_global_tonicity == 0.0 {
        1.0
    } else if child_global_tonicity == 0.0 {
        50.0
    } else {
        parent_global_tonicity / child_global_tonicity
    };
    let global_tonicity_alignment_ratio = global_tonicity_alignment_ratio.clamp(1.0 / 50.0, 50.0);

    let global_tonicity_logistic = 1.0 / (1.0 + (1.0 - global_tonicity_alignment_ratio).exp());
    let global_tonicity_alignment =
        GLOBAL_TONICITY_LIKELIHOOD_SCALING.powf(global_tonicity_logistic - 0.5);

    let dyadic_tonicity_alignment = DYADIC_TONICITY_LIKELIHOOD_SCALING.powf(dyadic_tonicity - 0.5);

    // kappa (lambda_i, c_i) in article.md
    let adjusted_subtree_likelihood = if subtree_complexity == 0.0 {
        // edge case, 0 complexity subtree probably means that the subtree is a leaf node.
        //
        // Deep tree penalty doesn't apply, only applies to non-leaf children.
        subtree_likelihood
    } else {
        subtree_likelihood.powf(1.0 + COMPLEXITY_LIKELIHOOD_SCALING * (0.5 - subtree_complexity))
            * 2.0f64.powf(
                COMPLEXITY_LIKELIHOOD_BIAS * (0.5 - subtree_complexity)
                    - DEEP_TREE_LIKELIHOOD_PENALTY,
            )
    };

    let lower_root_bias =
        LOW_NOTE_ROOT_LIKELIHOOD_SCALING.powf(parent_child_interval_cents / 1200.0);

    // TODO: besides geometric mean, any better aggregation methods?
    let likelihood = (global_tonicity_alignment
        * dyadic_tonicity_alignment
        * adjusted_subtree_likelihood
        * lower_root_bias)
        .powf(1.0 / 4.0);

    likelihood
}

/// Evaluate spanning tree complexity & likelihood using DFS.
///
/// **IMPORTANT ⚠️⚠️⚠️ USE ASCENDING PITCH ORDER:**
///
///   This function requires input notes to be indexed in ASCENDING PITCH order. The
///   freqs provided from the visualizer are in no particular order, so a sorted index mapping
///   must be obtained first.
///
///   The search space of precomputed trees is pruned so that pre-order inversions are capped
///   at `max_inversions`, which models the natural low-to-high precendence of pitch perception.
///
/// See `paper/article.md` for details.
///
/// Same algo as `dfs()` in `paper/triad_sts_computation_example.py`
///
/// - `cents_asc_order`: list of notes in cents, indexed in **ascending pitch order**.
///
/// - `tree`: the spanning tree to evaluate
///
/// - `tonicity_ctx`: tonicity context per node in **ascending pitch order**.
///
/// - `dyad_comp`: function that returns the dyadic/edge additive complexity (range 0-1) of (from,
///   to) edge in **ascending pitch order**.
///
/// - `dyad_tonicity`: function that takes in `(from, to)` indices in **ascending pitch order** and
///   returns the tonicity of the `from` node (i.e., between the two notes in the from-to dyad, how
///   tonic is `from`).
///
/// ## Returns
///
/// The (complexity, likelihood) of this interpretation tree.
///
/// Complexity is a value within [0, 1] representing the minimum/maximum complexity of all trees of
/// the same size.
///
/// Likelihood is an unbounded value [0, infty), representing relative likelihood of the tree.
///
fn dfs_st_comp_likelihood<F, G>(
    cents_asc_order: &[f64],
    tree: &ST,
    tonicity_ctx: &[f64],
    dyad_comp: F,
    dyad_tonicity: G,
) -> (f64, f64)
where
    F: Fn(usize, usize) -> f64,
    G: Fn(usize, usize) -> f64,
{
    // stack contains (node index, children visited?)
    //
    // For post-order traversal.
    let mut stack: Vec<(usize, bool)> = vec![(tree.root, false)];

    // store computed results per node
    //
    // Indexed by same order as tree nodes.
    let mut results: Vec<Option<DFSResult>> = vec![None; tonicity_ctx.len()];

    while let Some((node, visited_children)) = stack.pop() {
        if !visited_children {
            // `node` children not yet visited: visit the children of node first.
            stack.push((node, true));
            for &ch in tree.children[node].iter() {
                stack.push((ch, false));
            }
        } else {
            let children = &tree.children[node];
            if children.is_empty() {
                // leaf node base case
                results[node] = Some(DFSResult {
                    subtree_tonicity: tonicity_ctx[node],
                    subtree_complexity: 0.0,
                    subtree_size: 1,
                    subtree_likelihood: 1.0,
                });
                continue;
            }

            // Sum of each child's complexity contribution
            let mut sum_complexities = 0.0;

            // multiplicative aggregate of each child's likelihood contribution
            // TODO: is there a better method than just multiplying likelihoods of child subtrees?
            let mut mult_likelihoods = 1.0;

            // vecs in order of children (ascending pitch)
            let mut child_subtree_sizes = Vec::with_capacity(children.len());
            let mut child_dyad_complexities = Vec::with_capacity(children.len());
            let mut child_subtree_complexities = Vec::with_capacity(children.len());
            let mut child_subtree_tonicities = Vec::with_capacity(children.len());

            for &ch in children {
                // let (child_ton, child_comp) =
                //     results[ch].expect("Child result missing. Check if tree is valid");
                let results = results[ch]
                    .as_ref()
                    .expect("Child result missing. Check if tree is valid");
                let dyadic_complexity = dyad_comp(node, ch);

                child_subtree_sizes.push(results.subtree_size);
                child_dyad_complexities.push(dyadic_complexity);
                child_subtree_complexities.push(results.subtree_complexity);
                child_subtree_tonicities.push(results.subtree_tonicity);
            }

            let child_weights =
                compute_subtree_weights(&child_subtree_tonicities, &child_subtree_sizes);

            for i in 0..children.len() {
                let results = results[children[i]]
                    .as_ref()
                    .expect("Child result missing. Check if tree is valid");

                let child_subtree_complexity = if child_subtree_sizes[i] == 1 {
                    compute_child_leaf_complexity_contribution(
                        child_weights[i],
                        child_dyad_complexities[i],
                    )
                } else {
                    compute_child_subtree_complexity_contribution(
                        child_weights[i],
                        child_dyad_complexities[i],
                        child_subtree_complexities[i],
                    )
                };

                sum_complexities += child_subtree_complexity;

                let likelihood_contribution = compute_child_likelihood_contribution(
                    cents_asc_order[children[i]] - cents_asc_order[node],
                    tonicity_ctx[node],
                    tonicity_ctx[children[i]],
                    dyad_tonicity(node, children[i]),
                    results.subtree_likelihood,
                    results.subtree_complexity,
                );

                // don't allow over-dominant likelihood contributions.
                mult_likelihoods *= likelihood_contribution.clamp(0.2, 5.0);
            }

            results[node] = Some(DFSResult {
                subtree_tonicity: tonicity_ctx[node] + child_subtree_tonicities.iter().sum::<f64>(),
                subtree_complexity: sum_complexities,
                subtree_size: 1 + child_subtree_sizes.iter().sum::<usize>(),
                subtree_likelihood: mult_likelihoods,
            });
        }
    }

    let result = results[tree.root]
        .as_ref()
        .expect("Root result missing. Check if tree is valid");
    (result.subtree_complexity, result.subtree_likelihood)
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
    /// and the j-th candidate note in `candidate_cents` is 30% tonic in the dyad between them.
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
///
/// See: https://www.desmos.com/calculator/thhgebi4po
fn lower_interval_limit_penalty(centroid_freq: f64) -> f64 {
    return (1.0 / (0.08 * centroid_freq.powf(0.815)) - 0.05).min(1.0);
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
    /// how likely it is to choose to hear that interval relation between two notes (modelled using
    /// roughness).
    ///
    /// - `mult_roughness`: multiplicative roughness >= 1.
    ///
    /// Returns a tuple: (tonicity of lower note, tonicity of higher note).
    ///
    /// TODO: figure out the best function for this.
    fn heuristic_function(dyad_tonicity: f64, mult_roughness: f64) -> (f64, f64) {
        // Only very slight scaling is needed - just by looking at the smoothed tonicity plot
        // visually, for complex intervals, the tonicity gap is already unopinionated.
        (
            dyad_tonicity / mult_roughness.powf(0.05),
            (1.0 - dyad_tonicity) / mult_roughness.powf(0.05),
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
    let smoothing = smoothing.powf(4.0 * elapsed_seconds);
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
    use std::{result, usize};

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

    /// Run this test to compute certain sanity metrics
    ///
    /// (e.g., whether results agree with intuition or not).
    #[test]
    fn test_sanity_metrics() {
        fn dis(cents: &[f64], name: &str) -> GraphDissTestResult {
            graph_diss(cents, name, 5.0, 1)[0].clone()
        }

        let p4 = dis(&[0.0, 500.0], "P4");
        let tritone = dis(&[0.0, 600.0], "Tritone");
        let p5 = dis(&[0.0, 700.0], "P5");
        let min3 = dis(&[0.0, 300.0], "m3");
        let maj3 = dis(&[0.0, 400.0], "M3");

        let maj_high = dis(&[3600.0, 4000.0, 4300.0], "Major +3 oct");
        let maj_low = dis(&[-3600.0, -3200.0, -2900.0], "Major -3 oct");

        let maj = dis(&[0.0, 400.0, 700.0], "Major");
        let min = dis(&[0.0, 300.0, 700.0], "Minor");

        println!("\n\n=============== SANITY METRICS ================\n");

        let min_maj_triad_gap = min.diss.dissonance - maj.diss.dissonance;
        let min_maj_dyad_gap = min3.diss.dissonance - maj3.diss.dissonance;
        println!("        min - maj: {}", min_maj_triad_gap);
        println!(
            " min - maj scaled: {}",
            min_maj_triad_gap / min_maj_dyad_gap
        );
        println!(
            "     tritone - p4: {}",
            tritone.diss.dissonance - p4.diss.dissonance
        );
        println!(
            "          p4 - p5: {}",
            p4.diss.dissonance - p5.diss.dissonance
        );
        println!(
            "  lower intv. lim: {}",
            maj_low.diss.dissonance - maj_high.diss.dissonance
        );
        println!(
            "  P5 tonicity gap: {}",
            p5.diss.tonicity_target[0] - p5.diss.tonicity_target[1]
        );
        println!(" targ. C conf maj: {}", maj.diss.tonicity_target[0]);
        println!(" targ. C conf min: {}", min.diss.tonicity_target[0]);
    }

    #[test]
    fn test_evolution() {
        let cents = vec![0.0, 400.0, 700.0];

        let results = graph_diss(&cents, "", 0.1, 50);

        for (i, res) in results.iter().enumerate() {
            println!("\nIter {}: diss: {}", i, res.diss.dissonance);
            for (j, note) in cents.iter().enumerate() {
                println!("    {}c: {}", note, res.diss.tonicity_context[j]);
            }
        }
    }

    #[test]
    fn test_graph_diss() {
        let time = 2.5;
        let iters = 1;
        graph_diss(&[0.0, 500.0], "P4", time, iters);
        graph_diss(&[0.0, 600.0], "Tritone", time, iters);
        graph_diss(&[0.0, 700.0], "P5", time, iters);
        graph_diss(&[0.0, 100.0], "Semitone", time, iters);

        graph_diss(&[0.0, 400.0, 700.0], "C maj", time, iters);
        graph_diss(&[0.0, 300.0, 700.0], "C min", time, iters);
        graph_diss(&[0.0, -800.0, 700.0], "C/E spread", time, iters);
        graph_diss(&[0.0, -900.0, 700.0], "Cm/Eb spread", time, iters);
        graph_diss(&[0.0, 500.0, 900.0], "F/C", time, iters);
        graph_diss(&[0.0, 500.0, 800.0], "Fm/C", time, iters);
        graph_diss(&[0.0, 300.0, 800.0], "Ab/C", time, iters);
        graph_diss(&[0.0, 400.0, 900.0], "Am/C", time, iters);
        graph_diss(&[0.0, 100.0, 200.0], "3 semi cluster", time, iters);
        graph_diss(&[0.0, 50.0, 100.0], "3 quarter cluster", time, iters);

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

        graph_diss(
            &[0.0, 400.0, 700.0, 1100.0, 1400.0, 1800.0, 2100.0],
            "Cmaj13#11",
            time,
            iters,
        );
        graph_diss(
            &[0.0, 300.0, 700.0, 1000.0, 1400.0, 1700.0, 2100.0],
            "Cmin13",
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

    /// One iteration's result in the graph_diss test below.
    #[derive(Clone, Debug)]
    struct GraphDissTestResult {
        diss: Dissonance,
        debug: Option<GraphDissDebug>,
    }

    fn graph_diss(
        cents: &[f64],
        name: &str,
        elapsed_seconds: f64,
        iters: usize,
    ) -> Vec<GraphDissTestResult> {
        // if non-zero, show this many lowest complexity trees per root
        const SHOW_N_TREES: usize = 2;

        // if true, show trees sorted by descending diss contribution (complexity * softmax likelihood)
        const SHOW_TOP_N_CONTRIBUTING_TREES: usize = 10;

        // if false, inits with uniform tonicity context.
        const INIT_WITH_DYADIC_HEURISTIC: bool = true;

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

        let heuristic_tonicities = context.clone();

        let mut results_per_iter = vec![];

        println!(
            "\n============  Graph diss: {}  =====================\n",
            name
        );
        for i in 0..iters {
            let mut debug = if SHOW_N_TREES | SHOW_TOP_N_CONTRIBUTING_TREES == 0 {
                None
            } else {
                Some(GraphDissDebug::new(SHOW_N_TREES, cents.len()))
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

            results_per_iter.push(GraphDissTestResult {
                diss: diss[0].clone(),
                debug: debug.clone(),
            });

            println!("Iteration {}/{iters}", i + 1);
            for (idx, cents) in cents.iter().enumerate() {
                println!(
                    "  {:>7.2}c: ton {:>6.4}, ton heur: {:>6.4}, ton tgt: {:>6.4}, diss raw: {:>6.4}, diss ctx: {:>6.4}",
                    cents,
                    diss[0].tonicity_context[idx],
                    heuristic_tonicities[idx],
                    diss[0].tonicity_target[idx],
                    debug.as_ref().map(|d| d.per_root_agg_complexities[idx]).unwrap_or(f64::NAN),
                    debug.as_ref().map(|d| d.per_root_agg_complexities_weighted_global[idx]).unwrap_or(f64::NAN)
                );
            }
            println!("Diss: {:.4}", diss[0].dissonance);
            println!("{}s: {:#?}", (i + 1) as f64 * elapsed_seconds, diss);

            // Print the lowest complexity trees per root

            if let Some(debug) = debug {
                let sum_exp_likelihood = debug
                    .trees_sorted_asc_contrib
                    .iter()
                    .map(|tree| tree.likelihood.exp())
                    .sum::<f64>();
                for (root_idx, root) in cents.iter().enumerate() {
                    if SHOW_N_TREES != 0 {
                        println!(
                            "Lowest {} complexity trees for root {:>7.2}c:",
                            debug.N, root
                        );
                        for tree in debug.n_lowest_comp_trees_per_root[root_idx].iter() {
                            println!(
                                " -> complexity {:>6.4}, likelihood {:>6.4}, contrib {:>6.4}:",
                                tree.complexity,
                                tree.likelihood,
                                tree.likelihood.exp() / sum_exp_likelihood * tree.complexity
                            );
                            tree.tree.print();
                        }
                        println!(
                            "Highest {} likelihood trees for root {:>7.2}c:",
                            debug.N, root
                        );
                        for tree in debug.n_highest_like_trees_per_root[root_idx].iter().rev() {
                            println!(
                                " -> complexity {:>6.4}, likelihood {:>6.4}, contrib {:>6.4}:",
                                tree.complexity,
                                tree.likelihood,
                                tree.likelihood.exp() / sum_exp_likelihood * tree.complexity
                            );
                            tree.tree.print();
                        }
                    }
                }
                if SHOW_TOP_N_CONTRIBUTING_TREES != 0 {
                    println!("Trees sorted by descending diss contribution:");

                    for tree in debug
                        .trees_sorted_asc_contrib
                        .iter()
                        .rev()
                        .take(SHOW_TOP_N_CONTRIBUTING_TREES)
                    {
                        let contrib = tree.likelihood.exp() / sum_exp_likelihood * tree.complexity;
                        println!(
                            " -> contrib {:>6.4} (complexity {:>6.4}, likelihood {:>6.4}):",
                            contrib, tree.complexity, tree.likelihood
                        );
                        tree.tree.print();
                    }
                }
            }
            context = diss[0].tonicity_context.clone();
        }

        results_per_iter
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
