//! Polyadic dissonance & tonicity of notes.

use core::f64;
use std::{collections::BTreeSet, ops::Deref};

use compute::prelude::{max, softmax};

use rapidhash::RapidHashMap as HashMap;

use crate::{
    dyad_lookup::{DyadLookup, RoughnessType, TonicityLookup},
    tree_gen::{is_node_part_of_subtree, SubtreeKey, ST, TREES},
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
///
/// HOW TO TUNE: If this value is too high, new candidate notes will too quickly be accepted into
/// the pitch memory/as a new root.
///
/// If this value is too low, the new candidate note can have such low tonicity that calculations
/// involving tonicity context (e.g. global tonicity alignment and tonicity update using softmax)
/// become unstable.
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

/// Controls the weight of global tonicity alignment, i.e., how much more likely is a subtree if the
/// parent of the subtree has global tonicity 1.0 and subtree root has global tonicity 0.0, than if
/// the parent had global tonicity 0.0 and subtree root had global tonicity 1.0.
///
/// A value of 1.0 means that global tonicity context does not affect likelihood.
///
/// A value equal to [DYADIC_TONICITY_LIKELIHOOD_SCALING] means that global tonicity context matters
/// about as much as dyadic tonicity alignment.
///
/// HOW TO TUNE: Increasing this value will increase the major-minor triad dissonance gap, and make
/// global tonicity scores more confident by increasing the variation in likelihood between
/// interpretation trees.
///
/// If this value is too high, the global tonicity alignment affects the likelihood of
/// interpretation trees, which in turn affects global tonicity context. If
/// [TONICITY_CONTEXT_TEMPERATURE_TARGET] is too low (opinionated), or if
/// [NEW_CANDIDATE_TONICITY_RATIO] is too low (where very low tonicity scores are assigned to new
/// notes), it is posible to get a feedback loop where tonicity becomes asymptotically 1.0 for one
/// note and 0.0 for the rest. This can be balanced slighly by increasing the softmax temperature
/// [TONICITY_CONTEXT_TEMPERATURE_TARGET].
///
/// If this value is too low, the global tonicity scores will not sufficiently affect likelihood of
/// trees, which may make the model insensitive to the current harmonic context/perceived root.
const GLOBAL_TONICITY_LIKELIHOOD_SCALING: f64 = (2u128 << 10) as f64;

/// The maximum increase/decrease in likelihood a subtree can have due to global tonicity alignment.
///
/// This value is the natural log of the maximum multiplicative factor the natural log is allowed to
/// have.
///
/// A value of 1.0 means that even if a parent note has significantly higher/lower global tonicity
/// than the child note, the maximum increase/decrease in likelihood the subtree can have is e^1.0 =
/// 2.718x.
///
/// HOW TO TUNE: if the global tonicity quickly becomes asymptotic (where only one note has nearly
/// 100% tonicity), try decreasing this value to limit the feedback effect of global tonicity
/// alignment on likelihoods.
const GLOBAL_TONICITY_LIKELIHOOD_MAX_LN: f64 = 1.0;

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
///
/// HOW TO TUNE: If this value is too low, the global tonicity scores will be too opinionated. If
/// [NEW_CANDIDATE_TONICITY_RATIO] is low, then a new note will enter the global tonicity context
/// with very low score, which will skew the global tonicity context very heavily towards the
/// argmax. If [GLOBAL_TONICITY_LIKELIHOOD_SCALING] is too high, there can be a feedback loop which
/// causes only one note to have the majority of the tonicity score: balance this by decreasing
/// [GLOBAL_TONICITY_LIKELIHOOD_MAX_LN].
///
/// If this value is too high, the global tonicity scores will be too uniform, which will cause the
/// model to lose the ability to make contextually informed dissonance ratings. To counteract the
/// uniformity of global tonicity context in dissonance calculation (without changing
/// [GLOBAL_TONICITY_LIKELIHOOD_SCALING]), you can increase [TONICITY_CONTEXT_TEMPERATURE_DISS]
/// which only affects the final dissonance calculation (but the effect of global tonicity alignment
/// will still be diluted).
const TONICITY_CONTEXT_TEMPERATURE_TARGET: f64 = 0.8;

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

    pub fn print(&self, cents: &[f64], show_top_n_contributing_trees: usize) {
        let sum_exp_likelihood = self
            .trees_sorted_asc_contrib
            .iter()
            .map(|tree| tree.likelihood.exp())
            .sum::<f64>();
        for (root_idx, root) in cents.iter().enumerate() {
            if self.N != 0 {
                println!(
                    "Lowest {} complexity trees for root {:>7.2}c:",
                    self.N, root
                );
                for tree in self.n_lowest_comp_trees_per_root[root_idx].iter() {
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
                    self.N, root
                );
                for tree in self.n_highest_like_trees_per_root[root_idx].iter().rev() {
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
        if show_top_n_contributing_trees != 0 {
            println!("Trees sorted by descending diss contribution:");

            for tree in self
                .trees_sorted_asc_contrib
                .iter()
                .rev()
                .take(show_top_n_contributing_trees)
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
}

/// Inline helper function to build a mask table for mapping bitmasks of nodes in ascending pitch
/// order to bitmasks of nodes in original frequency order (where candidate note, if any, is the
/// last element).
///
/// Input: bitmask of nodes in ascending pitch order where LSB = lowest pitch note.
///
/// Output: bitmask of nodes in original frequency order where LSB = freqs[0] (according to
/// `asc_idx_to_og_idx`)
#[inline]
fn build_mask_table(asc_idx_to_og_idx: &[u8]) -> Vec<u8> {
    let mut table = vec![0u8; 1 << asc_idx_to_og_idx.len()];
    for m in 0..table.len() {
        let mut out = 0u8;
        for asc in 0..asc_idx_to_og_idx.len() {
            if (m >> asc) & 1 == 1 {
                let og = asc_idx_to_og_idx[asc];
                out |= 1 << og;
            }
        }
        table[m] = out;
    }
    table
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
///   If the sum of tonicity_context is 0.0 (i.e., no context), the model will assume initialize the
///   tonicity context based on the dyadic heuristic tonicities computed from
///   [dyadic_tonicity_heur].
///
/// - `smoothing`: How quickly the updated tonicity context should approach the target tonicity. 0.0
///   = no smoothing, 1.0 = no movement at all.
///
/// - `elapsed_seconds`: time elapsed used to scale smoothing - higher elapsed time = less
///   smoothing.
///
/// - `max_trees`: maximum number of interpretation trees to use. If there are more than `max_trees`
///   possible interpretation trees, a random subset of pre-computed trees will be used.
///
/// - `debug`: optional debug object to collect per-root complexities and likelihoods. Only use in
///   tests.
///
/// Rhythmic entrainment can be implemented by setting smoothing lower at regular time intervals and
/// higher at others.
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
    max_trees: usize,
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

    let tonicity_context = if tonicity_context.iter().sum::<f64>() == 0.0 {
        heuristic_tonicities.tonicities_no_cand.clone()
    } else {
        tonicity_context.to_vec()
    };

    let dyad_roughs = heuristic_tonicities.add_roughness_map; // use additive roughness.
    let cand_roughs = heuristic_tonicities.cand_add_roughness_map;
    let dyad_tonics = heuristic_tonicities.tonicity_map;
    let cand_tonics = heuristic_tonicities.cand_tonicity_map;

    let mut results = vec![];

    let st_trees = TREES
        .get(num_notes)
        .expect("No precomputed trees for this number of notes");

    let st_trees_indices = if st_trees.len() > max_trees {
        // Randomly sample max_trees number of trees.
        fastrand::choose_multiple(0..st_trees.len(), max_trees)
    } else {
        (0..st_trees.len()).collect::<Vec<usize>>()
    };

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
        let asc_to_og_idxs: Vec<u8> = {
            let mut pairs: Vec<(usize, f64)> = cents.iter().cloned().enumerate().collect();
            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            pairs.iter().map(|(idx, _)| *idx as u8).collect::<Vec<_>>()
        };

        // Input: original freqs idx, Output: ascending pitch order.
        let og_to_asc_idx: Vec<u8> = {
            let mut inv = vec![0; asc_to_og_idxs.len()];
            for (sorted_idx, &orig_idx) in asc_to_og_idxs.iter().enumerate() {
                inv[orig_idx as usize] = sorted_idx as u8;
            }
            inv
        };

        // `tonicity_context` sorted in low-to-high pitch order.
        let sorted_tonicity_ctx: Vec<f64> = asc_to_og_idxs
            .iter()
            .map(|&idx| tonicity_context[idx as usize])
            .collect();

        // Function to obtain precomputed dyad complexity of pitches indexed in ASCENDING PITCH
        // order.
        let dyad_comp = |from: u8, to: u8| {
            let idx_from = asc_to_og_idxs[from as usize];
            let idx_to = asc_to_og_idxs[to as usize];
            let bitmask = (1 << idx_from) | (1 << idx_to);
            *dyad_roughs
                .get(&bitmask)
                .expect("Missing edge in precomputed dyadic roughness lookup!")
        };

        let dyad_tonicity = |from: u8, to: u8| {
            let idx_from = asc_to_og_idxs[from as usize];
            let idx_to = asc_to_og_idxs[to as usize];
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

        let mut memoized_dfs_results: HashMap<SubtreeKey, DFSResult> = HashMap::default();

        for tree_idx in &st_trees_indices {
            let tree = &st_trees[*tree_idx];
            let (comp, like) = dfs_st_comp_likelihood(
                &cents_asc_order,
                tree,
                &sorted_tonicity_ctx,
                dyad_comp,
                dyad_tonicity,
                &mut memoized_dfs_results,
                // For non-candidate graph diss, these lookup tables are not required.
                u8::MAX,
                &[],
                &[],
            );
            let exp_likelihood = (like / TONICITY_CONTEXT_TEMPERATURE_TARGET).exp();
            comp_like_per_root_lo_to_hi[tree.root as usize].push((comp, exp_likelihood));
            likelihood_exp_sum += exp_likelihood;

            if let Some(d) = &mut debug {
                let entry = TreeResult {
                    tree: &tree,
                    complexity: comp,
                    likelihood: like,
                };

                let lowest_comp_trees = &mut d.n_lowest_comp_trees_per_root[tree.root as usize];
                lowest_comp_trees.insert(TreeResultCompAsc(entry.clone()));

                // Keep only the N lowest complexity trees
                while lowest_comp_trees.len() > d.N {
                    lowest_comp_trees.pop_last();
                }

                let highest_like_trees = &mut d.n_highest_like_trees_per_root[tree.root as usize];
                highest_like_trees.insert(TreeResultLikeAsc(entry.clone()));

                while highest_like_trees.len() > d.N {
                    highest_like_trees.pop_first();
                }

                d.trees_sorted_asc_contrib
                    .insert(TreeResultContribAsc(entry));
            }
        }

        let comp_like_per_root_og_order = (0..num_notes)
            .map(|i| comp_like_per_root_lo_to_hi[og_to_asc_idx[i] as usize].as_slice())
            .collect::<Vec<_>>();

        results.push(compute_interpretation_trees(
            &comp_like_per_root_og_order,
            likelihood_exp_sum,
            &tonicity_context,
            smoothing,
            elapsed_seconds,
            debug,
        ));

        return results;
    }

    // If candidates are supplied, we have to repeat the above process with each candidate
    // as the last indexed note provided freqs.

    let mut memoized_dfs_results: HashMap<OGIdxSubtreeKey, DFSResult> = HashMap::default();

    // When testing different candidate notes, high probability that the ordering of notes end up
    // being the same. Don't recompute the mask table every time.
    //
    // - Key: og_to_asc_idxs hash where the i-th 3 bits correspond to og_to_asc_idxs[i]
    // - Value: mask table mapping asc pitch order bitmask to og order bitmask.
    //
    // If og_to_asc_idxs order is same, just reuse the same mask table.
    let mut memoized_mask_tables: HashMap<u32, Vec<u8>> = HashMap::default();

    for candidate_idx in 0..num_candidates {
        let curr_candidate_cents = candidate_cents[candidate_idx];
        let cents_asc_order = {
            let mut combined_cents = cents.clone();
            combined_cents.push(curr_candidate_cents);
            combined_cents.sort_by(|a, b| a.partial_cmp(b).unwrap());
            combined_cents
        };
        let asc_to_og_idxs: Vec<u8> = {
            let mut pairs: Vec<(usize, f64)> = cents
                .iter()
                .cloned()
                .enumerate()
                .collect::<Vec<(usize, f64)>>();
            pairs.push((freqs.len(), curr_candidate_cents));
            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            pairs.iter().map(|(idx, _)| *idx as u8).collect::<Vec<_>>()
        };

        let og_to_asc_idxs: Vec<u8> = {
            let mut inv = vec![0; asc_to_og_idxs.len()];
            for (sorted_idx, &orig_idx) in asc_to_og_idxs.iter().enumerate() {
                inv[orig_idx as usize] = sorted_idx as u8;
            }
            inv
        };

        let og_to_asc_hash: u32 = og_to_asc_idxs
            .iter()
            .enumerate()
            .fold(0u32, |acc, (i, &v)| acc | ((v as u32) << (i * 3)));

        let og_idx_mask_lut = if let Some(mask_table) = memoized_mask_tables.get(&og_to_asc_hash) {
            mask_table.clone()
        } else {
            let mask_table = build_mask_table(&asc_to_og_idxs);
            memoized_mask_tables.insert(og_to_asc_hash, mask_table.clone());
            mask_table
        };

        // Since we are adding candidate notes, we have to use the dyadic tonicity heuristic to
        // find what initial tonicity score to use for the candidate note and scale accordingly.

        // To model the intuition that "new notes are less accepted", I will only use a fraction of
        // the heuristic tonicity score, and scale down the rest of the existing notes' tonicities
        // linearly. This scaling is arbitrary and can be adjusted later.
        //
        // Other initialization options: preset 0.001 tonicity, 100% heuristic tonicity, etc...

        let heur_candidate_tonicity = heuristic_tonicities.tonicities[candidate_idx][num_notes - 1];

        // The new tonicity context including the candidate note, in original order of `freqs`
        // where the candidate note is at the last index.
        let new_tonicity_ctx: Vec<f64> =
            scale_candidate_toncity(&tonicity_context, heur_candidate_tonicity);

        // Computes dyad complexities for (from, to) indices based on ascending pitch order.
        let dyad_comp = |from: u8, to: u8| {
            let idx_from = asc_to_og_idxs[from as usize];
            let idx_to = asc_to_og_idxs[to as usize];

            let bitmask = (1 << idx_from) | (1 << idx_to);

            if let Some(roughness) = dyad_roughs.get(&bitmask) {
                // both notes are non-candidate notes, the complexity is found in the dyad_roughs lookup.
                return *roughness;
            }
            // implicitly, if idx_from == freqs.len() || idx_to == freqs.len() {

            let existing_pitch_idx = idx_from.min(idx_to) as usize;
            cand_roughs[existing_pitch_idx]
                .get(candidate_idx) // which candidate to use
                .cloned()
                .expect("Missing edge in candidate dyadic roughness lookup!")
        };

        let dyad_tonicity = |from: u8, to: u8| {
            let idx_from = asc_to_og_idxs[from as usize];
            let idx_to = asc_to_og_idxs[to as usize];

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

            let from_is_candidate = idx_from as usize == freqs.len();

            let idx_of_existing_note = if from_is_candidate { idx_to } else { idx_from };

            let tonicity_of_existing_note = cand_tonics[idx_of_existing_note as usize]
                .get(candidate_idx)
                .cloned()
                .expect("Missing edge in candidate dyadic tonicity lookup!");

            let tonicity_of_from = if from_is_candidate {
                1.0 - tonicity_of_existing_note
            } else {
                tonicity_of_existing_note
            };

            tonicity_of_from
        };

        let mut comp_like_per_root_lo_to_hi: Vec<Vec<(f64, f64)>> = vec![vec![]; num_notes];
        let mut likelihood_exp_sum = 0.0;

        for tree_idx in &st_trees_indices {
            let tree = &st_trees[*tree_idx];
            let (complexity, likelihood) = dfs_st_comp_likelihood(
                &cents_asc_order,
                tree,
                &new_tonicity_ctx,
                dyad_comp,
                dyad_tonicity,
                &mut memoized_dfs_results,
                candidate_idx as u8,
                &asc_to_og_idxs,
                &og_idx_mask_lut,
            );

            let exp_likelihood = (likelihood / TONICITY_CONTEXT_TEMPERATURE_TARGET).exp();
            comp_like_per_root_lo_to_hi[tree.root as usize].push((complexity, exp_likelihood));
            likelihood_exp_sum += exp_likelihood;
        }

        let comp_like_per_root_og_order = (0..num_notes)
            .map(|i| comp_like_per_root_lo_to_hi[og_to_asc_idxs[i] as usize].as_slice())
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

/// Helper for scaling existing tonicity values & adding candidate tonicity value given the
/// candidate's heuristic toncity.
///
/// - `existing_tonicities`: existing tonicities of notes in original order.
/// - `heur_candidate_tonicity`: heuristic tonicity of the candidate note.
///
/// Returns new tonicity vector including candidate tonicity at the last index.
fn scale_candidate_toncity(existing_tonicities: &[f64], heur_candidate_tonicity: f64) -> Vec<f64> {
    let new_cand_tonicity = heur_candidate_tonicity * NEW_CANDIDATE_TONICITY_RATIO;
    let scale = 1.0 - new_cand_tonicity;
    existing_tonicities
        .iter()
        .map(|x| x * scale)
        .chain(std::iter::once(new_cand_tonicity))
        .collect()
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
    subtree_size: u8,

    /// Likelihood of this subtree interpretation, computed as some aggregate of child likelihood contributions.
    ///
    /// Likelihood is in the range [0, infty), and has no units/relative scale. The likelihoods will be normalized
    /// after all trees are evaluated.
    ///
    /// If subtree is a single leaf node, this is 1.0.
    subtree_likelihood: f64,

    /// Which candidate frequency was being used to compute this result.
    ///
    /// If no candidate frequency appears in this subtree, the result from the previous candidate
    /// computation can be copied over by updating the candidate_idx.
    candidate_idx: u8,
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
    child_subtree_sizes: &[u8],
) -> Vec<f64> {
    let sum_subtree_sizes = child_subtree_sizes.iter().sum::<u8>() as f64;
    let uniform_weighted_nodes: Vec<f64> = child_subtree_sizes
        .iter()
        .map(|s| *s as f64 / sum_subtree_sizes)
        .collect();

    let total_tonicity: f64 = unnormalized_child_subtree_tonicities.iter().sum();
    let mut normalized_and_scaled: Vec<f64> =
        Vec::with_capacity(unnormalized_child_subtree_tonicities.len());

    for ct in unnormalized_child_subtree_tonicities.iter() {
        normalized_and_scaled.push(ct / total_tonicity * LOCAL_TONICITY_TEMP);
    }

    let softmax_local_tonicities = softmax(&normalized_and_scaled);

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
    let global_tonicity_alignment_ln =
        (GLOBAL_TONICITY_LIKELIHOOD_SCALING.ln() * (global_tonicity_logistic - 0.5)).clamp(
            -GLOBAL_TONICITY_LIKELIHOOD_MAX_LN,
            GLOBAL_TONICITY_LIKELIHOOD_MAX_LN,
        );

    let dyadic_tonicity_alignment_ln =
        DYADIC_TONICITY_LIKELIHOOD_SCALING.ln() * (dyadic_tonicity - 0.5);

    // kappa (lambda_i, c_i) in article.md
    let adjusted_subtree_likelihood_ln = if subtree_complexity == 0.0 {
        // edge case, 0 complexity subtree probably means that the subtree is a leaf node.
        //
        // Deep tree penalty doesn't apply, only applies to non-leaf children.
        subtree_likelihood.ln()
    } else {
        subtree_likelihood.ln() * (1.0 + COMPLEXITY_LIKELIHOOD_SCALING * (0.5 - subtree_complexity))
            + f64::consts::LN_2
                * (COMPLEXITY_LIKELIHOOD_BIAS * (0.5 - subtree_complexity)
                    - DEEP_TREE_LIKELIHOOD_PENALTY)
    };

    let lower_root_bias_ln =
        LOW_NOTE_ROOT_LIKELIHOOD_SCALING.ln() * (parent_child_interval_cents / 1200.0);

    // TODO: besides geometric mean, any better aggregation methods?
    let log_likelihood = (global_tonicity_alignment_ln
        + dyadic_tonicity_alignment_ln
        + adjusted_subtree_likelihood_ln
        + lower_root_bias_ln)
        * 0.25;

    log_likelihood.exp()
}

/// Represents SubtreeKey but with node indices remapped to original freq indices + candidate note at
/// the end, instead of increasing pitch order indexing.
type OGIdxSubtreeKey = SubtreeKey;

/// The generated [SubtreeKey]s for each [ST] in [TREES] have node index sorted in ascending pitch
/// order.
///
/// However, pitch order may change if candidate note changes, so in order to memoize results across
/// different candidate choices (where we reuse precomputed trees that do not rely candidate pitch),
/// we have to store [SubtreeKey]s with node indices remapped to original freq indices and candidate
/// note at index `freqs.len()`.
///
/// - `subtree_key`: the original subtree key with ascending pitch order node indices.
/// - `asc_idx_to_og_idx`: mapping from ascending pitch order index to original freq index.
/// - `og_idx_mask_lut`: lookup table to remap bitmask of nodes in ascending pitch order to original
///   freq order. Generated by [build_mask_table].
fn remap_subtree_key_to_og_indexing(
    subtree_key: SubtreeKey,
    asc_idx_to_og_idx: &[u8],
    og_idx_mask_lut: &[u8],
) -> OGIdxSubtreeKey {
    let mut new_key: SubtreeKey = 0;

    // Bits 0-23: The i-th 3-bit segment from LSB corresponds to the i-th node's parent (or 0 if root).
    //
    // Remap each parent idx
    for node_idx in 0..asc_idx_to_og_idx.len() {
        let asc_order_parent_idx = ((subtree_key >> (node_idx * 3)) & 0b111) as usize; // 3 bits per node

        let og_order_parent_idx = asc_idx_to_og_idx[asc_order_parent_idx];

        new_key |= (og_order_parent_idx as SubtreeKey) << (node_idx * 3);
    }

    // Bits 24-31: Relative to bit 24 LSB,, the i-th bit indicates whether the i-th node is part of
    // the subtree or not. Remaps the bitmask according to og_idx_mask_lut.

    let asc_order_bitmask = ((subtree_key >> 24) & 0xFF) as u8;
    let og_order_bitmask = og_idx_mask_lut[asc_order_bitmask as usize];
    new_key |= (og_order_bitmask as SubtreeKey) << 24;

    // Bits 32-34: Remap root node index from asc idx to og idx.

    let asc_order_root_idx = (subtree_key >> 32) & 0b111;
    let og_order_root_idx = asc_idx_to_og_idx[asc_order_root_idx as usize] as u64;
    new_key |= og_order_root_idx << 32;

    new_key
}

/// Evaluate spanning tree complexity & likelihood using DFS.
///
/// **IMPORTANT ⚠️⚠️⚠️ USE ASCENDING PITCH ORDER:**
///
///   This function requires input notes to be indexed in ASCENDING PITCH order. The freqs provided
///   from the visualizer are in no particular order, so a sorted index mapping must be obtained
///   first.
///
///   The search space of precomputed trees is pruned so that pre-order inversions are capped at
///   `max_inversions`, which models the natural low-to-high precendence of pitch perception.
///
/// See `paper/article.md` for details.
///
/// Same algo as `dfs()` in `paper/triad_sts_computation_example.py`
///
/// - `cents_asc_order`: list of notes in cents, indexed in **ascending pitch order**.
///
/// - `tree`: the spanning tree to evaluate
///
/// - `tonicity_ctx`: tonicity context per node in **ascending pitch order**. If doing dfs with an
///   added candidate note, the tonicity context should include the candidate note in ascending
///   pitch order.
///
/// - `dyad_comp`: function that returns the dyadic/edge additive complexity (range 0-1) of (from,
///   to) edge in **ascending pitch order**.
///
/// - `dyad_tonicity`: function that takes in `(from, to)` indices in **ascending pitch order** and
///   returns the tonicity of the `from` node (i.e., between the two notes in the from-to dyad, how
///   tonic is `from`).
///
/// - `memoized_dfs_results`: memoization map to speed up repeated DFS calls on identical subtrees.
///
/// - `curr_candidate_idx`: if evaluating multiple candidate notes, this is the index of which
///   candidate in candidate_freqs is being evaluated. Used to memoize trees across different
///   candidates. If no candidate notes are supplied, set to u8::MAX = 255.
///
/// - `asc_idx_to_og_idx`: Map from ascending index order to original index order in freqs (where
///   last idx is the candidate frequency). Only used if curr_candidate_idx != u8::MAX, otherwise
///   just pass an empty slice.
///
/// - `og_idx_mask_lut`: Lookup table to remap subtree bitmask from ascending pitch order to
///   original freq order. Use `build_mask_table` to generate. Only used if curr_candidate_idx !=
///   u8::MAX, otherwise just pass an empty slice.
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
    memoized_dfs_results: &mut HashMap<OGIdxSubtreeKey, DFSResult>,
    curr_candidate_idx: u8,
    asc_idx_to_og_idx: &[u8],
    og_idx_mask_lut: &[u8],
) -> (f64, f64)
where
    F: Fn(u8, u8) -> f64,
    G: Fn(u8, u8) -> f64,
{
    // stack contains (node index, children visited?)
    //
    // For post-order traversal.
    let mut stack: Vec<(u8, bool)> = vec![(tree.root, false)];

    // store computed results per node
    //
    // Indexed by same order as tree nodes.
    let mut results: Vec<Option<DFSResult>> = vec![None; tonicity_ctx.len()];

    while let Some((node, visited_children)) = stack.pop() {
        let node = node as usize;

        if !visited_children {
            // `node` children not yet visited: visit the children of node first.
            stack.push((node as u8, true));
            for &ch in tree.children[node as usize].iter() {
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
                    candidate_idx: curr_candidate_idx,
                });
                continue;
            }

            let asc_order_subtree_key = &tree.subtree_key[node];
            let reidx_subtree_key = if curr_candidate_idx != u8::MAX {
                remap_subtree_key_to_og_indexing(
                    *asc_order_subtree_key,
                    asc_idx_to_og_idx,
                    og_idx_mask_lut,
                )
            } else {
                // We don't have to care about preserving original freqs index through automorphisms if
                // no candidate notes are being evaluated.
                *asc_order_subtree_key
            };

            if let Some(memoized_result) = memoized_dfs_results.get_mut(&reidx_subtree_key) {
                // We can only reuse memoized result if
                // - Subtree does not contain candidate index, or
                // - Subtree contains candidate index, but this particular subtree has already been
                //   computed with curr_candidate_idx.

                let subtree_already_computed_with_curr_cand =
                    memoized_result.candidate_idx == curr_candidate_idx;

                let candidate_in_subtree =
                    is_node_part_of_subtree(reidx_subtree_key, cents_asc_order.len() - 1);

                if !candidate_in_subtree && !subtree_already_computed_with_curr_cand {
                    // Memoized result can carry forward from previous candidate's computation,
                    // since candidate is not part of the subtree.

                    memoized_result.candidate_idx = curr_candidate_idx;
                    results[node] = Some(memoized_result.clone());
                    continue;
                } else if subtree_already_computed_with_curr_cand {
                    // Subtree is already memoized for this candidate, so we can reuse it, whether
                    // or not it contains the candidate.

                    results[node] = Some(memoized_result.clone());
                    continue;
                }
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
                let results = results[ch as usize]
                    .as_ref()
                    .expect("Child result missing. Check if tree is valid");
                let dyadic_complexity = dyad_comp(node as u8, ch);

                child_subtree_sizes.push(results.subtree_size);
                child_dyad_complexities.push(dyadic_complexity);
                child_subtree_complexities.push(results.subtree_complexity);
                child_subtree_tonicities.push(results.subtree_tonicity);
            }

            let child_weights =
                compute_subtree_weights(&child_subtree_tonicities, &child_subtree_sizes);

            for i in 0..children.len() {
                let child = children[i] as usize;
                let results = results[child]
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
                    cents_asc_order[child] - cents_asc_order[node],
                    tonicity_ctx[node],
                    tonicity_ctx[child],
                    dyad_tonicity(node as u8, child as u8),
                    results.subtree_likelihood,
                    results.subtree_complexity,
                );

                // don't allow over-dominant likelihood contributions.
                mult_likelihoods *= likelihood_contribution.clamp(0.2, 5.0);
            }

            results[node] = Some(DFSResult {
                subtree_tonicity: tonicity_ctx[node] + child_subtree_tonicities.iter().sum::<f64>(),
                subtree_complexity: sum_complexities,
                subtree_size: 1 + child_subtree_sizes.iter().sum::<u8>(),
                subtree_likelihood: mult_likelihoods,
                candidate_idx: curr_candidate_idx,
            });

            memoized_dfs_results.insert(reidx_subtree_key, results[node].as_ref().unwrap().clone());
        }
    }

    let result = results[tree.root as usize]
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

    /// The tonicity score of only the existing notes in `cents`, without considering any candidate notes.
    ///
    /// If no candidate notes are provided, this is equivalent to `tonicities[0]`.
    pub tonicities_no_cand: Tonicities,

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

    let mut tonicity_map = HashMap::default();
    let mut mult_roughness_map = HashMap::default();
    let mut add_roughness_map = HashMap::default();

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

    // Apply softmax on existing_tonicities.
    //
    // Normalize sum to 1 first.
    let tonicities_no_candidates: Vec<f64> = existing_tonicities
        .iter()
        .map(|x| x / existing_sum)
        .collect();

    let max_tonicity = max(tonicities_no_candidates.as_slice());

    let tonicities_no_candidates = softmax(
        &tonicities_no_candidates
            .iter()
            .map(|x| (x - max_tonicity) / tonicity_temperature)
            .collect::<Vec<f64>>(),
    );

    if candidate_cents.len() == 0 {
        return TonicityHeuristic {
            tonicities: vec![tonicities_no_candidates.clone()],
            tonicities_no_cand: tonicities_no_candidates,
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
        tonicities_no_cand: tonicities_no_candidates,
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
    use std::time::Instant;
    use std::{result, usize};

    const MAX_TREES: usize = usize::MAX;

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

        // This part tests whether newly added candidate notes with very low toncity cause
        // asymptotic numerical instability issues, which results in high difference between the
        // dissonance & tonicity scores obtained from when all 4 notes are evaluated together with
        // heuristically initialized tonicity vs. when the first 3 notes are existing and the last
        // note is added as a candidate with very low tonicity (scaled by
        // `NEW_CANDIDATE_TONICITY_RATIO`)
        let cents = [0.0, 700.0, 1400.0, 1600.0];
        let freqs = cents
            .iter()
            .map(|x| cents_to_hz(440.0, *x))
            .collect::<Vec<f64>>();

        let diss_existing = graph_dissonance(
            &freqs,
            &[],
            &vec![0f64; freqs.len()],
            0.9,
            5.0,
            MAX_TREES,
            None,
        )[0]
        .clone();

        let diss_candidate = graph_dissonance(
            &freqs[..(freqs.len() - 1)],
            &[*freqs.last().unwrap()],
            &vec![0f64; freqs.len() - 1],
            0.9,
            5.0,
            MAX_TREES,
            None, // NOTE: debug values are not implemented for candidate graph diss.
        )[0]
        .clone();
        let existing_vs_cand_diss_gap = diss_existing.dissonance - diss_candidate.dissonance;

        // This part is a benchmark where we perform 201 iterations of a 8-note update (which is
        // about the same complexity as 20 7-note selectCandidate ops with 10 candidates).

        let mut bench_ctx = vec![0.0; 8];
        const bench_cents_1: [f64; 8] = [
            // C+7b5#9 voicing: C E F# Bb, E Ab C D#
            // held for 1000 iterations at 0.01s per iteration.
            0.0, 400.0, 600.0, 1000.0, 1400.0, 1800.0, 2400.0, 2700.0,
        ];
        const bench_cents_2: [f64; 8] = [
            // Fmaj6/9 voicing: F A C D, G A D F
            0.0, 400.0, 700.0, 900.0, 1400.0, 1600.0, 2100.0, 2400.0,
        ];
        let bench_freqs_1 = bench_cents_1
            .iter()
            .map(|x| cents_to_hz(130.812, *x))
            .collect::<Vec<f64>>();
        let bench_freqs_2 = bench_cents_2
            .iter()
            .map(|x| cents_to_hz(174.614, *x))
            .collect::<Vec<f64>>();

        let start_time = Instant::now();
        for iter in 0..=200 {
            let freqs = if iter <= 100 {
                &bench_freqs_1
            } else {
                &bench_freqs_2
            };
            let res = &graph_dissonance(freqs, &[], &bench_ctx, 0.9, 0.01, MAX_TREES, None)[0];
            bench_ctx.copy_from_slice(&res.tonicity_context);

            if iter % 10 == 0 {
                println!(
                    "Bench iter {}: diss: {}, ctx: {:?}",
                    iter, res.dissonance, res.tonicity_context
                );
            }
        }
        let elapsed = start_time.elapsed();

        // This part tests common intervals have a generally accepted relative dissonance ranking.

        let min3 = dis(&[0.0, 300.0], "m3");
        let maj3 = dis(&[0.0, 400.0], "M3");

        let maj_high = dis(&[3600.0, 4000.0, 4300.0], "Major +3 oct");
        let maj_low = dis(&[-3600.0, -3200.0, -2900.0], "Major -3 oct");

        let maj = dis(&[0.0, 400.0, 700.0], "Major");
        let min = dis(&[0.0, 300.0, 700.0], "Minor");

        let p4 = dis(&[0.0, 500.0], "P4");
        let tritone = dis(&[0.0, 600.0], "Tritone");
        let p5 = dis(&[0.0, 700.0], "P5");

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
        println!(" existing vs cand: {}", existing_vs_cand_diss_gap);

        println!(
            "\nBenchmark time (201 8-note iters): {} seconds",
            elapsed.as_secs_f64()
        );
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
            vec![0.0; cents.len()] // passing a 0 vector will make graph_dissonance initialize with dyadic tonicity heuristic.
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
                MAX_TREES,
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
                debug.print(cents, SHOW_TOP_N_CONTRIBUTING_TREES);
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
        let diss = graph_dissonance(
            &freqs,
            &[],
            &vec![0f64; cents.len()],
            0.9,
            1.0,
            MAX_TREES,
            None,
        )[0]
        .clone();
        println!("Starting context");
        println!("{:#?}\n", diss);

        println!("Obtained tonicity context {:?}.\n", diss.tonicity_context);

        let cands = graph_dissonance(
            &freqs,
            &candidate_freqs,
            &diss.tonicity_context,
            0.9,
            1.0,
            MAX_TREES,
            None,
        );

        for (idx, cand) in cands.iter().enumerate() {
            println!("\nCandidate {}c: {:#?}", candidate_cents[idx], cand);
        }
    }

    #[test]
    /// This tests the difference between having [a, b, c, d] as existing notes, and adding the new
    /// candidate [d] to [a, b, c].
    ///
    /// Since no debug data is available for candidate graph dissonance, we add a simulated version
    /// with candidate tonicity heuristic initialized to the same value as in candidate
    /// graph_dissonance.
    ///
    /// If the difference between the existing and candidate dissonance & tonicity context scores is
    /// too high, that could possibly indicate that [GLOBAL_TONICITY_LIKELIHOOD_SCALING] is too
    /// high, or [TONICITY_CONTEXT_TEMPERATURE_TARGET] is too low.
    fn test_single_candidate_graph_diss_same_treatment() {
        // the last note is the candidate
        let cents = [0.0, 700.0, 1400.0, 1600.0];
        let freqs = cents
            .iter()
            .map(|x| cents_to_hz(440.0, *x))
            .collect::<Vec<f64>>();

        let mut debug_existing = Some(GraphDissDebug::new(2, freqs.len()));

        let diss_existing = graph_dissonance(
            &freqs,
            &[],
            &vec![0f64; freqs.len()],
            0.9,
            0.1,
            MAX_TREES,
            debug_existing.as_mut(),
        )[0]
        .clone();

        let diss_candidate = graph_dissonance(
            &freqs[..(freqs.len() - 1)],
            &[*freqs.last().unwrap()],
            &vec![0f64; freqs.len() - 1],
            0.9,
            0.1,
            MAX_TREES,
            None, // NOTE: debug values are not implemented for candidate graph diss.
        )[0]
        .clone();

        let mut debug_candidate_simul = Some(GraphDissDebug::new(2, freqs.len()));

        let heur_dyadic_tonicity_without_cand = &dyadic_tonicity_heur(
            440.0,
            &freqs[..(freqs.len() - 1)],
            &[],
            HEURISTIC_DYAD_TONICITY_TEMP,
        )
        .tonicities_no_cand;

        let heur_tonicity_of_cand = *dyadic_tonicity_heur(
            440.0,
            &freqs[..(freqs.len() - 1)],
            &[*freqs.last().unwrap()],
            HEURISTIC_DYAD_TONICITY_TEMP,
        )
        .tonicities[0]
            .last()
            .unwrap();

        let heur_tonicities_with_cand_as_new_note =
            scale_candidate_toncity(heur_dyadic_tonicity_without_cand, heur_tonicity_of_cand);

        let diss_cand_simul = graph_dissonance(
            &freqs,
            &[],
            &heur_tonicities_with_cand_as_new_note,
            0.9,
            0.1,
            MAX_TREES,
            debug_candidate_simul.as_mut(),
        )[0]
        .clone();

        println!("\nExisting: {:#?}", diss_existing);
        debug_existing.unwrap().print(&cents, 10);

        println!("\nCandidate: {:#?}", diss_candidate);

        println!("\nCandidate simulated: {:#?}", diss_cand_simul);
        debug_candidate_simul.unwrap().print(&cents, 10);

        println!(
            "\nInit with simul heur tonicities: {:#?}",
            heur_tonicities_with_cand_as_new_note
        );

        println!(
            "Dissonance difference existing vs. candidate: {}",
            diss_existing.dissonance - diss_candidate.dissonance
        );

        println!(
            "Dissonance difference candidate vs. simulated cand.: {}\n",
            diss_candidate.dissonance - diss_cand_simul.dissonance
        );

        println!("  Tonicities existing: {:?}", diss_existing.tonicity_target);
        println!(
            "      Tonicities cand: {:?}",
            diss_candidate.tonicity_target
        );
        println!(
            "Tonicities cand simul: {:?}",
            diss_cand_simul.tonicity_target
        );
        println!(
            "\nCand tonicity - heur cand init tonicity (+ve is better): {}",
            diss_candidate.tonicity_target.last().unwrap()
                - heur_tonicities_with_cand_as_new_note.last().unwrap()
        );
    }

    /// This tests the expectation that having [a, b, c, d] as existing notes, and having one
    /// candidate [d] to add to [a, b, c] should both return the same results for [dyadic_tonicity_heur].
    #[test]
    fn test_single_candidate_tonicity_heur_same_treatment() {
        let cents = [0.0, 700.0, 1400.0, 1600.0];
        let freqs = cents
            .iter()
            .map(|x| cents_to_hz(261.63, *x))
            .collect::<Vec<f64>>();

        let heur_existing = dyadic_tonicity_heur(261.63, &freqs, &[], HEURISTIC_DYAD_TONICITY_TEMP);

        let heur_candidate = dyadic_tonicity_heur(
            261.63,
            &freqs[..(freqs.len() - 1)],
            &[*freqs.last().unwrap()],
            HEURISTIC_DYAD_TONICITY_TEMP,
        );

        println!("\nExisting: {:#?}", heur_existing.tonicities[0]);
        println!("\nCandidate: {:#?}", heur_candidate.tonicities[0]);
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

    /// Benchmark [graph_dissonance] with N notes, and 0 candidates.
    #[test]
    fn bench_n_notes() {
        const ITERS: usize = 1000;
        const N_NOTES: usize = 7;
        let cents =
            vec![0.0, 200.0, 400.0, 700.0, 900.0, 1400.0, 1600.0, 1800.0][..N_NOTES].to_vec();
        let freqs = cents
            .iter()
            .map(|x| cents_to_hz(130.812, *x))
            .collect::<Vec<f64>>();

        let mut tonicity_context = vec![0.0; cents.len()];
        let start_time = Instant::now();
        for iter in 0..ITERS {
            let res =
                &graph_dissonance(&freqs, &[], &tonicity_context, 0.9, 0.01, MAX_TREES, None)[0];
            tonicity_context.copy_from_slice(&res.tonicity_context);

            if iter % (ITERS / 20) == 0 {
                println!(
                    "Bench iter {}: diss: {}, ctx: {:?}",
                    iter, res.dissonance, res.tonicity_context
                );
            }
        }
        let elapsed = start_time.elapsed();
        println!(
            "Benchmark time ({} {}-note iters): {} seconds",
            ITERS,
            N_NOTES,
            elapsed.as_secs_f64()
        );
    }

    /// Benchmark [graph_dissonance] with N existing notes and 10 candidates, for ITERS iterations.
    ///
    /// IMPORTANT: Run test with --release flag.
    #[test]
    fn bench_candidate_n_notes() {
        const ITERS: usize = 500;
        const N_NOTES: usize = 7; // how many notes per iter including candidate
        const N_CANDIDATES: usize = 10;
        let cents = vec![0.0, 200.0, 400.0, 700.0, 900.0, 1400.0, 1600.0][..(N_NOTES - 1)].to_vec();
        let candidate_cents = vec![
            1500.0, 1550.0, 1650.0, 1700.0, 1750.0, 1800.0, 1900.0, 2000.0, 2100.0, 2170.0, 1486.0,
            1536.0, 1636.0, 1686.0, 1736.0, 1786.0, 1886.0, 1986.0, 2086.0, 2186.0,
        ][..N_CANDIDATES]
            .to_vec();
        let freqs = cents
            .iter()
            .map(|x| cents_to_hz(130.812, *x))
            .collect::<Vec<f64>>();
        let candidate_freqs = candidate_cents
            .iter()
            .map(|x| cents_to_hz(130.812, *x))
            .collect::<Vec<f64>>();

        bench_candidates(&freqs, &candidate_freqs, ITERS, 1000, MAX_TREES, true);
    }

    /// Benchmark [graph_dissonance] with N existing notes and M candidates, for ITERS iterations.
    ///
    /// Returns the last iterations Vec of Dissonance results for each candidate.
    fn bench_candidates(
        freqs: &[f64],
        candidate_freqs: &[f64],
        iters: usize,
        max_runtime_secs: u64,
        max_trees: usize,
        show_progress: bool,
    ) -> Vec<Dissonance> {
        let mut tonicity_context = vec![0.0; freqs.len() + 1];

        let mut last_iter_diss = vec![];

        let start_time = Instant::now();

        let mut last_iter = 0;

        for iter in 0..iters {
            last_iter_diss = graph_dissonance(
                &freqs,
                &candidate_freqs,
                &tonicity_context[..freqs.len()],
                0.9,
                0.01,
                max_trees,
                None,
            );

            let max_tonicity_cand_idx = last_iter_diss
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1.tonicity_context
                        .last()
                        .unwrap()
                        .partial_cmp(b.1.tonicity_context.last().unwrap())
                        .unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap();

            tonicity_context
                .copy_from_slice(&last_iter_diss[max_tonicity_cand_idx].tonicity_context);

            if iter % (iters / 20) == 0 && show_progress {
                println!(
                    "Bench iter {}: max tonicity cands idx: {}, tonicities: {:?}",
                    iter, max_tonicity_cand_idx, tonicity_context
                );
            }

            last_iter = iter;

            let elapsed = start_time.elapsed().as_secs();

            if elapsed >= max_runtime_secs {
                break;
            }
        }
        let elapsed = start_time.elapsed();

        println!(
            "Benchmark time ({} {}-note {}-candidate iters): {} seconds. ({} iter/sec)",
            last_iter + 1,
            freqs.len(),
            candidate_freqs.len(),
            elapsed.as_secs_f64(),
            (last_iter + 1) as f64 / elapsed.as_secs_f64(),
        );

        last_iter_diss
    }

    /// Compares the scores for various MAX_TREES settings for candidate selection & resulting tonicity.
    #[test]
    fn bench_max_trees_deterioration() {
        let max_trees_vals = vec![usize::MAX, 20000, 10000, 5000, 2500, 1000, 500, 250, 100];

        const N_NOTES: usize = 7; // how many notes per iter including candidate
        const N_CANDIDATES: usize = 10;

        const RUN_TIME_PER_MAX_TREES: u64 = 10; // how many seconds to run per MAX_TREES setting

        const ITERS: usize = 500;

        let cents = vec![0.0, 200.0, 400.0, 700.0, 900.0, 1400.0, 1600.0][..(N_NOTES - 1)].to_vec();
        let candidate_cents = vec![
            1500.0, 1550.0, 1650.0, 1700.0, 1750.0, 1800.0, 1900.0, 2000.0, 2100.0, 2170.0, 1486.0,
            1536.0, 1636.0, 1686.0, 1736.0, 1786.0, 1886.0, 1986.0, 2086.0, 2186.0,
        ][..N_CANDIDATES]
            .to_vec();
        let freqs = cents
            .iter()
            .map(|x| cents_to_hz(130.812, *x))
            .collect::<Vec<f64>>();
        let candidate_freqs = candidate_cents
            .iter()
            .map(|x| cents_to_hz(130.812, *x))
            .collect::<Vec<f64>>();

        for &max_trees in &max_trees_vals {
            println!("\n\n=== MAX_TREES = {} ===", max_trees);
            let last_diss = bench_candidates(
                &freqs,
                &candidate_freqs,
                ITERS,
                RUN_TIME_PER_MAX_TREES,
                max_trees,
                false,
            );

            let max_tonicity_cand_idx = last_diss
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1.tonicity_context
                        .last()
                        .unwrap()
                        .partial_cmp(b.1.tonicity_context.last().unwrap())
                        .unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap();

            let max_tonicity_cand = &last_diss[max_tonicity_cand_idx];

            println!(
                "Max tonicity candidate: {} ({:.2}c). dissonance: {}, tonicity: {:?}",
                max_tonicity_cand_idx,
                candidate_cents[max_tonicity_cand_idx],
                max_tonicity_cand.dissonance,
                max_tonicity_cand.tonicity_context,
            );

            let tonicities_idx_highest_to_lowest = {
                let mut idxs: Vec<usize> = (0..max_tonicity_cand.tonicity_context.len()).collect();
                idxs.sort_by(|&a, &b| {
                    max_tonicity_cand
                        .tonicity_context[b]
                        .partial_cmp(&max_tonicity_cand.tonicity_context[a])
                        .unwrap()
                });
                idxs
            };

            println!("Tonicity ranking of candidates: {:?}", tonicities_idx_highest_to_lowest);
        }
    }

    /// Test the distribution of dissonance scores for N = 2, ..., 8 notes randomly distributed over C3 - C6.
    #[test]
    fn test_diss_distribution() {

        use compute::statistics::*;
        use histo::Histogram;

        fn gen_rand_cents(min: f64, max: f64) -> f64 {
            let rand_unif = fastrand::f64();

            rand_unif * (max - min) + min
        }

        for n in 2..=8 {
            let mut diss_scores = vec![];
            let mut histogram = Histogram::with_buckets(10);
            for _ in 0..1000 {
                let cents: Vec<f64> = (0..n)
                    .map(|_| gen_rand_cents(0.0, 3600.0))
                    .collect();
                let freqs = cents
                    .iter()
                    .map(|x| cents_to_hz(261.63 / 2.0, *x))
                    .collect::<Vec<f64>>();
                let diss = graph_dissonance(
                    &freqs,
                    &[],
                    &vec![0f64; cents.len()],
                    0.9,
                    0.1,
                    800,
                    None,
                )[0]
                .clone();
                diss_scores.push(diss.dissonance);
                histogram.add((diss.dissonance * 10000.0) as u64);
            }

            println!("Dissonance stats for {n} random notes in C3 - C6:");
            println!("   Min: {}", min(&diss_scores));
            println!("   Max: {}", max(&diss_scores));
            println!("  Mean: {}", mean(&diss_scores));
            println!("   Std: {}", std(&diss_scores));
            println!("{}", histogram);
        }
    }
}
