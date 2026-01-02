//! Polyadic dissonance & tonicity of notes.

use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
};

use binary_heap_plus::BinaryHeap;

// ⚠️ NOTE: This requires get-size2 crate.
use get_size2::GetSize;
use rand::random;
use termtree::Tree;

use crate::{
    dyad_lookup::{DyadLookup, RoughnessType, TonicityLookup},
    utils::hz_to_cents,
};

const PRINT_GRAPH: bool = false;

/// The tonicity vector is a list of heuristic probabilities of each note being perceived as tonic
/// relative to the entire chord.
///
/// This should sum to 1.
pub type Tonicities = Vec<f64>;

#[derive(Clone, Copy, GetSize)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    /// The dyadic roughness of the dyad. Scaled to fit within [0, 1].
    pub roughness: f64,

    /// Relative to the other edges to unvisited nodes (w.r.t. state of the traversal) that could
    /// have been chosen.
    ///
    /// This is the reciprocal of the roughness of this dyad over by the sum of reciprocals of all
    /// options.
    ///
    /// Any further contributions to dissonance by edges traversed AFTER this one should be
    /// multiplied by this (but not the contribution of the edge itself).
    ///
    /// If only one edge can be chosen (which means one unvisited node left), this should be 1,
    /// although, that doesn't matter as there are no further computations.
    ///
    /// The sum of all edge tonicities from one visited node to all unvisited nodes for a traversal
    /// state should be 1.
    pub edge_tonicity: f64,

    /// The absolute edge tonicity. `edge_tonicity` multiplied by `parent_edge_tonicity_abs`,
    /// represents the absolute edge_tonicity effect applied to edges after this one.
    pub edge_tonicity_abs: f64,

    /// The edge_tonicity of the most recent edge that was added to `from`, before this one.
    ///
    /// Adding the same edge to the exact same graph state can give two different results depending
    /// on the order of traversal.
    ///
    /// If this was the first edge added (from the root node), this should be 1.
    pub parent_edge_tonicity: f64,

    /// Absolute edge tonicity multiplier for final dissonance calculation. Equal to the parent's
    /// (`from`) last added edge's `edge_tonicity_abs`.
    pub parent_edge_tonicity_abs: f64,

    /// The relative single note tonicity of `from`, at the time before traversal of this new edge,
    /// calculated recursively from dyadic tonicities of the traversed spanning tree so far.
    ///
    /// This value must be normalized at the end of the traversal by dividing by the sum of all
    /// `parent_rel_dyadic_tonicity` of all nodes.
    ///
    /// The contribution to dissonance by this edge at the local level is `roughness *
    /// dyad_tonicity` or something similar.
    ///
    /// For more info, see [Traversal.note_tonicities].
    pub parent_rel_dyadic_tonicity: f64,

    /// The raw pairwise dyadic tonicity of `to` being perceived as the tonic over `from`. This
    /// value is directly from the [TonicityLookup].
    pub pairwise_dyadic_tonicity_to: f64,
}

impl Edge {
    /// The relative edge contribution to dissonance without edge_tonicity context, only using
    /// pairwise dyadic tonicity.
    ///
    /// `rel_dyadic_tonicity_sum`: The sum of all relative dyadic note tonicities values in the
    /// graph, which is required to normalize the relative dyadic tonicity into an absolute dyadic
    /// tonicity that sums to 1.
    ///
    /// `num_visited`: number of notes visited so far in this traversal.
    pub fn edge_contribution_relative(
        &self,
        rel_dyadic_tonicity_sum: f64,
        num_visited: usize,
    ) -> f64 {
        // TODO: try other functions.
        let abs_dyadic_tonicity = self.parent_rel_dyadic_tonicity / rel_dyadic_tonicity_sum;

        // here roughness is within [0, 1], which means that the raising it to a higher exponent
        // actually lowers it.

        let roughness_impact = abs_dyadic_tonicity * num_visited as f64;
        self.roughness.powf(1.0 / roughness_impact)
    }

    /// The absolute edge contribution to dissonance with context.
    ///
    /// This value scales with the parent edge tonicity, which models the likelihood of choosing to
    /// perceive a particular edge (interval) first, over other edges (intervals) to unvisited nodes
    /// (notes that haven't been 'analyzed' yet).
    ///
    /// ### Parameters
    ///
    /// `rel_dyadic_tonicity_sum`: The sum of all relative dyadic note tonicities values in the
    /// graph, which is required to normalize the relative dyadic tonicity into an absolute dyadic
    /// tonicity that sums to 1.
    ///
    /// `num_visited`: The number of notes visited so far in this traversal, which is required to
    /// obtain the expected relative dyadic tonicity.
    ///
    /// `parent_edge_tonicity_sum`: The sum of `parent_edge_tonicity_abs` for all edges in the
    /// traversal (plus 1)
    pub fn edge_contribution(
        &self,
        rel_dyadic_tonicity_sum: f64,
        num_visited: usize,
        parent_edge_tonicity_sum: f64,
    ) -> f64 {
        // This function MUST NOT return NaN or Inf.
        self.edge_contribution_relative(rel_dyadic_tonicity_sum, num_visited)
            * self.parent_edge_tonicity_abs
            / parent_edge_tonicity_sum
    }

    /// The tonicity score only considering dyadic tonicity. The result is normalized from 0 to 1.
    pub fn tonicity_score_dyadic(&self, rel_dyadic_tonicity_sum: f64) -> f64 {
        self.parent_rel_dyadic_tonicity / rel_dyadic_tonicity_sum
    }

    /// Tonicity score of dyadic tonicity + edge tonicity.
    pub fn tonicity_score(
        &self,
        rel_dyadic_tonicity_sum: f64,
        parent_edge_tonicity_sum: f64,
    ) -> f64 {
        self.parent_edge_tonicity_abs / parent_edge_tonicity_sum * self.parent_rel_dyadic_tonicity
            / rel_dyadic_tonicity_sum
    }
}

/// Represents a spanning tree traversal of the fully connected chord graph.
///
/// Traversals intend to model one particular way the listener chooses tonal organization (i.e., a
/// spanning tree), such that by exhausting all traversals, we may aggregate interpretations of
/// chords where certain notes/dyads or substructures take precedence over others, and the
/// 'logicalness' of the traversal scales the probability of that interpretation.
#[derive(Clone)]
pub struct Traversal {
    /// Index of freqs started.
    ///
    /// If start == freqs.len(), then this traversal starts with an `candidate_freqs` note.
    pub start: usize,

    /// Which candidate note in `candidate_freqs` is being considered.
    ///
    /// This must be set to 0 if there are no candidate frequencies.
    pub candidate_idx: usize,

    /// Bitmask of which vertices haven't been visited.
    ///
    /// Once this is 0, the traversal is complete.
    pub remaining_verts: u32,

    /// The most recent edge that is incident to a node.
    ///
    /// Keys correspond to both `from` and `to`. Upon a new edge incident along an existing node
    /// key, the edge is overwritten.
    pub last_added_edge: HashMap<usize, Edge>,

    /// Directed edges that were traversed.
    ///
    /// The keys correspond to the `from` index of the edges, and the values correspond to edges
    /// pointing to their children.
    ///
    /// There can be multiple nodes coming from the same `from` index (but no two edges can share
    /// the same `to`).
    pub edges_children: HashMap<usize, Vec<Edge>>,

    /// Directed edges that were traversed.
    ///
    /// The keys correspond to `to` index of an edge, and the values correspond to the unique edge
    /// that points to their parent.
    ///
    /// Each node can only have one parent (since this is a spanning tree).
    ///
    /// The `start` node has no parent.
    pub edges_parents: HashMap<usize, Edge>,

    /// The relative dyadic note tonicities of all nodes in the graph. Index corresponds to the
    /// index of the note. If candidate notes are given, the last index corresponds to the candidate
    /// note.
    ///
    /// To clarify, the key is the index of the `to` node of an edge, and it gives the relative
    /// tonicity score of that node relative to the `from` node (i.e., how likely is for `from` to
    /// be perceived as tonic over `to`). Since each node only has one parent, this is a unique
    /// value.
    ///
    /// To obtain the absolute value, divide by `note_tonicities_sum`.
    ///
    /// METHOD:
    ///
    /// There's many notions of 'single note tonicity', including the simple summative, nonrecursive
    /// dyadic tonicity heuristic used at the start, but the issue with that heuristic is that it is
    /// agnostic of the order of perception. We want traversals to model the prioritazation of
    /// earlier structures nearer to the 'perceived tonic', so in this case, the tonicity of the
    /// root of the spanning tree is 1, and the tonicity all children is recursively evaluated as
    /// its dyadic tonicity multiplied by the parent_rel_dyadic_tonicity of its parent.
    ///
    /// E.g., assume our traversal goes X->Y->Z, and that the tonicity of X with respect to Y is
    /// T(X, Y) = 0.7 and the tonicity of Y with respect to Z is T(Y, Z) = 0.4.
    ///
    /// I.e., with respect to dyads only, X compared to {X, Y} is 70% tonic, and Y compared to {X,
    /// Y} is 30% tonic.
    ///
    /// Also, Y compared to {Y, Z} is 40% tonic, and Z compared to {Y, Z} is 60% tonic.
    ///
    /// We model this with X starting with relative tonicity 1. The relative tonicity of X:Y should
    /// also have the same 70:30 ratio, so the relative tonicity of Y is (30/70) * 1 = 0.4285...
    ///
    /// The relative tonicity of Y:Z should also have the same 40:60 ratio, so the relative tonicity
    /// of Z is (60/40) * 0.4285... = 0.643...
    ///
    /// Although, since there are no more child nodes after Z, the relative tonicity of Z is not
    /// directly used in any computation except to contribute to the final dyadic_tonicity_sum.
    ///
    /// Now we obtain the absolute tonicity by dividing by the sum  (1 + 0.4285 + 0.643) which is
    /// represented by the `rel_dyadic_tonicity_sum` parameter.
    pub note_tonicities: Vec<f64>,

    /// Sum of all relative dyadic note tonicities values in the graph.
    pub note_tonicities_sum: f64,

    /// Sum of `parent_edge_tonicity_abs` for all edges in the traversal.
    pub parent_edge_tonicities_sum: f64,
}

/// The result of a dissonance calculation.
#[derive(Debug, Clone)]
pub struct Dissonance {
    /// dissonance score
    pub dissonance: f64,

    /// the target tonicity of the notes in the chord.
    pub tonicity_target: Tonicities,

    /// The new tonicity context applying smoothing towards the target.
    pub tonicity_context: Tonicities,

    pub iterations: usize,

    /// `num_traversals` counts the number of traversals excluding completed traversals.
    ///
    /// This counts the number of ways to traverse spanning trees of subgraphs of the chord graph.
    ///
    /// Relative to the metatree of graph traversals, this is the number of internal nodes.
    pub num_traversals: usize,

    /// `num_completed_traversals` counts the number of full traversals of all spanning trees of the
    /// chord graph.
    ///
    /// If no pruning is done, this value should be N! * (N-1)! * M, where N is the number of notes
    /// in the chord (including the candidate note, if any)
    ///
    /// Relative to the metatree of graph traversals, this is the number of leaf nodes.
    pub num_completed_traversals: usize,

    /// This is the number of traversals left incomplete at the last iteration.
    pub num_incomplete_traversals: usize,

    pub num_completed_traversals_per_note: Vec<usize>,
    pub avg_diss_per_traversal_per_note: Vec<f64>,

    // TOOD: decide which of the following fields below are not part of the main dissonance &
    // tonicity calculation and move them to an optional info struct. Don't calculate them unless
    // explicitly required.
    /// Entropy (bits) of the distribution of roughness contributed by each traversal, calculated
    /// for each note.
    pub diss_entropy_per_note: Vec<f64>,

    /// Entropy (bits) of the distribution of tonicity scores contributed by each traversal,
    /// calculated for each note.
    pub tonicity_entropy_per_note: Vec<f64>,

    /// Entropy (bits) of the distribution of roughness scores between notes after tabulating
    /// all traversals.
    pub entropy_between_notes_pre_tonicity: f64,

    /// Entropy (bits) of the distribution of target tonicity scores of each note (higher = less certain of the tonic)
    pub note_tonicities_target_entropy: f64,

    /// Entropy (bits) of the distribution of tonicity scores in the returned updated tonicity
    /// context (as a model of the currently perceived tonicity)
    pub tonicity_context_entropy: f64,

    /// Tonicities calculated using only dyadic tonicities + traversal order, without considering
    /// edge tonicity. Analogous to `tonicity_target`.
    pub tonicities_dyadic_only: Tonicities,

    /// Dissonances contributed by each note, influenced only by dyadic tonicities and traversal
    /// order considering edge tonicity. Analogous to `avg_diss_per_traversal_per_note`.
    pub diss_per_note_dyadic_only: Vec<f64>,
}

/// Pruning method for reducing computations on graph traversal.
#[derive(Clone, Copy)]
pub enum PruneMethod {
    /// Prune away edges with high dissonance contributions.
    PrioritizeMinDiss,
    /// Prune away edges that contribute the least tonicity.
    PrioritizeMaxTonicity,
    /// Prune randomly
    Random,
}

impl Display for PruneMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PruneMethod::PrioritizeMinDiss => write!(f, "PrioritizeMinDiss"),
            PruneMethod::PrioritizeMaxTonicity => write!(f, "PrioritizeMaxTonicity"),
            PruneMethod::Random => write!(f, "Random"),
        }
    }
}

/// Polyadic dissonance using a graph.
///
/// The maximum expected running time is N! * (N-1!) * M where N is the number of `freqs` and M is
/// the number of `candidate_freqs`. However, this can be reduced by setting lower pruning
/// parameters.
///
/// ### Parameters
///
/// `freqs` is a list of frequencies of the notes in the chord.
///
/// `candidate_freqs` is a list of possible frequency interpretations of a note that will be added
/// to the chord (by detempering). If empty, only `freqs` will be computed. However, if provided,
/// dissonances will be calculated for each choice of frequency in `candidate_freqs`.
///
/// `tonicity_context` should correspond to existing tonicity scores of corresponding freqs.
///
/// If `tonicity_context` will be normalized to sum to 1.
///
/// If a new loud note is played, the tonicities can be jerked by setting smoothing lower or
/// elapsed_seconds higher than usual. Rhythmic entrainment can be implemented by setting smoothing
/// lower at regular time intervals and higher at others.
///
/// `max_attempts_per_trv` is the maximum number of note-to-note edge candidates to consider for
/// each existing traversal record in one iteration. If the maximum is reached, only the least
/// dissonant edges will be considered. This will only have an effect if this value is set lower
/// than the number of notes in the chord (including the candidate note).
///
/// `max_attempts_per_iteration` is the maximum number of note-to-note edge candidates to consider
/// between all existing traversal records in one iteration.
///
/// `target_num_traversals` is the number of completed traversals before stopping early. All
/// starting notes should admit the same number of completed traversals, so if this number doesn't
/// divide the number of notes in the chords evenly, then this number will be rounded up. If there
/// are multiple candidate notes in `candidate_freqs`, the actual number of completed traversals
/// will be multiplied by the number of candidates. Changing this value does not change the number
/// of intermediate traversals, so it may not have a large effect on the final result.
///
/// If all of the above computation limits are set arbitrarily high, one can expect N! * (N-1)! * M
/// unique spanning tree traversals of the graph.
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
    max_attempts_per_trv: usize,
    max_attempts_per_iteration: usize,
    target_num_traversals: usize,
    prune_method: PruneMethod,
) -> Vec<Dissonance> {
    assert!(
        tonicity_context.len() == freqs.len(),
        "tonicity_context must correspond to notes in freqs"
    );

    let has_candidate_notes = candidate_freqs.len() != 0;
    let num_candidates = candidate_freqs.len().max(1); // if no candidates, we can treat last note as the sole candidate.
    let num_notes = freqs.len() + if has_candidate_notes { 1 } else { 0 };

    if num_notes < 2 {
        return vec![Dissonance {
            dissonance: 1.0,
            tonicity_target: tonicity_context.to_vec(),
            tonicity_context: tonicity_context.to_vec(),
            iterations: 0,
            num_traversals: 0,
            num_completed_traversals: 0,
            num_incomplete_traversals: 0,
            diss_entropy_per_note: vec![0.0; num_notes],
            tonicity_entropy_per_note: vec![0.0; num_notes],
            entropy_between_notes_pre_tonicity: 0.0,
            note_tonicities_target_entropy: 0.0,
            tonicity_context_entropy: 0.0,
            num_completed_traversals_per_note: vec![0; num_notes],
            avg_diss_per_traversal_per_note: vec![0.0; num_notes],
            tonicities_dyadic_only: vec![0.0; num_notes],
            diss_per_note_dyadic_only: vec![0.0; num_notes],
        }];
    }

    let cents: Vec<f64> = freqs.iter().map(|x| hz_to_cents(440.0, *x)).collect();

    let candidate_cents: Vec<f64> = candidate_freqs
        .iter()
        .map(|x: &f64| hz_to_cents(440.0, *x))
        .collect();

    // normalize tonicity context to sum to 1.
    let tonicity_context_sum: f64 = tonicity_context.iter().sum();
    let tonicity_context: Tonicities = if tonicity_context_sum > 0.0 {
        // we may need to add one more slot for the candidate note if provided.
        let mut c: Vec<f64> = tonicity_context
            .iter()
            .map(|x| {
                x / tonicity_context_sum
                    * if has_candidate_notes {
                        1.0 - 1.0 / num_notes as f64
                    } else {
                        1.0
                    }
            })
            .collect();

        if has_candidate_notes {
            c.push(1.0 / num_notes as f64);
        }

        c
    } else {
        // if the context is all zeroes, we assume equal tonicity.
        vec![1.0 / num_notes as f64; num_notes]
    };

    // Total number of completed traversals between all candidates and starting notes.
    let mut num_completed_traversals = 0;

    // Keeps track of how many completed traversals for each starting note. Once a starting note has reached
    // target_completed_traversals_per_start, no more computations will be done for that starting note.
    let mut num_completed_traversals_per_cand_per_note = vec![vec![0; num_notes]; num_candidates];

    // The target number of completed traversals for each choice of candidate note and starting note.
    let target_completed_traversals =
        (target_num_traversals as f64 / num_notes as f64).ceil() as usize;

    let mut completed_traversals_per_candidate: Vec<Vec<Traversal>> = vec![vec![]; num_candidates];
    let mut traversals = vec![];

    // Create lookup table for all dyads.
    let heuristic_tonicities = dyadic_tonicity_heur(&cents, &candidate_cents);
    let dyad_tonics = heuristic_tonicities.tonicity_map;
    let dyad_roughs = heuristic_tonicities.add_roughness_map; // use additive roughness.
    let cand_tonicity_map = heuristic_tonicities.cand_tonicity_map;
    let cand_roughness_map = heuristic_tonicities.cand_add_roughness_map;

    // The dissonance contribution per perceived tonic is scaled by this. If smoothing is high (>
    // 0.9) or elapsed seconds is low (< 0.1), this heavily favours the existing context, rather
    // than the target tonicity.
    //
    // Each Tonicities in this list corresponds to the choice of a candidate note in `candidate_freqs`.
    let smoothed_tonicity_contexts: Vec<Tonicities> = heuristic_tonicities
        .tonicities
        .iter()
        .map(|tonicities: &Tonicities| {
            tonicity_smoothing(&tonicities, &tonicity_context, smoothing, elapsed_seconds)
        })
        .collect();

    let full_bitmask = (1 << num_notes) - 1; // bitmask of N ones, where N is the number of frequencies.

    for candidate in 0..num_candidates {
        for start_note_idx in 0..num_notes {
            // The initial dissonance multiplier is the heuristic tonicity of the note.
            // The stronger the tonicity of a note, the stronger its dissonances branching out from that note
            // are perceived.
            let mut from_note_tonicities = vec![0.0; num_notes];
            from_note_tonicities.insert(start_note_idx, 1.0);
            traversals.push(Traversal {
                start: start_note_idx,
                candidate_idx: candidate,
                remaining_verts: full_bitmask & !(1 << start_note_idx),
                last_added_edge: HashMap::new(),
                edges_children: HashMap::new(),
                edges_parents: HashMap::new(),
                note_tonicities: from_note_tonicities,
                note_tonicities_sum: 1.0,
                parent_edge_tonicities_sum: 0.0,
            })
        }
    }

    // -------------------------------- ITERATING GRAPH TRAVERSALS ---------------------------------------

    // The iterations also count how many nodes has been visited. Iteration 1 denotes one node
    // already visited (starting node) and the iteration will look to traverse the second node.
    let mut iterations = 0;
    let mut num_traversals = 0;

    loop {
        iterations += 1;
        print!("Iteration {iterations}: ");
        // Contains traversals for the next iteration to process.
        //
        // The f64 in the tuple is a key such that higher values denotes the traversals to prune
        // first.
        let mut new_traversals =
            BinaryHeap::new_by(|a: &(Traversal, f64), b| a.1.partial_cmp(&b.1).unwrap());

        let mut continue_loop = false;

        for t in traversals.iter_mut() {
            // 1. First we find all valid edges to add to this current traversal t,
            // stored in edge_candidates.

            if num_completed_traversals_per_cand_per_note[t.candidate_idx][t.start]
                >= target_completed_traversals
            {
                continue;
            }

            num_traversals += 1;

            // Heap comparators should be such that candidates.into_sorted_vec() will put the
            // edges to prune at the end of the vec.
            let mut edge_candidates =
                match prune_method {
                    PruneMethod::PrioritizeMinDiss => {
                        // puts min diss at start of the vec
                        BinaryHeap::new_by(Box::new(|a: &Edge, b: &Edge| {
                            a.edge_contribution(
                                t.note_tonicities_sum,
                                iterations,
                                t.parent_edge_tonicities_sum,
                            )
                            .partial_cmp(&b.edge_contribution(
                                t.note_tonicities_sum,
                                iterations,
                                t.parent_edge_tonicities_sum,
                            ))
                            .unwrap()
                        })
                            as Box<dyn Fn(&Edge, &Edge) -> std::cmp::Ordering>)
                    }
                    PruneMethod::PrioritizeMaxTonicity => {
                        // puts max tonicity at start of vec.
                        BinaryHeap::new_by(Box::new(|a: &Edge, b: &Edge| {
                            b.tonicity_score(t.note_tonicities_sum, t.parent_edge_tonicities_sum)
                                .partial_cmp(&a.tonicity_score(
                                    t.note_tonicities_sum,
                                    t.parent_edge_tonicities_sum,
                                ))
                                .unwrap()
                        })
                            as Box<dyn Fn(&Edge, &Edge) -> std::cmp::Ordering>)
                    }
                    PruneMethod::Random => BinaryHeap::new_by(Box::new(|_: &Edge, _: &Edge| {
                        random::<u32>().cmp(&random())
                    })
                        as Box<dyn Fn(&Edge, &Edge) -> std::cmp::Ordering>),
                };
            for from_idx in 0..num_notes {
                if (t.remaining_verts & (1 << from_idx)) != 0 {
                    // must come from a vertex that is already visited.
                    continue;
                }
                for to_idx in 0..num_notes {
                    if from_idx == to_idx {
                        continue;
                    }
                    if (t.remaining_verts & (1 << to_idx)) == 0 {
                        // must go to a vertex that is not yet visited.
                        continue;
                    }
                    // assert!(
                    //     t.edges_parents.get(&from_idx).is_some()
                    //         || from_idx == t.start
                    //         || t.last_added_edge.len() == 0
                    // );
                    // assert!(t.edges_parents.get(&to_idx).is_none());
                    let (dyad_tonicity, roughness) = {
                        if from_idx == freqs.len() {
                            // From new candidate note to to_idx
                            (
                                cand_tonicity_map[to_idx][t.candidate_idx],
                                cand_roughness_map[to_idx][t.candidate_idx],
                            )
                        } else if to_idx == freqs.len() {
                            // From from_idx to new candidate note
                            (
                                cand_tonicity_map[from_idx][t.candidate_idx],
                                cand_roughness_map[from_idx][t.candidate_idx],
                            )
                        } else {
                            (
                                *dyad_tonics.get(&(1 << from_idx | 1 << to_idx)).unwrap(),
                                *dyad_roughs.get(&(1 << from_idx | 1 << to_idx)).unwrap(),
                            )
                        }
                    };

                    // The tonicity is always w.r.t. lower index being tonic, so if we're going
                    // from a lower index to a higher index, we need to invert the dyad tonicity. We
                    // want the probability of `to` being tonic, not `from`.
                    let pairwise_dyadic_tonicity_to = if to_idx < from_idx {
                        dyad_tonicity
                    } else {
                        1.0 - dyad_tonicity
                    };

                    let (parent_edge_tonicity, parent_edge_tonicity_abs) = t
                        .last_added_edge
                        .get(&from_idx)
                        .map(|x| (x.edge_tonicity, x.edge_tonicity_abs))
                        .unwrap_or((1.0, 1.0));

                    edge_candidates.push(Edge {
                        from: from_idx,
                        to: to_idx,
                        roughness: roughness,
                        // to be assigned after all obtaining all candidates coming from the same
                        // node
                        edge_tonicity: 0.0,
                        edge_tonicity_abs: 0.0, // calculated later
                        parent_edge_tonicity,
                        parent_edge_tonicity_abs,
                        parent_rel_dyadic_tonicity: t.note_tonicities[from_idx],
                        pairwise_dyadic_tonicity_to,
                    });
                }
            }

            // 2. Prune edges if there are too many edge candidates (for latency reasons)

            // The current strategy is to keep edges with minimum edge_contribution, but that
            // skews the results because the more 'tonic' nodes will have more traversals.
            //
            // Other strategies to try include:
            //
            // - Random pruning.
            //
            // - Prune edges with minimum and maximum edge contribution first.
            //
            // - Prune edges with average edge contribution first.
            //
            // - train reinforcement learning model to choose which edges to prune, using
            //   full-traversal output as loss function.
            //

            // TODO: If we can get this pruning method to work, we store candidates in a max-heap
            // so we don't have to sort the whole thing.

            let mut edge_candidates = if edge_candidates.len() > max_attempts_per_trv {
                let mut c = if let PruneMethod::Random = prune_method {
                    edge_candidates.into_vec()
                } else {
                    edge_candidates.into_sorted_vec()
                };
                c.truncate(max_attempts_per_trv);
                c
            } else {
                edge_candidates.into_vec()
            };

            // 3. For each edge candidate, create a new traversal by adding that edge to the
            //    current traversal t (factorial growth if unpruned).

            let sum_edge_tonicities: f64 = edge_candidates.iter().map(|x| 1.0 / x.roughness).sum();

            for c in edge_candidates.iter_mut() {
                c.edge_tonicity = (1.0 / c.roughness) / sum_edge_tonicities;
                c.edge_tonicity_abs = c.edge_tonicity * c.parent_edge_tonicity_abs;

                let mut new_traversal = t.clone();
                new_traversal.remaining_verts &= !(1 << c.to);
                new_traversal
                    .edges_children
                    .entry(c.from)
                    .or_insert_with(Vec::new)
                    .push(c.clone());

                new_traversal.edges_parents.insert(c.to, c.clone());
                new_traversal.last_added_edge.insert(c.from, c.clone());
                new_traversal.last_added_edge.insert(c.to, c.clone());

                // See [Traversal.note_tonicities] for more info.
                let rel_dyadic_tonicity_to = c.parent_rel_dyadic_tonicity
                    * c.pairwise_dyadic_tonicity_to
                    / (1.0 - c.pairwise_dyadic_tonicity_to);

                new_traversal.note_tonicities[c.to] = rel_dyadic_tonicity_to;
                new_traversal.note_tonicities_sum += rel_dyadic_tonicity_to;

                new_traversal.parent_edge_tonicities_sum += c.parent_edge_tonicity_abs;

                if new_traversal.remaining_verts == 0 {
                    completed_traversals_per_candidate[new_traversal.candidate_idx]
                        .push(new_traversal.clone());
                    num_completed_traversals_per_cand_per_note[new_traversal.candidate_idx]
                        [new_traversal.start] += 1;
                    num_completed_traversals += 1;

                    // TODO: remove after debug

                    if PRINT_GRAPH {
                        println!("Completed traversal:");
                        print_tree(&new_traversal);
                    }
                } else {
                    let from_note_tonicities_sum = new_traversal.note_tonicities_sum;
                    let parent_edge_tonicities_sum = new_traversal.parent_edge_tonicities_sum;

                    // The priority key has a higher value for traversals to be pruned first.
                    let priority_key = match prune_method {
                        PruneMethod::PrioritizeMinDiss => c.edge_contribution(
                            from_note_tonicities_sum,
                            iterations + 1,
                            parent_edge_tonicities_sum,
                        ),
                        PruneMethod::PrioritizeMaxTonicity => {
                            // negate tonicity score since we want max tonicity to be pruned last.
                            -c.tonicity_score(from_note_tonicities_sum, parent_edge_tonicities_sum)
                        }
                        PruneMethod::Random => random::<f64>(),
                    };
                    new_traversals.push((new_traversal, priority_key));

                    // A new traversal with unvisited nodes was added, reason to continue.
                    continue_loop = true;
                }
            }
        }

        let mut new_traversals = if let PruneMethod::Random = prune_method {
            new_traversals.into_vec()
        } else {
            new_traversals.into_sorted_vec()
        };

        if new_traversals.len() > max_attempts_per_iteration {
            // If too many traversal options in this iteration, prune them.
            new_traversals.truncate(max_attempts_per_iteration);
        }

        // replace old traversals with new ones.
        traversals = new_traversals.iter().map(|(x, _)| x.clone()).collect();

        println!(
            "{} traversals so far, {} completed, {} next",
            num_traversals,
            num_completed_traversals,
            traversals.len()
        );

        if !continue_loop || traversals.len() == 0 {
            break;
        }
    }

    // --------------------------------- CALCULATING DISSONANCE ----------------------------------------

    let mut results = vec![];

    // Normalize final score by dividing by 2 - 0.5^(N - 1)
    // This only applies to the current multiplicative edge_contribution function.
    // let normalizer = 1.0 / (2.0 - 0.5f64.powi(num_notes as i32 - 1));

    for cand_idx in 0..num_candidates {
        // TODO: figure out how we want to normalize dissonances between different number of notes.
        // Currently, the model returns a number between 0 and 1 corresponding to the least to most
        // dissonant chord possible relative to all other chords with the same number of notes.
        //
        // We can try a different way of normalizing using a stack of octaves or a stack of fifths
        // (i.e., dissonance should maintain constant if we keep repeatedly adding octaves/fifths)
        // which results in a more human output (although the dissonance scores are relatively the
        // same)
        let normalizer = 1.0;

        // Entropy of the distribution of roughness over traversals, calculated for each starting
        // note, summing over roughness -r log2(r). To normalize, divide by sum of
        // r and add
        //
        //   log2(sum of r).
        //
        // Sum of r = diss_per_note before normalization.
        let mut diss_entropy_per_note = vec![0.0; num_notes];

        // Sum over all traversals of dissonance scores per starting note
        let mut diss_per_note = vec![0.0; num_notes];
        let mut num_completed_traversals_per_note = vec![0; num_notes];

        // Sum over all traversals of relative tonicity scores per note.
        let mut tonicity_scores_per_note: Tonicities = vec![0.0; num_notes];

        // Entropy of the distribution of tonicity scores of each note over traversals.
        let mut tonicity_entropy_per_note = vec![0.0; num_notes];

        // Tonicity scores calculated using only dyadic tonicities + traversal order, without
        // considering edge tonicity.
        let mut ton_scores_dyadic_only: Tonicities = vec![0.0; num_notes];

        // Dissonance scores calculated using only dyadic tonicities + traversal order, without
        // considering edge tonicity.
        let mut diss_per_note_dyadic_only: Vec<f64> = vec![0.0; num_notes];

        for t in completed_traversals_per_candidate[cand_idx].iter() {
            let mut diss = 0.0;
            let mut tonicities = vec![0.0; num_notes];

            let mut diss_dyadic_only = 0.0;
            let mut tonicities_dyadic_only = vec![0.0; num_notes];
            for (_, e) in t.edges_parents.iter() {
                diss += e.edge_contribution(
                    t.note_tonicities_sum,
                    num_notes,
                    t.parent_edge_tonicities_sum,
                );
                tonicities[e.to] +=
                    e.tonicity_score(t.note_tonicities_sum, t.parent_edge_tonicities_sum);

                diss_dyadic_only += e.edge_contribution_relative(t.note_tonicities_sum, num_notes);
                tonicities_dyadic_only[e.to] += e.tonicity_score_dyadic(t.note_tonicities_sum);
            }
            tonicities[t.start] += 1.0 / t.note_tonicities_sum; // the starting note has tonicity of 1.
            diss_per_note[t.start] += diss;
            num_completed_traversals_per_note[t.start] += 1;

            // TODO: figure out how to scale contribution of tonicity scores. By multiplying
            // tonicity contribution by overall dissonance contribution of the traversal, we get the
            // most difference between dissonance of major and minor triads (major diss < minor
            // diss) If we divide by diss instead, we get minor diss < major diss, which is not what
            // we want.
            for idx in 0..num_notes {
                let tonicity_contribution = tonicities[idx] * diss; // contribution of this traversal
                tonicity_scores_per_note[idx] += tonicity_contribution;
                tonicity_entropy_per_note[idx] -=
                    tonicity_contribution * tonicity_contribution.log2();
            }

            diss_entropy_per_note[t.start] -= diss * diss.log2();

            tonicities_dyadic_only[t.start] += 1.0 / t.note_tonicities_sum;
            diss_per_note_dyadic_only[t.start] += diss_dyadic_only;

            ton_scores_dyadic_only
                .iter_mut()
                .zip(tonicities_dyadic_only.iter())
                .for_each(|(a, b)| *a += *b * diss_dyadic_only);
        }

        let avg_diss_per_traversal_per_note: Vec<f64> = diss_per_note
            .iter()
            .zip(num_completed_traversals_per_note.iter())
            .map(|(d, n)| if *n != 0 { d / *n as f64 } else { 0.0 })
            .collect();

        diss_per_note_dyadic_only = diss_per_note_dyadic_only
            .iter()
            .zip(num_completed_traversals_per_note.iter())
            .map(|(d, n)| if *n != 0 { d / *n as f64 } else { 0.0 })
            .collect();

        // normalize diss entropy since distribution of diss doesn't add up to 1.
        diss_entropy_per_note
            .iter_mut()
            .zip(diss_per_note.iter())
            .for_each(|(entropy, diss)| *entropy = *entropy / diss + diss.log2());

        // normalize tonicity entropy
        tonicity_entropy_per_note
            .iter_mut()
            .zip(tonicity_scores_per_note.iter())
            .for_each(|(entropy, ton)| *entropy = *entropy / ton + ton.log2());

        // This is the second attempt at calculating polyadic tonicity. This is done before
        // dissonance calculation to obtain the recursive tonicity first.
        //
        // Our traversals keep track of `from_note_tonicity` which is the recursive, relative
        // tonicity of a particular note in the chord, which is effectively obtained by traversing
        // the spanning tree from the root (starting note) to that note. This value is obtained
        // using only dyadic tonicities, which provides no new structural information.
        //
        // To obtain structural information (substructures/upper structures) we also have
        // `parent_edge_tonicity_abs`, which is tonicity of the previous edge (edge between parent
        // and grandparent) which models the "edge tonicity", which is the likelihood of choosing to
        // traverse this edge relative to other edges to go to other unvisited nodes at that point
        // in the traversal state.

        // TODO: Should we multiply tonicity scores by the pairwise heuristic as below?
        //       If not, comment out the lines below.

        // Without pairwise heuristic, the third of min and maj triads are given the most tonicity,
        // which is not what we want.
        tonicity_scores_per_note
            .iter_mut()
            .zip(smoothed_tonicity_contexts[cand_idx].iter())
            .for_each(|(rec_tonicity, pairwise_tonicity_heur)| {
                *rec_tonicity *= *pairwise_tonicity_heur
            });
        ton_scores_dyadic_only
            .iter_mut()
            .zip(smoothed_tonicity_contexts[cand_idx].iter())
            .for_each(|(rec_tonicity, pairwise_tonicity_heur)| {
                *rec_tonicity *= *pairwise_tonicity_heur
            });

        let sum_tonicity_scores: f64 = tonicity_scores_per_note.iter().sum();
        let note_tonicities_target: Tonicities = tonicity_scores_per_note
            .iter()
            .map(|x| x / sum_tonicity_scores)
            .collect();
        let sum_tonicity_scores_dyadic_only: f64 = ton_scores_dyadic_only.iter().sum();
        let tonicities_dyadic_only: Tonicities = ton_scores_dyadic_only
            .iter()
            .map(|x| x / sum_tonicity_scores_dyadic_only)
            .collect();

        // Uncomment this if we're doing some funky thing to note_tonicities_target such that we
        // have to normalize again.
        // let sum_target: f64 = note_tonicities_target.iter().sum();
        // let note_tonicities_target: Tonicities = note_tonicities_target
        //     .iter()
        //     .map(|x| x / sum_target)
        //     .collect::<Vec<_>>();

        // Smooth the tonicity context towards the target tonicity.
        let tonicity_context: Tonicities = tonicity_smoothing(
            &note_tonicities_target,
            &tonicity_context,
            smoothing,
            elapsed_seconds,
        );

        let mut sum_between_notes_pre_tonicity = 0.0;
        let mut entropy_between_notes_pre_tonicity = 0.0;

        // Sum dissonances between notes, scaled by the tonicity context.
        // The dissonances used are normalized by how many traversals started from that note.
        let sum_diss: f64 = avg_diss_per_traversal_per_note
            .iter()
            .enumerate()
            .map(|(idx, diss)| {
                sum_between_notes_pre_tonicity += diss;
                entropy_between_notes_pre_tonicity -= diss * diss.log2();

                // we scale dissonance contribution by the heuristic context.
                diss * tonicity_context[idx] // * smoothed_tonicity_contexts[cand_idx][idx]
            })
            .sum();
        entropy_between_notes_pre_tonicity = entropy_between_notes_pre_tonicity
            / sum_between_notes_pre_tonicity
            + sum_between_notes_pre_tonicity.log2();
        let final_diss = sum_diss * normalizer;

        // In the initial attempt to evaluate polyadic tonicity, we calculate note tonicities by
        // dissonance contribution, specifically, the hypothesis is that the probability a note is
        // perceived as tonic is inversely correlated with the amount of dissonance is contributed
        // by choosing to perceive that note as the root.
        //
        // This method requires calculating dissonances before graph-based tonicity, relying on the
        // smoothed tonicity context heuristic to multiply dissonance scores.
        //
        // This is may not be such a good idea. If we consider the minor triad, it is the inversion
        // of a major triad, meaning the dissonance contributed by the highest note in the minor
        // triad will be the lowest, since its edges are M3 and P5 (as opposed to m3-P5 and m3-M3),
        // even after performing the necessary dyading scaling by dyadic tonicities (which sensibly
        // outputs that the m3 should have more likely been the root, which is true when considering
        // the major tonality, followed by the root P1). This brings us back to the same problem of
        // "negative harmony duality" which we want to solve here, i.e., otonal and utonal
        // structures having the same dissonance score.

        // let note_tonicities_target: Tonicities = avg_diss_per_traversal_per_note
        //     .iter()
        //     .zip(smoothed_tonicity_contexts[cand_idx].iter())
        //     .map(|(diss, t_ctx)| {
        //         if *diss != 0.0 {
        //             t_ctx / diss
        //         } else {
        //             eprintln!("Warning: found no traversals starting from a note.");
        //             0.0
        //         }
        //     })
        //     .collect();

        let note_tonicities_target_entropy = note_tonicities_target
            .iter()
            .map(|x| if *x != 0.0 { -x * x.log2() } else { 0.0 })
            .sum::<f64>();

        let tonicity_context_entropy = tonicity_context.iter().map(|x| -x * x.log2()).sum::<f64>();

        let diss = Dissonance {
            dissonance: final_diss,
            tonicity_target: note_tonicities_target,
            tonicity_context,
            iterations,
            num_traversals,
            num_completed_traversals: completed_traversals_per_candidate[cand_idx].len(),
            num_incomplete_traversals: traversals
                .iter()
                .filter(|x| x.candidate_idx == cand_idx)
                .count(),
            diss_entropy_per_note,
            tonicity_entropy_per_note,
            entropy_between_notes_pre_tonicity,
            note_tonicities_target_entropy,
            tonicity_context_entropy,
            num_completed_traversals_per_note,
            avg_diss_per_traversal_per_note,
            diss_per_note_dyadic_only,
            tonicities_dyadic_only,
        };

        results.push(diss);
    }

    results
}

/// Obtain tonicities of notes in a chord using only dyadic tonicities.
///
/// The initial tonicity heuristic is a sum of dyadic tonicities ^ roughness
/// (increased roughness will reduce tonicity, as tonicity is in the interval [0, 1])
///
/// Also populates a lookup table for dyadic tonicity and roughness, if provided.
///
/// The lookup table is a Hashmap of bitstrings containing two 1s, where the power of 2 is the index
/// of each note in the dyad. The tonicity in the lookup is with respect to the note with lower
/// index (not lower pitch), e.g. if the bitstring is 0b1001, then the tonicity is with respect to hearing
/// the note at index 0 as tonic.
///
/// The indices of the returned tonicities corresponds to the note at input `cents`.
///
/// n(n-1) iterations, O(n^2).
pub fn dyadic_tonicity(
    cents: &[f64],
    tonicity_map: &mut Option<HashMap<u32, f64>>,
    roughness_map: &mut Option<HashMap<u32, f64>>,
) -> Vec<f64> {
    if cents.len() == 0 {
        return vec![];
    }
    if cents.len() == 1 {
        return vec![1.0];
    }
    let mut tonicities = vec![0.0; cents.len()];
    for i in 0..cents.len() {
        for j in (i + 1)..cents.len() {
            let i_is_higher = cents[i] >= cents[j];
            let higher = if i_is_higher { i } else { j };
            let lower = if i_is_higher { j } else { i };
            let dyad_tonicity = TonicityLookup::dyad_tonicity(cents[higher] - cents[lower]);
            let roughness = DyadLookup::get_roughness(
                cents[higher] - cents[lower],
                RoughnessType::Multiplicative,
            );
            // tonicities[lower] += dyad_tonicity.powf(roughness / 2.0);
            // tonicities[higher] += (1.0 - dyad_tonicity).powf(roughness / 2.0);
            tonicities[lower] += dyad_tonicity / roughness;
            tonicities[higher] += (1.0 - dyad_tonicity) / roughness;
            let bitmask = (1 << higher) | (1 << lower);
            if let Some(tonicity_map) = tonicity_map {
                if !tonicity_map.contains_key(&bitmask) {
                    // If i is higher, then dyad_tonicity is the probability of hearing the lower note j as tonic.
                    // We want the probability of i.
                    let tonicity_lower_index = if i_is_higher {
                        1.0 - dyad_tonicity
                    } else {
                        dyad_tonicity
                    };
                    tonicity_map.insert(bitmask, tonicity_lower_index);
                }
            }
            if let Some(roughness_map) = roughness_map {
                if !roughness_map.contains_key(&bitmask) {
                    roughness_map.insert(bitmask, roughness);
                }
            }
        }
    }
    let sum = tonicities.iter().sum::<f64>();
    tonicities.iter().map(|x| x / sum).collect()
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

    /// `mult_roughness_map` is a lookup of multiplicative roughnesses of the existing notes in `cents`.
    /// The power of 2 in the bitstring key is the index of each note in the dyad in `cents`
    pub mult_roughness_map: HashMap<u32, f64>,

    /// `add_roughness_map` is a lookup of additive roughnesses of the existing notes in `cents`.
    pub add_roughness_map: HashMap<u32, f64>,

    /// `cand_tonicity_map` is a lookup of dyad tonicities is indexed by the existing note, followed
    /// by the choice of new note. Its tonicity is with respect to the existing note against the new
    /// note.
    ///
    /// E.g., `cand_tonicity_map[i][j] == 0.7` means that i-th existing note in `cents` is 70% tonic
    /// relative to the new j-th note in `candidate_cents`.
    ///
    /// This convention agrees with the tonicities in `tonicity_map` being with respect to the lower
    /// index note, and that the new note is regarded as the last index.
    pub cand_tonicity_map: Vec<Vec<f64>>,

    /// `cand_roughness_map` is indexed by the existing note, followed by the choice of new note.
    ///
    /// E.g., `cand_roughness_map[i][j] == 1.5` means that the roughness of the dyad between the i-th
    /// note in `cents` and the j-th candidate note in `candidate_cents` is 1.5.
    pub cand_mult_roughness_map: Vec<Vec<f64>>,

    /// Same as `cand_mult_roughness_map`, but for additive roughness.
    pub cand_add_roughness_map: Vec<Vec<f64>>,
}

/// Heuristic initial tonicity of notes in a chord using only dyadic tonicities, where a new note
/// with several possible interpretations is added.
///
/// The heuristic tonicity is a probability distribution of each note being perceived as tonic
/// relative to the other notes.
///
/// ### Parameters
///
/// `cents` is the list of already existing notes in the chord.
///
/// `candidate_cents` is the list of possible interpretations of the new note, relative to the same
/// offset as `cents`.
///
/// ### Returns
///
/// See [TonicityHeuristic].
///
/// If `cand_cents` is not provided, the return value will have a single value in `tonicities`,
/// which is the tonicity of the notes in `cents` amongst themselves only.
///
pub fn dyadic_tonicity_heur(cents: &[f64], candidate_cents: &[f64]) -> TonicityHeuristic {
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
            let roughness = DyadLookup::get_roughness(
                cents[hi_idx] - cents[lo_idx],
                RoughnessType::Multiplicative,
            );
            let add_roughness =
                DyadLookup::get_roughness(cents[hi_idx] - cents[lo_idx], RoughnessType::Additive);

            let (lo_contrib, hi_contrib) = heuristic_function(dyad_tonicity, roughness);
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
            mult_roughness_map.insert(bitmask, roughness);
            add_roughness_map.insert(bitmask, add_roughness);
        }
    }

    let existing_sum = existing_tonicities.iter().sum::<f64>();

    if candidate_cents.len() == 0 {
        return TonicityHeuristic {
            tonicities: vec![existing_tonicities
                .iter()
                .map(|x| x / existing_sum)
                .collect()],
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
            let dyad_tonicity = TonicityLookup::dyad_tonicity(hi_cents - lo_cents);
            let roughness =
                DyadLookup::get_roughness(hi_cents - lo_cents, RoughnessType::Multiplicative);
            let add_roughness =
                DyadLookup::get_roughness(hi_cents - lo_cents, RoughnessType::Additive);
            let (lo_contrib, hi_contrib) = heuristic_function(dyad_tonicity, roughness);
            tonicities_per_candidate[cand_idx][lo_idx] += lo_contrib;
            tonicities_per_candidate[cand_idx][hi_idx] += hi_contrib;
            cand_tonicity_map[existing_idx].push(if existing_higher {
                1.0 - dyad_tonicity
            } else {
                dyad_tonicity
            });
            cand_mult_roughness_map[existing_idx].push(roughness);
            cand_add_roughness_map[existing_idx].push(add_roughness);
        }

        // Turning raw tonicity scores into a normalized prob distribution by dividing by sum.
        // let sum = tonicities_per_candidate[cand_idx].iter().sum::<f64>();
        // tonicities_per_candidate[cand_idx] = tonicities_per_candidate[cand_idx]
        //     .iter()
        //     .map(|x| x / sum)
        //     .collect();

        // Version 2: normalizing by softmax.
        // TODO: adjust softness temperature if needed.

        // Higher temperature = softer/less confident distribution.
        const SOFTMAX_TEMPERATURE: f64 = 1.0;

        // subtract max raw tonicity from raw tonicity scores to improve numerical stability
        let max_tonicity = tonicities_per_candidate[cand_idx]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = tonicities_per_candidate[cand_idx]
            .iter()
            .map(|x| ((x - max_tonicity) / SOFTMAX_TEMPERATURE).exp())
            .sum();
        tonicities_per_candidate[cand_idx] = tonicities_per_candidate[cand_idx]
            .iter()
            .map(|x| ((x - max_tonicity) / SOFTMAX_TEMPERATURE).exp() / exp_sum)
            .collect();
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

fn print_tree(t: &Traversal) {
    let children = &t.edges_children;
    let start = t.start;
    for (from, edges) in children.iter() {
        println!("{}:", from);
        for e in edges.iter() {
            println!(
                "  {}: {}",
                e.to,
                e.edge_contribution(
                    t.note_tonicities_sum,
                    t.edges_parents.len() + 1,
                    t.parent_edge_tonicities_sum
                )
            );
        }
    }

    let tree = make_termtree(start, children);
    println!("{}", tree);
}

fn make_termtree(start: usize, children: &HashMap<usize, Vec<Edge>>) -> Tree<usize> {
    let mut root = Tree::new(start);
    for edge in children.get(&start).unwrap_or(&vec![]) {
        root.push(make_termtree(edge.to, children));
    }
    root
}

#[cfg(test)]
mod tests {
    use compute::prelude::quad5;

    use super::*;
    use crate::utils::cents_to_hz;
    use std::{time::Instant, usize};

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
        let tonicities = dyadic_tonicity(cents, &mut None, &mut None);
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
        let out = dyadic_tonicity_heur(cents, cand_cents);
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
        let iters = 10;

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
        let freqs = cents
            .iter()
            .map(|x| cents_to_hz(440.0, *x))
            .collect::<Vec<f64>>();
        let mut context = vec![1.0 / freqs.len() as f64; freqs.len()];

        println!("\nGraph diss: {}", name);
        for i in 0..iters {
            let diss = graph_dissonance(
                &freqs,
                &[],
                &context,
                0.9,
                elapsed_seconds,
                usize::MAX,
                usize::MAX,
                usize::MAX,
                PruneMethod::PrioritizeMaxTonicity,
            );
            // sum_dyad_diss is an experiment to see what diss would be if we didn't consider
            // subtrees/upper structures but only dyads and their tonicity scores.
            let mut sum_dyad_diss = 0.0;
            for idx in 0..cents.len() {
                println!(
                    "{:>9.2}c: {:>7.4} (dyad: {:>7.4}) [diss] x {:>7.4} (target: {:>7.4}, dyad: {:>7.4}) [tonicity] = {:.4} (dyad: {:.4})",
                    cents[idx],
                    diss[0].avg_diss_per_traversal_per_note[idx],
                    diss[0].diss_per_note_dyadic_only[idx],
                    diss[0].tonicity_context[idx],
                    diss[0].tonicity_target[idx],
                    diss[0].tonicities_dyadic_only[idx],
                    diss[0].avg_diss_per_traversal_per_note[idx] * diss[0].tonicity_context[idx],
                    diss[0].diss_per_note_dyadic_only[idx] * diss[0].tonicities_dyadic_only[idx],
                );
                sum_dyad_diss +=
                    diss[0].diss_per_note_dyadic_only[idx] * diss[0].tonicities_dyadic_only[idx];
            }
            println!(
                "Diss: {:.4} (dyad: {:.4})",
                diss[0].dissonance, sum_dyad_diss
            );
            println!("{}s: {:#?}", (i + 1) as f64 * elapsed_seconds, diss);
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
        let diss = graph_dissonance(
            &freqs,
            &[],
            &vec![0f64; cents.len()],
            0.9,
            1.0,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            PruneMethod::PrioritizeMaxTonicity,
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
            usize::MAX,
            usize::MAX,
            usize::MAX,
            PruneMethod::PrioritizeMaxTonicity,
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

        let diss_existing = graph_dissonance(
            &freqs,
            &[],
            &vec![0f64; freqs.len()],
            0.9,
            0.1,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            PruneMethod::PrioritizeMaxTonicity,
        )[0]
        .clone();

        let diss_candidate = graph_dissonance(
            &freqs[..(freqs.len() - 1)],
            &[*freqs.last().unwrap()],
            &vec![0f64; freqs.len() - 1],
            0.9,
            0.1,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            PruneMethod::PrioritizeMaxTonicity,
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

        let heur_existing = dyadic_tonicity_heur(&freqs, &[]);

        let heur_candidate =
            dyadic_tonicity_heur(&freqs[..(freqs.len() - 1)], &[*freqs.last().unwrap()]);

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

        let forwards = dyadic_tonicity_heur(&freqs, &[]);
        let backwards =
            dyadic_tonicity_heur(&freqs.iter().rev().copied().collect::<Vec<f64>>(), &[]);

        println!("\nForwards: {:#?}", forwards.tonicities);
        println!(
            "\nBackwards: {:#?}",
            backwards.tonicities.iter().rev().collect::<Vec<_>>()
        );
    }

    /// It appears the most efficient pruning setting that preserves tonicity & minimizes inversions
    /// on the relative order of note tonicities is (3, MAX, 2000 to 4000) on the
    /// PrioritizeMaxTonicity method
    ///
    /// However, for dissonance calculations, (2, MAX, 400-2000) with Random method appears to work
    /// really fast while the dissonance scores are not too far off from the best pruning settings.
    ///
    /// We still have to test relative order dissonance scores between different candidates though...
    #[test]
    fn test_pruning_deterioration() {
        // let cents = [0., 600., 1000., 1300., 1600., 2100.]; // C13b9b5 (6 notes)
        // let cents = [0., 700., 1100., 1400., 1500., 1900., 2100.]; // Cm69maj7 (7 notes)
        let cents = [0., 400., 700., 1100., 1400., 1800., 2100., 2500.]; // Cmaj#15 (8 notes)
        let freqs = cents
            .iter()
            .map(|x| cents_to_hz(440.0, *x))
            .collect::<Vec<f64>>();
        let max_ram_usage_allowed = 10_000_000_000;
        let max_traversals =
            max_ram_usage_allowed / (size_of::<Traversal>() + size_of::<Edge>() * cents.len() * 3);
        let prune_settings = [
            // general settings to figure out what works best
            // (10_000_000, 10_000_000, 100_000),
            // (10_000_000, 10_000_000, 10_000),
            // (6, 10_000_000, 10_000_000),
            // (5, 10_000_000, 10_000_000),
            // (4, 10_000_000, 10_000_000),
            // (3, 10_000_000, 10_000_000),
            // (2, 10_000_000, 10_000_000),
            // (10_000_000, 20000, 10_000_000),
            // (10_000_000, 10000, 10_000_000),
            // (10_000_000, 1000, 10_000_000),
            // (10_000_000, 100, 10_000_000),

            // trying out specific combos that seem to work well for max attempts (traversals) per
            // iter, we have no choice but to limit it, otherwise there's not enough RAM for stack
            // buffer. Traversal is 208 bytes at the time of writing.
            (4, usize::MAX, max_traversals),
            (4, usize::MAX, 10_000),
            (4, usize::MAX, 6_000),
            (4, usize::MAX, 3_000),
            (4, usize::MAX, 1_000),
            (3, usize::MAX, max_traversals),
            (3, usize::MAX, 4_000),
            (3, usize::MAX, 2_000),
            (3, usize::MAX, 1_000),
            (2, usize::MAX, max_traversals),
            (2, usize::MAX, 400),
            (2, usize::MAX, 200),
            (2, usize::MAX, 100),
        ];
        let prune_methods = [
            PruneMethod::PrioritizeMaxTonicity,
            PruneMethod::PrioritizeMinDiss,
            PruneMethod::Random,
        ];
        println!("Computing lookups...");
        TonicityLookup::dyad_tonicity(800.0);

        println!(
            "Maximum traversals are capped at {max_traversals} to cap ram usage at {max_ram_usage_allowed} bytes.\n\
            Because of memory constraints, the baseline is capped at 7 attempts per traversal, so N >= 8 is not accurate."
        );
        let max_attempts_per_trv = 7;

        // TODO: for full traversal of cases N >= 8, implement memory mapped files using memmap2.
        // Figure out how to use lazy loading for the traversal process.

        let start = Instant::now();
        let baseline = graph_dissonance(
            &freqs,
            &[],
            &vec![0f64; freqs.len()],
            0.9,
            0.1,
            max_attempts_per_trv,
            usize::MAX,
            max_traversals,
            PruneMethod::PrioritizeMaxTonicity,
        )[0]
        .clone();
        let baseline_duration = start.elapsed();

        println!("Baseline: {:#?}", baseline);
        println!("Time taken: {:?}", baseline_duration);
        println!("\n---------------------------------------------\n");

        for prune_method in &prune_methods {
            for (max_attempts_per_trv, max_attempts_per_iteration, target_num_traversals) in
                prune_settings
            {
                let start = Instant::now();
                let pruned = graph_dissonance(
                    &freqs,
                    &[],
                    &vec![0f64; freqs.len()],
                    0.9,
                    0.1,
                    max_attempts_per_trv,
                    max_attempts_per_iteration,
                    target_num_traversals,
                    *prune_method,
                )[0]
                .clone();
                let duration = start.elapsed();
                println!(
                "\nPruning settings: {}, Max attempts per traversal: {}, Max attempts per iteration: {}, Num traversals: {}",
                prune_method, max_attempts_per_trv, max_attempts_per_iteration, target_num_traversals
            );
                println!("{:#?}", pruned);
                println!("Time taken: {:?}", duration);

                // Compare with baseline.

                let dissonance_difference = pruned.dissonance - baseline.dissonance;

                // Use root mean square error to compare the tonicity vectors.
                let tonicity_difference = pruned
                    .tonicity_context
                    .iter()
                    .zip(baseline.tonicity_context.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                // Count how many inversions there are in the tonicity vectors. The more inversions,
                // the more inconsistencies in the relative order of tonicities of notes.
                let tonicity_inversions =
                    count_inversions(&pruned.tonicity_context, &baseline.tonicity_context);

                let time_reduction = baseline_duration.as_secs_f64() / duration.as_secs_f64();

                // number in parentheses scales the error by the time reduction factor.
                println!(
                    "Dissonance difference: {:.4} ({:.4}), Tonicity difference: {:.4} ({:.4}), Tonicity inversions: {}",
                    dissonance_difference,
                    dissonance_difference / time_reduction * 1e5,
                    tonicity_difference,
                    tonicity_difference / time_reduction * 1e5,
                    tonicity_inversions
                );
                println!(
                    "Intermediate traversal reduction: {:.3}x, Completed traversal reduction: {:.3}x",
                    baseline.num_traversals as f64 / pruned.num_traversals as f64,
                    baseline.num_completed_traversals as f64 / pruned.num_completed_traversals as f64
                );
                println!("\n------------------------------------------\n")
            }
        }
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

    #[test]
    fn test_binary_heap() {
        let mut heap = BinaryHeap::new_by_key(|a| *a);
        heap.push(5);
        heap.push(1);
        heap.push(3);
        heap.push(9);
        heap.push(2);
        heap.push(6);
        println!("By key: {:?}", heap.into_sorted_vec());
        // will put min first, max last.

        let mut heap = BinaryHeap::new_by(|a: &f64, b| a.partial_cmp(b).unwrap());
        heap.push(1.);
        heap.push(5.);
        heap.push(3.);
        heap.push(6.);
        heap.push(2.);
        heap.push(9.);

        // Notice that the unsorted into_vec does not guarantee a fair random ordering.
        // The first half of the vec will always be greater than the second half.
        println!("Unsorted: {:?}", heap.clone().into_vec());

        println!("By partial cmp: {:?}", heap.into_sorted_vec());
        // a.cmp(b) will put min first, max last.

        let mut heap = BinaryHeap::new_by(|a: &f64, b| random::<u32>().cmp(&random()));
        heap.push(1.);
        heap.push(5.);
        heap.push(3.);
        heap.push(6.);
        heap.push(2.);
        heap.push(9.);
        println!("By random unsorted: {:?}", heap.clone().into_sorted_vec());
        println!("By random: {:?}", heap.into_sorted_vec());
    }
}
