//! For generating interpretation trees (spanning trees) of the complete graph of notes.

use std::sync::LazyLock;

use termtree::Tree;

/// Precomputed spanning trees of n nodes where n is the index of the outer vec.
pub static TREES: LazyLock<Vec<Vec<ST>>> = LazyLock::new(|| {
    let mut all_trees = vec![vec![], vec![]];

    // Generate STs for 1 to 8 nodes
    for n in 2..=8 {
        let sts = gen_sts(n, 3, 3, 3);
        all_trees.push(sts);
    }

    all_trees
});

/// An edge in a ST.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
}

/// A spanning tree of the complete graph of N notes.
#[derive(Debug, Clone)]
pub struct ST {
    /// Which node is the root.
    pub root: usize,

    /// Lookup of which parent each node has in the ST.
    ///
    /// The 0-based index corresponds which node index to get the parent of.
    ///
    /// Root node has no parent.
    pub parents: Vec<Option<usize>>,

    /// Lookup of which children each node has in the ST.
    ///
    /// The 0-based index corresponds to which node index to get the children of.
    pub children: Vec<Vec<usize>>,

    /// All edges in the ST.
    pub edges: Vec<Edge>,
}

impl ST {
    pub fn print(&self) {
        print_tree(self);
    }
}

/// Generate all spanning trees (STs) of a complete graph with `n` nodes.
///
/// Subject to the constraints:
/// - Maximum depth of `max_depth`
/// - Maximum number of siblings/children per node according to `max_siblings`
/// - Maximum number of inversions in the pre-order traversal of `max_inversions`
///
/// Notes/nodes must be indexed in ascending pitch order so that the pre-order derangements heuristic
/// is valid.
pub fn gen_sts(n: usize, max_depth: usize, max_siblings: usize, max_inversions: usize) -> Vec<ST> {
    if n == 0 { return vec![]; }
    let mut results = Vec::new();

    for r in 0..n {
        let mut st = ST {
            root: r,
            parents: vec![None; n],
            children: vec![vec![]; n],
            edges: Vec::with_capacity(n - 1),
        };
        let mut depths = vec![0; n];
        let mut visited_mask = 1u64 << r;
        let mut preorder = vec![r];
        let mut queue = vec![r]; // Nodes whose children we need to assign

        gen_sts_recursive(
            n,
            max_depth,
            max_siblings,
            max_inversions,
            &mut st,
            &mut depths,
            visited_mask,
            &mut preorder,
            &mut queue,
            0, // Start processing queue at index 0
            &mut results,
        );
    }
    results
}

/// Recursive helper for generating STs.
///
/// - `n`: number of nodes in complete graph
/// - `max_depth`: maximum depth of ST
/// - `max_siblings`: maximum number of children/siblings per node
/// - `max_inversions`: maximum number of inversions in pre-order traversal
/// - `st`: current incomplete ST
/// - `depths`: current depths of nodes in the ST
/// - `visited_mask`: bitmask of which nodes have been added to the ST
/// - `preorder`: current pre-order traversal of nodes in the ST
/// - `queue`: first in the queue represents the current parent node to assign children to
/// - `queue_idx`: index of first unprocessed node in the queue
/// - `results`: contains completed STs
fn gen_sts_recursive(
    n: usize,
    max_depth: usize,
    max_siblings: usize,
    max_inversions: usize,
    st: &mut ST,
    depths: &mut [usize],
    visited_mask: u64,
    preorder: &mut Vec<usize>,
    queue: &mut Vec<usize>,
    queue_idx: usize,
    results: &mut Vec<ST>,
) {
    // Prune based on the partial preorder's inversion count

    // Do not count inversions involving the root node, as we want equal exploration of all choices
    // of root nodes.
    if count_inversions(&preorder[1..]) > max_inversions {
        return;
    }

    if visited_mask.count_ones() as usize == n {
        // All nodes visited, push the complete ST.
        let mut final_st = st.clone();
        for (child, p_opt) in final_st.parents.iter().enumerate() {
            if let Some(p) = *p_opt {
                final_st.edges.push(Edge { from: p, to: child });
            }
        }
        results.push(final_st);
        return;
    }

    // If we already processed all nodes in the queue but haven't visited all nodes, it's an invalid
    // branch (shouldn't happen in a complete graph with n > 1)
    //
    // This can happen if certain constraints prevent any further nodes from being added.
    if queue_idx >= queue.len() {
        return;
    }

    let parent = queue[queue_idx];
    let parent_depth = depths[parent];

    let unvisited: Vec<usize> = (0..n).filter(|&i| (visited_mask & (1 << i)) == 0).collect();

    let possible_num_children = if parent_depth < max_depth {
        0..=std::cmp::min(max_siblings, unvisited.len())
    } else {
        0..=0
    };

    // The strat is to iterate over the number of children to add to this current node, then
    // generate unique combinations of possible children and clone a new state for each possible
    // assignment of children.
    //
    // This prevents any duplicate trees caused by two edges being added in different orders.
    //
    // Once children are assigned, they get added to the queue as potential parents to assign more
    // children to.

    for num_children in possible_num_children {
        if num_children == 0 {
            // if this cannot have children: go on to the next candidate parent node to assign
            // children to.
            gen_sts_recursive(
                n, max_depth, max_siblings, max_inversions,
                st, depths, visited_mask, preorder, queue, queue_idx + 1, results
            );
        } else {
            // Generate all combinations of `num_children` elements from the `unvisited` pool
            //
            // Combinations are already sorted in ascending order.
            for combination in combinations(&unvisited, num_children) {

                let mut new_visited = visited_mask;
                let mut added_nodes = Vec::new();

                for &child in &combination {
                    st.parents[child] = Some(parent);
                    st.children[parent].push(child);
                    depths[child] = parent_depth + 1;
                    new_visited |= 1 << child;
                    added_nodes.push(child);
                }

                let mut next_preorder = preorder.clone();
                next_preorder.extend(&added_nodes);

                let mut next_queue = queue.clone();
                next_queue.extend(&added_nodes);

                gen_sts_recursive(
                    n, max_depth, max_siblings, max_inversions,
                    st, depths, new_visited, &mut next_preorder, &mut next_queue, queue_idx + 1, results
                );

                // backtrack
                for &child in &combination {
                    st.parents[child] = None;
                    st.children[parent].pop();
                    depths[child] = 0;
                }
            }
        }
    }
}

/// Generates all combinations of `k` items from the given slice `items`.
///
/// Each combination is sorted in ascending order.
///
/// For example, combinations(&[0,1,2], 2) will return [[0,1],[0,2],[1,2]].
fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    let mut res = Vec::new();
    let mut current = Vec::new();
    fn run(items: &[usize], k: usize, start: usize, current: &mut Vec<usize>, res: &mut Vec<Vec<usize>>) {
        if current.len() == k {
            res.push(current.clone());
            return;
        }
        for i in start..items.len() {
            current.push(items[i]);
            run(items, k, i + 1, current, res);
            current.pop();
        }
    }
    run(items, k, 0, &mut current, &mut res);
    res
}

fn count_inversions(arr: &[usize]) -> usize {
    let mut count = 0;
    for i in 0..arr.len() {
        for j in i + 1..arr.len() {
            if arr[i] > arr[j] { count += 1; }
        }
    }
    count
}

fn print_tree(t: &ST) {
    let children = &t.children;
    let start = t.root;

    let tree = make_termtree(start, children);
    println!("{}", tree);
}

fn make_termtree(start: usize, children: &Vec<Vec<usize>>) -> Tree<usize> {
    let mut root = Tree::new(start);
    for edge in &children[start] {
        root.push(make_termtree(*edge, children));
    }
    root
}


mod tests {
    use super::*;

    #[test]
    fn test_combinations() {
        let items = vec![0, 1, 2, 3];
        let combs = combinations(&items, 2);
        for c in combs {
            println!("{:?}", c);
        }
    }

    #[test]
    fn test_gen_st() {

        let sts = gen_sts(4, 2, 2, 3);

        for (idx, st) in sts.iter().enumerate() {
            println!("ST {idx}");

            print_tree(st);
        }

        println!("^^^ STs of 4 notes, max 2 children per node, max depth 2 ^^^")
    }

    #[test]
    fn test_count_sts() {
        TREES.iter().enumerate().for_each(|(n, sts)| {
            println!("Number of STs with {n} nodes: {}", sts.len());
        });
    }

    #[test]
    fn test_count_inversions() {
        // This pattern represents the pre-order traversal of a Cmaj13#11 tertian voicing in
        // ascending pitch order C E G B D F# A, where the interpretation is according to the
        // chain-of-fifths from the root and third
        //
        // C -> E -> B -> F#
        //   -> G -> D -> A
        let arr = vec![0, 1, 3, 5, 2, 4, 6];

        let inv = count_inversions(&arr);
        println!("Inversions: {}", inv);
        assert_eq!(inv, 3);

        // This tree has 3 inversions and is still very intuitive.
        //
        // So 3 inversions is a good cutoff for generating trees.

        println!("Inversions for C4, E4, F#5, G4, B4, A5, D5: {}", count_inversions(&vec![0,1,5,2,3,6,4]));
        println!("Inversions for C4, E4, B4, G4, D5, F#5, A5: {}", count_inversions(&vec![0,1,3,2,4,5,6]));
        println!("Inversions for C4, E4, C#4, A4, G4, C5: {}", count_inversions(&vec![0,2,1,4,3,5]));
        println!("Inversions for C4, G4, A4, C#4, E4, C5: {}", count_inversions(&vec![0,3,4,1,2,5]));
    }
}