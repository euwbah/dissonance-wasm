//! For generating interpretation trees (spanning trees) of the complete graph of notes.

use std::sync::LazyLock;

use termtree::Tree;

/// Precomputed spanning trees of n nodes where n is the index of the outer vec.
pub static TREES: LazyLock<Vec<Vec<ST>>> = LazyLock::new(|| {
    let mut all_trees = vec![vec![], vec![]];

    // Generate STs for 1 to 8 nodes
    //
    // IMPORTANT: Because of the usage of bitmasks for optimizing edges and subtree keys, we cannot
    // make trees with more than 8 nodes!
    for n in 2..=8 {
        let sts = gen_sts(n, 3, 3, 3);

        all_trees.push(sts);
    }

    all_trees
});

/// An edge in a ST.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: u8,
    pub to: u8,
}

/// A key for representing the entire subtree rooted at a given node.
///
/// This key is optimized so that it is easy to convert a subtree into a key, and optimized for
/// space such that every parent node for every tree can have one pregenerated key in RAM.
///
/// No two subtrees should have the same key, but we don't have to care whether or not the subtree
/// is easily reconstructable from the key.
///
/// - Bits 0-23 (from LSB) are reserved to store an adjacency list where the i-th group of 3 bits
///   (from LSB) is the parent node number of node i. If node i is a root, that those 3 bits will be
///   set to 0.
///
/// - Bits 24-31 (from LSB) stores the nodes present in the subtree as a bitmask (where node 0 is bit 24).
///
/// - Bits 32-34 (from LSB) store the root node number (0-7).
pub type SubtreeKey = u64;

/// Check if the given node is part of the subtree represented by the given [SubtreeKey].
///
/// `node` is presented as a 0-based index, so valid values are 0 to 7 inclusive.
pub fn is_node_part_of_subtree(key: SubtreeKey, node: usize) -> bool {
    key & (1 << (24 + node)) != 0
}

/// A spanning tree of the complete graph of N notes.
#[derive(Debug, Clone)]
pub struct ST {
    /// Which node is the root.
    pub root: u8,

    /// Lookup of which parent each node has in the ST.
    ///
    /// The 0-based index corresponds which node index to get the parent of.
    ///
    /// Root node has no parent.
    pub parents: Vec<Option<u8>>,

    /// Unique [SubtreeKey] identifier of the subtree at each node. The index corresponds to which
    /// node (0 to 7 inclusive). Across, different [ST]s, identical subtrees should have identical
    /// subtree keys.
    ///
    /// If the node is a leaf, the subtree key will contain empty adjacency list but will record the
    /// leaf node itself as the subtree root (bits 32-34 from LSB).
    ///
    /// This lookup is used in computations across many trees so that the results for identical
    /// subtrees can be memoized.
    pub subtree_key: Vec<SubtreeKey>,

    /// Lookup of which children each node has in the ST.
    ///
    /// The 0-based index corresponds to which node index to get the children of.
    pub children: Vec<Vec<u8>>,

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
    if n == 0 {
        return vec![];
    }
    let mut results = Vec::new();

    for r in 0..n {
        let mut st = ST {
            root: r as u8,
            parents: vec![None; n],
            subtree_key: vec![0; n],
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

/// Recursively computes subtree keys for all nodes in the ST.
///
/// - `node`: current node to compute subtree key for
/// - `final_st`: the complete [ST]
/// - `subtree_keys`: mutable slice to store computed subtree keys, indexed by node.
fn compute_subtree_keys(node: usize, final_st: &ST, subtree_keys: &mut [u64]) {
    // Process all children first (post-order)
    for &child in &final_st.children[node] {
        compute_subtree_keys(child as usize, final_st, subtree_keys);
    }

    // Now compute this node's subtree key by merging children's keys
    let mut key = 0u64;

    // Mark this node as present in its own subtree
    key |= 1u64 << (24 + node);

    // Merge in all children's subtrees
    for &child in &final_st.children[node] {
        let child_usize = child as usize;
        let child_key = subtree_keys[child_usize];

        // Merge: OR in all nodes and adjacency bits from child
        key |= child_key & 0xFFFFFF; // adjacency bits
        key |= child_key & (0xFF << 24); // nodes mask

        // Set the parent pointer for this child to current node
        let parent_table_idx = 3 * child_usize;
        key &= !(0b111u64 << parent_table_idx); // clear old parent bits
        key |= (node as u64) << parent_table_idx; // set new parent
    }

    // Set root bits to this node
    key |= (node as u64) << 32;

    subtree_keys[node] = key;
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
/// - `subtree_keys_in_progress`: subtree keys being built up
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

    let num_visited = visited_mask.count_ones() as usize;

    if num_visited == n {
        // All nodes visited, push the complete ST.
        let mut final_st = st.clone();

        // populate adjacency list edges
        for (child, p_opt) in final_st.parents.iter().enumerate() {
            if let Some(p) = *p_opt {
                final_st.edges.push(Edge {
                    from: p,
                    to: child as u8,
                });
            }
        }

        // Traverse all subtrees of completed ST to compute subtree keys.

        let mut subtree_keys = vec![0u64; n];

        compute_subtree_keys(final_st.root as usize, &final_st, &mut subtree_keys);

        final_st.subtree_key = subtree_keys;

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
    assert!(parent < n);

    let parent_depth = depths[parent];

    let unvisited: Vec<usize> = (0..n).filter(|&i| (visited_mask & (1 << i)) == 0).collect();

    // What is the range of the number of children this parent can have?
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
                n,
                max_depth,
                max_siblings,
                max_inversions,
                st,
                depths,
                visited_mask,
                preorder,
                queue,
                queue_idx + 1,
                results,
            );
        } else {
            // Generate all combinations of `num_children` elements from the `unvisited` pool
            //
            // Combinations are already sorted in ascending order.
            for combination in combinations(&unvisited, num_children) {
                let mut new_visited = visited_mask;
                let mut added_nodes = Vec::new();

                // add edges to all children in combination. We backtrack later.
                for &child in &combination {
                    st.parents[child] = Some(parent as u8);
                    st.children[parent].push(child as u8);
                    depths[child] = parent_depth + 1;
                    new_visited |= 1 << child;
                    added_nodes.push(child);
                }

                let mut next_preorder = preorder.clone();
                next_preorder.extend(&added_nodes);

                let mut next_queue = queue.clone();
                next_queue.extend(&added_nodes);

                gen_sts_recursive(
                    n,
                    max_depth,
                    max_siblings,
                    max_inversions,
                    st,
                    depths,
                    new_visited,
                    &mut next_preorder,
                    &mut next_queue,
                    queue_idx + 1,
                    results,
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
    fn run(
        items: &[usize],
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        res: &mut Vec<Vec<usize>>,
    ) {
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
            if arr[i] > arr[j] {
                count += 1;
            }
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

fn make_termtree(start: u8, children: &Vec<Vec<u8>>) -> Tree<u8> {
    let mut root = Tree::new(start);
    for edge in &children[start as usize] {
        root.push(make_termtree(*edge, children));
    }
    root
}

mod tests {
    use super::*;

    #[test]
    fn test_combinations() {
        let items = vec![0, 2, 3, 7, 8];
        let combs = combinations(&items, 3);
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

        println!(
            "Inversions for C4, E4, F#5, G4, B4, A5, D5: {}",
            count_inversions(&vec![0, 1, 5, 2, 3, 6, 4])
        );
        println!(
            "Inversions for C4, E4, B4, G4, D5, F#5, A5: {}",
            count_inversions(&vec![0, 1, 3, 2, 4, 5, 6])
        );
        println!(
            "Inversions for C4, E4, C#4, A4, G4, C5: {}",
            count_inversions(&vec![0, 2, 1, 4, 3, 5])
        );
        println!(
            "Inversions for C4, G4, A4, C#4, E4, C5: {}",
            count_inversions(&vec![0, 3, 4, 1, 2, 5])
        );
    }

    #[test]
    fn test_subtree_keys() {
        use std::collections::{HashMap, HashSet};

        // Maps SubtreeKey -> (root, nodes_mask, adjacency_list as string for debugging)
        let mut key_to_subtree: HashMap<SubtreeKey, (usize, u8, String)> = HashMap::new();

        // Maps (root, nodes_mask, adjacency_list) -> SubtreeKey
        let mut subtree_to_key: HashMap<(usize, u8, String), SubtreeKey> = HashMap::new();

        let mut collision_count = 0;
        let mut duplicate_subtree_count = 0;

        for (n, trees) in TREES.iter().enumerate() {
            if n == 0 || n == 1 {
                continue;
            }

            println!("Checking {} trees with {} nodes", trees.len(), n);

            for tree in trees {
                for node in 0..n {
                    let key = tree.subtree_key[node];

                    // 00011110 000 000 000 001 110 001 000 000

                    // Extract components from SubtreeKey
                    let adjacency_bits = key & 0xFFFFFF; // bits 0-23
                    let nodes_mask = ((key >> 24) & 0xFF) as u8; // bits 24-31
                    let root = ((key >> 32) & 0b111) as usize; // bits 32-34

                    // Build adjacency list as per key.
                    //
                    // Formatted as (parent, child).
                    let mut adj_list_from_key = Vec::new();
                    for i in 0..n {
                        let parent_of_i = (adjacency_bits >> (3 * i)) & 0b111;
                        let i_in_subtree = (nodes_mask & (1 << i)) != 0;
                        if i_in_subtree {
                            // ensure that adj_list doesn't include nodes not in the subtree
                            assert!(
                                (parent_of_i as usize) < n,
                                "Invalid parent {} for node {} in subtree. Parent not inside node count {}",
                                parent_of_i,
                                i,
                                n
                            );

                            if i != root {
                                adj_list_from_key.push((parent_of_i as usize, i));
                            } else {
                                assert!(
                                    parent_of_i == 0,
                                    "Root node {} has non-zero parent {} in subtree",
                                    i,
                                    parent_of_i
                                );
                            }
                        } else {
                            assert!(
                                parent_of_i == 0,
                                "Node {} not in subtree but has parent {}",
                                i,
                                parent_of_i
                            );
                        }
                    }
                    adj_list_from_key.sort();

                    // Verify adjacency list matches actual subtree structure via DFS
                    let mut actual_nodes_in_subtree = HashSet::new();
                    let mut actual_adj_list = Vec::new();

                    // DFS from node to collect all nodes and edges in its subtree
                    let mut stack = vec![node];
                    actual_nodes_in_subtree.insert(node);

                    while let Some(current) = stack.pop() {
                        for &child in &tree.children[current] {
                            if !actual_nodes_in_subtree.contains(&(child as usize)) {
                                actual_nodes_in_subtree.insert(child as usize);
                                actual_adj_list.push((current, child as usize));
                                stack.push(child as usize);
                            }
                        }
                    }

                    actual_adj_list.sort();

                    // Verify nodes_mask matches actual nodes
                    let mut expected_mask = 0u8;
                    for &n in &actual_nodes_in_subtree {
                        expected_mask |= 1 << n;
                    }

                    assert_eq!(
                        nodes_mask,
                        expected_mask,
                        "Node mask mismatch for subtree at node {} in tree with root {}\n  \
                        Key mask: {:08b}, Actual mask: {:08b}, {:?}",
                        node,
                        tree.root,
                        nodes_mask,
                        expected_mask,
                        tree.print()
                    );

                    // Verify adjacency list matches
                    assert_eq!(
                        adj_list_from_key.len(), actual_adj_list.len(),
                        "Adjacency list length mismatch for subtree at node {} in tree with root {}\n  \
                        Key adj: {:?}, Actual adj: {:?}",
                        node, tree.root, adj_list_from_key, actual_adj_list
                    );

                    for (key_entry, actual_entry) in adj_list_from_key.iter().zip(&actual_adj_list)
                    {
                        assert_eq!(
                            key_entry, actual_entry,
                            "Adjacency list entry mismatch for subtree at node {} in tree with root {}\n  \
                            Key entry: {:?}, Actual entry: {:?}\n  \
                            Full key adj: {:?}, Full actual adj: {:?}\nError in subtree key: {:b} {:?}",
                            node, tree.root, key_entry, actual_entry, adj_list_from_key, actual_adj_list, key, tree.print()
                        );
                    }

                    // 1 00000011 000 000 000 000 000 000 000 000

                    // Verify root matches
                    assert_eq!(
                        root, node,
                        "Root mismatch: key says {}, but subtree is rooted at {}",
                        root, node
                    );

                    let adj_str = format!("{:?}", adj_list_from_key);
                    let subtree_signature = (root, nodes_mask, adj_str.clone());

                    // Check if this exact subtree (same structure) was seen before
                    if let Some(&existing_key) = subtree_to_key.get(&subtree_signature) {
                        if existing_key != key {
                            println!(
                                "ERROR: Same subtree structure has different keys!\n  \
                             Root: {}, Nodes: {:08b}, Adj: {}\n  \
                             Key1: 0x{:016x}, Key2: 0x{:016x}",
                                root, nodes_mask, adj_str, existing_key, key
                            );
                            collision_count += 1;
                        } else {
                            duplicate_subtree_count += 1;
                        }
                    } else {
                        subtree_to_key.insert(subtree_signature.clone(), key);
                    }

                    // Check if this key was seen before with a different subtree
                    if let Some(existing_subtree) = key_to_subtree.get(&key) {
                        if existing_subtree != &subtree_signature {
                            println!(
                                "ERROR: Same key maps to different subtrees!\n  \
                             Key: 0x{:016x}\n  \
                             Subtree1: {:?}\n  \
                             Subtree2: {:?}",
                                key, existing_subtree, subtree_signature
                            );
                            collision_count += 1;
                        }
                    } else {
                        key_to_subtree.insert(key, subtree_signature);
                    }
                }
            }
        }

        println!("\nSubtree Key Statistics:");
        println!("  Total unique keys: {}", key_to_subtree.len());
        println!("  Total unique subtrees: {}", subtree_to_key.len());
        println!(
            "  Duplicate subtrees found (expected): {}",
            duplicate_subtree_count
        );
        println!("  Collisions found (should be 0): {}", collision_count);

        assert_eq!(
            collision_count, 0,
            "Found {} key collisions - same key for different subtrees or vice versa!",
            collision_count
        );

        assert_eq!(
            key_to_subtree.len(),
            subtree_to_key.len(),
            "Mismatch between unique keys and unique subtrees!"
        );
    }
}
