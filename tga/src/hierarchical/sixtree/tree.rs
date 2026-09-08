use super::*;
use std::{collections::VecDeque, ops::Range};

#[derive(Debug, Clone)]
struct BuildNode {
    seed_start: usize,
    seed_end: usize,
    children: Vec<BuildNode>,
    dimension_stack: Vec<usize>,
}

impl BuildNode {
    fn new(seed_start: usize, seed_end: usize) -> Self {
        Self {
            seed_start,
            seed_end,
            children: Vec::new(),
            dimension_stack: Vec::new(),
        }
    }

    fn seed_count(&self) -> usize {
        self.seed_end - self.seed_start
    }

    fn seed_range(&self) -> Range<usize> {
        self.seed_start..self.seed_end
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

fn split_node(node: &mut BuildNode, leaf_max: usize, vectors: &[Address], layout: DigitLayout) {
    if node.seed_count() <= leaf_max {
        return;
    }

    let Some(split_dim) = first_variable_dim(node.seed_range(), vectors, layout) else {
        return;
    };

    let mut start = node.seed_start;
    while start < node.seed_end {
        let mut end = start + 1;
        while end < node.seed_end
            && layout.digit(vectors[end], split_dim) == layout.digit(vectors[start], split_dim)
        {
            end += 1;
        }
        node.children.push(BuildNode::new(start, end));
        start = end;
    }

    visit_children_mut(node, |child| split_node(child, leaf_max, vectors, layout));
}

fn first_variable_dim(
    seed_range: Range<usize>,
    vectors: &[Address],
    layout: DigitLayout,
) -> Option<usize> {
    if seed_range.len() <= 1 {
        return None;
    }

    (0..layout.dimensions()).find(|&dim| !is_steady(seed_range.clone(), dim, vectors, layout))
}

fn initialize_ds(
    node: &mut BuildNode,
    parent_ds: &[usize],
    vectors: &[Address],
    layout: DigitLayout,
) {
    let mut ds = parent_ds.to_vec();

    for dim in 0..layout.dimensions() {
        if !ds.contains(&dim) && is_steady(node.seed_range(), dim, vectors, layout) {
            ds.push(dim);
        }
    }

    if node.is_leaf() {
        for dim in 0..layout.dimensions() {
            if !ds.contains(&dim) {
                ds.push(dim);
            }
        }
    } else {
        visit_children_mut(node, |child| initialize_ds(child, &ds, vectors, layout));
    }

    node.dimension_stack = ds;
}

fn visit_children_mut<F>(node: &mut BuildNode, f: F)
where
    F: Fn(&mut BuildNode) + Send + Sync,
{
    if node.seed_count() >= PARALLEL_NODE_SEED_THRESHOLD {
        node.children.par_iter_mut().for_each(f);
    } else {
        node.children.iter_mut().for_each(f);
    }
}

fn is_steady(
    seed_range: Range<usize>,
    dim: usize,
    vectors: &[Address],
    layout: DigitLayout,
) -> bool {
    let first = layout.digit(vectors[seed_range.start], dim);
    vectors[seed_range]
        .iter()
        .all(|vector| layout.digit(*vector, dim) == first)
}

pub(super) fn build_model(
    vectors: &[Address],
    leaf_max: usize,
    batch_percent: u8,
    layout: DigitLayout,
) -> SixTreeModel {
    let mut nodes = Vec::new();
    let mut roots = Vec::new();
    let mut start = 0;
    while start < vectors.len() {
        let mut end = start + 1;
        // Omitted leading bits remain fixed outside the address vectors.
        while end < vectors.len() && layout.scope(vectors[start]) == layout.scope(vectors[end]) {
            end += 1;
        }
        let mut root = BuildNode::new(start, end);
        split_node(&mut root, leaf_max, vectors, layout);
        initialize_ds(&mut root, &[], vectors, layout);
        roots.push(flatten_tree(&root, None, vectors, &mut nodes, layout));
        start = end;
    }
    let current_batch = initial_leaf_queue(&nodes, roots);
    let mut model = SixTreeModel {
        nodes,
        current_batch,
        batch_percent,
        ..Default::default()
    };
    model.replace_descendants();
    model
}

fn flatten_tree(
    node: &BuildNode,
    parent: Option<NodeId>,
    vectors: &[Address],
    out: &mut Vec<SixTreeNode>,
    layout: DigitLayout,
) -> NodeId {
    let node_id = out.len();
    out.push(SixTreeNode::new(parent, node.dimension_stack.clone()));
    out[node_id].layout = layout;

    if node.is_leaf() {
        let leaf = &mut out[node_id];
        leaf.targets = RegionSet::from_regions(
            vectors[node.seed_range()]
                .iter()
                .copied()
                .map(Region::singleton),
        );
        leaf.expand();
    } else {
        let children = node
            .children
            .iter()
            .map(|child| flatten_tree(child, Some(node_id), vectors, out, layout))
            .collect();
        out[node_id].children = children;
    }

    node_id
}

fn initial_leaf_queue(nodes: &[SixTreeNode], roots: Vec<NodeId>) -> Vec<NodeId> {
    let mut queue = VecDeque::from(roots);
    let mut leaves = Vec::new();

    while let Some(node_id) = queue.pop_front() {
        let node = &nodes[node_id];
        if node.children.is_empty() {
            leaves.push(node_id);
        } else {
            queue.extend(node.children.iter().copied());
        }
    }

    leaves
}
