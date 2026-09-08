use super::*;
use crate::GenerationState;

fn train_dense() -> SixTreeModel {
    let mut model = SixTree::default()
        .train(&[Observation {
            address: 1u128.to_be_bytes(),
            active: true,
        }])
        .unwrap();
    for _ in 0..2 {
        let feedback: Vec<_> = model
            .batch_frontier()
            .cursor()
            .map(|address| Feedback::Active(address.into()))
            .collect();
        model.apply_feedback(&feedback).unwrap();
    }
    assert!(matches!(model.phase, SearchPhase::Detecting(_)));
    model
}

fn probe_feedback(model: &SixTreeModel, active: bool) -> Vec<Feedback> {
    model
        .batch_frontier()
        .cursor()
        .enumerate()
        .map(|(index, address)| {
            if active && index == 0 {
                Feedback::Active(address.into())
            } else {
                Feedback::Inactive(address.into())
            }
        })
        .collect()
}

fn expand_detection_to(model: &mut SixTreeModel, dimensions: u32) {
    while model.nodes[model.current_batch[0]]
        .targets
        .regions()
        .next()
        .unwrap()
        .wildcard_dimensions(model.nodes[model.current_batch[0]].layout)
        < dimensions
    {
        model.apply_feedback(&probe_feedback(model, true)).unwrap();
        assert!(matches!(model.phase, SearchPhase::Detecting(_)));
    }
}

#[test]
fn equation_nine_triggers_only_after_dense_scans() {
    let model = train_dense();
    let node = &model.nodes[model.current_batch[0]];
    assert_eq!(node.probe_count(), 256);
    assert_eq!(node.active.len(), 256);
    assert_eq!(node.targets.size_capped(), 4096);
    assert!(node.is_abnormal());
    let mut node = node.clone();
    node.active = node.active.into_iter().take(188).collect();
    assert!(!node.is_abnormal());
    node.active.insert(u128::MAX.to_be_bytes());
    assert!(node.is_abnormal());
}

#[test]
fn sampled_probes_balance_the_last_dimension_and_preserve_fixed_bits() {
    let model = train_dense();
    let node = &model.nodes[model.current_batch[0]];
    let round = AliasProbeRound::sample(node, 0, &RegionSet::default());
    let mut counts = [0; 16];
    for address in round.pending() {
        assert!(node.targets.contains(address));
        counts[usize::from(node.layout.digit(address, node.last_expanded.unwrap()))] += 1;
    }
    assert!(counts.iter().all(|&count| (1..=10).contains(&count)));
    assert!(
        round
            .pending()
            .any(|address| node.scanned.contains(address))
    );
    let same = AliasProbeRound::sample(node, 0, &RegionSet::default());
    let different = AliasProbeRound::sample(node, 1, &RegionSet::default());
    assert_eq!(
        round.pending().collect::<Vec<_>>(),
        same.pending().collect::<Vec<_>>()
    );
    assert_ne!(
        round.pending().collect::<Vec<_>>(),
        different.pending().collect::<Vec<_>>()
    );
}

#[test]
fn any_positive_probe_expands_without_changing_scan_density() {
    let mut model = train_dense();
    let feedback = probe_feedback(&model, true);
    model.apply_feedback(&feedback[..1]).unwrap();
    model.apply_feedback(&feedback[..1]).unwrap();
    assert_eq!(model.nodes[0].targets.size_capped(), 4096);
    model.apply_feedback(&feedback[1..]).unwrap();
    assert_eq!(model.nodes[0].targets.size_capped(), 65536);
    assert_eq!(model.nodes[0].probe_count(), 256);
    assert_eq!(model.nodes[0].active.len(), 256);
    assert_eq!(model.nodes[0].scanned.size_capped(), 256);
    assert!(matches!(model.phase, SearchPhase::Detecting(_)));
    assert!(model.detected_aliases().is_empty());
}

#[test]
fn positive_feedback_overrides_an_earlier_negative_before_the_round_completes() {
    let mut model = train_dense();
    let feedback = probe_feedback(&model, false);
    let Feedback::Inactive(address) = feedback[0] else {
        panic!("inactive probe");
    };
    model.apply_feedback(&feedback[..1]).unwrap();
    model.apply_feedback(&[Feedback::Active(address)]).unwrap();
    model.apply_feedback(&feedback[1..]).unwrap();
    assert!(matches!(model.phase, SearchPhase::Detecting(_)));
    assert_eq!(model.nodes[0].targets.size_capped(), 65536);
}

#[test]
fn external_aliases_remove_previously_received_probe_responses() {
    let mut model = train_dense();
    let feedback = probe_feedback(&model, false);
    let Feedback::Inactive(address) = feedback[0] else {
        panic!("inactive probe");
    };
    model.apply_feedback(&[Feedback::Active(address)]).unwrap();
    model
        .apply_feedback(&[Feedback::Aliased(Ipv6Prefix::new(address, 128).unwrap())])
        .unwrap();
    model.apply_feedback(&feedback[1..]).unwrap();
    assert!(matches!(model.phase, SearchPhase::Scanning));
    assert_eq!(model.nodes[0].targets.size_capped(), 4096);
    assert!(model.detected_aliases().is_empty());
}

#[test]
fn small_unresponsive_regions_resume_normal_scanning() {
    let mut model = train_dense();
    model
        .apply_feedback(&probe_feedback(&model, false))
        .unwrap();
    assert!(matches!(model.phase, SearchPhase::Scanning));
    assert_eq!(model.batch_frontier().size_capped(), 3840);
    assert!(model.detected_aliases().is_empty());
    assert_eq!(model.nodes[0].probe_count(), 256);
}

#[test]
fn alias_threshold_is_strictly_greater_than_two_to_the_twentieth() {
    let mut model = train_dense();
    expand_detection_to(&mut model, 5);
    model
        .apply_feedback(&probe_feedback(&model, false))
        .unwrap();
    assert!(matches!(model.phase, SearchPhase::Scanning));
    assert!(model.detected_aliases().is_empty());
    assert_eq!(model.nodes[0].targets.size_capped(), 1 << 20);
}

#[test]
fn large_unresponsive_regions_record_prefixes_and_lower_queue_priority() {
    let mut model = train_dense();
    expand_detection_to(&mut model, 6);
    let mut other = SixTreeNode::new(None, vec![0]);
    other.targets = RegionSet::from_regions([Region::singleton(u128::MAX.to_be_bytes())]);
    other.probed = RegionSet::from_regions([Region::singleton(2u128.to_be_bytes())]);
    other.active.insert(2u128.to_be_bytes());
    model.nodes.push(other);
    model.queue.push(1);
    model
        .apply_feedback(&probe_feedback(&model, false))
        .unwrap();
    assert_eq!(
        model.detected_aliases(),
        &["::/104".parse::<Ipv6Prefix>().unwrap()]
    );
    assert_eq!(model.current_batch, vec![1]);
    assert_eq!(model.queue, vec![0]);
    assert_eq!(model.nodes[0].density(), 1.0);
    assert_eq!(model.nodes[0].scanned.size_capped(), 256);
}

#[test]
fn skipped_or_excluded_probes_cannot_establish_aliases() {
    for excluded in [false, true] {
        let mut model = train_dense();
        expand_detection_to(&mut model, 6);
        let mut feedback = probe_feedback(&model, false);
        let Feedback::Inactive(address) = feedback[0] else {
            panic!("inactive probe");
        };
        feedback[0] = if excluded {
            Feedback::Aliased(Ipv6Prefix::new(address, 128).unwrap())
        } else {
            Feedback::Skipped(address)
        };
        model.apply_feedback(&feedback).unwrap();
        assert!(model.detected_aliases().is_empty());
        assert!(matches!(model.phase, SearchPhase::Scanning));
    }
}

#[test]
fn alias_probe_progress_survives_checkpoints_and_feedback_chunks() {
    let mut model = train_dense();
    let mut output = [[0; 16]; 17];
    model.generate(&mut output).unwrap();
    let first = output.to_vec();
    model
        .apply_feedback(&[Feedback::Active(first[0].into())])
        .unwrap();
    let bytes = SixTree::encode_model(&model).unwrap();
    let mut restored = SixTree::decode_model(SixTree::MODEL_VERSION, &bytes).unwrap();
    let mut probes = Vec::new();
    loop {
        let batch = model.generate(&mut output).unwrap();
        let mut copy = [[0; 16]; 17];
        let copied = restored.generate(&mut copy).unwrap();
        assert_eq!(batch.written, copied.written);
        assert_eq!(batch.state, copied.state);
        assert_eq!(output[..batch.written], copy[..copied.written]);
        probes.extend_from_slice(&output[..batch.written]);
        if batch.state == GenerationState::AwaitingFeedback {
            break;
        }
    }
    assert!(probes.iter().all(|address| !first.contains(address)));
    let feedback: Vec<_> = first[1..]
        .iter()
        .chain(&probes)
        .map(|&address| Feedback::Inactive(address.into()))
        .collect();
    for chunk in feedback.chunks(13) {
        restored.apply_feedback(chunk).unwrap();
    }
    model.apply_feedback(&feedback).unwrap();
    assert_eq!(model.nodes[0].targets.size_capped(), 65536);
    assert_eq!(
        model.batch_frontier().cursor().collect::<Vec<_>>(),
        restored.batch_frontier().cursor().collect::<Vec<_>>()
    );
}

#[test]
fn alias_expansion_promotes_the_parent_and_retires_deeper_descendants() {
    let mut model = train_dense();
    let mut parent = SixTreeNode::new(None, (0..28).collect());
    parent.children = vec![1, 2];
    let mut child = model.nodes[0].clone();
    child.parent = Some(0);
    let mut branch = SixTreeNode::new(Some(0), (0..29).collect());
    branch.children = vec![3];
    let mut grandchild = SixTreeNode::new(Some(2), (0..30).collect());
    grandchild.scanned = RegionSet::from_regions([Region::singleton(4096u128.to_be_bytes())]);
    grandchild.probed = grandchild.scanned.clone();
    grandchild.active.insert(4096u128.to_be_bytes());
    model.nodes = vec![parent, child, branch, grandchild];
    model.current_batch = vec![1];
    model.queue = vec![3];
    model.start_alias_round();
    model.apply_feedback(&probe_feedback(&model, true)).unwrap();
    assert_eq!(model.current_batch, vec![0]);
    assert!(model.queue.is_empty());
    assert_eq!(model.nodes[0].last_expanded, Some(28));
    assert_eq!(model.nodes[0].probe_count(), 257);
    assert_eq!(model.nodes[0].active.len(), 257);
    assert_eq!(model.nodes[0].targets.size_capped(), 65536);
    assert!(matches!(model.phase, SearchPhase::Detecting(_)));
}

#[test]
fn responsive_full_space_stops_expanding_without_claiming_a_negative_boundary() {
    let mut model = train_dense();
    model.nodes[0].targets = RegionSet::from_regions([Region::prefix("::/0".parse().unwrap())]);
    model.nodes[0].dimension_stack.clear();
    model.nodes[0].last_expanded = Some(0);
    model.start_alias_round();
    model.apply_feedback(&probe_feedback(&model, true)).unwrap();
    assert!(matches!(model.phase, SearchPhase::Scanning));
    assert!(model.detected_aliases().is_empty());
    let generated = model.generate(&mut [[0; 16]; 4]).unwrap();
    assert_eq!(generated.written, 4);
}

#[test]
fn prefix_translation_starts_at_the_first_wildcard() {
    let mut first = Region::singleton(
        "2001:db8:abcd:1234:5678:9abc:def0:1234"
            .parse::<std::net::Ipv6Addr>()
            .unwrap()
            .octets(),
    );
    first.expand(12, DigitLayout::default());
    first.expand(31, DigitLayout::default());
    assert_eq!(
        first.covering_prefix(),
        "2001:db8:abcd::/48".parse::<Ipv6Prefix>().unwrap()
    );
    let mut second = Region::singleton(
        "3001:db8:abcd:1234:5678:9abc:def0:1234"
            .parse::<std::net::Ipv6Addr>()
            .unwrap()
            .octets(),
    );
    second.expand(24, DigitLayout::default());
    assert_eq!(
        second.covering_prefix(),
        "3001:db8:abcd:1234:5678:9abc::/96"
            .parse::<Ipv6Prefix>()
            .unwrap()
    );
}

#[test]
fn detection_is_enabled_by_default_and_can_be_disabled_for_reference_baselines() {
    assert!(SixTree::default().alias_detection);
    let config = SixTree {
        alias_detection: false,
        ..Default::default()
    };
    let mut model = config
        .train(&[Observation {
            address: 1u128.to_be_bytes(),
            active: true,
        }])
        .unwrap();
    for _ in 0..2 {
        let feedback: Vec<_> = model
            .batch_frontier()
            .cursor()
            .map(|address| Feedback::Active(address.into()))
            .collect();
        model.apply_feedback(&feedback).unwrap();
    }
    assert!(matches!(model.phase, SearchPhase::Scanning));
    assert_eq!(model.batch_frontier().size_capped(), 3840);
}

#[test]
fn density_trigger_and_probe_balance_follow_the_selected_base() {
    for base in [2, 4, 8, 16, 32] {
        let mut model = SixTree {
            base,
            ..Default::default()
        }
        .train(&[Observation {
            address: 1u128.to_be_bytes(),
            active: true,
        }])
        .unwrap();
        while matches!(model.phase, SearchPhase::Scanning) {
            let feedback: Vec<_> = model
                .batch_frontier()
                .cursor()
                .map(|address| Feedback::Active(address.into()))
                .collect();
            assert!(!feedback.is_empty());
            model.apply_feedback(&feedback).unwrap();
        }
        let node = &model.nodes[model.current_batch[0]];
        let dimensions = node
            .targets
            .regions()
            .next()
            .unwrap()
            .wildcard_dimensions(node.layout);
        assert_eq!(dimensions, 10u32.div_ceil(base.trailing_zeros()));
        assert_eq!(node.layout.radix(), usize::from(base));
        let mut counts = vec![0; usize::from(base)];
        for address in model.batch_frontier().cursor() {
            counts[usize::from(node.layout.digit(address, node.last_expanded.unwrap()))] += 1;
        }
        assert!(counts.iter().all(|&count| (1..=10).contains(&count)));
        let original_size = node.targets.size_capped();
        model.apply_feedback(&probe_feedback(&model, true)).unwrap();
        assert_eq!(
            model.nodes[model.current_batch[0]].targets.size_capped(),
            original_size * usize::from(base)
        );
    }
}
