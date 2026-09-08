use sixseven_core::{
    Address, BitRange, Generated, GenerationOptions, GenerationState, TargetModel, TgaError,
    generation::generate,
};

struct Source {
    left: usize,
}
impl TargetModel for Source {
    fn generate(&mut self, output: &mut [Address]) -> Result<Generated, TgaError> {
        sixseven_core::generation::validate_output(output)?;
        let count = self.left.min(output.len());
        self.left -= count;
        output[..count].fill([1; 16]);
        Ok(Generated {
            written: count,
            state: if self.left == 0 {
                GenerationState::Exhausted
            } else {
                GenerationState::Ready
            },
        })
    }
}

fn options(count: usize, unique: bool, max_attempts: u64) -> GenerationOptions {
    GenerationOptions {
        count,
        unique,
        max_attempts,
        exclude: Vec::new(),
    }
}

#[test]
fn duplicate_support_stops_at_attempt_budget() {
    let mut source = Source { left: usize::MAX };
    let mut emitted = 0;
    let error = generate(&mut source, options(2, true, 5), |_| {
        emitted += 1;
        Ok(())
    })
    .unwrap_err();
    assert_eq!(emitted, 1);
    assert!(error.to_string().contains("attempt budget"));
}

#[test]
fn finite_generation_does_not_restart() {
    let mut source = Source { left: 1 };
    let mut emitted = 0;
    assert!(
        generate(&mut source, options(2, false, 10), |_| {
            emitted += 1;
            Ok(())
        })
        .is_err()
    );
    assert_eq!(emitted, 1);
}

#[test]
fn non_unique_output_does_not_claim_unique_count() {
    let stats = generate(&mut Source { left: 2 }, options(2, false, 10), |_| Ok(())).unwrap();
    assert_eq!(stats.written, 2);
    assert_eq!(stats.unique, None);
}

#[test]
fn bit_ranges_validate_after_deserialization() {
    assert!(serde_json::from_str::<BitRange>("[0,129]").is_err());
    assert!(serde_json::from_str::<BitRange>("[4,4]").is_err());
    assert_eq!(
        serde_json::from_str::<BitRange>("[0,128]").unwrap().end(),
        128
    );
}

#[test]
fn algorithm_ids_validate_namespace_components() {
    for value in ["", "/", "a/", "/a", "a//b", "-a", "a-", "a b"] {
        assert!(sixseven_core::AlgorithmId::new(value).is_err());
    }
    assert_eq!(
        sixseven_core::AlgorithmId::new("Example/My-Tga")
            .unwrap()
            .as_str(),
        "example/my-tga"
    );
}

#[test]
fn zero_count_has_known_unique_count_when_requested() {
    let stats = generate(&mut Source { left: 1 }, options(0, true, 0), |_| Ok(())).unwrap();
    assert_eq!(stats.unique, Some(0));
}

#[test]
fn an_empty_buffer_does_not_advance_the_source() {
    let mut source = Source { left: 3 };
    assert!(source.generate(&mut []).is_err());
    assert_eq!(source.left, 3);
}

#[test]
fn buffers_can_be_reused_for_partial_results() {
    let mut source = Source { left: 5 };
    let mut output = [[0; 16]; 3];
    let first = source.generate(&mut output).unwrap();
    assert_eq!(first.written, 3);
    assert_eq!(first.state, GenerationState::Ready);
    output.fill([42; 16]);
    let second = source.generate(&mut output).unwrap();
    assert_eq!(second.written, 2);
    assert_eq!(second.state, GenerationState::Exhausted);
    assert_eq!(output[..second.written], [[1; 16]; 2]);
}

#[test]
fn invalid_generated_counts_are_rejected() {
    use sixseven_core::generation::validate_generated;
    assert!(
        validate_generated(
            Generated {
                written: 2,
                state: GenerationState::Ready
            },
            1
        )
        .is_err()
    );
    assert!(
        validate_generated(
            Generated {
                written: 0,
                state: GenerationState::Ready
            },
            1
        )
        .is_err()
    );
    assert!(
        validate_generated(
            Generated {
                written: 0,
                state: GenerationState::Exhausted
            },
            1
        )
        .is_ok()
    );
}

#[test]
fn prefix_exclusions_consume_attempts_without_consuming_target_budget() {
    use sixseven_core::{GenerationOptions, PrefixSet, generation::CandidateFilter};
    let mut filter = CandidateFilter::new(GenerationOptions {
        count: 1,
        max_attempts: 2,
        unique: true,
        exclude: vec![],
    });
    let aliases: PrefixSet = ["::/0".parse().unwrap()].into_iter().collect();
    assert!(!filter.accept_excluding([0; 16], &aliases).unwrap());
    assert!(!filter.accept_excluding([0; 16], &aliases).unwrap());
    assert!(filter.accept_excluding([0; 16], &aliases).is_err());
    assert_eq!(filter.remaining(), 1);
    assert_eq!(filter.stats.excluded, 2);
}
