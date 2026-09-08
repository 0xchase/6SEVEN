use tga::{Algorithm, GenerationState, Observation, TargetModel};

fn check<A: Algorithm>(algorithm: A) {
    let observations = [1u8, 2, 3].map(|last| {
        let mut address = [0; 16];
        address[..4].copy_from_slice(&[0x20, 1, 0x0d, 0xb8]);
        address[15] = last;
        Observation {
            address,
            active: true,
        }
    });
    let model = algorithm.train(&observations).unwrap();
    let bytes = A::encode_model(&model).unwrap();
    let mut whole = A::decode_model(A::MODEL_VERSION, &bytes).unwrap();
    let mut split = A::decode_model(A::MODEL_VERSION, &bytes).unwrap();
    assert!(split.generate(&mut []).is_err());
    let mut expected = [[0; 16]; 17];
    let generated = whole.generate(&mut expected).unwrap();
    let mut actual = Vec::new();
    loop {
        let mut output = [[0; 16]; 3];
        let capacity = (17 - actual.len()).min(output.len());
        if capacity == 0 {
            break;
        }
        let next = split.generate(&mut output[..capacity]).unwrap();
        assert!(next.written <= capacity);
        actual.extend_from_slice(&output[..next.written]);
        if next.state != GenerationState::Ready {
            break;
        }
    }
    assert_eq!(actual, expected[..generated.written], "{}", A::ID);
}

#[test]
fn buffer_size_does_not_change_builtin_candidate_order() {
    check(tga::Det {
        delta_base: 16,
        leaf_max: 16,
    });
    check(tga::SixTree::default());
    check(tga::SixGraph::default());
    check(tga::SixForest::default());
    check(
        serde_json::from_value::<tga::SixProbe>(serde_json::json!({
            "beta": 12, "tree_num": 1, "mode": "forest", "dhc_type": "left-vdps",
            "split_array_type": "sequential", "split_order": "right", "random_seed": 0
        }))
        .unwrap(),
    );
    check(
        serde_json::from_value::<tga::SixGen>(serde_json::json!({
            "budget": 32, "range_mode": "Loose", "seed": 0
        }))
        .unwrap(),
    );
    check(tga::EntropyIp::default());
}
