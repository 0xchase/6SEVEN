use rayon::prelude::*;
use std::collections::BTreeMap;

use super::encode::get_nybble_window_value;

pub(super) fn compute_position_entropies(
    nybbles_list: &[[u8; 32]],
    step: usize,
    al: usize,
) -> Vec<f64> {
    let n_addr = nybbles_list.len() as f64;
    (0..al)
        .into_par_iter()
        .map(|pos| {
            let mut counts: BTreeMap<u128, usize> = BTreeMap::new();
            for nybbles in nybbles_list {
                let value = get_nybble_window_value(nybbles, pos * step, step);
                *counts.entry(value).or_insert(0) += 1;
            }
            let mut entropy = 0.0;
            for &count in counts.values() {
                let p = count as f64 / n_addr;
                entropy -= p * p.log2();
            }
            entropy / (4.0 * step as f64)
        })
        .collect()
}

pub(super) fn define_segments(
    entropies: &[f64],
    isp_pos: usize,
    net_pos: usize,
    thresholds: &[f64],
    hysteresis: f64,
) -> Vec<(usize, usize)> {
    let al = entropies.len();
    let mut segments_def = Vec::new();
    let mut last_segment_start = 0usize;
    for i in 1..al {
        if i < isp_pos {
            continue;
        }
        if i == isp_pos || i == net_pos {
            segments_def.push((last_segment_start, i - 1));
            last_segment_start = i;
            continue;
        }

        let mut cross_threshold = false;
        for &t in thresholds {
            if ((entropies[i - 1] < t && entropies[i] >= t)
                || (entropies[i - 1] >= t && entropies[i] < t))
                && (entropies[i] - entropies[i - 1]).abs() > hysteresis
            {
                cross_threshold = true;
                break;
            }
        }
        if cross_threshold {
            segments_def.push((last_segment_start, i - 1));
            last_segment_start = i;
        }
    }
    segments_def.push((last_segment_start, al - 1));
    segments_def
}
