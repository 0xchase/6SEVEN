use crate::Address;
use rand::{
    Rng, SeedableRng,
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
};

const DIMENSIONS: usize = 100;
const WINDOW: usize = 5;
const EPOCHS: usize = 5;
const NEGATIVES: usize = 5;
const EPSILON: f64 = 0.0085;

pub(super) fn classify(
    seeds: &[Address],
    batch_size: usize,
    seed: u64,
) -> Result<Vec<String>, String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut counts = [0usize; 512];
    let sentences = seeds
        .iter()
        .map(|address| {
            std::array::from_fn::<_, 32, _>(|position| {
                let byte = address[position / 2];
                let digit = if position % 2 == 0 {
                    byte >> 4
                } else {
                    byte & 15
                };
                let token = position * 16 + digit as usize;
                counts[token] += 1;
                token
            })
        })
        .collect::<Vec<_>>();
    let vocabulary = (0..512)
        .filter(|&token| counts[token] > 0)
        .collect::<Vec<_>>();
    let embeddings = cbow(&sentences, &counts, &mut rng);
    let words = vocabulary
        .iter()
        .map(|&token| embeddings[token].clone())
        .collect::<Vec<_>>();
    let projected = tsne(&words, &mut rng);
    let mut positions = [[0.0; 2]; 512];
    for (&token, point) in vocabulary.iter().zip(projected) {
        positions[token] = point;
    }
    let addresses = sentences
        .iter()
        .map(|sentence| {
            let mut point = [0.0; 2];
            for &token in sentence {
                point[0] += positions[token][0] / 32.0;
                point[1] += positions[token][1] / 32.0;
            }
            point
        })
        .collect::<Vec<_>>();
    let labels = dbscan(&addresses, EPSILON, batch_size);
    let count = labels.iter().copied().max().unwrap_or(-1) + 1;
    if count > 10 {
        return Err(format!(
            "IPv62Vec produced {count} clusters, exceeding the reference limit of 10"
        ));
    }
    Ok(labels
        .into_iter()
        .map(|label| format!("cluster_{label}"))
        .collect())
}

fn cbow(sentences: &[[usize; 32]], counts: &[usize; 512], rng: &mut StdRng) -> Vec<Vec<f64>> {
    let mut input = (0..512)
        .map(|_| {
            (0..DIMENSIONS)
                .map(|_| rng.gen_range(-0.005..0.005))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut output = vec![vec![0.0; DIMENSIONS]; 512];
    let negatives = WeightedIndex::new(counts.iter().map(|&count| (count as f64).powf(0.75)))
        .expect("IPv62Vec corpus is nonempty");
    let total = counts.iter().sum::<usize>() as f64;
    let keep = counts.map(|count| {
        let frequency = count as f64 / total;
        if count == 0 {
            0.0
        } else {
            ((frequency / 0.001).sqrt() + 1.0) * 0.001 / frequency
        }
    });
    for epoch in 0..EPOCHS {
        for (row_index, row) in sentences.iter().enumerate() {
            let progress =
                (epoch * sentences.len() + row_index) as f64 / (EPOCHS * sentences.len()) as f64;
            let rate = (0.025 * (1.0 - progress)).max(0.0001);
            let row = row
                .iter()
                .copied()
                .filter(|&token| rng.gen_range(0.0..1.0) < keep[token])
                .collect::<Vec<_>>();
            for (position, &target) in row.iter().enumerate() {
                let window = rng.gen_range(1..=WINDOW);
                let context = (position.saturating_sub(window)
                    ..(position + window + 1).min(row.len()))
                    .filter(|&index| index != position)
                    .map(|index| row[index])
                    .collect::<Vec<_>>();
                if context.is_empty() {
                    continue;
                }
                let mut hidden = vec![0.0; DIMENSIONS];
                for &token in &context {
                    for (value, weight) in hidden.iter_mut().zip(&input[token]) {
                        *value += weight / context.len() as f64;
                    }
                }
                let mut error = vec![0.0; DIMENSIONS];
                for attempt in 0..=NEGATIVES {
                    let (token, label) = if attempt == 0 {
                        (target, 1.0)
                    } else {
                        (negatives.sample(rng), 0.0)
                    };
                    if attempt > 0 && token == target {
                        continue;
                    }
                    let dot = hidden
                        .iter()
                        .zip(&output[token])
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
                    let gradient = (label - 1.0 / (1.0 + (-dot).exp())) * rate;
                    for ((error, weight), value) in
                        error.iter_mut().zip(&mut output[token]).zip(&hidden)
                    {
                        *error += gradient * *weight;
                        *weight += gradient * value;
                    }
                }
                for token in context {
                    for (weight, error) in input[token].iter_mut().zip(&error) {
                        *weight += error;
                    }
                }
            }
        }
    }
    input
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum()
}

fn joint_probabilities(words: &[Vec<f64>]) -> Vec<f64> {
    let n = words.len();
    let target_entropy = 30.0f64.ln();
    let mut probabilities = vec![0.0; n * n];
    for i in 0..n {
        let distances = words
            .iter()
            .map(|word| squared_distance(&words[i], word))
            .collect::<Vec<_>>();
        let (mut beta, mut lower, mut upper) = (1.0, 0.0, f64::INFINITY);
        for _ in 0..60 {
            let mut sum = 0.0;
            let mut weighted = 0.0;
            for j in 0..n {
                let value = if i == j {
                    0.0
                } else {
                    (-distances[j] * beta).exp()
                };
                probabilities[i * n + j] = value;
                sum += value;
                weighted += value * distances[j];
            }
            let entropy = sum.max(1e-300).ln() + beta * weighted / sum.max(1e-300);
            for j in 0..n {
                probabilities[i * n + j] /= sum.max(1e-300);
            }
            if (entropy - target_entropy).abs() < 1e-5 {
                break;
            }
            if entropy > target_entropy {
                lower = beta;
                beta = if upper.is_infinite() {
                    beta * 2.0
                } else {
                    (beta + upper) / 2.0
                };
            } else {
                upper = beta;
                beta = (beta + lower) / 2.0;
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            let joint =
                ((probabilities[i * n + j] + probabilities[j * n + i]) / (2 * n) as f64).max(1e-12);
            probabilities[i * n + j] = joint;
            probabilities[j * n + i] = joint;
        }
    }
    probabilities
}

fn tsne(words: &[Vec<f64>], rng: &mut StdRng) -> Vec<[f64; 2]> {
    let n = words.len();
    let probabilities = joint_probabilities(words);
    let mut points = (0..n)
        .map(|_| {
            let radius = (-2.0 * rng.gen_range(f64::MIN_POSITIVE..1.0).ln()).sqrt() * 1e-4;
            let angle = rng.gen_range(0.0..std::f64::consts::TAU);
            [radius * angle.cos(), radius * angle.sin()]
        })
        .collect::<Vec<_>>();
    let mut velocities = vec![[0.0f64; 2]; n];
    let mut gains = vec![[1.0f64; 2]; n];
    let mut affinities = vec![0.0; n * n];
    for iteration in 0..1000 {
        if iteration == 250 {
            velocities.fill([0.0; 2]);
            gains.fill([1.0; 2]);
        }
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..i {
                let value = 1.0 / (1.0 + squared_distance(&points[i], &points[j]));
                affinities[i * n + j] = value;
                affinities[j * n + i] = value;
                sum += 2.0 * value;
            }
        }
        let mut gradients = vec![[0.0; 2]; n];
        let exaggeration = if iteration < 250 { 12.0 } else { 1.0 };
        for i in 0..n {
            for j in 0..n {
                let q = affinities[i * n + j];
                let factor = 4.0 * (exaggeration * probabilities[i * n + j] - q / sum) * q;
                for axis in 0..2 {
                    gradients[i][axis] += factor * (points[i][axis] - points[j][axis]);
                }
            }
        }
        let momentum = if iteration < 250 { 0.5 } else { 0.8 };
        let mut center = [0.0; 2];
        for i in 0..n {
            for (axis, center) in center.iter_mut().enumerate() {
                let gain = &mut gains[i][axis];
                *gain = if velocities[i][axis] * gradients[i][axis] < 0.0 {
                    *gain + 0.2
                } else {
                    (*gain * 0.8).max(0.01)
                };
                velocities[i][axis] =
                    momentum * velocities[i][axis] - 200.0 * *gain * gradients[i][axis];
                points[i][axis] += velocities[i][axis];
                *center += points[i][axis] / n as f64;
            }
        }
        for point in &mut points {
            for axis in 0..2 {
                point[axis] -= center[axis];
            }
        }
    }
    points
}

fn dbscan(points: &[[f64; 2]], epsilon: f64, min_samples: usize) -> Vec<i32> {
    use std::collections::BTreeMap;
    let cell = |point: &[f64; 2]| {
        (
            (point[0] / epsilon).floor() as i64,
            (point[1] / epsilon).floor() as i64,
        )
    };
    let mut grid = BTreeMap::<_, Vec<usize>>::new();
    for (index, point) in points.iter().enumerate() {
        grid.entry(cell(point)).or_default().push(index);
    }
    let neighbors = |index: usize| {
        let (x, y) = cell(&points[index]);
        let mut found = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(indices) = grid.get(&(x + dx, y + dy)) {
                    found.extend(indices.iter().copied().filter(|&other| {
                        squared_distance(&points[index], &points[other]) <= epsilon * epsilon
                    }));
                }
            }
        }
        found.sort_unstable();
        found
    };
    let mut labels = vec![-2; points.len()];
    let mut cluster = 0;
    for index in 0..points.len() {
        if labels[index] != -2 {
            continue;
        }
        let mut queue = neighbors(index);
        if queue.len() < min_samples {
            labels[index] = -1;
            continue;
        }
        labels[index] = cluster;
        let mut queued = vec![false; points.len()];
        for &next in &queue {
            queued[next] = true;
        }
        let mut cursor = 0;
        while cursor < queue.len() {
            let next = queue[cursor];
            cursor += 1;
            if labels[next] == -1 {
                labels[next] = cluster;
            }
            if labels[next] != -2 {
                continue;
            }
            labels[next] = cluster;
            let adjacent = neighbors(next);
            if adjacent.len() >= min_samples {
                for neighbor in adjacent {
                    if !queued[neighbor] {
                        queued[neighbor] = true;
                        queue.push(neighbor);
                    }
                }
            }
        }
        cluster += 1;
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affinities_match_sklearn_reference() {
        let words = (0..32)
            .map(|i| vec![i as f64 / 32.0, (i % 5) as f64 / 5.0])
            .collect::<Vec<_>>();
        let probabilities = joint_probabilities(&words);
        let expected = [
            (0, 1, 0.0013957485453459531),
            (0, 31, 0.0006631141074134336),
            (7, 17, 0.0011980198033008046),
            (10, 20, 0.0012739768073165535),
            (30, 31, 0.0014120383484636785),
        ];
        for (i, j, expected) in expected {
            assert!((probabilities[i * 32 + j] - expected).abs() < 1e-9);
        }
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn density_clusters_preserve_noise_and_duplicates() {
        let points = [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.005, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [3.0, 3.0],
        ];
        assert_eq!(dbscan(&points, EPSILON, 2), vec![0, 0, 0, 1, 1, -1]);
    }

    #[test]
    fn vector_classification_is_repeatable() {
        let seeds = [[0; 16], [0; 16]];
        assert_eq!(
            classify(&seeds, 2, 42).unwrap(),
            classify(&seeds, 2, 42).unwrap()
        );
    }
}
