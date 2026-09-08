use super::{TOTAL_LEN, TokenCorpus, TrainBackend};
use burn::{
    module::{AutodiffModule, Initializer, Module},
    nn::{Embedding, EmbeddingConfig},
    optim::{GradientsParams, Optimizer, SgdConfig},
    tensor::{Int, Tensor, TensorData, activation, backend::Backend},
};

pub(super) struct Word2VecConfig {
    pub embedding_dim: usize,
    pub window_size: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub min_learning_rate: f64,
    pub seed: u64,
}

#[derive(Module, Debug)]
struct SkipGram<B: Backend> {
    input: Embedding<B>,
    output: Embedding<B>,
}

impl<B: Backend> SkipGram<B> {
    fn new(vocabulary: usize, dimension: usize, device: &B::Device) -> Self {
        let bound = 0.5 / dimension as f64;
        Self {
            input: EmbeddingConfig::new(vocabulary, dimension)
                .with_initializer(Initializer::Uniform {
                    min: -bound,
                    max: bound,
                })
                .init(device),
            output: EmbeddingConfig::new(vocabulary, dimension)
                .with_initializer(Initializer::Zeros)
                .init(device),
        }
    }

    fn loss(&self, centers: Tensor<B, 2, Int>, contexts: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let [batch, _] = centers.dims();
        let [_, dimension] = self.input.weight.val().dims();
        let hidden = self.input.forward(centers).reshape([batch, dimension]);
        let logits = hidden.matmul(self.output.weight.val().transpose());
        activation::log_softmax(logits, 1)
            .gather(1, contexts)
            .neg()
            .mean()
    }
}

struct Vocabulary {
    tokens: Vec<usize>,
    indices: Vec<usize>,
}

impl Vocabulary {
    fn from_counts(counts: &[usize]) -> Self {
        let tokens: Vec<_> = counts
            .iter()
            .enumerate()
            .filter_map(|(token, &count)| (count > 0).then_some(token))
            .collect();
        let mut indices = vec![usize::MAX; counts.len()];
        for (index, &token) in tokens.iter().enumerate() {
            indices[token] = index;
        }
        Self { tokens, indices }
    }

    fn encode(
        &self,
        tokens: Vec<usize>,
        device: &<TrainBackend as Backend>::Device,
    ) -> Tensor<TrainBackend, 2, Int> {
        let batch = tokens.len();
        let indices: Vec<_> = tokens
            .into_iter()
            .map(|token| self.indices[token] as i64)
            .collect();
        Tensor::from_data(TensorData::new(indices, [batch, 1]), device)
    }
}

pub(super) fn train_ipv62vec(
    config: &Word2VecConfig,
    corpus: &TokenCorpus,
    token_counts: &[usize],
) -> Result<Vec<f32>, String> {
    let _rng = crate::ml::rng_guard();
    let vocabulary = Vocabulary::from_counts(token_counts);
    if corpus.is_empty() || vocabulary.tokens.len() < 2 {
        return Err("IPv62Vec requires a nonempty corpus with at least two address words".into());
    }
    let pairs_per_row: usize = (0..TOTAL_LEN)
        .map(|center| {
            let radius = config.window_size / 2;
            center.min(radius) + (TOTAL_LEN - center - 1).min(radius)
        })
        .sum();
    let pairs_per_epoch = pairs_per_row
        .checked_mul(corpus.len())
        .ok_or("IPv62Vec training pair count overflow")?;
    let batches = pairs_per_epoch.div_ceil(config.batch_size);
    let total_steps = batches
        .checked_mul(config.epochs)
        .ok_or("IPv62Vec training step count overflow")?;
    let device = crate::ml::train_device();
    TrainBackend::seed(&device, config.seed);
    let mut model =
        SkipGram::<TrainBackend>::new(vocabulary.tokens.len(), config.embedding_dim, &device);
    let mut optimizer = SgdConfig::new().init();
    let mut step = 0;

    for epoch in 0..config.epochs {
        let mut pairs = training_pairs(corpus, config.window_size);
        let mut total_loss = 0.0_f64;
        loop {
            let (centers, contexts): (Vec<_>, Vec<_>) =
                pairs.by_ref().take(config.batch_size).unzip();
            if centers.is_empty() {
                break;
            }
            let batch = centers.len();
            let loss = model.loss(
                vocabulary.encode(centers, &device),
                vocabulary.encode(contexts, &device),
            );
            let value = loss.clone().into_scalar();
            if !value.is_finite() {
                return Err("IPv62Vec training produced non-finite loss".into());
            }
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            let rate = learning_rate(config, step, total_steps);
            model = optimizer.step(rate, model, gradients);
            total_loss += f64::from(value) * batch as f64;
            step += 1;
        }
        tracing::info!(target: "sixveclm", epoch, loss = total_loss / pairs_per_epoch as f64, "IPv62Vec skip-gram training");
    }

    let compact = model
        .valid()
        .input
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| format!("Failed to export IPv62Vec embeddings: {error:?}"))?;
    if compact.iter().any(|value| !value.is_finite()) {
        return Err("IPv62Vec training produced non-finite weights".into());
    }
    let size = token_counts
        .len()
        .checked_mul(config.embedding_dim)
        .ok_or("IPv62Vec embedding dimensions overflow")?;
    let mut weights = vec![0.0; size];
    for (&token, vector) in vocabulary
        .tokens
        .iter()
        .zip(compact.chunks_exact(config.embedding_dim))
    {
        let start = token * config.embedding_dim;
        weights[start..start + config.embedding_dim].copy_from_slice(vector);
    }
    Ok(weights)
}

fn training_pairs<'a>(
    corpus: &'a TokenCorpus<'_>,
    window: usize,
) -> impl Iterator<Item = (usize, usize)> + 'a {
    let radius = window / 2;
    (0..corpus.len()).flat_map(move |row| {
        let tokens = corpus.row(row).expect("corpus row is in bounds");
        (0..TOTAL_LEN).flat_map(move |center| {
            let start = center.saturating_sub(radius);
            let end = (center + radius + 1).min(TOTAL_LEN);
            (start..end)
                .filter(move |&context| context != center)
                .map(move |context| (tokens[center] as usize, tokens[context] as usize))
        })
    })
}

fn learning_rate(config: &Word2VecConfig, step: usize, steps: usize) -> f64 {
    let progress = step as f64 / steps.saturating_sub(1).max(1) as f64;
    config.learning_rate
        + (config.min_learning_rate - config.learning_rate) * progress.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests;
