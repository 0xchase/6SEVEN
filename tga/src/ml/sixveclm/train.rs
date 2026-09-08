use super::*;
use crate::{Algorithm, Observation};
use embedding::{Word2VecConfig, train_ipv62vec};
use transformer::{TransformerHyperParams, TransformerTrainConfig, train_transformer};

impl SixVecLm {
    pub(super) fn train_inner(&self, seeds: Vec<[u8; 16]>) -> Result<SixVecLmModel, String> {
        self.validate()?;

        if seeds.is_empty() {
            return Err("Need at least one active seed address".into());
        }

        let tokenizer = IPv6Tokenizer::new();
        let vocab_size = tokenizer.vocab_size();
        let mut token_counts = vec![0usize; vocab_size];
        let mut raw_sequences = Vec::new();
        let mut raw_seed_addresses = Vec::new();

        for addr in seeds {
            if let Some(tokens) = encode_tokens(&tokenizer, &addr) {
                count_tokens(&tokens, &mut token_counts);
                raw_sequences.push(tokens);
                raw_seed_addresses.push(addr);
            }
        }

        if raw_sequences.is_empty() {
            return Err("No usable sequences produced from seeds".into());
        }

        let vocab_token_mask = token_counts
            .iter()
            .map(|&count| count > 0)
            .collect::<Vec<_>>();
        let position_candidates = build_position_candidates(&tokenizer, &vocab_token_mask)?;
        let corpus = TokenCorpus::new(&raw_sequences);

        let w2v_config = Word2VecConfig {
            embedding_dim: self.embedding_dim,
            window_size: self.embedding_window,
            epochs: self.embedding_epochs,
            batch_size: self.embedding_batch_size,
            learning_rate: self.embedding_lr,
            min_learning_rate: self.embedding_min_lr,
            seed: self.seed,
        };

        let word_embeddings = train_ipv62vec(&w2v_config, &corpus, &token_counts)?;

        let embedding_tensor =
            tensor_from_embeddings(&word_embeddings, vocab_size, self.embedding_dim);

        let hyper = TransformerHyperParams {
            d_model: self.embedding_dim,
            n_heads: self.heads,
            d_ff: self.ff_dim,
            n_layers: self.layers,
            dropout: self.dropout,
            prefix_len: PREFIX_LEN,
            decoder_len: DECODER_LEN,
        };

        let train_cfg = TransformerTrainConfig {
            epochs: self.transformer_epochs,
            batch_size: self.transformer_batch_size,
            noam_factor: self.noam_factor,
            noam_warmup: self.noam_warmup,
            seed: self.seed,
            shuffle: false,
        };

        let transformer_artifacts =
            train_transformer(&hyper, &train_cfg, &corpus, &embedding_tensor)?;

        Ok(SixVecLmModel {
            generation: Default::default(),
            tokenizer,
            embedding_dim: self.embedding_dim,
            vocab_size,
            word_embeddings: Arc::new(word_embeddings),
            transformer: Arc::new(transformer_artifacts),
            position_candidates: Arc::new(position_candidates),
            seed_addresses: Arc::new(raw_seed_addresses),
            generation_seed: self.seed,
            generation_temperature: self.generation_temperature,
            runtime: OnceLock::new(),
        })
    }
}

fn tensor_from_embeddings(
    weights: &[f32],
    vocab_size: usize,
    embedding_dim: usize,
) -> Tensor<TrainBackend, 2> {
    let device = crate::ml::train_device();
    Tensor::from_data(
        TensorData::new(weights.to_vec(), [vocab_size, embedding_dim]),
        &device,
    )
}

impl Algorithm for SixVecLm {
    const ID: &'static str = "sixveclm";
    const DESCRIPTION: &'static str = "6VecLM encoder-decoder transformer with IPv62Vec embeddings and cosine-temperature sampling";

    const MODEL_VERSION: u32 = 2;

    type Model = SixVecLmModel;

    fn train(&self, observations: &[Observation]) -> Result<Self::Model, TgaError> {
        let seeds: Vec<[u8; 16]> = observations
            .iter()
            .filter(|obs| obs.active)
            .map(|obs| obs.address)
            .collect();
        self.train_inner(seeds).map_err(TgaError::Training)
    }
}
