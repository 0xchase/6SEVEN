use serde::{Deserialize, Serialize};
use sixseven_core::{
    Address, Algorithm, Feedback, Generated, GenerationState, Observation, TargetModel, TgaError,
};

#[derive(Serialize, Deserialize)]
pub struct Example {
    start: u8,
}

#[derive(Serialize, Deserialize)]
pub struct Model {
    next: u8,
    waiting: bool,
}

impl Algorithm for Example {
    const ID: &'static str = "example/native";
    const DESCRIPTION: &'static str = "Adaptive native plugin example";
    type Model = Model;
    fn train(&self, _: &[Observation]) -> Result<Model, TgaError> {
        Ok(Model {
            next: self.start,
            waiting: false,
        })
    }
}

impl TargetModel for Model {
    fn generate(&mut self, output: &mut [Address]) -> Result<Generated, TgaError> {
        sixseven_core::generation::validate_output(output)?;
        let written = if self.waiting {
            0
        } else {
            self.waiting = true;
            let mut address = [0; 16];
            address[..4].copy_from_slice(&[0x20, 1, 0x0d, 0xb8]);
            address[15] = self.next;
            output[0] = address;
            1
        };
        Ok(Generated {
            written,
            state: GenerationState::AwaitingFeedback,
        })
    }
    fn apply_feedback(&mut self, items: &[Feedback]) -> Result<(), TgaError> {
        if items.iter().any(Feedback::is_address_observation) {
            self.next = self
                .next
                .checked_add(1)
                .ok_or_else(|| TgaError::Feedback("example exhausted".into()))?;
            self.waiting = false;
        }
        Ok(())
    }
}

sixseven_plugins::export_plugin!(Example);
