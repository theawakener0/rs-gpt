use crate::model::layers::*;
use crate::model::value::*;
use rand::seq::IndexedRandom;
use std::collections::HashMap;

pub struct Sampler {
    pub keys: Vec<Matrix>,
    pub values: Vec<Matrix>,
    pub token_id: usize,
    pub pos_id: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub temperature: f64,
    pub bos: usize,
    pub current_sample: Vec<char>,
    pub done: bool,
    pub samples_generated: usize,
    pub target_samples: usize,
    pub chars: Vec<char>,
}

impl Sampler {
    pub fn new(
        n_layer: usize,
        n_head: usize,
        head_dim: usize,
        block_size: usize,
        vocab_size: usize,
        bos: usize,
        chars: Vec<char>,
        target_samples: usize,
    ) -> Self {
        Self {
            keys: vec![Vec::new(); n_layer],
            values: vec![Vec::new(); n_layer],
            token_id: bos,
            pos_id: 0,
            block_size,
            n_layer,
            n_head,
            head_dim,
            vocab_size,
            temperature: 0.5,
            bos,
            current_sample: Vec::new(),
            done: false,
            samples_generated: 0,
            target_samples,
            chars,
        }
    }

    pub fn reset(&mut self) {
        self.keys = vec![Vec::new(); self.n_layer];
        self.values = vec![Vec::new(); self.n_layer];
        self.token_id = self.bos;
        self.pos_id = 0;
        self.current_sample.clear();
        self.done = false;
    }

    /// Generate one token; returns Some(char) if emitted, or None if step was BOS break/reset.
    /// Returns None when all samples done.
    pub fn step(
        &mut self,
        state_dict: &HashMap<String, Matrix>,
        rng: &mut impl rand::Rng,
    ) -> Option<Option<char>> {
        if self.samples_generated >= self.target_samples {
            return None;
        }
        if self.pos_id >= self.block_size {
            // sample complete without BOS — force break and start next
            self.samples_generated += 1;
            self.reset();
            return Some(None);
        }

        let logits = gpt(
            self.token_id,
            self.pos_id,
            self.n_layer,
            self.n_head,
            self.head_dim,
            &mut self.keys,
            &mut self.values,
            state_dict,
        );
        let probs: Vec<ValueRef> = softmax(
            &logits
                .iter()
                .map(|l| l.truediv(&Value::new(self.temperature)))
                .collect(),
        );

        let token_id = *(0..self.vocab_size)
            .collect::<Vec<usize>>()
            .choose_weighted(rng, |&i| probs[i].borrow().data)
            .unwrap();

        self.token_id = token_id;
        self.pos_id += 1;

        if token_id == self.bos {
            self.samples_generated += 1;
            let was_empty = self.current_sample.is_empty();
            self.reset();
            if was_empty {
                return Some(None);
            }
            return Some(None);
        }

        let ch = self.chars[token_id];
        self.current_sample.push(ch);
        Some(Some(ch))
    }
}
