use crate::model::layers::AttnSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Training,
    Sampling,
    Done,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub num_params: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub block_size: usize,
    pub head_dim: usize,
    pub num_steps: usize,
}

#[derive(Debug)]
pub struct App {
    pub phase: Phase,
    pub step: usize,
    pub num_steps: usize,
    pub losses: Vec<(f64, f64)>,      // (step, loss)
    pub grad_norms: Vec<f64>,
    pub current_loss: f64,
    pub current_lr: f64,
    pub current_grad_norm: f64,
    pub attn: Option<AttnSnapshot>,
    pub selected_layer: usize,
    pub selected_head: usize,
    pub config: Config,
    // inference streaming
    pub inference_buf: String,
    pub inference_samples: Vec<String>,
    pub inference_done: bool,
    pub dataset_size: usize,
    pub vocab_size: usize,
    pub num_params: usize,
    pub chars: Vec<char>,
    pub bos: usize,
}

impl App {
    pub fn new(config: Config, chars: Vec<char>, bos: usize, dataset_size: usize) -> Self {
        Self {
            phase: Phase::Training,
            step: 0,
            num_steps: config.num_steps,
            losses: Vec::with_capacity(config.num_steps),
            grad_norms: Vec::with_capacity(config.num_steps),
            current_loss: 0.0,
            current_lr: 0.0,
            current_grad_norm: 0.0,
            attn: None,
            selected_layer: 0,
            selected_head: 0,
            config: config.clone(),
            inference_buf: String::new(),
            inference_samples: Vec::new(),
            inference_done: false,
            dataset_size,
            vocab_size: config.vocab_size,
            num_params: config.num_params,
            chars,
            bos,
        }
    }

    pub fn push_step(&mut self, step: usize, loss: f64, lr: f64, grad_norm: f64, attn: Option<AttnSnapshot>) {
        self.step = step + 1; // 1-indexed for display
        self.current_loss = loss;
        self.current_lr = lr;
        self.current_grad_norm = grad_norm;
        self.losses.push((step as f64, loss));
        self.grad_norms.push(grad_norm);
        if attn.is_some() {
            self.attn = attn;
        }
        if self.step >= self.num_steps {
            self.phase = Phase::Sampling;
        }
    }

    pub fn cycle_layer(&mut self, delta: i32) {
        let n = self.config.n_layer as i32;
        if n == 0 {
            return;
        }
        let cur = self.selected_layer as i32;
        self.selected_layer = ((cur + delta).rem_euclid(n)) as usize;
    }

    pub fn cycle_head(&mut self, delta: i32) {
        let n = self.config.n_head as i32;
        if n == 0 {
            return;
        }
        let cur = self.selected_head as i32;
        self.selected_head = ((cur + delta).rem_euclid(n)) as usize;
    }

    pub fn progress_pct(&self) -> f64 {
        if self.num_steps == 0 {
            1.0
        } else {
            self.step as f64 / self.num_steps as f64
        }
    }
}
