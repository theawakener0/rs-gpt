use crate::model::layers::*;
use crate::model::value::*;
use crate::tui::app::App;
use std::collections::HashMap;

pub struct TrainState {
    pub m: Vec<f64>,
    pub v: Vec<f64>,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

impl TrainState {
    pub fn new(num_params: usize) -> Self {
        Self {
            m: vec![0.0; num_params],
            v: vec![0.0; num_params],
            learning_rate: 0.01,
            beta1: 0.85,
            beta2: 0.99,
            eps: 1e-8,
        }
    }
}

pub struct Dataset {
    pub names: Vec<String>,
    pub chars: Vec<char>,
    pub bos: usize,
    pub vocab_size: usize,
}

impl Dataset {
    pub fn load(path: &str) -> Self {
        let file_contents = std::fs::read_to_string(path).expect("Couldn't read dataset");
        let mut names: Vec<String> = file_contents.lines().map(|s| s.to_string()).collect();
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();
        names.shuffle(&mut rng);
        let uchars: std::collections::BTreeSet<char> =
            std::collections::BTreeSet::from_iter(names.iter().flat_map(|s| s.chars()));
        let chars: Vec<char> = uchars.into_iter().collect();
        let bos = chars.len();
        let vocab_size = chars.len() + 1;
        Self {
            names,
            chars,
            bos,
            vocab_size,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn train_step(
    app: &App,
    step: usize,
    state_dict: &HashMap<String, Matrix>,
    params: &[&ValueRef],
    train_state: &mut TrainState,
    dataset: &Dataset,
    n_layer: usize,
    n_head: usize,
    head_dim: usize,
    block_size: usize,
) -> (f64, f64, f64, Option<AttnSnapshot>) {
    let data: &str = &dataset.names[step % dataset.names.len()];
    let bos = dataset.bos;

    let mut tokens = vec![bos];
    tokens.extend(
        data.chars()
            .map(|ch| dataset.chars.iter().position(|&c| c == ch).unwrap()),
    );
    tokens.push(bos);
    let n = usize::min(block_size, tokens.len() - 1);

    let mut keys: Vec<Matrix> = vec![Vec::new(); n_layer];
    let mut values: Vec<Matrix> = vec![Vec::new(); n_layer];
    let mut losses: Vec<ValueRef> = Vec::new();
    let mut attn_snapshot: AttnSnapshot = Vec::new();

    for pos_id in 0..n {
        let (token_id, target_id) = (tokens[pos_id], tokens[pos_id + 1]);
        // capture attn for TUI
        let logits = gpt_with_attn(
            token_id,
            pos_id,
            n_layer,
            n_head,
            head_dim,
            &mut keys,
            &mut values,
            state_dict,
            Some(&mut attn_snapshot),
        );
        let probs = softmax(&logits);
        let loss_t: ValueRef = probs[target_id].log().neg();
        losses.push(loss_t);
    }

    let loss: ValueRef = Value::new(1.0 / n as f64).mul(&sum(losses));
    loss.backward();
    let loss_val = loss.borrow().data;

    // grad norm before zeroing
    let grad_norm: f64 = params
        .iter()
        .map(|p| p.borrow().grad.powi(2))
        .sum::<f64>()
        .sqrt();

    let lr_t = train_state.learning_rate * (1.0 - (step as f64) / (app.num_steps as f64));
    for (i, p) in params.iter().enumerate() {
        train_state.m[i] =
            train_state.beta1 * train_state.m[i] + (1.0 - train_state.beta1) * p.borrow().grad;
        train_state.v[i] = train_state.beta2 * train_state.v[i]
            + (1.0 - train_state.beta2) * p.borrow().grad.powi(2);
        let m_hat = train_state.m[i] / (1.0 - train_state.beta1.powi(step as i32 + 1));
        let v_hat = train_state.v[i] / (1.0 - train_state.beta2.powi(step as i32 + 1));
        p.borrow_mut().data -= lr_t * m_hat / (v_hat.powf(0.5) + train_state.eps);
        p.borrow_mut().grad = 0.0;
    }

    (loss_val, lr_t, grad_norm, Some(attn_snapshot))
}
