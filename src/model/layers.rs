use super::value::*;
use rand::rng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;


pub type Matrix = Vec<Vec<ValueRef>>;

pub fn matrix(nout: usize, nin: usize) -> Matrix {
    let mut rng = rng();
    let normal = Normal::new(0.0, 0.08).unwrap();

    (0..nout)
        .map(|_| {
            (0..nin)
                .map(|_| Value::new(normal.sample(&mut rng)))
                .collect()
        })
        .collect()
}

pub fn sum(x: impl IntoIterator<Item = ValueRef>) -> ValueRef {
    x.into_iter().fold(Value::new(0.0), |acc, x| acc.add(&x))
}

pub fn linear(x: &Vec<ValueRef>, w: &Matrix) -> Vec<ValueRef> {
    w.iter().
        map(|wo| sum(wo.iter().zip(x).map(|(wi, xi)| wi.mul(xi))))
        .collect()
}

pub fn softmax(logits: &Vec<ValueRef>) -> Vec<ValueRef> {
    let max_val = Value::new(
        logits
            .iter()
            .map(|v| v.borrow().data)
            .max_by(|x, y| x.total_cmp(y))
            .unwrap(),
    );
    let exps = logits.iter().map(|vals| vals.sub(&max_val).exp());
    let total: ValueRef = sum(exps.clone().map(|v| v));

    exps.map(|e| e.truediv(&total)).collect()
}

pub fn rmsnorm(x: &Vec<ValueRef>) -> Vec<ValueRef> {
    let ms = sum(x.iter().map(|xi| xi.mul(xi))).truediv(&Value::new(x.len() as f64));
    let scale = ms.add(&Value::new(1e-5)).pow(-0.5);
    x.iter().map(|xi| xi.mul(&scale)).collect()
}

/// Snapshot shape: [layer][head][query_pos][key_pos] as f64 weights (triangular)
pub type AttnSnapshot = Vec<Vec<Vec<Vec<f64>>>>;

pub fn gpt(
    token_id: usize,
    pos_id: usize,
    n_layer: usize,
    n_head: usize,
    head_dim: usize,
    keys: &mut Vec<Matrix>,
    values: &mut Vec<Matrix>,
    state_dict: &HashMap<String, Matrix>,
) -> Vec<ValueRef> {
    gpt_with_attn(
        token_id, pos_id, n_layer, n_head, head_dim, keys, values, state_dict, None,
    )
}

pub fn gpt_with_attn(
    token_id: usize,
    pos_id: usize,
    n_layer: usize,
    n_head: usize,
    head_dim: usize,
    keys: &mut Vec<Matrix>,
    values: &mut Vec<Matrix>,
    state_dict: &HashMap<String, Matrix>,
    mut attn_out: Option<&mut AttnSnapshot>,
) -> Vec<ValueRef> {
    let tok_emb: &Vec<ValueRef> = &state_dict["wtc"][token_id];
    let pos_emb: &Vec<ValueRef> = &state_dict["wpe"][pos_id];

    let mut x: Vec<ValueRef> = tok_emb.iter().zip(pos_emb).map(|(t, p)| t.add(p)).collect();
    x = rmsnorm(&x);

    for li in 0..n_layer {
        let x_residual = x.clone();
        x = rmsnorm(&x);
        let q: Vec<ValueRef> = linear(&x, &state_dict[&format!("layer{li}.attn_wq")]);
        let k: Vec<ValueRef> = linear(&x, &state_dict[&format!("layer{li}.attn_wk")]);
        let v: Vec<ValueRef> = linear(&x, &state_dict[&format!("layer{li}.attn_wv")]);
        keys[li].push(k);
        values[li].push(v);

        let mut x_attn: Vec<ValueRef> = Vec::new();
        for h in 0..n_head {
            let hs = h * head_dim;

            let q_h = &q[hs..hs + head_dim];
            let k_h: Vec<&[ValueRef]> = keys[li].iter().map(|ki| &ki[hs..hs + head_dim]).collect();
            let v_h: Vec<&[ValueRef]> = values[li].iter().map(|vi| &vi[hs..hs + head_dim]).collect();
            let attn_logits: Vec<ValueRef> = (0..k_h.len())
                .map(|t| {
                    let dot_product: ValueRef = sum((0..head_dim).map(|j| q_h[j].mul(&k_h[t][j])));
                    dot_product.truediv(&Value::new((head_dim as f64).sqrt()))
                })
                .collect();
            let atten_weights = softmax(&attn_logits);
            if let Some(ref mut out) = attn_out {
                // ensure structure exists: out[li][h] is Vec<Vec<f64>> (query -> keys)
                let snapshot: Vec<f64> = atten_weights.iter().map(|w| w.borrow().data).collect();
                // attn_out shape is [layer][head][query][key]; grow lazily
                while out.len() <= li {
                    out.push(vec![Vec::new(); n_head]);
                }
                if out[li][h].len() <= pos_id {
                    out[li][h].resize(pos_id + 1, Vec::new());
                }
                out[li][h][pos_id] = snapshot;
            }
            let head_out: Vec<ValueRef> = (0..head_dim)
                .map(|j| sum((0..k_h.len()).map(|t| atten_weights[t].mul(&v_h[t][j]))))
                .collect();
            x_attn.extend(head_out);
        }
        x = linear(&x_attn, &state_dict[&format!("layer{li}.attn_wo")]);
        x = x
            .iter()
            .zip(x_residual)
            .map(|(a, b)| a.add(&b))
            .collect();

        let x_residual = x.clone();
        x = rmsnorm(&x);
        x = linear(&x, &state_dict[&format!("layer{li}.mlp_fc1")]);
        x = x.iter().map(ValueRef::relu).collect();
        x = linear(&x, &state_dict[&format!("layer{li}.mlp_fc2")]);
        x = x.iter().zip(x_residual).map(|(a, b)| a.add(&b)).collect();
    }

    let logits: Vec<Rc<RefCell<Value>>> = linear(&x, &state_dict["lm_head"]);
    logits
}

