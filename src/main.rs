use core::f64;
use std::cell::RefCell;
use std::error::Error;
use std::fs;
use rand::seq::SliceRandom;
use rand::rng;
use rand_distr::{Distribution, Normal};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;

// currently this is not used
// fn dataset() -> Vec<&str> {
//     let mut dataset = Vec::new();
//
//     let file_contents = fs::read_to_string("dataset/input.txt").expect("Couldn't read the file");
//
//     for line in file_contents.lines() {
//         dataset.push(line);
//     }
//
//     let mut rng = rng();
//     dataset.shuffle(&mut rng);
//
//     dataset
// }

type ValueRef = Rc<RefCell<Value>>;
type Matrix = Vec<Vec<ValueRef>>;

#[derive(Debug, Clone)]
struct Value {
    data: f64,
    grad: f64,
    childern: Vec<ValueRef>,
    local_grads: Vec<f64>,
}

impl Value {
    fn new(data: f64) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data: data,
            grad: 0.0,
            childern: Vec::new(),
            local_grads: Vec::new(),
        }))
    }
}

trait ValueOps {
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn pow(&self, other: f64) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn truediv(&self, other: &Self) -> Self;
    fn neg(&self) -> Self;
    fn log(&self) -> Self;
    fn exp(&self) -> Self;
    fn relu(&self) -> Self;
    fn backward(&self);
}

impl ValueOps for ValueRef {
    fn add(&self, other: &Self) -> Self {
        Rc::new(RefCell::new(Value {
            data: self.borrow().data + other.borrow().data,
            grad: 0.0,
            childern: vec![self.clone(), other.clone()],
            local_grads: vec![1.0, 1.0],
        }))
    }

    fn mul(&self, other: &Self) -> Self {
        Rc::new(RefCell::new(Value {
            data: self.borrow().data * other.borrow().data,
            grad: 0.0,
            childern: vec![self.clone(), other.clone()],
            local_grads: vec![other.borrow().data, self.borrow().data],
        }))
    }

    fn pow(&self, other: f64) -> Self {
        Rc::new(RefCell::new(Value {
            data: self.borrow().data.powf(other),
            grad: 0.0,
            childern: vec![self.clone()],
            local_grads: vec![other * self.borrow().data.powf(other - 1.0)],
        }))
    }

    fn neg(&self) -> Self {
        self.mul(&Value::new(-1.0))
    }

    fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    fn truediv(&self, other: &Self) -> Self {
        self.mul(&other.pow(-1.0))
    }

    fn log(&self) -> Self {
        Rc::new(RefCell::new(Value {
            data: self.borrow().data.ln(),
            grad: 0.0,
            childern: vec![self.clone()],
            local_grads: vec![1.0 / self.borrow().data],
        }))
    }

    fn exp(&self) -> Self {
        Rc::new(RefCell::new(Value {
            data: self.borrow().data.exp(),
            grad: 0.0,
            childern: vec![self.clone()],
            local_grads: vec![self.borrow().data.exp()]
        }))
    }

    fn relu(&self) -> Self {
        Rc::new(RefCell::new(Value {
            data: f64::max(0.0, self.borrow().data),
            grad: 0.0,
            childern: vec![self.clone()],
            local_grads: vec![f64::from(self.borrow().data > 0.0)],
        }))
    }

    fn backward(&self) {
        fn build_topo(
            v: &ValueRef,
            visited: &mut HashSet<*const RefCell<Value>>,
            topo: &mut Vec<ValueRef>,
        ) {
            let addr = Rc::as_ptr(v);
            if !visited.contains(&addr) {
                visited.insert(addr);
                for child in &v.borrow().childern {
                    build_topo(child, visited, topo);
                }
                topo.push(v.clone());
            }
        }

        let mut topo: Vec<ValueRef> = Vec::new();
        let mut visited: HashSet<*const RefCell<Value>> = HashSet::new();

        build_topo(self, &mut visited, &mut topo);

        self.borrow_mut().grad = 1.0;

        for v in topo.iter().rev() {
            for (child, local_grad) in self.borrow().childern.iter().zip(&v.borrow().local_grads) {
                child.borrow_mut().grad += local_grad * v.borrow().grad;
            }
        }
    }

}

fn matrix(nout: usize, nin: usize) -> Matrix {
    let mut rng = rng();
    let normal = Normal::new(0.0, 0.8).unwrap();

    (0..nout)
        .map(|_| {
            (0..nin)
                .map(|_| Value::new(normal.sample(&mut rng)))
                .collect()
        })
        .collect()
}

fn sum(x: impl IntoIterator<Item = ValueRef>) -> ValueRef {
    x.into_iter().fold(Value::new(0.0), |acc, x| acc.add(&x))
}

fn linear(x: &Vec<ValueRef>, w: &Matrix) -> Vec<ValueRef> {
    w.iter().
        map(|wo| sum(wo.iter().zip(x).map(|(wi, xi)| wi.mul(xi))))
        .collect()
}

fn softmax(logits: &Vec<ValueRef>) -> Vec<ValueRef> {
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

fn rmsnorm(x: &Vec<ValueRef>) -> Vec<ValueRef> {
    let ms = sum(x.iter().map(|xi| xi.mul(xi))).truediv(&Value::new(x.len() as f64));
    let scale = ms.add(&Value::new(1e-5)).pow(0.5);
    x.iter().map(|xi| xi.mul(&scale)).collect()
}

fn gpt(
    token_id: usize,
    pos_id: usize,
    n_layer: usize,
    n_head: usize,
    head_dim: usize,
    keys: &mut Vec<Matrix>,
    values: &mut Vec<Matrix>,
    state_dict: &HashMap<String, Matrix>,
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
            let head_out: Vec<ValueRef> = (0..head_dim)
                .map(|j| sum((0..k_h.len()).map(|t| atten_weights[t].mul(&v_h[t][j]))))
                .collect();
            x_attn.extend(head_out);
        }
        x = linear(&x, &state_dict[&format!("layer{li}.attn_wo")]);
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

fn main() -> Result<(), Box<dyn Error>> {
    // set up dataset
    let mut dataset: Vec<&str> = Vec::new();

    let file_contents = fs::read_to_string("dataset/input.txt").expect("Couldn't read the file");

    for line in file_contents.lines() {
        dataset.push(line);
    }

    let mut rng = rng();
    dataset.shuffle(&mut rng);

    let uchars: BTreeSet<char> = BTreeSet::from_iter(dataset.iter().flat_map(|s| s.chars()));
    let uchars: Vec<&char> = uchars.iter().collect();
    
    let BOS = uchars.len();
    let vocab_size = uchars.len() + 1;

    println!(" dataset_size: {}", dataset.len());
    println!("vocab_size: {}", vocab_size);

    // set up model parameters, optimizer, and training loop
    let n_layer = 1;
    let n_embd = 16;
    let block_size = 16;
    let n_head = 4;
    let head_dim = n_embd / n_head;

    let mut state_dict: HashMap<String, Matrix> = HashMap::new();
    state_dict.insert(String::from("wtc"), matrix(vocab_size, n_embd));
    state_dict.insert(String::from("wpe"), matrix(block_size, n_embd));
    state_dict.insert(String::from("lm_head"), matrix(vocab_size, n_embd));

    for i in 0..n_layer {
        state_dict.insert(format!("layer{i}.attn_wq"), matrix(n_embd, n_embd));
        state_dict.insert(format!("layer{i}.attn_wk"), matrix(n_embd, n_embd));
        state_dict.insert(format!("layer{i}.attn_wv"), matrix(n_embd, n_embd));
        state_dict.insert(format!("layer{i}.attn_wo"), matrix(n_embd, n_embd));

        state_dict.insert(format!("layer{i}.mlp_fc1"), matrix(4 * n_embd, n_embd));
        state_dict.insert(format!("layer{i}.mlp_fc2"), matrix(n_embd, 4 *n_embd));
    }

    let params: Vec<&ValueRef> = state_dict.values().flatten().flatten().collect();
    println!("num params: {}", params.len());

    let (learning_rate, beta1, beta2, eps_adam) = (0.01, 0.85, 0.99, 1e-8);
    let mut m = vec![0.0; params.len()];
    let mut v = vec![0.0; params.len()];

    let num_steps = 1000;
    for step in 0..num_steps {

        let data: &str = dataset[step % dataset.len()];

        let mut tokens = vec![BOS];
        tokens.extend(
            data.chars()
                .map(|ch| uchars.iter().position(|&&c| c == ch).unwrap()),
        );

        tokens.push(BOS);
        let n = usize::min(block_size, tokens.len() - 1);
        
        let (mut keys, mut values): (Vec<Matrix>, Vec<Matrix>) = (vec![Vec::new(); n_layer], vec![Vec::new(); n_layer]);
        let mut losses: Vec<ValueRef> = Vec::new();
        for pos_id in 0..n {
            let (token_id, target_id) = (tokens[pos_id], tokens[pos_id + 1]);
            let logits = gpt(
                token_id,
                pos_id,
                n_layer,
                n_head,
                head_dim,
                &mut keys,
                &mut values,
                &state_dict,
            );
            let probs = softmax(&logits);
            let loss_t: ValueRef = probs[target_id].log().neg();
            losses.push(loss_t);
        }
        let loss: ValueRef = Value::new(1.0 / n as f64).mul(&sum(losses));
        loss.backward();

        let lr_t = learning_rate * (1.0 - (step as f64) / (num_steps as f64));
        for (i, p) in params.iter().enumerate() {
            m[i] = beta1 * m[i] + (1.0 - beta1) * p.borrow().grad;
            v[i] = beta2 * v[i] + (1.0 - beta2) * p.borrow().grad.powi(2);
            let m_hat = m[i] / (1.0 - beta1.powi(step as i32));
            let v_hat = v[i] / (1.0 - beta2.powi(step as i32));
            p.borrow_mut().data -= lr_t * m_hat / (v_hat.powf(0.5) + eps_adam);
            p.borrow_mut().grad = 0.0;
        }

        println!(
            "step {:4} / {:4} | loss {:.4}\r",
            step + 1,
            num_steps,
            loss.borrow().data
        )

    }
    Ok(())
}
