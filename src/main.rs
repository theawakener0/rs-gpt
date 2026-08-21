use core::f64;
use std::cell::RefCell;
use std::fs;
use rand::seq::SliceRandom;
use rand::rng;
use rand_distr::{Distribution, Normal};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;

fn dataset() -> Vec<String> {
    let mut dataset = Vec::new();

    let file_contents = fs::read_to_string("dataset/input.txt").expect("Couldn't read the file");

    for line in file_contents.lines() {
        dataset.push(line.to_string());
    }

    let mut rng = rng();
    dataset.shuffle(&mut rng);

    dataset
}

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
    state_dict: HashMap<String, Matrix>,
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

fn main() {
    let dataset = dataset();

    let uchar: BTreeSet<char> = dataset.iter().flat_map(|data| data.chars()).collect();
    
    // BOS is currently not used
    // let BOS = uchar.len();
    let vocab_size = uchar.len() + 1;

    println!("{}", dataset.len());
    println!("vocab_size: {}", vocab_size);
}
