use core::f64;
use std::cell::RefCell;
use std::fs;
use rand::seq::SliceRandom;
use rand::rng;
use std::collections::{BTreeSet, HashSet};
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


fn main() {
    let dataset = dataset();

    let uchar: BTreeSet<char> = dataset.iter().flat_map(|data| data.chars()).collect();
    
    // BOS is currently not used
    // let bos = uchar.len();
    let vocab_size = uchar.len() + 1;

    println!("{}", dataset.len());
    println!("vocab_size: {}", vocab_size);
}
