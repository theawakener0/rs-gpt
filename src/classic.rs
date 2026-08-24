use crate::model::layers::*;
use crate::model::value::*;
use rand::rng;
use rand::seq::{IndexedRandom, SliceRandom};
use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fs;

pub fn run() -> Result<(), Box<dyn Error>> {
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

    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;

    println!("dataset_size: {}", dataset.len());
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
        state_dict.insert(format!("layer{i}.mlp_fc2"), matrix(n_embd, 4 * n_embd));
    }

    let params: Vec<&ValueRef> = state_dict.values().flatten().flatten().collect();
    println!("num params: {}", params.len());

    let (learning_rate, beta1, beta2, eps_adam) = (0.01, 0.85, 0.99, 1e-8);
    let mut m = vec![0.0; params.len()];
    let mut v = vec![0.0; params.len()];

    let num_steps = 1000;
    for step in 0..num_steps {
        let data: &str = dataset[step % dataset.len()];

        let mut tokens = vec![bos];
        tokens.extend(
            data.chars()
                .map(|ch| uchars.iter().position(|&&c| c == ch).unwrap()),
        );

        tokens.push(bos);
        let n = usize::min(block_size, tokens.len() - 1);

        let (mut keys, mut values): (Vec<Matrix>, Vec<Matrix>) =
            (vec![Vec::new(); n_layer], vec![Vec::new(); n_layer]);
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
            let m_hat = m[i] / (1.0 - beta1.powi(step as i32 + 1));
            let v_hat = v[i] / (1.0 - beta2.powi(step as i32 + 1));
            p.borrow_mut().data -= lr_t * m_hat / (v_hat.powf(0.5) + eps_adam);
            p.borrow_mut().grad = 0.0;
        }

        println!(
            "step {:4} / {:4} | loss {:.4}\r",
            step + 1,
            num_steps,
            loss.borrow().data
        );
    }

    // The inference loop
    let temperature = 0.5;
    println!("\n --- inference (new hallucinated names) ---");
    for sample_idx in 0..20 {
        let (mut keys, mut vlaues) = (vec![Vec::new(); n_layer], vec![Vec::new(); n_layer]);
        let mut token_id = bos;
        let mut sample = Vec::new();

        for pos_id in 0..block_size {
            let logits = gpt(
                token_id,
                pos_id,
                n_layer,
                n_head,
                head_dim,
                &mut keys,
                &mut vlaues,
                &state_dict,
            );
            let probs: Vec<ValueRef> = softmax(
                &logits
                    .iter()
                    .map(|l| l.truediv(&Value::new(temperature)))
                    .collect(),
            );
            token_id = *(0..vocab_size)
                .collect::<Vec<usize>>()
                .choose_weighted(&mut rng, |&i| probs[i].borrow().data)?;
            if token_id == bos {
                break;
            }
            sample.push(uchars[token_id]);
        }
        print!("sample {:2}: {}\n", sample_idx + 1, String::from_iter(sample));
    }

    Ok(())
}
