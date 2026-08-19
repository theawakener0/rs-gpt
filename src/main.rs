use std::fs;
use rand::seq::SliceRandom;
use rand::rng;

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

fn main() {
    let dataset = dataset();
    println!("{:#?}", dataset);
}
