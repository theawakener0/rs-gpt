mod classic;
mod model;
mod tui;

use clap::Parser;
use std::error::Error;

// include the dataset in the binary
const DATASET: &str = include_str!("../dataset/input.txt");

#[derive(Parser, Debug)]
#[command(
    name = "rs-gpt",
    about = "A Rust implementation of Andrej Karpathy's microgpt"
)]
struct Args {
    #[arg(long)]
    tui: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.tui { tui::run(DATASET) } else { classic::run(DATASET) }
}
