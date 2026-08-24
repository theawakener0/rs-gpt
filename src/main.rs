mod classic;
mod model;
mod tui;

use clap::Parser;
use std::error::Error;

#[derive(Parser, Debug)]
#[command(name = "rs-gpt", about = "microGPT in Rust — classic CLI and ratatui TUI")]
struct Args {
    /// Enable TUI dashboard (loss chart, heatmap, streaming inference). Without it, runs classic CLI.
    #[arg(long)]
    tui: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.tui {
        tui::run()
    } else {
        classic::run()
    }
}
