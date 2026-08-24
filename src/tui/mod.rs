pub mod app;
pub mod sample;
pub mod train;
pub mod ui;

use crate::model::layers::matrix;
use crate::model::value::*;
use crate::tui::app::{App, Config, Phase};
use crate::tui::sample::Sampler;
use crate::tui::train::{Dataset, TrainState};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::collections::HashMap;
use std::error::Error;
use std::io;
use std::time::{Duration, Instant};

pub fn run() -> Result<(), Box<dyn Error>> {
    let dataset = Dataset::load("dataset/input.txt");
    let n_layer = 1;
    let n_embd = 16;
    let block_size = 16;
    let n_head = 4;
    let head_dim = n_embd / n_head;
    let vocab_size = dataset.vocab_size;
    let num_steps = 1000;

    let mut state_dict: HashMap<String, crate::model::layers::Matrix> = HashMap::new();
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
    let num_params = params.len();

    let config = Config {
        vocab_size,
        num_params,
        n_layer,
        n_head,
        n_embd,
        block_size,
        head_dim,
        num_steps,
    };

    let mut app = App::new(config, dataset.chars.clone(), dataset.bos, dataset.names.len());
    let mut train_state = TrainState::new(num_params);

    // terminal setup
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let orig_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
        orig_hook(info);
    }));

    let mut sampler: Option<Sampler> = None;
    let mut rng = rand::rng();
    let mut last_sample_tick = Instant::now();
    let sample_interval = Duration::from_millis(33);

    let mut step: usize = 0;

    loop {
        if event::poll(Duration::from_millis(16))? {
            match event::read()? {
                Event::Key(k) => match k.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') => break,
                    KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => break,
                    KeyCode::Esc if app.show_help => app.show_help = false,
                    KeyCode::Char('?') => app.show_help = !app.show_help,
                    KeyCode::Char('/') if !k.modifiers.contains(KeyModifiers::CONTROL) => app.show_help = !app.show_help,
                    KeyCode::Left => app.cycle_layer(-1),
                    KeyCode::Right => app.cycle_layer(1),
                    KeyCode::Up => app.cycle_head(-1),
                    KeyCode::Down => app.cycle_head(1),
                    KeyCode::Char('j') => {
                        app.inference_scroll = app.inference_scroll.saturating_add(1);
                        app.inference_follow = false;
                    }
                    KeyCode::Char('k') => {
                        app.inference_scroll = app.inference_scroll.saturating_sub(1);
                        app.inference_follow = false;
                    }
                    KeyCode::PageDown => {
                        let h = app.inference_inner_height.max(1);
                        app.inference_scroll = app.inference_scroll.saturating_add(h);
                        app.inference_follow = false;
                    }
                    KeyCode::PageUp => {
                        let h = app.inference_inner_height.max(1);
                        app.inference_scroll = app.inference_scroll.saturating_sub(h);
                        app.inference_follow = false;
                    }
                    KeyCode::Home | KeyCode::Char('g') => {
                        app.inference_scroll = 0;
                        app.inference_follow = false;
                    }
                    KeyCode::End | KeyCode::Char('G') => {
                        app.inference_follow = true;
                    }
                    _ => {}
                },
                Event::Mouse(me) => {
                    use crossterm::event::MouseEventKind;
                    match me.kind {
                        MouseEventKind::ScrollDown => {
                            app.inference_scroll = app.inference_scroll.saturating_add(3);
                            app.inference_follow = false;
                        }
                        MouseEventKind::ScrollUp => {
                            app.inference_scroll = app.inference_scroll.saturating_sub(3);
                            app.inference_follow = false;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        match app.phase {
            Phase::Training => {
                if step < num_steps {
                    let (loss, lr, grad_norm, attn) = train::train_step(
                        &app,
                        step,
                        &state_dict,
                        &params,
                        &mut train_state,
                        &dataset,
                        n_layer,
                        n_head,
                        head_dim,
                        block_size,
                    );
                    app.push_step(step, loss, lr, grad_norm, attn);
                    step += 1;
                } else {
                    app.phase = Phase::Sampling;
                }
            }
            Phase::Sampling => {
                if sampler.is_none() {
                    sampler = Some(Sampler::new(
                        n_layer,
                        n_head,
                        head_dim,
                        block_size,
                        vocab_size,
                        dataset.bos,
                        dataset.chars.clone(),
                        20,
                    ));
                    last_sample_tick = Instant::now();
                }
                if last_sample_tick.elapsed() >= sample_interval {
                    last_sample_tick = Instant::now();
                    if let Some(s) = sampler.as_mut() {
                        match s.step(&state_dict, &mut rng) {
                            None => {
                                if !app.inference_buf.is_empty() {
                                    app.inference_samples.push(app.inference_buf.clone());
                                    app.inference_buf.clear();
                                }
                                app.phase = Phase::Done;
                            }
                            Some(opt_ch) => {
                                match opt_ch {
                                    Some(ch) => {
                                        app.inference_buf.push(ch);
                                    }
                                    None => {
                                        if !app.inference_buf.is_empty() {
                                            let completed = app.inference_buf.clone();
                                            app.inference_samples.push(completed);
                                            app.inference_buf.clear();
                                        } else if s.samples_generated > app.inference_samples.len() {
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Phase::Done => {
                // stay interactive for inspection
            }
        }

        terminal.draw(|f| ui::draw(f, &mut app))?;

    }

    // restore
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    let _ = std::panic::take_hook();

    Ok(())
}
