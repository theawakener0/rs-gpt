use crate::tui::app::{App, Phase};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Chart, Dataset, Gauge, GraphType, Paragraph, Row, Sparkline, Table, Wrap},
    Frame,
};

// const FORGE_CHAR: Color = Color::Rgb(18, 16, 18);
// const ANVIL: Color = Color::Rgb(59, 46, 42);
const SMITHY: Color = Color::Rgb(43, 30, 22);
const KILN: Color = Color::Rgb(80, 44, 30); 
const HARD_RUST: Color = Color::Rgb(115, 64, 43);
const RUST_ORANGE: Color = Color::Rgb(206, 66, 43);
const EMBER: Color = Color::Rgb(232, 94, 37);
const SAND: Color = Color::Rgb(222, 165, 132);
const ASH: Color = Color::Rgb(255, 231, 194);
const CREAM: Color = Color::Rgb(235, 219, 178);
const STONE: Color = Color::Rgb(103, 115, 122);

fn interpolate_color(t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.3 {
        let k = t / 0.3;
        (18.0 + k * 97.0, 16.0 + k * 48.0, 18.0 + k * 25.0)
    } else if t < 0.6 {
        let k = (t - 0.3) / 0.3;
        (115.0 + k * 91.0, 64.0 + k * 2.0, 43.0 + k * 0.0)
    } else if t < 0.85 {
        let k = (t - 0.6) / 0.25;
        (206.0 + k * 16.0, 66.0 + k * 99.0, 43.0 + k * 89.0)
    } else {
        let k = (t - 0.85) / 0.15;
        (222.0 + k * 33.0, 165.0 + k * 66.0, 132.0 + k * 62.0)
    };
    Color::Rgb(r as u8, g as u8, b as u8)
}

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    if area.width < 60 || area.height < 20 {
        let msg = Paragraph::new("Terminal too small — resize to at least 80x24 (q / Ctrl-C to quit)")
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(HARD_RUST))
                    .title(Span::styled(" rs-gpt ", Style::default().fg(RUST_ORANGE).add_modifier(Modifier::BOLD)))
                    .style(Style::default().fg(CREAM)),
            )
            .style(Style::default().fg(STONE))
            .wrap(Wrap { trim: true });
        frame.render_widget(msg, area);
        return;
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Percentage(35),
            Constraint::Percentage(35),
            Constraint::Min(8),
            Constraint::Length(1),
        ])
        .split(area);

    draw_header(frame, app, chunks[0]);
    draw_loss_row(frame, app, chunks[1]);
    draw_attention(frame, app, chunks[2]);
    draw_inference(frame, app, chunks[3]);
    draw_footer(frame, app, chunks[4]);
    if app.show_help {
        draw_help(frame, area);
    }
}

fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let pct = (app.progress_pct() * 100.0) as u16;
    let loss_str = if app.losses.is_empty() {
        "—".to_string()
    } else {
        format!("{:.4}", app.current_loss)
    };
    let grad_str = if app.grad_norms.is_empty() {
        "—".to_string()
    } else {
        format!("{:.3}", app.current_grad_norm)
    };
    let title = format!(
        " rs-gpt {} │ step {}/{} │ lr {:.4} │ loss {} │ |grad| {} │ vocab {} │ params {} ",
        match app.phase {
            Phase::Training => "▸ training",
            Phase::Sampling => "▸ sampling",
            Phase::Done => "✓ done",
        },
        app.step,
        app.num_steps,
        app.current_lr,
        loss_str,
        grad_str,
        app.vocab_size,
        app.num_params,
    );

    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(KILN))
                .title(Span::styled(title, Style::default().fg(SAND).add_modifier(Modifier::BOLD))),
        )
        .gauge_style(Style::default().fg(RUST_ORANGE).add_modifier(Modifier::BOLD))
        .percent(pct)
        .label(Span::styled(
            format!("{pct}%"),
            Style::default().fg(ASH).add_modifier(Modifier::BOLD),
        ));
    frame.render_widget(gauge, area);
}

fn draw_loss_row(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(area);

    // Loss chart
    let loss_data: Vec<(f64, f64)> = app.losses.clone();
    let max_x = app.num_steps as f64;
    let (y_min, y_max) = if loss_data.is_empty() {
        (0.0, 4.0)
    } else {
        let min = loss_data.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        let max = loss_data.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
        let pad = (max - min) * 0.15 + 0.2;
        ((min - pad).max(0.0), max + pad)
    };

    let datasets = if loss_data.is_empty() {
        vec![]
    } else {
        vec![
            Dataset::default()
                .name("loss")
                .marker(ratatui::symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(RUST_ORANGE).add_modifier(Modifier::BOLD))
                .data(&loss_data),
        ]
    };

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(KILN))
                .title(Span::styled(
                    format!(" Loss  ({} points) ", loss_data.len()),
                    Style::default().fg(SAND).add_modifier(Modifier::BOLD),
                )),
        )
        .x_axis(
            ratatui::widgets::Axis::default()
                .title(Span::styled("step", Style::default().fg(STONE)))
                .style(Style::default().fg(STONE))
                .bounds([0.0, max_x])
                .labels(vec![
                    Span::styled("0", Style::default().fg(STONE)),
                    Span::styled(format!("{}", app.num_steps / 2), Style::default().fg(STONE)),
                    Span::styled(format!("{}", app.num_steps), Style::default().fg(STONE)),
                ]),
        )
        .y_axis(
            ratatui::widgets::Axis::default()
                .title(Span::styled("loss", Style::default().fg(STONE)))
                .style(Style::default().fg(STONE))
                .bounds([y_min, y_max])
                .labels(vec![
                    Span::styled(format!("{:.1}", y_min), Style::default().fg(STONE)),
                    Span::styled(format!("{:.1}", (y_min + y_max) / 2.0), Style::default().fg(STONE)),
                    Span::styled(format!("{:.1}", y_max), Style::default().fg(STONE)),
                ]),
        );
    frame.render_widget(chart, cols[0]);

    // Right column: grad sparkline + stats
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(cols[1]);

    // Sparkline: sliding window with local max so it keeps changing instead of flattening
    const GRAD_WINDOW: usize = 64;
    let spark_data: Vec<u64> = if app.grad_norms.is_empty() {
        vec![0]
    } else {
        let window: &[f64] = if app.grad_norms.len() > GRAD_WINDOW {
            &app.grad_norms[app.grad_norms.len() - GRAD_WINDOW..]
        } else {
            &app.grad_norms
        };
        let local_max = window.iter().cloned().fold(0.0_f64, f64::max).max(1e-6);
        window
            .iter()
            .map(|v| ((v / local_max) * 8.0).round().clamp(0.0, 8.0) as u64)
            .collect()
    };
    let window_len = if app.grad_norms.len() > GRAD_WINDOW { GRAD_WINDOW } else { app.grad_norms.len() };
    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(KILN))
                .title(Span::styled(
                    format!(" Grad · forge (last {} ) ", if window_len == 0 { 0 } else { window_len }),
                    Style::default().fg(SAND).add_modifier(Modifier::BOLD),
                )),
        )
        .data(&spark_data)
        .max(8)
        .style(Style::default().fg(EMBER));
    frame.render_widget(sparkline, right[0]);

    // Stats block
    let last_loss = app.losses.last().map(|(_, v)| format!("{:.4}", v)).unwrap_or_else(|| "—".to_string());
    let best_loss = if app.losses.is_empty() {
        "—".to_string()
    } else {
        let m = app.losses.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
        format!("{:.4}", m)
    };
    let stats_text = vec![
        Line::from(vec![
            Span::styled("current: ", Style::default().fg(STONE)),
            Span::styled(last_loss, Style::default().fg(RUST_ORANGE).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("best:    ", Style::default().fg(STONE)),
            Span::styled(best_loss, Style::default().fg(ASH).add_modifier(Modifier::ITALIC)),
        ]),
        Line::from(vec![
            Span::styled("phase:   ", Style::default().fg(STONE)),
            Span::styled(
                format!("{:?}", app.phase),
                Style::default().fg(SAND).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(Span::styled(
            "q quit │ ←→ layer │ ↑↓ head",
            Style::default().fg(STONE).add_modifier(Modifier::ITALIC),
        )),
    ];
    let stats = Paragraph::new(stats_text).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(KILN))
            .title(Span::styled(" Stats ", Style::default().fg(SAND).add_modifier(Modifier::BOLD))),
    );
    frame.render_widget(stats, right[1]);
}

fn draw_attention(frame: &mut Frame, app: &App, area: Rect) {
    let has_attn = app.attn.is_some();
    let title = if has_attn {
        format!(
            " Attention Heatmap  layer {}/{}  head {}/{}  (query ↓ × key →) ",
            app.selected_layer + 1,
            app.config.n_layer,
            app.selected_head + 1,
            app.config.n_head
        )
    } else {
        " Attention Heatmap  (warming up — no snapshot yet) ".to_string()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(KILN))
        .title(Span::styled(title, Style::default().fg(SAND).add_modifier(Modifier::BOLD)));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if !has_attn {
        let p = Paragraph::new("Training… attention weights will appear after first step")
            .style(Style::default().fg(STONE).add_modifier(Modifier::ITALIC))
            .wrap(Wrap { trim: true });
        frame.render_widget(p, inner);
        return;
    }

    let attn = app.attn.as_ref().unwrap();
    // attn shape: [layer][head][query][key]
    if app.selected_layer >= attn.len() {
        let p = Paragraph::new("No attention for selected layer").style(Style::default().fg(EMBER));
        frame.render_widget(p, inner);
        return;
    }
    let layer = &attn[app.selected_layer];
    if app.selected_head >= layer.len() {
        let p = Paragraph::new("No attention for selected head").style(Style::default().fg(EMBER));
        frame.render_widget(p, inner);
        return;
    }
    let head = &layer[app.selected_head];
    if head.is_empty() {
        let p = Paragraph::new("Empty attention matrix").style(Style::default().fg(STONE));
        frame.render_widget(p, inner);
        return;
    }

    let n_key = app.config.block_size;
    let n_query = app.config.block_size;

    // Build table: header row = key positions, then each query row
    let mut rows: Vec<Row> = Vec::new();

    let constraints: Vec<Constraint> = std::iter::once(Constraint::Length(4))
        .chain((0..n_key).map(|_| Constraint::Length(3)))
        .collect();

    let header_cells: Vec<Cell> = std::iter::once(Cell::from("").style(Style::default().fg(STONE)))
        .chain((0..n_key).map(|k| {
            Cell::from(format!("{k}")).style(Style::default().fg(STONE))
        }))
        .collect();
    let header = Row::new(header_cells).height(1);
    for qi in 0..n_query {
        let row_weights = head.get(qi);
        let mut cells: Vec<Cell> = Vec::with_capacity(n_key + 1);
        cells.push(Cell::from(format!("{qi}")).style(Style::default().fg(STONE)));
        for ki in 0..n_key {
            let w = row_weights.and_then(|r| r.get(ki).copied()).unwrap_or(0.0);
            let beyond_data = row_weights.is_none() || ki >= row_weights.unwrap().len();
            let is_future = ki > qi;
            let is_pad = beyond_data;
            let bg = if is_pad || is_future {
                SMITHY
            } else {
                interpolate_color(w)
            };
            // high heat gets ASH fg for contrast on ember
            let fg = if !is_pad && !is_future && w > 0.6 { ASH } else { CREAM };
            let cell = Cell::from("  ").style(Style::default().bg(bg).fg(fg));
            cells.push(cell);
        }
        rows.push(Row::new(cells).height(1));
    }

    let table = Table::new(rows, constraints)
        .header(header.style(Style::default().fg(SAND).add_modifier(Modifier::BOLD)).height(1))
        .block(Block::default())
        .column_spacing(0);

    frame.render_widget(table, inner);
}

fn draw_inference(frame: &mut Frame, app: &mut App, area: Rect) {
    let scroll_hint = if app.inference_follow { "follow" } else { "scroll" };
    let title = format!(
        " Inference — {} samples  [{}]  (j/k PgUp/PgDn scroll, End follow) ",
        app.inference_samples.len(),
        scroll_hint
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(KILN))
        .title(Span::styled(title, Style::default().fg(SAND).add_modifier(Modifier::BOLD)));

    let mut lines: Vec<Line> = Vec::new();
    for (i, s) in app.inference_samples.iter().enumerate() {
        lines.push(Line::from(vec![
            Span::styled(
                format!("sample {:2}: ", i + 1),
                Style::default().fg(STONE).add_modifier(Modifier::ITALIC),
            ),
            Span::styled(s.clone(), Style::default().fg(CREAM)),
        ]));
    }
    if !app.inference_buf.is_empty() && app.phase == Phase::Sampling {
        lines.push(Line::from(vec![
            Span::styled(
                format!("sample {:2}: ", app.inference_samples.len() + 1),
                Style::default().fg(STONE).add_modifier(Modifier::ITALIC),
            ),
            Span::styled(app.inference_buf.clone(), Style::default().fg(CREAM)),
            Span::styled("█", Style::default().fg(EMBER).add_modifier(Modifier::SLOW_BLINK | Modifier::BOLD)),
        ]));
    } else if app.phase == Phase::Done {
        lines.push(Line::from(Span::styled(
            "— done —  (footer: q to quit, ←→/↑↓ inspect attention, ? help)",
            Style::default().fg(SAND).add_modifier(Modifier::ITALIC),
        )));
    } else if app.inference_samples.is_empty() && app.phase == Phase::Training {
        lines.push(Line::from(Span::styled(
            "Waiting for training to finish…",
            Style::default().fg(STONE).add_modifier(Modifier::ITALIC),
        )));
    }

    // Auto-scroll logic: Paragraph::scroll offset is in wrapped rows, not logical lines.
    let inner = block.inner(area);
    app.inference_inner_height = inner.height;
    let viewport_h = inner.height as usize;
    let inner_w = inner.width as usize;
    let content_h: usize = if inner_w == 0 {
        lines.len()
    } else {
        lines
            .iter()
            .map(|l| {
                let w = l.width() as usize;
                if w == 0 {
                    1
                } else {
                    (w + inner_w - 1) / inner_w
                }
            })
            .sum()
    };
    let max_scroll = content_h.saturating_sub(viewport_h) as u16;
    if app.inference_follow {
        app.inference_scroll = max_scroll;
    } else {
        app.inference_scroll = app.inference_scroll.min(max_scroll);
    }

    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((app.inference_scroll, 0));

    frame.render_widget(para, area);
}

fn draw_footer(frame: &mut Frame, app: &App, area: Rect) {
    let key_style = Style::default().fg(RUST_ORANGE).add_modifier(Modifier::BOLD);
    let desc_style = Style::default().fg(CREAM);
    let sep = Span::styled(" │ ", Style::default().fg(STONE));
    let quit_style = if app.phase == Phase::Done {
        Style::default().fg(ASH).add_modifier(Modifier::BOLD)
    } else {
        key_style
    };
    // Truncate gracefully on narrow terminals
    let width = area.width as usize;
    let full = vec![
        Span::styled(" q ", quit_style),
        Span::styled("quit", desc_style),
        sep.clone(),
        Span::styled(" Ctrl-C ", key_style),
        Span::styled("exit", desc_style),
        sep.clone(),
        Span::styled(" ←/→ ", key_style),
        Span::styled("layer", desc_style),
        sep.clone(),
        Span::styled(" ↑/↓ ", key_style),
        Span::styled("head", desc_style),
        sep.clone(),
        Span::styled(" j/k ", key_style),
        Span::styled("scroll", desc_style),
        sep.clone(),
        Span::styled(" PgUp/PgDn ", key_style),
        Span::styled("page", desc_style),
        sep.clone(),
        Span::styled(" ? ", key_style),
        Span::styled("help", desc_style),
    ];
    let short = vec![
        Span::styled(" q", quit_style),
        Span::styled(":quit", desc_style),
        Span::styled(" ^C:exit ", key_style),
        Span::styled(" ?:help", key_style),
    ];
    let line = if width < 80 {
        Line::from(short)
    } else {
        Line::from(full)
    };
    let para = Paragraph::new(line).style(Style::default().fg(CREAM));
    frame.render_widget(para, area);
}

fn draw_help(frame: &mut Frame, area: Rect) {
    use ratatui::widgets::Clear;
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(HARD_RUST))
        .title(Span::styled(
            " Help — ?/Esc to close ",
            Style::default().fg(SAND).add_modifier(Modifier::BOLD),
        ))
        .style(Style::default().fg(CREAM));
    // centered popup 60% x 50%
    let popup_area = {
        let v = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(25), Constraint::Percentage(50), Constraint::Percentage(25)])
            .split(area);
        let h = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(20), Constraint::Percentage(60), Constraint::Percentage(20)])
            .split(v[1]);
        h[1]
    };
    frame.render_widget(Clear, popup_area);
    let help_lines = vec![
        Line::from(Span::styled("Keys", Style::default().fg(EMBER).add_modifier(Modifier::BOLD))),
        Line::from(""),
        Line::from(vec![Span::styled(" q / Q / Ctrl-C  ", Style::default().fg(RUST_ORANGE).add_modifier(Modifier::BOLD)), Span::raw("quit (any phase)")]),
        Line::from(vec![Span::styled(" ← / →          ", Style::default().fg(RUST_ORANGE).add_modifier(Modifier::BOLD)), Span::raw("cycle attention layer")]),
        Line::from(vec![Span::styled(" ↑ / ↓          ", Style::default().fg(RUST_ORANGE).add_modifier(Modifier::BOLD)), Span::raw("cycle attention head")]),
        Line::from(vec![Span::styled(" j / k          ", Style::default().fg(RUST_ORANGE).add_modifier(Modifier::BOLD)), Span::raw("scroll inference 1 line (auto-follow off)")]),
        Line::from(vec![Span::styled(" PgUp / PgDn    ", Style::default().fg(RUST_ORANGE).add_modifier(Modifier::BOLD)), Span::raw("scroll page")]),
        Line::from(vec![Span::styled(" Home / End / G ", Style::default().fg(RUST_ORANGE).add_modifier(Modifier::BOLD)), Span::raw("top / bottom (End re-enables follow)")]),
        Line::from(vec![Span::styled(" ? / Esc        ", Style::default().fg(RUST_ORANGE).add_modifier(Modifier::BOLD)), Span::raw("toggle / close help")]),
        Line::from(""),
        Line::from(Span::styled("Inference auto-follows newest samples; scroll to inspect history.", Style::default().fg(STONE).add_modifier(Modifier::ITALIC))),
        Line::from(Span::styled("Header shows training progress; heatmap fixed 16×16 (q/ctx to quit).", Style::default().fg(STONE).add_modifier(Modifier::ITALIC))),
    ];
    let para = Paragraph::new(help_lines)
        .block(block)
        .wrap(Wrap { trim: true });
    frame.render_widget(para, popup_area);
}
