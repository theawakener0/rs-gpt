use crate::tui::app::{App, Phase};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Chart, Dataset, Gauge, GraphType, Paragraph, Row, Sparkline, Table, Wrap},
    Frame,
};

fn interpolate_color(t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    // blue (low) -> red (high) via purple
    // simple lerp: low=blue, high=red
    let r = (t * 255.0) as u8;
    let b = ((1.0 - t) * 255.0) as u8;
    let g = (20.0 + t * 40.0 * (1.0 - t) * 4.0) as u8; // slight purple mid
    Color::Rgb(r, g, b)
}

pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    if area.width < 60 || area.height < 20 {
        let msg = Paragraph::new("Terminal too small — resize to at least 80x24 (q to quit)")
            .block(Block::default().borders(Borders::ALL).title(" rs-gpt "))
            .wrap(Wrap { trim: true });
        frame.render_widget(msg, area);
        return;
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Percentage(35), // loss + grad
            Constraint::Percentage(35), // attention
            Constraint::Min(8),     // inference
        ])
        .split(area);

    draw_header(frame, app, chunks[0]);
    draw_loss_row(frame, app, chunks[1]);
    draw_attention(frame, app, chunks[2]);
    draw_inference(frame, app, chunks[3]);
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
        .block(Block::default().borders(Borders::ALL).title(title))
        .gauge_style(Style::default().fg(Color::Cyan).bg(Color::DarkGray).add_modifier(Modifier::BOLD))
        .percent(pct)
        .label(Span::styled(
            format!("{pct}%"),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
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
                .style(Style::default().fg(Color::Cyan))
                .data(&loss_data),
        ]
    };

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Loss  ({} points) ", loss_data.len()))
                .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        )
        .x_axis(
            ratatui::widgets::Axis::default()
                .title("step")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, max_x])
                .labels(vec![
                    Span::raw("0"),
                    Span::raw(format!("{}", app.num_steps / 2)),
                    Span::raw(format!("{}", app.num_steps)),
                ]),
        )
        .y_axis(
            ratatui::widgets::Axis::default()
                .title("loss")
                .style(Style::default().fg(Color::Gray))
                .bounds([y_min, y_max])
                .labels(vec![
                    Span::raw(format!("{:.1}", y_min)),
                    Span::raw(format!("{:.1}", (y_min + y_max) / 2.0)),
                    Span::raw(format!("{:.1}", y_max)),
                ]),
        );
    frame.render_widget(chart, cols[0]);

    // Right column: grad sparkline + stats
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(cols[1]);

    // Sparkline needs &[u64]; scale grad_norms
    let spark_data: Vec<u64> = if app.grad_norms.is_empty() {
        vec![0]
    } else {
        let max = app.grad_norms.iter().cloned().fold(0.0_f64, f64::max).max(1e-6);
        app.grad_norms
            .iter()
            .map(|v| ((v / max) * 8.0).round() as u64)
            .collect()
    };

    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Grad |g| ")
                .title_style(Style::default().fg(Color::Magenta)),
        )
        .data(&spark_data)
        .max(8)
        .style(Style::default().fg(Color::Magenta));
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
            Span::styled("current: ", Style::default().fg(Color::DarkGray)),
            Span::styled(last_loss, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("best:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(best_loss, Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled("phase:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:?}", app.phase),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(Span::styled(
            "q quit │ ←→ layer │ ↑↓ head",
            Style::default().fg(Color::DarkGray),
        )),
    ];
    let stats = Paragraph::new(stats_text).block(Block::default().borders(Borders::ALL).title(" Stats "));
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

    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if !has_attn {
        let p = Paragraph::new("Training… attention weights will appear after first step")
            .style(Style::default().fg(Color::DarkGray))
            .wrap(Wrap { trim: true });
        frame.render_widget(p, inner);
        return;
    }

    let attn = app.attn.as_ref().unwrap();
    // attn shape: [layer][head][query][key]
    if app.selected_layer >= attn.len() {
        let p = Paragraph::new("No attention for selected layer").style(Style::default().fg(Color::Red));
        frame.render_widget(p, inner);
        return;
    }
    let layer = &attn[app.selected_layer];
    if app.selected_head >= layer.len() {
        let p = Paragraph::new("No attention for selected head").style(Style::default().fg(Color::Red));
        frame.render_widget(p, inner);
        return;
    }
    let head = &layer[app.selected_head];
    if head.is_empty() {
        let p = Paragraph::new("Empty attention matrix").style(Style::default().fg(Color::DarkGray));
        frame.render_widget(p, inner);
        return;
    }

    let _n_query = head.len();
    // infer max key len
    let n_key = head.iter().map(|row| row.len()).max().unwrap_or(0);
    if n_key == 0 {
        return;
    }

    // Build table: header row = key positions, then each query row
    let mut rows: Vec<Row> = Vec::new();

    // Constraints: each column fixed width 3 (for "  " with bg + space) or 4
    let constraints: Vec<Constraint> = std::iter::once(Constraint::Length(4))
        .chain((0..n_key).map(|_| Constraint::Length(3)))
        .collect();

    // Header: empty corner + key indices
    let header_cells: Vec<Cell> = std::iter::once(Cell::from("").style(Style::default().fg(Color::DarkGray)))
        .chain((0..n_key).map(|k| {
            Cell::from(format!("{k}")).style(Style::default().fg(Color::DarkGray))
        }))
        .collect();
    let header = Row::new(header_cells).height(1);
    // We will render manually with Table header
    // Build rows for each query
    for (qi, row_weights) in head.iter().enumerate() {
        let mut cells: Vec<Cell> = Vec::with_capacity(n_key + 1);
        cells.push(Cell::from(format!("{qi}")).style(Style::default().fg(Color::Gray)));
        for ki in 0..n_key {
            let w = row_weights.get(ki).copied().unwrap_or(0.0);
            // future masking: query should not attend to future keys; weight ~0 => dim
            let is_future = ki > qi;
            let bg = if is_future {
                Color::Rgb(20, 20, 20)
            } else {
                interpolate_color(w)
            };
            // Use "  " with bg to form heat cell; overlay weight as narrow block
            let cell = Cell::from("  ").style(Style::default().bg(bg).fg(Color::White));
            cells.push(cell);
        }
        rows.push(Row::new(cells).height(1));
    }

    let table = Table::new(rows, constraints)
        .header(header.style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)).height(1))
        .block(Block::default())
        .column_spacing(0);

    frame.render_widget(table, inner);
}

fn draw_inference(frame: &mut Frame, app: &App, area: Rect) {
    let title = format!(
        " Inference — {} samples  (streaming ~30fps) ",
        app.inference_samples.len()
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .title_style(Style::default().fg(Color::Green).add_modifier(Modifier::BOLD));

    let mut lines: Vec<Line> = Vec::new();
    for (i, s) in app.inference_samples.iter().enumerate() {
        lines.push(Line::from(vec![
            Span::styled(
                format!("sample {:2}: ", i + 1),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(s.clone(), Style::default().fg(Color::White)),
        ]));
    }
    if !app.inference_buf.is_empty() && app.phase == Phase::Sampling {
        lines.push(Line::from(vec![
            Span::styled(
                format!("sample {:2}: ", app.inference_samples.len() + 1),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(app.inference_buf.clone(), Style::default().fg(Color::White)),
            Span::styled("█", Style::default().fg(Color::Green).add_modifier(Modifier::SLOW_BLINK)),
        ]));
    } else if app.phase == Phase::Done {
        lines.push(Line::from(Span::styled(
            "— done —  (q to quit, ←→/↑↓ to inspect attention)",
            Style::default().fg(Color::DarkGray),
        )));
    } else if app.inference_samples.is_empty() && app.phase == Phase::Training {
        lines.push(Line::from(Span::styled(
            "Waiting for training to finish…",
            Style::default().fg(Color::DarkGray),
        )));
    }

    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });

    frame.render_widget(para, area);
}
