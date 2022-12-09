#![warn(clippy::cargo,
  clippy::complexity,
  clippy::correctness,
  clippy::nursery,
  clippy::perf,
  clippy::style,
  clippy::suspicious,
  clippy::todo,
  // clippy::pedantic,
  // clippy::unwrap_used,
  // clippy::expect_used,
)]

extern crate dotenv;
extern crate ocl;
extern crate piston_window;

use dotenv::dotenv;
use ndarray::s;
use piston_window::{clear, rectangle, EventLoop, PistonWindow, WindowSettings};
use std::cmp;
use std::time::Instant;
use human_bytes::human_bytes;

mod constants;
mod env;
mod kernels;
mod parameters;
mod simulation;

// TODO: split up properly between modes
fn main() {
  dotenv().ok();

  let headless_mode: bool = env::var_bool("HEADLESS", Some(false));

  let params = parameters::create_params();
  let mut sim = simulation::Simulation::new(&params);

  for ((w,h,d), _) in sim.geometry.clone().indexed_iter() {
    if w > 2 * sim.params.w_parts / 3 && d > 2 * sim.params.d_parts / 3 {
      sim.geometry[[ w,h, d]] = 1i8;
    }
  }

  sim.sources.push(simulation::Source {
    frequency: 800.0,
    invert_phase: false,
    position: [params.w_parts / 3, params.h_parts / 2, params.d_parts / 3],
    // pulses: i64::MAX,
    pulses: 1,
    start_at: 0.0,
  });


  // TODO: geometry
  sim.calculate_neighbours();

  println!("{}", human_bytes(params.grid_size as f64 * 8f64 * 5f64));
  println!("Starting simulation!");
  for key in ["MAX_FREQUENCY", "HEADLESS"] {
    println!("[ENV] {}: {}", key, dotenv::var(key).unwrap());
  }

  if headless_mode {
    let iteration_count: i64 = env::var_num::<i64>("SIM_ITERATIONS", None);
    let now = Instant::now();
    for _ in 0..iteration_count {
      // while true {
      sim.step();
    }
    // ns
    let simulated = sim.time * 1000f64 * 1000f64 / (iteration_count as f64);
    let elapsed = (now.elapsed().as_micros() as f64) / (iteration_count as f64);
    println!(
      "Elapsed: {:.2?} IRL, {:.2?} simulated",
      now.elapsed(),
      sim.time
    );
    println!("Average: {:.2?}us per simulation", elapsed);
    println!("Factor: {:.2}x", elapsed / simulated);
    println!("Ran simulation!");
  } else {
    let max_fps: u64 = 60;
    let target_window_size: u32 = env::var_num::<u32>("WINDOW_SIZE", Some(500u32));
    let iterations_per_step: u32 = env::var_num::<u32>("ITERATIONS_PER_STEP", Some(1u32));

    let cell_size =(target_window_size) as f64
      / cmp::max(params.w_parts, params.d_parts) as f64;

    let w_width = (cell_size * (params.w_parts as f64)) as u32;
    let w_height = (cell_size * (params.d_parts as f64)) as u32;

    let mut window: PistonWindow = WindowSettings::new("Simulation Viewer", [w_width, w_height])
      .exit_on_esc(true)
      .build()
      .unwrap();

    window.set_max_fps(max_fps);

    while let Some(e) = window.next() {
      window.draw_2d(&e, |c, gfx, _device| {
        for _ in 0..iterations_per_step {
          sim.step();
        }
        clear([1.0; 4], gfx);

        let h = params.h_parts / 2 + 1;
        // sim.pressure.slice_axis(axis, indices);
        let slice = sim.pressure.slice(s![.., h, ..]);

        let mut max_value = 0f64;
        let mut max_analysis_value = 0f64;
        for ((w, d), v) in slice.indexed_iter() {
          if sim.is_source_position(w, h, d) || sim.neighbours[[w,h,d]] < 6 {
            continue;
          }
          max_value = v.abs().max(max_value);
          max_analysis_value = sim.analysis[[w,h,d]].abs().max(max_analysis_value);
        }
        let scalar = max_value as f32;
        let scalar_analysis = max_analysis_value as f32;
        println!("Time: {}s ({}) {}", sim.time, sim.iteration, max_value);

        if max_value == 0.0f64 {
          return;
        }

        for ((w, d), v) in slice.indexed_iter() {
          let geometry = sim.geometry[[w, h, d]];
          let neighbours = sim.neighbours[[w, h, d]];
          let analysis = sim.analysis[[w, h, d]];
          let p = (*v as f32) / scalar;
          let analysis = (analysis as f32) / scalar_analysis;
          let r = if p > 0f32 { p } else { 0f32 };
          let b = if p < 0f32 { -p } else { 0f32 };
          // let r = if analysis > 0f32 { analysis } else { 0f32 };
          // let b = if analysis < 0f32 { -analysis } else { 0f32 };
          let g = if neighbours > 0 && neighbours < 6 {(6 - neighbours) as f32 / 5f32} else {0f32};

          let x = w as f64 * cell_size;
          let y = d as f64 * cell_size;

          rectangle(
            [r, g, b, 1.0], // red
            [x, y, cell_size, cell_size],
            c.transform,
            gfx,
          );
        }
      });
    }
  }
}
