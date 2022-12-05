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

use ndarray::s;
use piston_window::{clear, rectangle, EventLoop, PistonWindow, WindowSettings};
use std::cmp;
use std::time::Instant;
extern crate piston_window;

extern crate ocl;

mod constants;
mod kernels;
mod parameters;
mod simulation;

const ITERATIONS: i64 = 50000;
const HEADLESS: bool = false;

const FPS: u64 = 60;
const TARGET_WINDOW_WIDTH: u32 = 500;
const TARGET_WINDOW_HEIGHT: u32 = 500;
const IT_PER_SETP: u32 = 3;

// TODO: split up properly between modes
fn main() {
  let params = parameters::create_params();
  let mut sim = simulation::Simulation::new(&params);

  sim.sources.push(simulation::Source {
    frequency: 200.0,
    invert_phase: false,
    position: [params.w_parts / 2, params.h_parts / 2, params.d_parts / 2],
    pulses: i64::MAX,
    start_at: 0.0,
  });

  println!("Starting simulation!");

  if HEADLESS {
    let now = Instant::now();
    for _ in 0..ITERATIONS {
      // while true {
      sim.step();
    }
    // ns
    let simulated = sim.time * 1000f64 * 1000f64 / (ITERATIONS as f64);
    let elapsed = (now.elapsed().as_micros() as f64) / (ITERATIONS as f64);
    println!("Elapsed: {:.2?} IRL, {:.2?} simulated", now.elapsed(), sim.time);
    println!("Average: {:.2?}us per simulation", simulated);
    println!("Factor: {:.2}x", elapsed / simulated);
    println!("Ran simulation!");
  } else {
    let cell_size = cmp::min(TARGET_WINDOW_WIDTH, TARGET_WINDOW_HEIGHT) as f64
      / cmp::max(params.w_parts, params.d_parts) as f64;

    let w_width = (cell_size * (params.w_parts as f64)) as u32;
    let w_height = (cell_size * (params.d_parts as f64)) as u32;

    let mut window: PistonWindow = WindowSettings::new("Simulation Viewer", [w_width, w_height])
      .exit_on_esc(true)
      .build()
      .unwrap();

    window.set_max_fps(FPS);

    while let Some(e) = window.next() {
      window.draw_2d(&e, |c, gfx, _device| {
        for _ in 0..IT_PER_SETP {
          sim.step();
        }
        clear([1.0; 4], gfx);

        // sim.pressure.slice_axis(axis, indices);
        let slice = sim.pressure.slice(s![.., params.d_parts / 2, ..]);

        let mut max_value = 1e-99f64;
        let h = params.h_parts / 2;
        for v in slice.iter() {
          max_value = v.abs().max(max_value);
        }
        let scalar = max_value as f32;
        println!("Time: {}s ({}) {}", sim.time, sim.iteration, max_value);

        for ((w, d), v) in slice.indexed_iter() {
          let geometry = sim.geometry[[w, h, d]] as f32;
          let p_full = sim.pressure[[w, h, d]] as f32;
          let p = p_full / scalar;
          let r = if p > 0f32 { p } else { 0f32 };
          let b = if p < 0f32 { -p } else { 0f32 };
          let g = 0.0;

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
