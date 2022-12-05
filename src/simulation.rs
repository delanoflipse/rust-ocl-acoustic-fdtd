use std::f64::consts::PI;

use ndarray::{Array3, Dim};

use crate::{parameters, kernels};

pub struct Source {
  pub position: [usize; 3],
  pub frequency: f64,
  pub pulses: i64,
  pub invert_phase: bool,
  pub start_at: f64,
}

pub struct Simulation<'a> {
  pub geometry: Array3<f64>,
  pub neighbours: Array3<f64>,
  pub pressure: Array3<f64>,
  pub velocity_x: Array3<f64>,
  pub velocity_y: Array3<f64>,
  pub velocity_z: Array3<f64>,
  pub analysis: Array3<f64>,
  pub time: f64,
  pub iteration: i64,
  pub sources: Vec<Source>,
  pub kernel_prog: kernels::KernalProgram,
  pub params: &'a parameters::SimulationParameters,
}

pub fn create_grid(params: &parameters::SimulationParameters) -> Array3<f64> {
  let shape = Dim([params.w_parts, params.h_parts, params.d_parts]);
  Array3::<f64>::zeros(shape)
}

impl<'a> Simulation<'a> {
  pub fn new(params: &'a parameters::SimulationParameters) -> Self {
    Self {
      geometry: create_grid(params),
      neighbours: create_grid(params),
      pressure: create_grid(params),
      velocity_x: create_grid(params),
      velocity_y: create_grid(params),
      velocity_z: create_grid(params),
      analysis: create_grid(params),
      time: 0f64,
      iteration: 0,
      sources: vec![],
      params,
      kernel_prog: kernels::create_program(params).expect("Failed to create kernel!"),
    }
  }
  
  pub fn sources_step(&mut self) {
    for source in self.sources.iter() {
      let radial_f = 2f64 * PI * source.frequency;
      let time_active = source.pulses as f64 / source.frequency;
      let break_off = source.start_at + time_active;
      let active = self.time >= source.start_at && (source.pulses == 0 || self.time <= break_off);
      if !active {
        continue;
      }
      let rel_time = self.time - source.start_at;
      let factor = if active {if source.invert_phase {-1f64} else {1f64}} else {0f64};
      let radial_t = radial_f * rel_time;
      let signal = radial_t.sin() * factor;
      self.pressure[source.position] = signal;
    }
  }
  
  pub fn step(&mut self) {
    kernels::run_kernels(self).expect("Cannot run simulation step");
    self.sources_step();

    self.time += self.params.dt;
    self.iteration += 1;
  }
}
