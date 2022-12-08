use std::f64::consts::PI;

use ndarray::{Array3, Dim};

use crate::{kernels, parameters};

pub struct Source {
  pub position: [usize; 3],
  pub frequency: f64,
  pub pulses: i64,
  pub invert_phase: bool,
  pub start_at: f64,
}

pub struct Simulation<'a> {
  pub geometry: Array3<i8>,
  pub neighbours: Array3<i8>,
  pub pressure: Array3<f64>,
  pub pressure_previous: Array3<f64>,
  pub analysis: Array3<f64>,
  pub time: f64,
  pub iteration: i64,
  pub sources: Vec<Source>,
  pub kernel_prog: kernels::KernalProgram,
  pub params: &'a parameters::SimulationParameters,
}

pub fn create_grid_f64(params: &parameters::SimulationParameters) -> Array3<f64> {
  let shape = Dim([params.w_parts, params.h_parts, params.d_parts]);
  Array3::<f64>::zeros(shape)
}
pub fn create_grid_i8(params: &parameters::SimulationParameters) -> Array3<i8> {
  let shape = Dim([params.w_parts, params.h_parts, params.d_parts]);
  Array3::<i8>::zeros(shape)
}

impl<'a> Simulation<'a> {
  pub fn new(params: &'a parameters::SimulationParameters) -> Self {
    Self {
      geometry: create_grid_i8(params),
      neighbours: create_grid_i8(params),
      pressure_previous: create_grid_f64(params),
      pressure: create_grid_f64(params),
      analysis: create_grid_f64(params),
      time: 0f64,
      iteration: 0,
      sources: vec![],
      params,
      kernel_prog: kernels::create_program(params).expect("Failed to create kernel!"),
    }
  }

  pub fn sources_step(&mut self) {
    for source in self.sources.iter() {
      if source.pulses == 1 {
        // https://stackoverflow.com/questions/35211841/gaussian-wave-generation-with-a-given-central-frequency

        let t0 = source.start_at;
        let t = self.time;
        let rel_t = t - t0;
        let sigma = 0.0015;
        let variance = sigma * sigma;
        let cos_factor = (2.0 * PI * rel_t * source.frequency).cos();
        let sqrt_var = (2.0 * PI * variance).sqrt();
        let exp_factor = -rel_t * rel_t / (2.0 * variance);

        let signal = cos_factor * (exp_factor.exp() / sqrt_var);
        self.pressure[source.position] += signal;
      } else {
        let radial_f = 2f64 * PI * source.frequency;
        let time_active = source.pulses as f64 / source.frequency;
        let break_off = source.start_at + time_active;
        let active = self.time >= source.start_at && (source.pulses == 0 || self.time <= break_off);
        if !active {
          continue;
        }
        let rel_time = self.time - source.start_at;
        let factor = if active {
          if source.invert_phase {
            -1f64
          } else {
            1f64
          }
        } else {
          0f64
        };
        let radial_t = radial_f * rel_time;
        let signal = radial_t.sin() * factor;
        self.pressure[source.position] += signal;
      }
    }
  }

  pub fn calculate_neighbours(&mut self) {
    for ((w, h, d), g) in self.geometry.indexed_iter() {
      let mut neighour_count: i8 = 0;
      if *g == 0 {
        if w > 0 && self.geometry[[w - 1, h, d]] == 0 {
          neighour_count += 1;
        }
        if w < self.params.w_parts - 1 && self.geometry[[w + 1, h, d]] == 0 {
          neighour_count += 1;
        }
        if h > 0 && self.geometry[[w, h - 1, d]] == 0 {
          neighour_count += 1;
        }
        if h < self.params.h_parts - 1 && self.geometry[[w, h + 1, d]] == 0 {
          neighour_count += 1;
        }
        if d > 0 && self.geometry[[w, h, d - 1]] == 0 {
          neighour_count += 1;
        }
        if d < self.params.d_parts - 1 && self.geometry[[w, h , d + 1]] == 0 {
          neighour_count += 1;
        }
      }

      self.neighbours[[w, h, d]] = neighour_count;
    }
  }

  pub fn step(&mut self) {
    kernels::run_kernels(self).expect("Cannot run simulation step");
    self.sources_step();

    self.time += self.params.dt;
    self.iteration += 1;
  }
}
