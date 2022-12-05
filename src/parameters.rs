use crate::constants::C_AIR;

// ROOM SIZE
const WIDTH: f64 = 3.6f64;
const DEPTH: f64 = 4.3f64;
const HEIGHT: f64 = 2.4f64;

// # SIMULATION
const MAX_FREQUENCY: f64 = 400f64;
const AIR_DAMPENING: f64 = 0.999;

pub struct SimulationParameters {
  pub width: f64,
  pub height: f64,
  pub depth: f64,
  pub max_frequency: f64,
  pub air_dampening: f64,

  pub min_wavelength: f64,
  pub d_spatial: f64,
  pub dx: f64,
  pub d_temporal: f64,
  pub dt: f64,
  pub dt_over_dx: f64,

  pub w_parts: usize,
  pub h_parts: usize,
  pub d_parts: usize,
  pub grid_size: usize,
}

// calculate parameters
pub fn create_params() -> SimulationParameters {
  let min_wavelength: f64 = C_AIR / MAX_FREQUENCY;
  let d_spatial: f64 = min_wavelength / 16.0; // 16 is slightly arbitrary;
  let dx: f64 = d_spatial;
  let d_temporal: f64 = dx / (3.0f64.sqrt() * C_AIR); // 3 -> 3D
  let dt: f64 = d_temporal;
  let dt_over_dx: f64 = dt / dx;

  // # CALCULATED
  let w_parts: usize = (WIDTH / dx).floor() as usize + 1;
  let h_parts: usize = (HEIGHT / dx).floor() as usize + 1;
  let d_parts: usize = (DEPTH / dx).floor() as usize + 1;
  let grid_size: usize = w_parts * h_parts * d_parts;

  SimulationParameters {
    width: WIDTH,
    height: HEIGHT,
    depth: DEPTH,
    max_frequency: MAX_FREQUENCY,
    air_dampening: AIR_DAMPENING,
    min_wavelength,
    d_spatial,
    dx,
    d_temporal,
    dt,
    dt_over_dx,
    w_parts,
    h_parts,
    d_parts,
    grid_size,
  }
}
