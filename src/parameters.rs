use crate::{constants::C_AIR, env};

#[derive(Copy, Clone)]
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
  pub lambda_courant: f64,
  pub a: f64,
  pub b: f64,
  pub d1: f64,
  pub d2: f64,
  pub d3: f64,
  pub d4: f64,

  pub w_parts: usize,
  pub h_parts: usize,
  pub d_parts: usize,
  pub grid_size: usize,
}

impl SimulationParameters {
  pub fn new(width: f64, height: f64, depth: f64) -> Self {
    // SIMULATION
    let max_frequency: f64 = env::var_num::<f64>("MAX_FREQUENCY", None);
    let air_dampening: f64 = env::var_num::<f64>("AIR_DAMPENING", Some(1f64));

    let min_wavelength: f64 = C_AIR / max_frequency;
    let d_spatial: f64 = min_wavelength / 16.0; // 16 is slightly arbitrary;
    let dx: f64 = d_spatial;
    let d_temporal: f64 = dx / (3.0f64.sqrt() * C_AIR); // 3 -> 3D
    let dt: f64 = d_temporal;
    let dt_over_dx: f64 = dt / dx;
    let lambda_courant: f64 = (C_AIR * dt) / dx;
    let lambda_2: f64 = lambda_courant * lambda_courant;

    // CALCULATED
    let w_parts: usize = (width / dx).floor() as usize + 1;
    let h_parts: usize = (height / dx).floor() as usize + 1;
    let d_parts: usize = (depth / dx).floor() as usize + 1;
    let grid_size: usize = w_parts * h_parts * d_parts;

    let a = 0f64;
    let b = 0f64;

    let d1 = lambda_2 * (1.0 - 4.0 * a + 4.0 * b);
    let d2 = lambda_2 * (a - 2.0 * b);
    let d3 = lambda_2 * (b);
    let d4 = 2.0 * (1.0 - 3.0 * lambda_2 + 6.0 * lambda_2 * a - 4.0 * b * lambda_2);

    Self {
      width,
      height,
      depth,
      max_frequency,
      air_dampening,
      min_wavelength,
      d_spatial,
      dx,
      d_temporal,
      dt,
      dt_over_dx,
      lambda_courant,
      w_parts,
      h_parts,
      d_parts,
      grid_size,

      a,
      b,
      d1,
      d2,
      d3,
      d4,
    }
  }

  pub fn scale(self, size: f64) -> usize {
    (size / self.dx).floor() as usize
  }
}

// calculate parameters
pub fn create_params() -> SimulationParameters {
  // ROOM SIZE
  let room_width: f64 = env::var_num::<f64>("ROOM_WIDTH", Some(1f64));
  let room_height: f64 = env::var_num::<f64>("ROOM_HEIGHT", Some(1f64));
  let room_depth: f64 = env::var_num::<f64>("ROOM_DEPTH", Some(1f64));
  
  return SimulationParameters::new(room_width, room_height, room_depth);
}
