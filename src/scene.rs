use crate::{simulation, parameters};

pub fn living_room<'a>() -> simulation::Simulation {
  let params = parameters::SimulationParameters::new(7.1, 2.5, 4.1);
  println!("{}w {}h {}d", params.w_parts, params.h_parts, params.d_parts);
  let mut sim = simulation::Simulation::new(params);

  let chimney_w = params.scale(1.1);
  let chimney_d = params.d_parts - params.scale(0.3);
  let chimney_x = params.w_parts - chimney_w - params.scale(2.1);

  for ((w,h,d), _) in sim.geometry.clone().indexed_iter() {
    if w > chimney_x && w < chimney_x + chimney_w && d > chimney_d {
      sim.geometry[[w, h, d]] |= simulation::WALL_FLAG;
    }
  }

  sim.sources.push(simulation::Source {
    frequency: 100.0,
    invert_phase: false,
    position: [chimney_x + chimney_w + params.scale(0.3), params.h_parts / 2, chimney_d + params.scale(0.15)],
    // pulses: i64::MAX,
    pulses: 1,
    start_at: 0.0,
  });

  sim.sources.push(simulation::Source {
    frequency: 100.0,
    invert_phase: false,
    position: [chimney_x / 2, params.h_parts / 2, chimney_d + params.scale(0.15)],
    // pulses: i64::MAX,
    pulses: 1,
    start_at: 0.0,
  });

  return sim;
}

// pub fn bed_room(sim: &mut simulation::simulation) {
  
// }