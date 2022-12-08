use ocl::{Buffer, Kernel, MemFlags, ProQue, OclPrm};

use crate::{constants, parameters, simulation};

const KERNEL_SOURCE: &str = include_str!("./kernels/fdtd.cl");

pub struct KernalProgram {
  queue: ProQue,
  pressure_kernel: Kernel,
  analysis_kernel: Kernel,

  pv_buffer: Buffer<f64>,
  p_buffer: Buffer<f64>,
  pn_buffer: Buffer<f64>,
  analysis_buffer: Buffer<f64>,
  geometry_buffer: Buffer<i8>,
  neighbour_buffer: Buffer<i8>,
}

pub fn create_buffer<T: OclPrm>(queue: &ProQue, flags: MemFlags) -> Buffer<T> {
  return queue
    .buffer_builder::<T>()
    .fill_val(Default::default())
    .flags(flags)
    .build()
    .expect("Cannot create buffer!");
}

// create buffers, queue and program and compile kernels
pub fn create_program(params: &parameters::SimulationParameters) -> ocl::Result<KernalProgram> {
  let platform = ocl::Platform::default();
  println!("Platform: {}", platform.name()?);

  let queue = ProQue::builder()
    .platform(platform)
    .src(KERNEL_SOURCE)
    .dims(params.w_parts * params.h_parts * params.d_parts)
    .build()?;

  let rw_flag = MemFlags::READ_WRITE | MemFlags::ALLOC_HOST_PTR;
  let pv_buffer = create_buffer::<f64>(&queue, rw_flag);
  let p_buffer = create_buffer::<f64>(&queue, rw_flag);
  let pn_buffer = create_buffer::<f64>(&queue, rw_flag);
  let analysis_buffer = create_buffer::<f64>(&queue, rw_flag);
  let geometry_buffer = create_buffer::<i8>(&queue, MemFlags::READ_ONLY | MemFlags::ALLOC_HOST_PTR);
  let neighbour_buffer = create_buffer::<i8>(&queue, MemFlags::READ_ONLY | MemFlags::ALLOC_HOST_PTR);

  let rho_param = -1f64 * constants::RHO_INVERSE * params.dt_over_dx;
  let pressure_kernel = queue
    .kernel_builder("compact_step")
    .arg(&pv_buffer)
    .arg(&p_buffer)
    .arg(&pn_buffer)
    .arg(&geometry_buffer)
    .arg(&neighbour_buffer)
    .arg(params.w_parts as u32)
    .arg(params.h_parts as u32)
    .arg(params.d_parts as u32)
    .arg(params.d1)
    .arg(params.d2)
    .arg(params.d3)
    .arg(params.d4)
    .build()
    .expect("Cannot create step kernel!");

  let analysis_kernel = queue
    .kernel_builder("analysis_step")
    .arg(&p_buffer)
    .arg(&analysis_buffer)
    .arg(&geometry_buffer)
    .arg(params.w_parts as u32)
    .arg(params.h_parts as u32)
    .arg(params.d_parts as u32)
    .arg(params.dt)
    .build()
    .expect("Cannot create analysis kernel!");

  Ok(KernalProgram {
    queue,
    pv_buffer,
    p_buffer,
    pn_buffer,
    analysis_buffer,
    geometry_buffer,
    neighbour_buffer,
    pressure_kernel,
    analysis_kernel,
  })
}

// first write data into buffers
// run program kernels
// read data from buffers
pub fn run_kernels(sim: &mut simulation::Simulation) -> ocl::Result<()> {
  let write_buffers_f64 = [
    (&mut sim.pressure_previous, &mut sim.kernel_prog.pv_buffer),
    (&mut sim.pressure, &mut sim.kernel_prog.p_buffer),
    (&mut sim.analysis, &mut sim.kernel_prog.analysis_buffer),
  ];

  let write_buffers_i8 = [
    (&mut sim.geometry, &mut sim.kernel_prog.geometry_buffer),
    (&mut sim.neighbours, &mut sim.kernel_prog.neighbour_buffer),
  ];

  for (ndarr, buf) in &write_buffers_f64 {
    buf
      .write(ndarr.as_slice().expect("Cannot create slice from array"))
      .enq()
      .expect("Cannot run write operation!");
  }
  
  for (ndarr, buf) in &write_buffers_i8 {
    buf
      .write(ndarr.as_slice().expect("Cannot create slice from array"))
      .enq()
      .expect("Cannot run write operation!");
  }

  unsafe {
    sim
      .kernel_prog
      .pressure_kernel
      .enq()
      .expect("Cannot run pressure kernel!");

    // sim
    //   .kernel_prog
    //   .analysis_kernel
    //   .enq()
    //   .expect("Cannot run analysis kernel!");
  }

  let read_buffers = [
    (&mut sim.pressure_previous, &mut sim.kernel_prog.p_buffer),
    (&mut sim.pressure, &mut sim.kernel_prog.pn_buffer),
    (&mut sim.analysis, &mut sim.kernel_prog.analysis_buffer),
  ];

  for (ndarr, buf) in read_buffers {
    buf
      .read(ndarr.as_slice_mut().expect("Cannot create slice from array"))
      .enq()
      .expect("Cannot run write operation!");
  }

  Ok(())
}
