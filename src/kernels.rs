use ocl::{flags, Buffer, Kernel, MemFlags, ProQue, core::Mem};

use crate::{constants, parameters, simulation};

const KERNEL_SOURCE: &str = include_str!("./kernels/fdtd.cl");

pub struct KernalProgram {
  queue: ProQue,
  pressure_kernel: Kernel,
  velocity_kernel: Kernel,

  p_buffer: Buffer<f64>,
  vx_buffer: Buffer<f64>,
  vy_buffer: Buffer<f64>,
  vz_buffer: Buffer<f64>,
  geometry_buffer: Buffer<f64>,
}

pub fn create_buffer(queue: &ProQue, flags: MemFlags) -> Buffer<f64> {
  return queue
    .buffer_builder::<f64>()
    .fill_val(Default::default())
    .flags(flags)
    .build()
    .expect("Cannot create buffer!");
}

// create buffers, queue and program and compile kernels
pub fn create_program(params: &parameters::SimulationParameters) -> ocl::Result<KernalProgram> {
  let queue = ProQue::builder()
    .src(KERNEL_SOURCE)
    .dims(params.w_parts * params.h_parts * params.d_parts)
    .build()?;

  let rw_flag = MemFlags::READ_WRITE | MemFlags::ALLOC_HOST_PTR;
  let p_buffer = create_buffer(&queue, rw_flag);
  let vx_buffer = create_buffer(&queue, rw_flag);
  let vy_buffer = create_buffer(&queue, rw_flag);
  let vz_buffer = create_buffer(&queue, rw_flag);
  let geometry_buffer = create_buffer(&queue, MemFlags::READ_ONLY | MemFlags::ALLOC_HOST_PTR);

  let rho_param = -1f64 * constants::RHO_INVERSE * params.dt_over_dx;
  let pressure_kernel = queue
    .kernel_builder("pressure_step")
    .arg(&p_buffer)
    .arg(&vx_buffer)
    .arg(&vy_buffer)
    .arg(&vz_buffer)
    .arg(&geometry_buffer)
    .arg(params.w_parts as u32)
    .arg(params.h_parts as u32)
    .arg(params.d_parts as u32)
    .arg(rho_param)
    .build()
    .expect("Cannot create pressure kernel!");

  let kappa_param = -1f64 * constants::KAPPA * params.dt_over_dx;
  let velocity_kernel = queue
    .kernel_builder("velocity_step")
    .arg(&p_buffer)
    .arg(&vx_buffer)
    .arg(&vy_buffer)
    .arg(&vz_buffer)
    .arg(&geometry_buffer)
    .arg(params.w_parts as u32)
    .arg(params.h_parts as u32)
    .arg(params.d_parts as u32)
    .arg(kappa_param)
    .build()
    .expect("Cannot create velocity kernel!");

  Ok(KernalProgram {
    queue,
    p_buffer,
    vx_buffer,
    vy_buffer,
    vz_buffer,
    geometry_buffer,
    pressure_kernel,
    velocity_kernel,
  })
}

// first write data into buffers
// run program kernels
// read data from buffers
pub fn run_kernels(sim: &mut simulation::Simulation) -> ocl::Result<()> {
  sim
    .kernel_prog
    .p_buffer
    .write(sim.pressure.as_slice().expect("Cannot copy pressure data"))
    .enq()
    .expect("Cannot run write operation!");

  sim
    .kernel_prog
    .vx_buffer
    .write(sim.velocity_x.as_slice().expect("Cannot copy vx data"))
    .enq()
    .expect("Cannot run write operation!");

  sim
    .kernel_prog
    .vy_buffer
    .write(sim.velocity_y.as_slice().expect("Cannot copy vy data"))
    .enq()
    .expect("Cannot run write operation!");

  sim
    .kernel_prog
    .vz_buffer
    .write(sim.velocity_z.as_slice().expect("Cannot copy vz data"))
    .enq()
    .expect("Cannot run write operation!");

  // TODO: write oncer
  sim
    .kernel_prog
    .geometry_buffer
    .write(sim.geometry.as_slice().expect("Cannot copy geometry data"))
    .enq()
    .expect("Cannot run write operation!");

  unsafe {
    sim
      .kernel_prog
      .velocity_kernel
      .enq()
      .expect("Cannot run velocity kernel!");

    sim
      .kernel_prog
      .pressure_kernel
      .enq()
      .expect("Cannot run pressure kernel!");
  }

  sim
    .kernel_prog
    .p_buffer
    .read(
      sim
        .pressure
        .as_slice_mut()
        .expect("Cannot write pressure data"),
    )
    .enq()
    .expect("Cannot run read operation!");

  sim
    .kernel_prog
    .vx_buffer
    .read(sim.velocity_x.as_slice_mut().expect("Cannot write vx data"))
    .enq()
    .expect("Cannot run read operation!");

  sim
    .kernel_prog
    .vy_buffer
    .read(sim.velocity_y.as_slice_mut().expect("Cannot write vy data"))
    .enq()
    .expect("Cannot run read operation!");

  sim
    .kernel_prog
    .vz_buffer
    .read(sim.velocity_z.as_slice_mut().expect("Cannot write vz data"))
    .enq()
    .expect("Cannot run read operation!");

  Ok(())
}
