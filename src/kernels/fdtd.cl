__kernel void pressure_step(__global double *pressure, __global double *vx,
                            __global double *vy, __global double *vz,
                            __global double *geometry, uint size_w, uint size_h,
                            uint size_d, double kappa_dt_dx) {
  size_t i = get_global_id(0);
  size_t w = (i / (size_h * size_d)) % size_w;
  size_t h = (i / (size_d)) % size_h;
  size_t d = i % size_d;

  size_t d_stride = 1;
  size_t h_stride = size_d;
  size_t w_stride = size_d * size_h;
  size_t size = size_d * size_h * size_w;

  if (i < size) {
    double current = pressure[i];
    double delta_velocity = 0;

    if (i + w_stride < size) {
      delta_velocity += vx[i + w_stride] - vx[i];
    }

    if (i + h_stride < size) {
      delta_velocity += vy[i + h_stride] - vy[i];
    }

    if (i + d_stride < size) {
      delta_velocity += vz[i + d_stride] - vz[i];
    }

    pressure[i] = (current + delta_velocity * kappa_dt_dx);
  }
}

__kernel void velocity_step(__global double *pressure, __global double *vx,
                            __global double *vy, __global double *vz,
                            __global double *geometry, uint size_w, uint size_h,
                            uint size_d, double rho_dt_dx) {
  size_t i = get_global_id(0);
  size_t w = (i / (size_h * size_d)) % size_w;
  size_t h = (i / (size_d)) % size_h;
  size_t d = i % size_d;

  size_t d_stride = 1;
  size_t h_stride = size_d;
  size_t w_stride = size_d * size_h;
  size_t size = size_d * size_h * size_w;

  if (i < size) {

    double current_pressure = pressure[i];
    double dpw = 0;
    double dv = 0;
    if (i - w_stride >= 0) {
      dpw = current_pressure - pressure[i - w_stride];
      dv = rho_dt_dx * dpw;
      vx[i] = vx[i] + dv;
    }

    if (i - h_stride >= 0) {
      dpw = current_pressure - pressure[i - h_stride];
      dv = rho_dt_dx * dpw;
      vy[i] = vy[i] + dv;
    }

    if (i - d_stride >= 0) {
      dpw = current_pressure - pressure[i - d_stride];
      dv = rho_dt_dx * dpw;
      vz[i] = vz[i] + dv;
    }
  }
}
