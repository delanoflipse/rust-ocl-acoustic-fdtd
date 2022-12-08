#define D1_SIZE 6
#define D2_SIZE 12
#define D3_SIZE 8

bool is_valid_position(uint size, uint index) {
  return index > 0 && index < size - 1;
}

__kernel void compact_step(__global double *previous_pressure,
                           __global double *pressure,
                           __global double *pressure_next,
                           __global char *geometry, __global char *neighbours,
                           uint size_w, uint size_h, uint size_d, double d1,
                           double d2, double d3, double d4) {
  size_t i = get_global_id(0);
  size_t w = (i / (size_h * size_d)) % size_w;
  size_t h = (i / (size_d)) % size_h;
  size_t d = i % size_d;

  size_t d_stride = 1;
  size_t h_stride = size_d;
  size_t w_stride = size_d * size_h;
  size_t size = size_d * size_h * size_w;
  char geometry_type = geometry[i];
  char neighbour_count = neighbours[i];

  if (i >= size || i < 0) {
    return;
  }

  if (geometry_type > 0 || neighbour_count == 0) {
    pressure_next[i] = 0.0;
    return;
  }

  double current = pressure[i];
  double previous = previous_pressure[i];
  double d_sum = 0.0;

  if (d1 != 0.0) {
    double sum = 0.0;
    uint positions[D1_SIZE] = {
        i + w_stride, i - w_stride, i + h_stride,
        i - h_stride, i + d_stride, i - d_stride,
    };

    for (int i = 0; i < D1_SIZE; i++) {
      uint pos = positions[i];
      if (is_valid_position(size, pos)) {
        sum += pressure[pos];
      }
    }

    d_sum += d1 * sum;
  }

  if (d2 != 0.0) {
    double sum = 0.0;
    uint positions[D2_SIZE] = {
        i + w_stride + h_stride, i - w_stride + h_stride,
        i + w_stride - h_stride, i - w_stride - h_stride,

        i + h_stride + d_stride, i - h_stride + d_stride,
        i + h_stride - d_stride, i - h_stride - d_stride,

        i + w_stride + d_stride, i - w_stride + d_stride,
        i + w_stride - d_stride, i - w_stride - d_stride,
    };

    for (int i = 0; i < D2_SIZE; i++) {
      uint pos = positions[i];
      if (is_valid_position(size, pos)) {
        sum += pressure[pos];
      }
    }

    d_sum += d2 * sum;
  }

  if (d3 != 0.0) {
    double sum = 0.0;
    uint positions[D3_SIZE] = {
        i + w_stride + h_stride + d_stride, i - w_stride + h_stride + d_stride,
        i + w_stride - h_stride + d_stride, i - w_stride - h_stride + d_stride,
        i + w_stride + h_stride - d_stride, i - w_stride + h_stride - d_stride,
        i + w_stride - h_stride - d_stride, i - w_stride - h_stride - d_stride,
    };

    for (int i = 0; i < D3_SIZE; i++) {
      uint pos = positions[i];
      if (is_valid_position(size, pos)) {
        sum += pressure[pos];
      }
    }

    d_sum += d3 * sum;
  }

  if (d4 != 0.0) {
    d_sum += d4 * current;
  }

  pressure_next[i] = d_sum - previous;
}

__kernel void analysis_step(__global double *pressure,
                            __global double *analysis,
                            __global char *geometry, uint size_w, uint size_h,
                            uint size_d, double rho_dt_dx) {
  size_t i = get_global_id(0);
  size_t w = (i / (size_h * size_d)) % size_w;
  size_t h = (i / (size_d)) % size_h;
  size_t d = i % size_d;

  size_t size = size_d * size_h * size_w;

  if (i < size) {
    double current_pressure = pressure[i];
  }
}
