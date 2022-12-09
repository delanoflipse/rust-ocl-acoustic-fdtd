#define D1_SIZE 6
#define D2_SIZE 12
#define D3_SIZE 8

bool in_range(uint size, uint index) { return index > 0 && index < size; }

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

  bool w_plus = in_range(size_w, w + 1);
  bool w_min = in_range(size_w, w - 1);
  bool h_plus = in_range(size_h, h + 1);
  bool h_min = in_range(size_h, h - 1);
  bool d_plus = in_range(size_d, d + 1);
  bool d_min = in_range(size_d, d - 1);

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
  double n_factor = 1.0;

  // TODO: better check (epsilon)
  if (d1 != 0.0) {
    double sum = 0.0;

    if (w_plus)
      sum += pressure[i + w_stride];
    if (w_min)
      sum += pressure[i - w_stride];
    if (h_plus)
      sum += pressure[i + h_stride];
    if (h_min)
      sum += pressure[i - h_stride];
    if (d_plus)
      sum += pressure[i + d_stride];
    if (d_min)
      sum += pressure[i - d_stride];

    d_sum += d1 * sum;
  }

  // TODO: better check (epsilon)
  if (d2 != 0.0) {
    double sum = 0.0;

    if (w_plus && h_plus)
      sum += pressure[i + w_stride + h_stride];
    if (w_min && h_plus)
      sum += pressure[i - w_stride + h_stride];
    if (w_min && h_min)
      sum += pressure[i - w_stride - h_stride];
    if (w_plus && h_min)
      sum += pressure[i + w_stride - h_stride];

    if (d_plus && h_plus)
      sum += pressure[i + d_stride + h_stride];
    if (d_min && h_plus)
      sum += pressure[i - d_stride + h_stride];
    if (d_min && h_min)
      sum += pressure[i - d_stride - h_stride];
    if (d_plus && h_min)
      sum += pressure[i + d_stride - h_stride];

    if (w_plus && d_plus)
      sum += pressure[i + w_stride + d_stride];
    if (w_min && d_plus)
      sum += pressure[i - w_stride + d_stride];
    if (w_min && d_min)
      sum += pressure[i - w_stride - d_stride];
    if (w_plus && d_min)
      sum += pressure[i + w_stride - d_stride];

    d_sum += d2 * sum;
  }

  // TODO: better check (epsilon)
  if (d3 != 0.0) {
    double sum = 0.0;

    if (w_plus && h_plus && d_plus)
      sum += pressure[i + w_stride + h_stride + d_stride];
    if (w_min && h_plus && d_plus)
      sum += pressure[i - w_stride + h_stride + d_stride];
    if (w_plus && h_min && d_plus)
      sum += pressure[i + w_stride - h_stride + d_stride];
    if (w_min && h_min && d_plus)
      sum += pressure[i - w_stride - h_stride + d_stride];
    if (w_plus && h_plus && d_min)
      sum += pressure[i + w_stride + h_stride - d_stride];
    if (w_min && h_plus && d_min)
      sum += pressure[i - w_stride + h_stride - d_stride];
    if (w_plus && h_min && d_min)
      sum += pressure[i + w_stride - h_stride - d_stride];
    if (w_min && h_min && d_min)
      sum += pressure[i - w_stride - h_stride - d_stride];

    d_sum += d3 * sum;
  }

  if (d4 != 0.0) {
    d_sum += d4 * current;
  }

  // if (neighbour_count < 6) {
  //   double n = (double) neighbour_count;
  //   n_factor = n / 12.0;
  // }

  pressure_next[i] = n_factor * (d_sum - previous);
}

__kernel void analysis_step(__global double *pressure,
                            __global double *analysis, __global char *geometry,
                            uint size_w, uint size_h, uint size_d, double dt) {
  size_t i = get_global_id(0);
  size_t w = (i / (size_h * size_d)) % size_w;
  size_t h = (i / (size_d)) % size_h;
  size_t d = i % size_d;

  size_t size = size_d * size_h * size_w;

  if (i >= size || i < 0) {
    return;
  }

  char geometry_type = geometry[i];

  if (geometry_type > 0) {
    return;
  }

  double current_pressure = pressure[i];
  analysis[i] += current_pressure * dt;
}
