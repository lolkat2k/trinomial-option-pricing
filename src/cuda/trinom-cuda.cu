#include <math.h>  // exp

/* Data structures */
struct YieldCurveData {
  double P; // Discount Factor Function
  double t; // Time [days]
};

struct OptionData {
  double strike;
  double maturity;
  int number_of_terms;
  double reversion_rate;
  double volatility;
};

/* Probability measures for tree construction */
// Exhibit 1A (-jmax < j < jmax)
#define PU_A(j, M) ((1/6) + (((double)(j*j))*M*M + ((double)j)*M) * (1/2))
#define PM_A(j, M) ((2/3) -  ((double)(j*j))*M*M)
#define PD_A(j, M) ((1/6) + (((double)(j*j))*M*M - ((double)j)*M) * (1/2))

// Exhibit 1B (j == -jmax)
#define PU_B(j, M) ( (1/6) + (((double)(j*j))*M*M  - ((double)j)*M) * (1/2))
#define PM_B(j, M) (-(1/3) - (((double)(j*j))*M*M) - (2*((double)j)*M))
#define PD_B(j, M) ( (7/6) + (((double)(j*j))*M*M  - (3*(double)j)*M) * (1/2))

// Exhibit 1C (j == jmax)
#define PU_C(j, M) ( (7/6) + (((double)(j*j))*M*M  + (3*(double)j)*M) * (1/2))
#define PM_C(j, M) (-(1/3) - (((double)(j*j))*M*M) - (2*((double)j)*M))
#define PD_C(j, M) ( (1/6) + (((double)(j*j))*M*M  + ((double)j)*M) * (1/2))

/* forward propagation helper */
double forward_helper(double M, double dr, double dt, double alphai,
		      double *QCopy, int beg_ind, int m, int i,
		      int imax, int jmax, int j)
{
  double eRdt_u1 = exp(-((double)(j+1)*dr+alphai)*dt);
  double eRdt    = exp(-((double)(j)  *dr+alphai)*dt);
  double eRdt_d1 = exp(-((double)(j-1)*dr+alphai)*dt);
  double res;
  double pu, pm, pd;

  if (i < jmax) {
    pu = PU_A(j-1, M);
    pm = PM_A(j,   M);
    pd = PD_A(j+1, M);

    if((i == 0) && (j == 0)) {
      res = pm*QCopy[beg_ind+j+m]*eRdt;
    } else if(j == (-imax + 1)) {
      res = pd*QCopy[beg_ind+j+m+1]*eRdt_u1 + pm*QCopy[beg_ind+j+m]*eRdt;
    } else if(j == ( imax - 1)) {
      res = pm*QCopy[beg_ind+j+m]*eRdt + pu*QCopy[beg_ind+j+m-1]*eRdt_d1;
    } else if(j == (-imax)) {
      res = pd*QCopy[beg_ind+j+m+1]*eRdt_u1;
    } else if(j == ( imax)) {
      res = pu*QCopy[beg_ind+j+m-1]*eRdt_d1;
    } else {
      res = // return
	pd*QCopy[beg_ind+j+m+1]*eRdt_u1 +
	pm*QCopy[beg_ind+j+m]*eRdt +
	pu*QCopy[beg_ind+j+m-1]*eRdt_d1;
    }
  } // END_OF: (i < jmax) {
  else if(j == jmax) {
    pm = PU_C(j,   M);
    pu = PU_A(j-1, M);
    res = pm*QCopy[beg_ind+j+m]*eRdt + pu*QCopy[beg_ind+j-1+m] * eRdt_d1; // return
  } // END_OF: (j == jmax)
  else if(j == (jmax - 1)) {
    pd = PM_C(j+1, M);
    pm = PM_A(j  , M);
    pu = PU_A(j-1, M);
    res =
      pd*QCopy[beg_ind+j+1+m]*eRdt_u1 +
      pm*QCopy[beg_ind+j+m]*eRdt +
      pu*QCopy[beg_ind+j-1+m]*eRdt_d1;
  } // END_OF: (j == (jmax - 1))
  else if(j == (jmax - 2)) {
    double eRdt_u2 = exp(-(((double)(j-2))*dr + alphai) * dt);
    double pd_c = PD_C(j + 2, M);
    pd = PD_A(j + 1, M);
    pm = PM_A(j, M);
    pu = PU_A(j - 1, M);
    res =
      pd_c*QCopy[beg_ind+j+2+m]*eRdt_u2 +
      pd*QCopy[beg_ind+j+1+m]*eRdt_u1 +
      pm*QCopy[beg_ind+j+m]*eRdt +
      pu*QCopy[beg_ind+j-1+m]*eRdt_d1;
  } // END_OF: (jmax - 2))
  else if(j == (-jmax + 2)) {
    double eRdt_d2 = exp(-(((double)(j-2))*dr + alphai) * dt);
    double pu_b = PU_B(j - 2, M);
    pd = PD_A(j + 1, M);
    pm = PM_A(j,     M);
    pu = PU_A(j - 1, M);
    res =
      pd*QCopy[beg_ind+j+1+m]*eRdt_u1 +
      pm*QCopy[beg_ind+j+m]*eRdt +
      pu*QCopy[beg_ind+j-1+m]*eRdt_d1 +
      pu_b*QCopy[beg_ind+j-2+m]*eRdt_d2;
  } // END_OF: (j == (-jmax + 2))
  else if(j == (-jmax + 1)) {
    pd = PD_A(j + 1, M);
    pm = PM_A(j, M);
    pu = PM_B(j - 1, M);
    res =
      pd*QCopy[beg_ind+j+1+m]*eRdt_u1 +
      pm*QCopy[beg_ind+j+m]*eRdt +
      pu*QCopy[beg_ind+j-1+m]*eRdt_d1;
  } // END_OF: (j == (-jmax + 1))
  else if(j == (-jmax)) {
    pd = PD_A(j + 1, M);
    pm = PD_B(j,     M);
    res = pd*QCopy[beg_ind+j+1+m]*eRdt_u1 + pm*QCopy[beg_ind+j+m]*eRdt;
  } // END_OF: (-jmax)
  else {
    pd = PD_A(j + 1, M);
    pm = PM_A(j,     M);
    pu = PU_A(j - 1, M);
    res =
      pd*QCopy[beg_ind+j+1+m]*eRdt_u1 +
      pm*QCopy[beg_ind+j+m]*eRdt +
      pu*QCopy[beg_ind+j-1+m]*eRdt_d1;
  } // END_OF: default

  return res;
}


/* backward propagation helper */
double backward_helper()
{
  double eRdt = exp(-((double)(j)  *dr+alphai)*dt);
  double res;
  double pu, pm, pd;

  // define res in big if-statement
  if (i < jmax) {
    // central node
    pu = PU_A(j, M);
    pm = PM_A(j, M);
    pd = PD_A(j, M);
    res = (pu*CallCopy[beg_ind+j+m+1] +
	   pm*CallCopy[beg_ind+j+m] +
	   pd*CallCopy[beg_ind+j+m-1]) *
	  eRdt;
  } else if(j == jmax) {
    // top node
    pu = PU_C(j, M);
    pm = PM_C(j, M);
    pd = PD_C(j, M);
    res = (pu*CallCopy[beg_ind+j+m] +
	   pm*CallCopy[beg_ind+j+m-1] +
	   pd*CallCopy[beg_ind+j+m-2]) *
          eRdt;
  } else if(j == -jmax) {
    // bottom node
    pu = PU_B(j, M);
    pm = PM_B(j, M);
    pd = PD_B(j, M);
    res = (pu*CallCopy[beg_ind+j+m+2] +
	   pm*CallCopy[beg_ind+j+m+1] +
	   pd*CallCopy[beg_ind+j+m]) *
          eRdt
  } else {
    // central node
    pu = PU_A(j, M);
    pm = PM_A(j, M);
    pd = PD_A(j, M);
    res = (pu*CallCopy[beg_ind+j+m+1] +
	   pm*CallCopy[beg_ind+j+m] +
	   pd*CallCopy[beg_ind+j+m-1]) *
          eRdt;
  }

  // OBS: need to define your own max function
  if(i == ((int)(3 / dt))) { res = max_for_doubles(X - res, 0); }

  return res;
}
// START: Cosmin's
// int option_chunk_x = options_in_chunk[blockIdx.x];
// int *option_in_chunk_x = option_indices + blockIdx.x * max_options_in_chunk;
// double X = strikes[options_in_chunk[i]];

/* trinomial chunk kernel */
__global__ void trinom_chunk_kernel(double yield_curve,
				    double *strikes,
				    double *maturities,
				    double *reversion_rates,
				    double *volatilities,
				    int *num_termss,
				    int n_max,          // maximum number of time steps
				    int *options_in_chunk, // size: [number_of_blocks]
				    int *option_indices,   // size: [number_of_blocks][maxOptionsInChunk]
				    int max_options_in_chunk)
{
  // computing option id
  unsigned int lid = threadIdx.x;
  int option_chunk = options_in_chunk[blockIdx.x];
  int *options_in_chunk = option_indices + blockIdx.x * max_options_in_chunk;
  int option_id = options_in_chunk[lid];

  // computing option specific values
  double X = strikes[option_id];
  double T = maturities[option_id];
  int n    = num_termss[option_id];
  double a = reversion_rates[option_id];
  double sigma = volatilities[option_id];
  double dt = T / ((double) n);
  double V  = sigma * sigma * (1 - (exp(0.0 - 2.0 * a * dt)) ) / (2.0 * a);
  double dr = sqrt((1.0 + 2.0) * V);
  double M  = exp(0.0 - a * dt) - 1.0;
  double jmax = ((int)(-0.184 / M)) + 1;
  int m = jmax + 2;

  if(lid < sum_of_qlens_in_block) { // maybe some guard here
    // 1. forward iteration
    for(int i=0; i<n_max; i++) {
      if() { // guard because of __synthreads
      }
      // barrier because of dependency between q_{i} and q_{i+1}
      __syncthreads();
    }

    // 2. backward iteration
    for(int i=0; i<n_max; i++) {
      if() { // guard because of __synthreads
      }
      // barrier because of dependency between c_{i-1} and c_{i}
      __syncthreads();
    }
  } // END: lid < sum_of_qlens_in_block
}


int main()
{
  // small.in - should be read from file
  double strike[1]         = 0.7965300572556244; // long double?
  double maturity[1]       = 9.0000;
  int num_terms[1]         = 108;
  double reversion_rate[1] = 0.1000;
  double volatility[1]     = 0.0100;

  // set maximum chunk size
  int w = 256;

  // start out with: (assuming that all options are equal)
  // n_max := options[0].n
  // m_max := options[0].m

  // compute: chunks
  // each thread should know here to read from when data is
  // in global device memory

  // copy data from host to device

  // compute block and grid dimensions
  // - block: (1, 1, w)
  // - grid:  (1, ceil(sum(Qlen) / w)

  // execute kernel

  // copy data from device to host

  return 0;
}
