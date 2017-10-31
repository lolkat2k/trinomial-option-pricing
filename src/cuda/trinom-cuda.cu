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
    double eRdt_u2 = exp(-(((double)(j+2))*dr + alphai) * dt);
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
double backward_helper(double X, double M, double dr, double dt, 
		       double alphai, double *CallCopy, int beg_ind, int m, 
		       int i, int jmax, int j)
{
  double eRdt = exp(-((double)(j)*dr + alphai)*dt);
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
          eRdt;
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
  // The 3 is the length of contract. Here 3 years. Maybe parameterize it?
  if(i == ((int)(3 / dt))) { res = max_for_doubles(X - res, 0); }

  return res;
}
// START: Cosmin's
// int option_chunk_x = options_in_chunk[blockIdx.x];
// int *option_in_chunk_x = option_indices + blockIdx.x * max_options_in_chunk;
// double X = strikes[options_in_chunk[i]];

/* trinomial chunk kernel */
__global__ void trinom_chunk_kernel(
            float *yield_curve,
				    double *strikes,
				    double *maturities,
				    double *reversion_rates,
				    double *volatilities,
				    int *num_terms,
				    int n_max,          // maximum number of time steps
				    int *options_in_chunk, // size: [number_of_blocks]
				    int *option_indices,   // size: [number_of_blocks][maxOptionsInChunk]
				    int max_options_in_chunk,
            float *alphass)
{
  extern __shared__ char sh_mem[];

  volatile float* tmp_scan = (float*) sh_mem;
  volatile float* Qs       = (float*) (tmp_scan + blockDim.x);
  volatile int* flags      = (int*) (Qs + blockDim.x);
  volatile int* sgm_inds   = (int*) (flags + blockDim.x);
  
  volatile float* Xs    = (float*)(sgm_inds + blockDim.x);
  volatile float* ns    = (float*)(Xs    + max_options_in_chunk);
  volatile float* dts   = (float*)(ns    + max_options_in_chunk);
  volatile float* drs   = (float*)(dts   + max_options_in_chunk);
  volatile float* Ms    = (float*)(drs   + max_options_in_chunk);
  volatile float* jmaxs = (float*)(Ms    + max_options_in_chunk);
  volatile float* ms    = (float*)(jmaxs + max_options_in_chunk);

  // computing option id
  //unsigned int lid = threadIdx.x;
  int num_options = options_in_chunk[blockIdx.x];
  int *options_in_chunk = option_indices + blockIdx.x * max_options_in_chunk;

  if (threadIdx.x < num_options) {
    int option_id = options_in_chunk[threadIdx.x];
    
    Xs[threadIdx.x] = strikes[option_id];
    float T = maturities[option_id];
    int n    = num_terms[option_id];
    Ns[threadIdx.x] = n;
    float a = reversion_rates[option_id];
    float sigma = volatilities[option_id];
    float dt = T / n;
    dts[threadIdx.x] = dt;
    float V  = sigma * sigma * (1 - (exp(0.0 - 2.0 * a * dt)) ) / (2.0 * a);
    float dr = sqrt((1.0 + 2.0) * V);
    drs[threadIdx.x] = dr;
    float M  = exp(0.0 - a * dt) - 1.0;
    Ms[threadIdx.x] = M;
    int jmax = ((int)(-0.184 / M)) + 1;
    jmaxs[threadIdx.x] = jmax;
    int m = jmax + 2;
    ms[threadIdx.x] = m;
  }
  __syncthreads();
  int w = blockDim.x;

  // Translating map_lens
  if (threadIdx.x < num_options) {
    tmp_scan[threadIdx.x] = 2 * ms[threadIdx.x] + 1;
  } else {
    tmp_scan[threadIdx.x] = 0;
  }
  __syncthreads();
  // Translating the scan (+) over map_len 
  T res = scanIncBlock < Add<int>, int >(tmp_scan, threadIdx.x);

  // Build the flags
  flags[threadIdx.x] = 0;
  __syncthreads();
  if (threadIdx.x < num_options) {
    if (threadIdx.x == 0) {
      flags[0] = 2 * ms[0] + 1;
    } else {
      flags[tmp_scan[threadIdx.x - 1]] = 2 * ms[threadIdx.x] + 1;
    }
  }
  __syncthreads();

  // Build sgm_inds
  sgm_inds[threadIdx.x] = 0;
  __syncthreads();
  if (threadIdx.x < num_options) {
    sgm_inds[tmp_scan[threadIdx.x - 1]] = threadIdx.x;
  }
  __syncthreads();
  sgmScanIncBlock <Add<int>, int>(sgm_inds, flags, threadIdx.x);
  
  // make the segmented iota across all ms
  iota2mp1[threadIdx.x] = 1;
  __syncthreads();
  sgmScanIncBlock <Add<int>, int>(iota2mp1, flags, threadIdx.x);
  iota2mp1[threadIdx.x] -= 1;

  // calculate Q
  Qs[threadIdx.x] = 0.0;
  __syncthreads();
  if (threadIdx.x < num_options) {
    if (iota2mp1[threadIdx.x] == ms[sgm_inds[threadIdx.x]]) {
      Qs[threadIdx.x] = 1.0;
    } else {
      Qs[threadIdx.x] = 0.0;
    }
  }

  int seq_len = n_max + 1;
  float *blockalphas = alphass + blockIdx * num_options * (n_max + 1);
  if (threadIdx.x < num_options) {
    blockalphas[threadIdx.x * seq_len] = yield_curve[0].P;
  }
  __syncthreads();

  for (int i = 0; i < n_max; i++) {
    bool go_on = i < ns[sgm_inds[threadIdx.x]];
    if (go_on) {
      //body
      if (threadIdx.x < num_options) {
        imaxs[threadIdx.x] = min(i + 1, jmaxs[threadIdx.x]);
      }
      QCopys'[threadIdx.x] = Qs[threadIdx.x];
    }
    __syncthreads();
    if (go_on) {
      int sgm_ind = sgm_inds[threadIdx.x];
      if (sgm_ind >= num_options || i >= ns[sgm_ind]) {
        Qs'[threadIdx.x] = 0.0;
      } else {
        int imax = imaxs[sgm_ind];
        int m    = ms[sgm_ind];
        int j    = iota2mp1[threadIdx.x] - m;
        if (j < (-imax) || (j > imax)) {
          Qs'[threadIdx.x] = 0.0;
        } else {
          int begind;
          if (sgm_ind == 0) {
            begind = 0;
          } else {
            begind = res[sgm_ind - 1];
          }
          Qs'[threadIdx.x] = forwardhelper (Ms[sgm_ind], drs[sgm_ind],
                         dts[sgm_ind],
                         alphass[sgm_ind*seq_len+i,//alphas or blockalphas???,
                         QCopys', begind, m,
                         i, imax, jmaxs[sgm_ind], j);
        }
      }
    }


    __syncthreads();
    if (go_on) {
      int sgm_ind = sgm_inds[threadIdx.x];
      int imax = sgm_inds[sgm_ind];
      int j = iota2mp1[threadIdx.x] - imax;
      if (j < (-imax) || j > imax || sgm_ind >= optionsInChunk || i >= ns[sgm_ind]) {
        tmpss[threadIdx.x] = 0.0;
      } else {
        int begind;
        if (sgm_ind == 0) {
          begind = 0;
        } else {
          begind = res[sgm_ind - 1];
        }
        tmpss[threadIdx.x] = Qs'[begind + j + ms[sgm_ind]] * exp(-(float)j) * drs[sgm_ind] * dts[sgm_ind];
      }
    }
    __syncthreads();
    
    //alpha_vals size is numoptionsinblock = num_options
    tmpss_scan = sgmScanIncBlock <Add<float>, float>(tmpss, flags, threadIdx.x);
    if (go_on) {
      if(threadIdx.x == blockDim.x - 1 ||
          flags[threadIdx.x + 1] > 0) {
        alpha_vals[sgm_ind[threadIdx.x]] = tmpss_scan[threadIdx.x];
      }
    }
    __syncthreads();
    
  }



  

  




  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  // Calculating alphas
  // alphas = replicate (n + 1) zero
  // alphas[0] = #P (h_YieldCurve[0])
  double *alphas = (double*) malloc(n + 1);
  alpha[0] = yield_curve[0].P;
  for (int i = 1; i < n + 1; i++){
    alphas[i] = 0;
  }
  
  // Computing QCopy
  // Q = map (\i -> if i == m then one else zero) (iota (2 * m + 1))
  int *Q = (int*) malloc(2 * m + 1);
  for (int i = 0; i < 2 * m + 1; i++) {
    if i == m
  }

  if(lid < sum_of_qlens_in_block) { // maybe some guard here
    // 1. forward iteration
    for(int i = 0; i<n_max; i++) {
      double alphai = alphas[i];
      if() { // guard because of __synthreads
      }
      forwardhelper(M, dr, dt, alphai,
		      double *QCopy, int beg_ind, m, i,
		      int imax, int jmax, int j);
      // barrier because of dependency between q_{i} and q_{i+1}
      __syncthreads();
    }

    // 2. backward iteration
    for(int i = 0; i<n_max; i++) {
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

  // third kernel argument til trinom-kernel should be the shared memory size.
  // Look at IncSgmScan sgmScanIncKernel function around line 230.
  // You do extern __shared__ something something
  // Then you can do volatile blah = pointer in your shared mem.

  // execute kernel

  // copy data from device to host

  return 0;
}
