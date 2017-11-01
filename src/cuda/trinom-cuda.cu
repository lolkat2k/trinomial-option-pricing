#include <math.h>  // exp

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
                     int num_all_options, // number of options in total
                     int max_options_in_chunk, // blockdimensions, i.e. w
                     int *options_in_chunk, // list of options per block size: [number_of_blocks]
                     int *option_indices   // size: [number_of_blocks][maxOptionsInChunk]

                     // These 4 have length numAllOptions
                     float *strikes,
                     float *maturities,
                     float *reversion_rates,
                     float *volatilities

                     int n_max,    // maximum number of time steps
                     int *num_terms,
                                          
                     int yc_count, // number of yield curves
                     float *yield_curve_P,
                     float *yield_curve_t,
                     float *alphass
                     )
{
  extern __shared__ char sh_mem[];
  
  int w = blockDim.x;
  int seq_len = n_max + 1;
  int num_options = options_in_chunk[blockIdx.x];
  int *options_in_chunk = option_indices + blockIdx.x * max_options_in_chunk;

  volatile float* tmp_scan   = (float*) sh_mem;
  volatile int* flags        = (int*) (tmp_scan + max_options_in_chunk);
  volatile int* sgm_inds     = (int*) (flags + w);
  volatile int* iota2mp1     = (int*) (sgm_inds + w);
  volatile float* Qs         = (float*) (iota2mp1 + w);
  volatile float* QCopys_    = (float*) (Qs + w);
  volatile int* imaxs        = (int*) (QCopys_ + w);
  volatile float* Qs_        = (float*) (imaxs + max_options_in_chunk);
  volatile float* tmpss      = (float*) (Qs_ + w);
  volatile float* tmpss_scan = (float*) (tmpss + w);
  volatile int* alpha_vals   = (float*) (tmpss_scan + w);
  volatile int* Ps           = (float*) (alpha_vals + max_options_in_chunk);
  // alphass_ is defined after the shared memory arrays
  volatile float* Qs__       = (float*) (Ps + max_options_in_chunk);
  volatile float* Calls      = (float*) (Qs__ + w);
  volatile float* is         = (float*) (Calls + w);
  volatile float* CallCopys_ = (float*) (is + max_options_in_chunk);
  volatile float* Calls_     = (float*) (CallCopys_ + w);
  volatile float* Calls__    = (float*) (Calls_ + w);
  volatile float* res        = (float*) (Calls__ + w);
  
  volatile float* Xs    = (float*)(res   + max_options_in_chunk);
  volatile float* ns    = (float*)(Xs    + max_options_in_chunk);
  volatile float* dts   = (float*)(ns    + max_options_in_chunk);
  volatile float* drs   = (float*)(dts   + max_options_in_chunk);
  volatile float* Ms    = (float*)(drs   + max_options_in_chunk);
  volatile float* jmaxs = (float*)(Ms    + max_options_in_chunk);
  volatile float* ms    = (float*)(jmaxs + max_options_in_chunk);

  float *alphass_ = alphass + blockIdx.x * num_options * (seq_len);
  // computing option id and values for this particular option
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

  // Translating map_lens
  if (threadIdx.x < num_options) {
    tmp_scan[threadIdx.x] = 2 * ms[threadIdx.x] + 1;
  } else {
    tmp_scan[threadIdx.x] = 0;
  }
  __syncthreads();
  // Translating the scan (+) over map_len 
  T scanned_lens = scanIncBlock < Add<int>, int >(tmp_scan, threadIdx.x);

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
  if (threadIdx.x < num_options && threadIdx.x > 0) {
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

  if (threadIdx.x < num_options) {
    alphass_[threadIdx.x * seq_len] = yield_curve[0].P;
  }
  __syncthreads();

  // BEGIN FOR LOOP
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
            begind = scanned_lens[sgm_ind - 1];
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
          begind = scanned_lens[sgm_ind - 1];
        }
        tmpss[threadIdx.x] = Qs'[begind + j + ms[sgm_ind]] * exp(-(float)j * drs[sgm_ind] * dts[sgm_ind]);
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
    
    if (go_on) {
      if (threadIdx.x >= num_options || i >= ns[threadIdx.x]) {
        Ps[threadIdx.x] = 1.0; 
      } else {
        float t = (float)(i+1) * dts[threadIdx.x] + 1.0;
        float t2 = roundf(t);
        float t1 = t2 - 1;
        if (t2 >= ycCount) {
          t2 = ycCount - 1;
          t1 = ycCount - 2;
        }
        float R = (yield_curve[t2].P - yield_curve[t1].P) / (yield_curve[t2].t - yield_curve[t1].t) * (t * 365 - (yield_curve[t1].t) + (yield_curve[t1].P);
        P = exp(-R * t);
        Ps[threadIdx.x] = P;
      }
    }
    __syncthreads();

    if(go_on) {
      alpha_vals[threadIdx.x] = log(alpha_vals[threadIdx.x] / Ps[threadIdx.x]);
    }

    if(go_on) {
      if (threadIdx.x < num_options && i < ns[threadIdx.x]) {
        alpha_inds[threadIdx.x] = threadIdx.x * seq_len + (i+1);
      } else {
        alpha_inds[threadIdx.x] = -1;
      }
    }
    __syncthreads();

    if(go_on) {
      alphass'[threadIdx.x] = alpha_vals[alpha_inds[threadIdx.x]];
    }

    // Qs''
    if(go_on) {
      if(threadIdx.x < num_options) {
        int sgm_ind = sgm_inds[threadIdx.x];
        if (sgm_ind < num_options && i < ns[sgm_ind]) {
          Qs''[threadIdx.x] = Qs'[threadIdx.x];
        }
      }
    }
  } //END OF FOR LOOP

  // Calls
  if(threadIdx.x < num_options) {
    int sgm_ind = sgm_inds[threadIdx.x];
    float jmax = jmaxs[sgm_ind];
    float m    = ms[sgm_ind];
    float j = iota2mp1[threadIdx.x];
    if (j >= -jmax+m && j <= jmax + m) {
      Calls[threadIdx.x] = 1.0;
    } else {
      Calls[threadIdx.x] = 0.0;
    }
  }

  __syncthreads();
  for (int ii = 0; ii < n_max; ii++) {
    bool go_on = ii < ns[sgm_inds[threadIdx.x]];

    if(go_on) {
      if (threadIdx.x >= num_options) {
        is[threadIdx.x] = 0;
      } else {
        is[threadIdx.x] = ns[sgm_ind] - 1 - ii;
      }
    }
    __syncthreads();

    if(go_on) {
      if (threadIdx.x < num_options) {
        imaxs[threadIdx.x] = min((is[threadIdx.x]+1), jmaxs[threadIdx.x]);
      }
    }
    CallCopys'[threadIdx.x] = Calls[threadIdx.x];
    __syncthread();

    if(go_on) {
      if (threadIdx.x < num_options) {
        int sgm_ind = sgm_inds[threadIdx.x];
        if (sgm_ind >= num_options || ii >= ns[sgm_ind]) {
          Calls'[threadIdx.x] = 0.0;
        } else {
          int imaxs[sgm_ind];
          int i = is[sgm_ind];
          int m = ms[sgm_ind];
          int j = iota2mp1[threadIdx.x] - m;
          if (j < (-imax) || j > imax) {
            Calls'[threadIdx.x] = 0.0;
          } else {
            int begind = (sgm_ind == 0) ? 0 : scanned_lens[sgm_ind-1];
            backwardhelper(Xs[sgm_ind], Ms[sgm_ind], drs[sgm_ind],
                           dts[sgm_ind], alphass[sgm_ind * seq_len + i],
                           CallCopys', begind, ms[sgm_ind], i,
                           jmaxs[sgm_ind], j);
          }
        }
      }
    }

    // Update calls
    __syncthreads();
    if(go_on) {
      if (threadIdx.x < num_options) {
        int sgm_ind = sgm_inds[threadIdx.x];
        if (sgm_ind < num_options && ii < ns[sgm_ind]) {
          Calls''[threadIdx.x] = Calls'[threadIdx.x];
        }
      }
    }
  } // END OF FOR LOOP
  __syncthreads();

  if (threadIdx.x < num_options) {
    if(threadIdx.x >= num_options) {
      res[-1] = 0.0; // out of range
    } else {
      int begind = (threadIdx.x == 0) ? 0 : scanned_lens[threadIdx.x - 1];
      int m_ind = begind + ms[threadIdx.x];
      res[threadIdx.x] = Calls[m_ind];
    }
  }
  __syncthreads();
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
  dim3 block_dim(w, 1, 1);
  dim3 grid_dim(w + (block_dim.x - 1) / block_dim.x, 1);

  // execute kernel
  trinom_chunk_kernel<<<grid_dim, block_dim, sh_mem_size>>>();

  // copy data from device to host

  return 0;
}
