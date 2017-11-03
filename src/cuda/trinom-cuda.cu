#include <stdio.h> // gettimeofday
#include <math.h>  // exp
#include <algorithm> // max
#include "ScanHost.cu.h"
#include "ScanKernels.cu.h"

/* Probability measures for tree construction */
// Exhibit 1A (-jmax < j < jmax)
#define PU_A(j, M) ((1.0/6.0) + (((float)(j*j))*M*M + ((float)j)*M) * (1.0/2.0))
#define PM_A(j, M) ((2.0/3.0) -  ((float)(j*j))*M*M)
#define PD_A(j, M) ((1.0/6.0) + (((float)(j*j))*M*M - ((float)j)*M) * (1.0/2.0))

// Exhibit 1B (j == -jmax)
#define PU_B(j, M) ( (1.0/6.0) + (((float)(j*j))*M*M  - ((float)j)*M) * (1.0/2.0))
#define PM_B(j, M) (-(1.0/3.0) - (((float)(j*j))*M*M) - (2*((float)j)*M))
#define PD_B(j, M) ( (7.0/6.0) + (((float)(j*j))*M*M  - (3*(float)j)*M) * (1.0/2.0))

// Exhibit 1C (j == jmax)
#define PU_C(j, M) ( (7.0/6.0) + (((float)(j*j))*M*M  + (3*(float)j)*M) * (1.0/2.0))
#define PM_C(j, M) (-(1.0/3.0) - (((float)(j*j))*M*M) - (2*((float)j)*M))
#define PD_C(j, M) ( (1.0/6.0) + (((float)(j*j))*M*M  + ((float)j)*M) * (1.0/2.0))

/* forward propagation helper */
__device__ float forward_helper(float M, float dr, float dt, float alphai,
		      volatile float *QCopy, int beg_ind, int m, int i,
		      int imax, int jmax, int j)
{
  /*
  printf("M: %.4f\n", M);
  printf("dr: %.4f\n", dr);
  printf("dt: %.4f\n", dt);
  printf("alphai: %.4f\n", alphai);
  printf("beg_ind: %d\n", beg_ind);
  printf("m: %d\n", m);
  printf("i: %d\n", i);
  printf("imax: %d\n", imax);
  printf("jmax: %d\n", jmax);
  printf("j: %d\n", j);
  */
  float eRdt_u1 = exp(-((float)(j+1)*dr+alphai)*dt);
  float eRdt    = exp(-((float)(j)  *dr+alphai)*dt);
  float eRdt_d1 = exp(-((float)(j-1)*dr+alphai)*dt);
  float pu, pm, pd;

  if (i < jmax) {
    pu = PU_A(j-1, M);
    pm = PM_A(j,   M);
    pd = PD_A(j+1, M);

    if((i == 0) && (j == 0)) {
      return pm*QCopy[beg_ind+j+m]*eRdt;
    } else if(j == (-imax + 1)) {
      return pd*QCopy[beg_ind+j+m+1]*eRdt_u1 + pm*QCopy[beg_ind+j+m]*eRdt;
    } else if(j == ( imax - 1)) {
      return pm*QCopy[beg_ind+j+m]*eRdt + pu*QCopy[beg_ind+j+m-1]*eRdt_d1;
    } else if(j == (-imax)) {
      return pd*QCopy[beg_ind+j+m+1]*eRdt_u1;
    } else if(j == ( imax)) {
      return pu*QCopy[beg_ind+j+m-1]*eRdt_d1;
    } else {
      return
	pd*QCopy[beg_ind+j+m+1]*eRdt_u1 +
	pm*QCopy[beg_ind+j+m]*eRdt +
	pu*QCopy[beg_ind+j+m-1]*eRdt_d1;
    }
  } // END_OF: (i < jmax) {
  else if(j == jmax) {
    pm = PU_C(j,   M);
    pu = PU_A(j-1, M);
    return pm*QCopy[beg_ind+j+m]*eRdt + pu*QCopy[beg_ind+j-1+m] * eRdt_d1; // return
  } // END_OF: (j == jmax)
  else if(j == (jmax - 1)) {
    pd = PM_C(j+1, M);
    pm = PM_A(j  , M);
    pu = PU_A(j-1, M);
    return
      pd*QCopy[beg_ind+j+1+m]*eRdt_u1 +
      pm*QCopy[beg_ind+j+m]*eRdt +
      pu*QCopy[beg_ind+j-1+m]*eRdt_d1;
  } // END_OF: (j == (jmax - 1))
  else if(j == (jmax - 2)) {
    float eRdt_u2 = exp(-(((float)(j+2))*dr + alphai) * dt);
    float pd_c = PD_C(j + 2, M);
    pd = PD_A(j + 1, M);
    pm = PM_A(j, M);
    pu = PU_A(j - 1, M);
    return 
      pd_c*QCopy[beg_ind+j+2+m]*eRdt_u2 +
      pd*QCopy[beg_ind+j+1+m]*eRdt_u1 +
      pm*QCopy[beg_ind+j+m]*eRdt +
      pu*QCopy[beg_ind+j-1+m]*eRdt_d1;
  } // END_OF: (jmax - 2))
  else if(j == (-jmax + 2)) {
    float eRdt_d2 = exp(-(((float)(j-2))*dr + alphai) * dt);
    float pu_b = PU_B(j - 2, M);
    pd = PD_A(j + 1, M);
    pm = PM_A(j,     M);
    pu = PU_A(j - 1, M);
    return 
      pd*QCopy[beg_ind+j+1+m]*eRdt_u1 +
      pm*QCopy[beg_ind+j+m]*eRdt +
      pu*QCopy[beg_ind+j-1+m]*eRdt_d1 +
      pu_b*QCopy[beg_ind+j-2+m]*eRdt_d2;
  } // END_OF: (j == (-jmax + 2))
  else if(j == (-jmax + 1)) {
    pd = PD_A(j + 1, M);
    pm = PM_A(j, M);
    pu = PM_B(j - 1, M);
    return
      pd*QCopy[beg_ind+j+1+m]*eRdt_u1 +
      pm*QCopy[beg_ind+j+m]*eRdt +
      pu*QCopy[beg_ind+j-1+m]*eRdt_d1;
  } // END_OF: (j == (-jmax + 1))
  else if(j == (-jmax)) {
    pd = PD_A(j + 1, M);
    pm = PD_B(j,     M);
    return pd*QCopy[beg_ind+j+1+m]*eRdt_u1 + pm*QCopy[beg_ind+j+m]*eRdt;
  } // END_OF: (-jmax)
  else {
    pd = PD_A(j + 1, M);
    pm = PM_A(j,     M);
    pu = PU_A(j - 1, M);
    return
      pd*QCopy[beg_ind+j+1+m]*eRdt_u1 +
      pm*QCopy[beg_ind+j+m]*eRdt +
      pu*QCopy[beg_ind+j-1+m]*eRdt_d1;
  } // END_OF: default
}


/* backward propagation helper */
__device__ float backward_helper(float X, float M, float dr, float dt, 
		       float alphai, volatile float *CallCopy, int beg_ind, int m, 
		       int i, int jmax, int j)
{
  float eRdt = exp(-((float)(j)*dr + alphai)*dt);
  float res;
  float pu, pm, pd;

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
  if(i == ((int)(3 / dt))) { res = max(X - res, 0.); }

  return res;
}

__device__ int scanInc(volatile int* ms, int ms_length)
{
  int res = 0;
  for (int i = 0; i < ms_length; i++) {
    if (ms[i] != 0) {
      res += 2 * ms[i] + 1;
    }
  }
  return res;
};

__device__ void sgmScanIncFloat(volatile float* input, volatile int* flags, int n)
{
  float acc = 0.0;

  for (int i = 0; i < n; i++) {
    if(flags[i] > 0) {
      acc = input[i];
    }
    else {
      input[i] += acc;
      acc = input[i];
    }
  }
}

__device__ void sgmScanIncInt(volatile int* input, volatile int* flags, int n)
{
  int acc = 0;

  for (int i = 0; i < n; i++) {
    if(flags[i] > 0) {
      acc = input[i];
    }
    else {
      input[i] += acc;
      acc = input[i];
    }
  }
}

/* trinomial chunk kernel */
__global__ void trinom_chunk_kernel(
                     int num_all_options, // number of options in total
                     int max_options_in_chunk,
                     int *options_per_chunk, // list of options per block size: [number_of_blocks]
                     int *option_indices,   // size: [number_of_blocks][maxOptionsInChunk]

                     // These 4 have length numAllOptions
                     float *strikes,
                     float *maturities,
                     float *reversion_rates,
                     float *volatilities,

                     int n_max,    // maximum number of time steps
                     int *num_terms,
                     
                     int yc_count, // number of yield curves
                     float *yield_curve_P,
                     float *yield_curve_t,
                     float *alphass,
                     float *d_output
                     )
{
  extern __shared__ char sh_mem[];

  int w = blockDim.x;
  int seq_len = n_max + 1;
  if (threadIdx.x == 0) {
    printf("blockIdx.x: %d\n", blockIdx.x);
  }
  int num_options = options_per_chunk[blockIdx.x];
  int *options_in_chunk = option_indices + blockIdx.x * max_options_in_chunk;

  volatile int* tmp_scan     = (int*) sh_mem;
  volatile int* sgm_inds     = (int*) (tmp_scan + w);
  volatile int* iota2mp1     = (int*) (sgm_inds + w);
  volatile int* flags        = (int*) (iota2mp1 + w);
  volatile float* Qs         = (float*) (flags + w);
  volatile float* Qs_        = (float*) (Qs + w);
  volatile float* QCopys_    = (float*) (Qs_ + w);
  volatile float* tmpss      = (float*) (QCopys_ + w);
  volatile float* Calls      = (float*) (tmpss + w);
  volatile float* Calls_     = (float*) (Calls + w);
  volatile float* CallCopys_ = (float*) (Calls_ + w);
  
  volatile float* alpha_vals = (float*) (CallCopys_ + max_options_in_chunk);
  volatile int* scanned_lens = (int*) (alpha_vals + max_options_in_chunk);
  volatile int* is           = (int*) (scanned_lens + max_options_in_chunk);
  volatile float* Ps     = (float*) (is + max_options_in_chunk);
  volatile float* Xs     = (float*) (Ps  + max_options_in_chunk);
  volatile int* ns       = (int*) (Xs  + max_options_in_chunk);
  volatile float* dts    = (float*) (ns  + max_options_in_chunk);
  volatile float* drs    = (float*) (dts + max_options_in_chunk);
  volatile float* Ms     = (float*) (drs + max_options_in_chunk);
  volatile int* imaxs    = (int*) (Ms + max_options_in_chunk);
  volatile int* jmaxs    = (int*) (imaxs + max_options_in_chunk);
  volatile int* ms       = (int*) (jmaxs + max_options_in_chunk);
  
  volatile int* w_guard_ = (int*) (ms + max_options_in_chunk);
  volatile int* alpha_inds   = (int*) (w_guard_ + sizeof(int));

  float *alphass_ = alphass + blockIdx.x * num_options * (seq_len);

  // computing option id and values for this particular option
  if (threadIdx.x < num_options) {
    int option_id = options_in_chunk[threadIdx.x];
    printf("threadID: %d  --  option_id: %d\n", threadIdx.x, option_id);
    
    Xs[threadIdx.x] = strikes[option_id];
    float T = maturities[option_id];
    int n    = num_terms[option_id];
    ns[threadIdx.x] = n;
    float a = reversion_rates[option_id];
    float sigma = volatilities[option_id];
    float dt = T / n;
    dts[threadIdx.x] = dt;
    float V  = sigma * sigma * (1 - (exp(-2.0 * a * dt)) ) / (2.0 * a);
    float dr = sqrt((1.0 + 2.0) * V);
    drs[threadIdx.x] = dr;
    float M  = exp(-a * dt) - 1.0;
    Ms[threadIdx.x] = M;
    int jmax = ((int)(-0.184 / M)) + 1;
    jmaxs[threadIdx.x] = jmax;
    int m = jmax + 2;
    ms[threadIdx.x] = m;
  }
  __syncthreads();

  unsigned int w_guard = 0;
  if (threadIdx.x == 0) {
    *w_guard_ = scanInc(ms, max_options_in_chunk);
    printf("ms[0]: %d", ms[0]);
    printf("num options er %d\n", num_options);
  }
  __syncthreads();
  w_guard = *w_guard_; 

  int tmp;
  // Translating map_lens
  if (threadIdx.x < num_options) {
    tmp = 2 * ms[threadIdx.x] + 1; 
    tmp_scan[threadIdx.x] = tmp;
    //map_lens[threadIdx.x] = tmp; We dont need maplens because we do it inline
  } else {
    tmp_scan[threadIdx.x] = 0;
  }
  __syncthreads();
  
  // Translating the scan (+) over map_len 
  tmp = scanIncBlock<Add<int>, int>(tmp_scan, threadIdx.x);
  __syncthreads();

  if(threadIdx.x < num_options) {
    scanned_lens[threadIdx.x] = tmp;
  }
  __syncthreads();

  // Build the flags. Arrays of size w are initialized like so flags[threadIdx.x] = 0.
  flags[threadIdx.x] = 0;
  sgm_inds[threadIdx.x] = 0;
  __syncthreads();
  if (threadIdx.x < num_options) {
    if (threadIdx.x == 0) {
      flags[0] = 2 * ms[0] + 1;
    } else {
      int flagindex = scanned_lens[threadIdx.x - 1];
      flags[flagindex] = 2 * ms[threadIdx.x] + 1;
      sgm_inds[flagindex] = threadIdx.x;
    }
  } else if (threadIdx.x < max_options_in_chunk) {
    if (threadIdx.x == num_options) {
      int last_ind = scanned_lens[num_options - 1];
      if (last_ind < w) {
        flags[last_ind] = w - last_ind;
        sgm_inds[last_ind] = threadIdx.x;
      }
    }
  }
  __syncthreads();

  /* Print flags and sgm_inds
  __syncthreads();
  if (threadIdx.x == 0) {
    printf("flags:\n");
    for (int i = 0; i < w; i++) {
      printf("%d ", flags[i]);
    }
    printf("--------\n");
    printf("sgm_inds:\n");
    for (int i = 0; i < w; i++) {
      printf("%d ", sgm_inds[i]);
    }
    printf("--------\n");
    printf("ns:\n");
    for (int i = 0; i < num_options; i++) {
      printf("%d ", ns[i]);
    }
    printf("\n");
  }
  */
  

  // We copy the flags into tmp_scan because sgmScanIncBlock updates its flag array and we want to keep the flags array intact
  if (threadIdx.x == 0) {
    sgmScanIncInt(sgm_inds, flags, w);
  }
  __syncthreads();
  
  // Print scanned sgm_inds
  __syncthreads();
  if (threadIdx.x == 0) {
    printf("initial: scanned sgm_inds:\n");
    for (int i = 0; i < w; i++) {
      printf("%d ", sgm_inds[i]);
    }
    printf("\n");
  } 
 
  // Print flags
  __syncthreads();
  if (threadIdx.x == 0) {
    printf("flags: \n");
    for (int i = 0; i < w; i++) {
      printf("%d ", flags[i]);
    }
    printf("\n");
  } 

  // make the segmented iota across all ms
  iota2mp1[threadIdx.x] = 1;
  __syncthreads();
  if (threadIdx.x == 0) {
    sgmScanIncInt(iota2mp1, flags, w);
  }
  __syncthreads();
  iota2mp1[threadIdx.x] -= 1;
  __syncthreads();
  
  // calculate Qs
  Qs[threadIdx.x] = 0.0;
  __syncthreads();
  if (threadIdx.x < w_guard) {
    if (iota2mp1[threadIdx.x] == ms[sgm_inds[threadIdx.x]]) {
      Qs[threadIdx.x] = 1.0;
    } else {
      Qs[threadIdx.x] = 0.0;
    }
  }
  __syncthreads();

  alphass_[threadIdx.x] = 0.0;
  __syncthreads();
  if (threadIdx.x < num_options) {
    alphass_[threadIdx.x * seq_len] = yield_curve_P[0];
  }
  __syncthreads();



  if (threadIdx.x == 0) {
    printf("Qs:\n");
    for (int i = 0; i < w_guard; i++) {
      printf("%.1f, ", Qs[i]);
    }
    printf("\n");
  }
  
  __syncthreads();
  
  // BEGIN FOR LOOP
  for (int i = 0; i < n_max; i++) {
    bool go_on = i < ns[sgm_inds[threadIdx.x]];
    if (go_on) {
      if (threadIdx.x < num_options) {
        imaxs[threadIdx.x] = min((int)i + 1, (int)jmaxs[threadIdx.x]);
      }
      QCopys_[threadIdx.x] = Qs[threadIdx.x];
    }
    __syncthreads();

    if (go_on) {
      int sgm_ind = sgm_inds[threadIdx.x];
      if (sgm_ind >= num_options || i >= ns[sgm_ind]) {
        Qs_[threadIdx.x] = 0.0; // Out of bounds on sgm_ind
      } else {
        int imax = imaxs[sgm_ind];
        int m    = ms[sgm_ind];
        int j    = iota2mp1[threadIdx.x] - m;
        if (j < (-imax) || (j > imax)) {
          Qs_[threadIdx.x] = 0.0; // Error
        } else {
          if (i == 1) printf("Hello I am thread %d\n", threadIdx.x);
          int begind;
          if (sgm_ind == 0) {
            begind = 0;
          } else {
            begind = scanned_lens[sgm_ind - 1];
          }
          float tmp = forward_helper (
                 Ms[sgm_ind], drs[sgm_ind],
                 dts[sgm_ind],
                 alphass_[sgm_ind*seq_len+i],
                 QCopys_, begind, m,
                 i, imax, jmaxs[sgm_ind], j);
          if (i == 0) {
            printf("forward_helper returns: %.3f\n", tmp);
          }
          Qs_[threadIdx.x] = tmp; 
        }
      }
    }
    __syncthreads();

    if (go_on) {
      if(threadIdx.x < w_guard) {
        int sgm_ind = sgm_inds[threadIdx.x];
        int imax = sgm_inds[sgm_ind];
        int j = iota2mp1[threadIdx.x] - imax;
        if (j < (-imax) || j > imax || sgm_ind >= num_options || i >= ns[sgm_ind]) {
          tmpss[threadIdx.x] = 0.0;
        } else {
          int begind;
          if (sgm_ind == 0) {
            begind = 0;
          } else {
            begind = scanned_lens[sgm_ind - 1];
          }
          int Qs_index = begind + j + ms[sgm_ind];
          tmpss[threadIdx.x] = Qs_[Qs_index] * exp(-(float)j * drs[sgm_ind] * dts[sgm_ind]);
        }
      }
    }
    __syncthreads();

        
    if (threadIdx.x == 0) {
      sgmScanIncFloat(tmpss, flags, w);
    }
    __syncthreads();

    if (go_on) {
      if(threadIdx.x < w_guard) {
        if(threadIdx.x == blockDim.x - 1 || flags[threadIdx.x + 1] > 0) {
          alpha_vals[sgm_inds[threadIdx.x]] = tmpss[threadIdx.x];
        }
      }
    }
    __syncthreads();
    
    if (go_on) {
      if (threadIdx.x < num_options) {
        int current_n = ns[threadIdx.x];
        if (i >= current_n) {
          Ps[threadIdx.x] = 1.0; 
        } else {
          float t = (float)(i+1) * dts[threadIdx.x] + 1.0;
          int t2 = (int) roundf(t);
          int t1 = t2 - 1;
          if (t2 >= yc_count) {
            t2 = yc_count - 1;
            t1 = yc_count - 2;
          }
          float R = (yield_curve_P[t2] - yield_curve_P[t1]) / (yield_curve_t[t2] - yield_curve_t[t1]) * (t * 365 - (yield_curve_t[t1]) + (yield_curve_P[t1]));
          float P = exp(-R * t);
          Ps[threadIdx.x] = P;
        }
      }
    }
    __syncthreads();
    
    /* Print Ps 
    __syncthreads();
    if (threadIdx.x == 0 && i == 0) {
      printf("Ps:\n");
      for (int i = 0; i < num_options; i++) {
        printf("%.4f ", Ps[i]);
      }
      printf("--------\n");
      printf("alpha_vals:\n");
      for (int i = 0; i < num_options; i++) {
        printf("%.2f ", alpha_vals[i]);
      }
      printf("\n");
    }
    */
    
    // Update alpha_vals
    if(go_on) {
      if (threadIdx.x < num_options) {
         alpha_vals[threadIdx.x] = log(alpha_vals[threadIdx.x] / Ps[threadIdx.x]);
      }
    }
    __syncthreads();
    
    // Print updated alpha_vals
    __syncthreads();
    if (threadIdx.x == 0 && i == 0) {
      printf("alpha_vals after:\n");
      for (int i = 0; i < num_options; i++) {
        printf("%.2f ", alpha_vals[i]);
      }
      printf("\n");
    }

    __syncthreads();
    if(go_on) {
      if (threadIdx.x < num_options && i < ns[threadIdx.x]) {
        alpha_inds[threadIdx.x] = threadIdx.x * seq_len + (i+1);
      } else {
        alpha_inds[threadIdx.x] = -1;
      }
    }
    __syncthreads();

    if(go_on) {
      if (threadIdx.x < num_options * seq_len) {
        //printf("(idx: %d, alpha_inds: %d, alphavals: %.3f)\n", threadIdx.x, alpha_inds[threadIdx.x], alpha_vals[alpha_inds[threadIdx.x]]);
        alphass_[threadIdx.x] = alpha_vals[alpha_inds[threadIdx.x]];
      }
    }


    // Print updated alpha_vals
    __syncthreads();
    if (threadIdx.x == 0 && i < 5) {
      printf("alphass_ after:\n");
      for (int i = 0; i < num_options; i++) {
        printf("%.2f ", alphass_[i]);
      }
      printf("\n");
    }
    // Qs''
    if(go_on) {
      if(threadIdx.x < w_guard) {
        int sgm_ind = sgm_inds[threadIdx.x];
        if (sgm_ind < num_options && i < ns[sgm_ind]) {
          Qs[threadIdx.x] = Qs_[threadIdx.x];
        }
      }
    }
  } //END OF FOR LOOP
  __syncthreads();

  /* TEST QS!
  if(threadIdx.x == 0) {
    for(int i = 0; i < w; i++) {
      printf("Qs: %.4f\n", Qs[i]);
    } 
  }
  
  __syncthreads(); 
  */

  // Calls
  if(threadIdx.x < w_guard) {
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

  // START OF LOOP
  for (int ii = 0; ii < n_max; ii++) {
    bool go_on = ii < ns[sgm_inds[threadIdx.x]];

    if(go_on) {
      if (threadIdx.x >= num_options) {
        is[threadIdx.x] = 0;
      } else {
        is[threadIdx.x] = ns[threadIdx.x] - 1 - ii;
      }
    }
    __syncthreads();

    if(go_on) {
      if (threadIdx.x < num_options) {
        imaxs[threadIdx.x] = min((is[threadIdx.x]+1), jmaxs[threadIdx.x]);
      }
    }
    
    CallCopys_[threadIdx.x] = Calls[threadIdx.x];
    __syncthreads();

    if(go_on) {
      if (threadIdx.x < w_guard) {
        int sgm_ind = sgm_inds[threadIdx.x];
        if (sgm_ind >= num_options || ii >= ns[sgm_ind]) {
          Calls_[threadIdx.x] = 0.0;
        } else {
          int imax = imaxs[sgm_ind];
          int i = is[sgm_ind];
          int m = ms[sgm_ind];
          int j = iota2mp1[threadIdx.x] - m;
          if (j < (-imax) || j > imax) {
            Calls_[threadIdx.x] = 0.0;
          } else {
            int begind = (sgm_ind == 0) ? 0 : scanned_lens[sgm_ind-1];
            Calls_[threadIdx.x] = backward_helper(Xs[sgm_ind], Ms[sgm_ind], drs[sgm_ind],
                           dts[sgm_ind], alphass_[sgm_ind * seq_len + i],
                           CallCopys_, begind, ms[sgm_ind], i,
                           jmaxs[sgm_ind], j);
          }
        }
      }
    }

    // Update calls
    __syncthreads();
    if(go_on) {
      if (threadIdx.x < w_guard) {
        int sgm_ind = sgm_inds[threadIdx.x];
        if (sgm_ind < num_options && ii < ns[sgm_ind]) {
          Calls[threadIdx.x] = Calls_[threadIdx.x];
        }
      }
    }
  } // END OF FOR LOOP
  __syncthreads();

  /* if(threadIdx.x >= num_options) {
      d_output[-1] = 0.0; // out of range
    } else {
    }
  */
  if (threadIdx.x < num_options) {
    int begind = (threadIdx.x == 0) ? 0 : scanned_lens[threadIdx.x - 1];
    int m_ind = begind + ms[threadIdx.x];
    d_output[threadIdx.x] = Calls[m_ind];
    printf("jeg er thread number %d og jeg siger %.3f\n", threadIdx.x, Calls[m_ind]);
  }
  __syncthreads();
}

int main()
{
  // small.in - should be read from file
  float h_strikes[3]          = { 0.7965300572556244, 0.7965300572556244, 0.7965300572556244 };
  float h_maturity[3]         = { 9.0000, 9.0000, 9.0000 };
  int   h_num_terms[3]        = { 108, 108, 108};
  float h_reversion_rate[3]   = { 0.1000, 0.1000, 0.1000};
  float h_volatilities[3]     = { 0.0100, 0.0100, 0.0100 };
  int h_options_per_chunk[1]  = { 3 };
  int h_option_indices[3]     = { 0, 1, 2 };

  float yield_curve_P[11] = { 0.0501772,
                              0.0509389,
                              0.0579733,
                              0.0630595,
                              0.0673464,
                              0.0694816,
                              0.0708807,
                              0.0727527,
                              0.0730852,
                              0.0739790,
                              0.0749015 };
  float yield_curve_t[11] = { 3.0   ,
                              367.0 , 
                              731.0 ,
                              1096.0,
                              1461.0,
                              1826.0,
                              2194.0,
                              2558.0,
                              2922.0,
                              3287.0,
                              3653.0 };

  int yc_count = sizeof(yield_curve_P) / sizeof(float);


  // set maximum chunk size
  int w = 256;
  int num_all_options = 3;
  int number_of_chunks = 1; // number of chunks?

  float T  = h_maturity[0];
  int n    = h_num_terms[0];
  float a  = h_reversion_rate[0];
  float dt = T / n;
  float M  = exp(-a * dt) - 1.0;
  int jmax = ((int)(-0.184 / M)) + 1;
  int m = jmax + 2;

  printf("%d\n", n);
  printf("%d\n", m);

  // start out with: (assuming that all options are equal)
  int n_max = n;
  int m_max = m;
  int max_options_in_chunk = w / m_max;
  //int mem_size = num_all_options * sizeof(float) * m_max * n_max;
  float* h_output = (float*) malloc(sizeof(float) * num_all_options);

  // compute: chunks
  // each thread should know here to read from when data is
  // in global device memory
  unsigned long int elapsed;
  struct timeval t_start, t_end, t_diff;
  gettimeofday(&t_start, NULL);

  { // Allocate arrays on device
    float* d_strikes;
    float* d_maturities;
    float* d_reversion_rates;
    float* d_volatilities;
    int* d_num_terms;
    float* d_output;
    int* d_options_per_chunk;
    int* d_option_indices;
    float* d_yield_curve_P;
    float* d_yield_curve_t;
    float* d_alphass;

    cudaMalloc((void**)&d_strikes, num_all_options * sizeof(float)); 
    cudaMalloc((void**)&d_maturities, num_all_options * sizeof(float)); 
    cudaMalloc((void**)&d_reversion_rates, num_all_options * sizeof(float)); 
    cudaMalloc((void**)&d_volatilities, num_all_options * sizeof(float)); 
    cudaMalloc((void**)&d_num_terms, num_all_options * sizeof(int));
    cudaMalloc((void**)&d_output, num_all_options * sizeof(float));
    cudaMalloc((void**)&d_options_per_chunk, num_all_options * sizeof(int));
    cudaMalloc((void**)&d_option_indices, num_all_options * sizeof(int));
    cudaMalloc((void**)&d_yield_curve_P, 11 * sizeof(float));
    cudaMalloc((void**)&d_yield_curve_t, 11 * sizeof(float));
    cudaMalloc((void**)&d_alphass, num_all_options * (n_max + 1) * sizeof(float));

    // some kind of guard (if an option is too large to fit into any block - abort)

    // copy data from host to device
    cudaMemcpy(d_strikes, h_strikes, num_all_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maturities, h_maturity, num_all_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reversion_rates, h_reversion_rate, num_all_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_volatilities, h_volatilities, num_all_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_terms, h_num_terms, num_all_options * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, num_all_options * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_options_per_chunk, h_options_per_chunk, number_of_chunks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_option_indices, h_option_indices, num_all_options * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yield_curve_P, yield_curve_P, sizeof(yield_curve_P), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yield_curve_t, yield_curve_t, sizeof(yield_curve_t), cudaMemcpyHostToDevice);

    // compute block and grid dimensions
    // - block: (1, 1, w)
    // - grid:  (1, ceil(sum(Qlen) / w)
    dim3 block_dim(w, 1, 1);
    dim3 grid_dim(number_of_chunks, 1, 1);
    unsigned int sh_mem_size = (11 * w + 12 * max_options_in_chunk) * 4 + sizeof(int) + (n_max + 1) * sizeof(float);


    // execute kernel
    trinom_chunk_kernel<<<grid_dim, block_dim, sh_mem_size>>> (
        num_all_options,
        max_options_in_chunk,
        d_options_per_chunk,
        d_option_indices,

        // These 4 have length num_all_options
        d_strikes,
        d_maturities,
        d_reversion_rates,
        d_volatilities,

        n_max,    // maximum number of time steps
        d_num_terms,
                                   
        yc_count,
        d_yield_curve_P,
        d_yield_curve_t,
        d_alphass,
        d_output
    );
    cudaThreadSynchronize();

    // copy data from device to host
    cudaMemcpy(h_output, d_output, num_all_options * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    cudaFree(d_strikes);
    cudaFree(d_maturities);
    cudaFree(d_reversion_rates);
    cudaFree(d_volatilities);
    cudaFree(d_num_terms);
    cudaFree(d_options_per_chunk);
    cudaFree(d_option_indices);
    cudaFree(d_alphass);
  }

  gettimeofday(&t_end, NULL);
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = (t_diff.tv_sec * 1e6 + t_diff.tv_usec);
  printf("Trinom kernel runs in %lu microsecs\n", elapsed);

  // Time to validate result
  bool success = true;
  /*for(int i = 0; i < sizeof(h_output) / sizeof(float); i++) {
    if (h_output[i] != 0.142982) {
      success = false;
      printf("answer: %.5f\n", h_output[i]);
    }
  }
  */
  printf("answer: %.5f\n", h_output[0]);
  if (success) printf("VALID RESULT!\n");
  else         printf("INVALID RESULT!\n");

  free(h_output);

  return 0;
}
