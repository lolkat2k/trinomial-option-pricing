#include <float.h> // double_max
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
  if(i == ((int)(3 / dt))) { res = double_max(X - res, 0); }

  return res;
}



int main()
{
  return 0;
}
