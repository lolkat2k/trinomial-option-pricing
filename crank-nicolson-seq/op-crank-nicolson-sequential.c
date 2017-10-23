/* This is an example-implementation of Crank-Nicholsons finite difference
 * method for finding option prices (EU).
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void eu_call(double *q, double strike, double xmin, int nx, double dx)
{
  for(int i=0; i<=nx; i++) {
    q[i] = (i * dx) + xmin - strike;
    if(q[i] < 0) { q[i] = 0; }
  }
}

void eu_put(double *q, double strike, double xmin, int nx, double dx)
{
  for(int i=0; i<=nx; i++) {
    q[i] = strike - xmin - (i * dx);
    if(q[i]<0) { q[i] = 0; }
  }
}

void finitedifference(double *q, double r, double sigma_in,
		      int xmin, double nt, double dt, int nx, double dx)
{
  // variables (for what?)
  double mu, sigma;

  // coefficients for system of equations - all nx long
  unsigned int mem_size = (nx + 1) * sizeof(double);
  double *a = (double *)malloc(mem_size);
  double *b = (double *)malloc(mem_size);
  double *c = (double *)malloc(mem_size);

  // variables for forward and backward substitution
  double e;
  double *f = (double *)malloc(mem_size);
  double *h = (double *)malloc(mem_size);
  double *x = (double *)malloc(mem_size);

  // underlying asset
  double tmp; // helper

  for(int i=0; i<=nx; i++) {
    mu    = (i * dx + xmin) * r;
    sigma = (i * dx + xmin) * sigma_in;

    tmp = (sigma * sigma) / (dx * dx);
    a[i] = 0.25 * dt * ((mu / dx) - tmp);
    b[i] = 1 + 0.5 * dt * (r + tmp);
    c[i] = -0.25 * dt * ((mu / dx) + tmp);
  }

  // 'closing' the system of equations
  a[0] = 0; // shouldn't be in the system
  b[0] = 1; // closes the system and ensures contingency
  c[0] = 0; // closes the system

  a[nx] = 0; // closes the system
  b[nx] = 1; // closes the system and ensures contingency
  c[nx] = 0; // closes the system

  // Thomas' algorithm
  for(int j=nt - 1; j>=0; j--) {

    // forward substition - step 1
    e    = b[0];
    f[0] = c[0] / e;
    h[0] = q[0] / e;

    // forward substition - step 2
    for(int i=1; i<=nx ; i++) {
      e    = b[i] - a[i] * f[i-1];
      f[i] = c[i] / e;
      h[i] = (q[i] - a[i] * h[i - 1]) / e;
    }

    // forward substition - step 3
    e     = b[nx] - a[nx] * f[nx];
    h[nx] = (q[nx] - a[nx] * h[nx - 1]) / e;

    // backward substitution
    x[nx] = h[nx];

    for(int i=nx - 1; i>=0; i--) {
      x[i] = h[i] - f[i] * x[i + 1];
    }

    // determine start and end values
    q[0]  = x[0];
    q[nx] = x[nx];

    for(int i=1; i<=nx; i++) {
      q[i] = -1 * a[i] * x[i-1] + (2 - b[i]) * x[i] - 1 * c[i] * x[i + 1];
    }
  }

  free(a); free(b); free(c);
  free(f); free(h); free(x);
}

int main()
{
  // underlying asset
  double xmin = 0;
  double xmax = 400;
  int nx = 1000;
  double dx = (xmax - xmin) / nx;

  // timeline
  double tmin = 0;
  double tmax = 1; // years
  int nt = 100;
  double dt = (tmax - tmin) / nt;

  // contract specification
  unsigned int mem_size = (nx + 1) * sizeof(double);
  double *q = (double *)malloc(mem_size);
  int strike = 100;
  int ua = 100;

  // market specifications
  double interest_rate = 0.05;
  double sigma_in = 0.1;

  // interpolating option array for correct pricing
  int tmp  = floor((ua - xmin) / dx);
  double w = ((ua - xmin) / dx - tmp);

  eu_call(q, strike, xmin, nx, dx);
  //  eu_put(q, strike, xmin, nx, dx);
  finitedifference(q, interest_rate, sigma_in, xmin, nt, dt, nx, dx);

  double optionprice = (1 - w) * q[tmp] + (w * q[tmp + 1]);
  printf("The option price is: %.5f\n", optionprice);

  free(q);
}
