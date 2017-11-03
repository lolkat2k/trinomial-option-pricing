#include <stdio.h>
#include <stdlib.h>

#define MAXSIZE 1000

//another alternative would be to pass the size as an argument and replacing that with MAXSIZE
void parser(const char filename, float *strikes, float *maturities, float *terms, float *rates, float *volatilities) {
	FILE *fptr;

	int i = 0;
	float strike, maturity, num_terms;
	float reversion_rate, volatility;

	float A[MAXSIZE], B[MAXSIZE], C[MAXSIZE], D[MAXSIZE], E[MAXSIZE];


	fptr = fopen(filename, 'rb');

	if (fptr == NULL) {
		printf("Error!");
		exit(1);
	}
	else {
		while (scanf("%f %f %f %f %f", &strike, &maturity, &num_terms, &reversion_rate, &volatility) == 5) {
			A[i] = strike;
			B[i] = maturity;
			C[i] = num_terms;
			D[i] = reversion_rate;
			E[i] = volatility;

			i++
		}
	}

	fclose(fptr);

}

// there is a possibility that it should be char instead of float. Check it before sending
int main(void) {
	char strikes[MAXSIZE] = { 0 };
	char maturities[MAXSIZE] = { 0 };
	char terms[MAXSIZE] = { 0 };
	char rates[MAXSIZE] = { 0 };
	char volatilities[MAXSIZE] = { 0 };
	
	const char myFile[] = "small - old.in";

	parser(myFile, strikes, maturities, terms, rates, volatilities);

	printf(strikes);
	return 0;

}
