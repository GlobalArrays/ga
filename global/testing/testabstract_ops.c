#define NUM_THREADS 4
#define TRIALS 1000000
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ga.h"
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "abstract_ops.h"
typedef struct cpl{
    double real;
    double imag;
} cpl;

double abs_val(cpl a);
void divide(cpl *a, cpl *b, cpl *c);

int main(int argc, char **argv)
{
    int i;
    int minsafe, maxsafe,divsafe;
#if defined(_OPENMP)
    srand(time(NULL));
    cpl * arr1 = (cpl*)malloc(TRIALS*sizeof(cpl));
    cpl * arr2 = (cpl*)malloc(TRIALS*sizeof(cpl));
    cpl * arr3 = (cpl*)malloc(TRIALS*sizeof(cpl));
    cpl * arr4 = (cpl*)malloc(TRIALS*sizeof(cpl));
    cpl * arr5 = (cpl*)malloc(TRIALS*sizeof(cpl));
    cpl * arr6 = (cpl*)malloc(TRIALS*sizeof(cpl));

    for(i = 0; i < TRIALS; i++)
    {
        arr2[i].real = (double) (rand() % 100)+1;
        arr2[i].imag = (double) (rand() % 100)+1;
        arr3[i].real = (double) (rand() % 100)+1;
        arr3[i].imag = (double) (rand() % 100)+1;
    }
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
    {
#pragma omp for
        for(i = 0; i < TRIALS; i++)
        {
            cpl a,b,c;
            b = arr2[i];
            c = arr3[i];
            assign_max_cpl(a,b,c);
            arr4[i] = a;
            assign_min_cpl(a,b,c);
            arr5[i] = a;
            assign_div_cpl(a,b,c);
            arr6[i] = a;
        }
    }

    maxsafe=1;
    minsafe = 1;
    divsafe = 1;
    for(i = 0; i < TRIALS; i++)
    {
        if(maxsafe)
        {
            arr1[i] = abs_val(arr2[i])>abs_val(arr3[i])\
                      ? arr2[i] : arr3[i];
            if(arr1[i].real != arr4[i].real ||\
                    arr1[i].imag != arr4[i].imag)
                maxsafe = 0;
        }
        if(minsafe)
        {
            arr1[i] = abs_val(arr2[i])<abs_val(arr3[i])\
                      ? arr2[i] : arr3[i];
            if(arr1[i].real != arr5[i].real ||\
                    arr1[i].imag != arr5[i].imag)
                minsafe =0;
        }
        if(divsafe)
        {
            divide(&arr1[i],&arr2[i],&arr3[i]);
            if(arr1[i].real != arr6[i].real ||\
                    arr1[i].imag != arr6[i].imag)
                divsafe = 0;
        }
    }
    if (minsafe == 0)
        printf("assign_min_cpl is not threadsafe\n");
    if (maxsafe == 0)
        printf("assign_max_cpl is not threadsafe\n");
    if (divsafe == 0)
        printf("assign_div_cpl is not threadsafe\n");

    free(arr1);
    free(arr2);
    free(arr3);
    free(arr4);
    free(arr5);
    free(arr6);

    return (minsafe + maxsafe+ divsafe != 3);
#else
    printf("OPENMP Disabled\n");
    return 1;
#endif
}

double abs_val(cpl a)
{
    return sqrt(a.real*a.real + a.imag *a.imag);
}

void divide(cpl *a, cpl *b, cpl *c)
{
    a->real = (b->real*c->real+b->imag*c->imag)/\
              (c->real*c->real+c->imag*c->imag);
    a->imag = (b->imag*c->real-b->real*c->imag)/\
              (c->real*c->real+c->imag*c->imag);
}
