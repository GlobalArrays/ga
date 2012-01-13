#include <iostream>

#include <omp.h>
#include <pthread.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  const long int maxSz = 1000000 * 1000;

  int maxt = omp_get_max_threads();
  cpu_set_t *sets = new cpu_set_t[maxt];
  double *dat = new double[maxSz], sum = 0.0;
  double *rnd = new double[maxSz];
  double t0, t1;

  for (int i = 0; i < maxt; i++)
    CPU_ZERO(&sets[i]);

  for (long int i = 0; i < maxSz; i++)
    rnd[i] = random();

  t0 = omp_get_wtime();
#pragma omp parallel default(none), shared(dat, rnd, sets, sum)
  {
    pthread_t tid = pthread_self();
    int otid = omp_get_thread_num();

    pthread_getaffinity_np(tid, sizeof(sets[otid]), &sets[otid]);

#pragma omp for reduction(+: sum), schedule(static)
    for (long int i = 0; i < maxSz; i++) {
      dat[i] = i + rnd[i];
      sum += dat[i];
    }
  }
  t1 = omp_get_wtime();

  for (int i = 0; i < maxt; i++) {
    std::cout << "Thread: " << i << ", CPU set: ";

    for (int j = 0; j < CPU_SETSIZE; j++)
      if (CPU_ISSET(j, &sets[i]))
	std::cout << j << " ";

    std::cout << std::endl;
  }

  std::cout << "Sum: " << std::fixed << sum << ", time: " << t1 - t0 << std::endl;

  delete [] dat;
  delete [] sets;

  return 0;
} /* main */
