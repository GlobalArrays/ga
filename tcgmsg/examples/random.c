double random_(seed)
     unsigned long *seed;
{
  *seed = *seed *1812433253 + 12345;
  return (double) ((*seed & 0x7fffffff) * 4.6566128752458e-10);
}
