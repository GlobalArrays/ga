int fred;

void Busy(int n)
{
  fred = 0;
  while (n-- >= 0)
    fred++;
}
