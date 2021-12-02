#include "cmx.hpp"
#include <iostream>
#include <stdio.h>

int main(int argc, char **argv)
{
  CMX::Environment *env = CMX::Environment::instance(&argc,&argv);
  CMX::Group *group = env->getWorldGroup();
  int size = group->size();
  int rank = group->rank();
  std::cout << "Printing from rank "<<rank<<" of "<<size<<std::endl;
  return 0;
}
