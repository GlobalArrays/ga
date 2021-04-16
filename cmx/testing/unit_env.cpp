#include "cmx.hpp"

int main(int argc, char **argv)
{
  CMX::Environment *env = CMX::Environment::instance(&argc,&argv);
  return 0;
}
