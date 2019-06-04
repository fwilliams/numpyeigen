#include <tuple>
#include <Eigen/Core>
#include <iostream>

#include <npe.h>

#define __NPE_REDIRECT_IO__

npe_function(redirect_io)
npe_begin_code()

std::cout << "this is a test of stdout" << std::endl;
std::cerr << "this is a test of stderr" << std::endl;

npe_end_code()


