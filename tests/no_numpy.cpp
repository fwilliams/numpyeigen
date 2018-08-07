#include <string>
#include <npe.h>

npe_function(no_numpy)
npe_arg(a, std::string)
npe_begin_code()

return a;

npe_end_code()
