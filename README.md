# Experiments for constructing "zero" overhead Python bindings for LibIGL

The goal is to support a wide range of reasonable input types and not do any data copies if possible. Ideally, I'd like libigl to work as a geometry extension to scipy and numpy.