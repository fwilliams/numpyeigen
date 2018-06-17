# Experiments for constructing "zero" overhead Python bindings for LibIGL

The goal is to support a wide range of reasonable input types and not do any data copies if possible. Ideally, I'd like libigl to work as a geometry extension to scipy and numpy.

## Building and running

Don't forget to `git clone --recursive`!!!

Building should be as easy as:
```
mkdir build
cd build
cmake ..
make
```

There's an `environment.yml` file which can be used to generate conda environment with all the right dependencies by running:
```
conda env create -f environment.yml
```
