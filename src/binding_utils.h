#ifndef BINDING_UTILS_H
#define BINDING_UTILS_H

#define MOVE_TO_NP(type, eig_var) \
  pybind11::detail::eigen_encapsulate< \
    pybind11::detail::EigenProps<type>>(new type(std::move(eig_var)))

#endif // BINDING_UTILS_H
