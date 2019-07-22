npe_function(one_d_arg)
npe_arg(v, dense_double, dense_float)
npe_arg(f, dense_int, dense_long)
npe_arg(p, npe_matches(v))
npe_begin_code()

  return std::make_tuple(npe::move(v), npe::move(p));

npe_end_code()


npe_function(one_d_arg_big)
npe_arg(v, dense_double, dense_float)
npe_arg(f, dense_int, dense_long)
npe_arg(p, npe_matches(v))
npe_arg(q, npe_matches(p))
npe_arg(r, npe_matches(q))
npe_arg(s, npe_matches(v))
npe_begin_code()

  return std::make_tuple(npe::move(v), npe::move(p));

npe_end_code()
