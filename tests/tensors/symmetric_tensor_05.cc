// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// make sure the tensor t_ijkl=delta_ik delta_jl + delta_il delta_jk
// actually maps a rank-2 tensor onto twice itself

#include <deal.II/base/symmetric_tensor.h>

#include "../tests.h"


template <int dim>
void
test()
{
  SymmetricTensor<4, dim> t;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      for (unsigned int k = 0; k < dim; ++k)
        for (unsigned int l = 0; l < dim; ++l)
          t[i][j][k][l] =
            (((i == k) && (j == l) ? 1 : 0) + ((i == l) && (j == k) ? 1 : 0));

  SymmetricTensor<2, dim> a, b;
  a[0][0] = 1;
  a[1][1] = 2;
  a[0][1] = 3;

  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      {
        double tmp_ij = 0;
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
            {
              deallog << i << ' ' << j << ' ' << k << ' ' << l << ": "
                      << t[i][j][k][l] << ' ' << a[k][l] << std::endl;
              tmp_ij += t[i][j][k][l] * a[k][l];
            }
        b[i][j] = tmp_ij;
      }

  AssertThrow(a == b / 2, ExcInternalError());

  // try the same thing with scaled
  // tensors etc
  t *= 2;
  b.clear();
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      {
        double tmp_ij = 0;
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
            tmp_ij += t[i][j][k][l] * a[k][l];
        b[i][j] = tmp_ij;
      }

  AssertThrow(a == b / 4, ExcInternalError());
}



int
main()
{
  initlog();
  deallog << std::setprecision(3);

  test<2>();
  test<3>();

  deallog << "OK" << std::endl;
}
