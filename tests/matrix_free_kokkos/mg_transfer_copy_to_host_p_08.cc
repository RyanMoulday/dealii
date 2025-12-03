// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


/**
 * Like mg_transfer_p_08.cc but for MGTwoLevelTransferCopyToHost.
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include "mg_transfer_copy_to_host_util.h"

using namespace dealii;

template <int dim, typename Number, typename MemorySpace>
void
do_test(const FiniteElement<dim> &fe_fine,
        const FiniteElement<dim> &fe_coarse,
        const Triangulation<dim> &tria)
{
  deallog << "Testing " << fe_fine.get_name() << " <-> " << fe_coarse.get_name()
          << std::endl;
  // setup dof-handlers
  DoFHandler<dim> dof_handler_fine(tria);
  dof_handler_fine.distribute_dofs(fe_fine);

  DoFHandler<dim> dof_handler_coarse(tria);
  dof_handler_coarse.distribute_dofs(fe_coarse);

  AffineConstraints<Number> constraints_coarse(
    dof_handler_coarse.locally_owned_dofs(),
    DoFTools::extract_locally_relevant_dofs(dof_handler_coarse));
  DoFTools::make_hanging_node_constraints(dof_handler_coarse,
                                          constraints_coarse);
  constraints_coarse.close();

  AffineConstraints<Number> constraints_fine(
    dof_handler_fine.locally_owned_dofs(),
    DoFTools::extract_locally_relevant_dofs(dof_handler_fine));
  DoFTools::make_hanging_node_constraints(dof_handler_fine, constraints_fine);
  constraints_fine.close();

  // setup transfer operator
  MGTwoLevelTransferCopyToHost<
    dim,
    LinearAlgebra::distributed::Vector<Number, MemorySpace>>
    transfer_ref;
  transfer_ref.reinit(dof_handler_fine,
                      dof_handler_coarse,
                      constraints_fine,
                      constraints_coarse);

  MappingQ1<dim>          mapping;
  MatrixFree<dim, Number> mf_coarse, mf_fine;
  mf_coarse.reinit(mapping,
                   dof_handler_coarse,
                   constraints_coarse,
                   QGauss<dim>(1),
                   typename MatrixFree<dim, Number>::AdditionalData());
  mf_fine.reinit(mapping,
                 dof_handler_fine,
                 constraints_fine,
                 QGauss<dim>(1),
                 typename MatrixFree<dim, Number>::AdditionalData());
  MGTwoLevelTransferCopyToHost<
    dim,
    LinearAlgebra::distributed::Vector<Number, MemorySpace>>
    transfer;
  transfer.reinit(mf_fine, 0, mf_coarse, 0);

  LinearAlgebra::distributed::Vector<Number, MemorySpace> vec_fine_0,
    vec_fine_1, vec_coarse_0, vec_coarse_1;

  LinearAlgebra::distributed::Vector<Number> tmp;

  mf_fine.initialize_dof_vector(vec_fine_0);
  mf_fine.initialize_dof_vector(vec_fine_1);
  mf_coarse.initialize_dof_vector(vec_coarse_0);
  mf_coarse.initialize_dof_vector(tmp);
  mf_coarse.initialize_dof_vector(vec_coarse_1);

  // prolongate
  for (auto &d : tmp)
    d = random_value<double>();
  copy_from_host(vec_coarse_0, tmp);
  transfer_ref.prolongate_and_add(vec_fine_0, vec_coarse_0);
  transfer.prolongate_and_add(vec_fine_1, vec_coarse_0);
  deallog << "Prolongate: " << vec_fine_0.l2_norm() << " "
          << vec_fine_1.l2_norm() << " ";
  vec_fine_1 -= vec_fine_0;
  deallog << vec_fine_1.l2_norm() << std::endl;

  vec_coarse_0 = 0;
  transfer_ref.restrict_and_add(vec_coarse_0, vec_fine_0);
  vec_coarse_1 = 0;
  transfer.restrict_and_add(vec_coarse_1, vec_fine_0);
  deallog << "Restrict: " << vec_coarse_0.l2_norm() << " "
          << vec_coarse_1.l2_norm() << " ";
  vec_coarse_1 -= vec_coarse_0;
  deallog << vec_coarse_1.l2_norm() << std::endl << std::endl;
}



template <int dim, typename Number, typename MemorySpace>
void
test(int fe_degree_fine, int fe_degree_coarse)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, -1., 1.);
  tria.refine_global(5 - dim);

  deallog << "Uniform grid" << std::endl << std::endl;
  do_test<dim, Number, MemorySpace>(FE_Q<dim>(fe_degree_fine),
                                    FE_Q<dim>(fe_degree_coarse),
                                    tria);
  do_test<dim, Number, MemorySpace>(FE_DGQ<dim>(fe_degree_fine),
                                    FE_Q<dim>(fe_degree_coarse),
                                    tria);
  do_test<dim, Number, MemorySpace>(FE_DGQ<dim>(fe_degree_fine),
                                    FE_DGQ<dim>(fe_degree_coarse),
                                    tria);

  for (const auto &cell : tria.active_cell_iterators())
    if (cell->center()[0] < 0 || cell->center()[1] < 0 ||
        cell->center()[dim - 1] < 0)
      cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  deallog << "Adaptively refined grid" << std::endl << std::endl;
  do_test<dim, Number, MemorySpace>(FE_Q<dim>(fe_degree_fine),
                                    FE_Q<dim>(fe_degree_coarse),
                                    tria);
  do_test<dim, Number, MemorySpace>(FE_DGQ<dim>(fe_degree_fine),
                                    FE_Q<dim>(fe_degree_coarse),
                                    tria);
  do_test<dim, Number, MemorySpace>(FE_DGQ<dim>(fe_degree_fine),
                                    FE_DGQ<dim>(fe_degree_coarse),
                                    tria);
}



int
main()
{
  initlog();

  MultithreadInfo::set_thread_limit(1);

  deallog.precision(8);

  test<2, double, MemorySpace::Default>(2, 1);
  test<2, double, MemorySpace::Default>(3, 1);
  test<2, double, MemorySpace::Default>(1, 1);
  test<3, double, MemorySpace::Default>(2, 1);
}
