// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2010 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



// we couldn't call DoFTools::dof_indices_with_subdomain_association
// without causing an abort
//
// in this test, let processor 1 output stuff since its index range
// is non-contiguous

#include <deal.II/base/tensor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include "../tests.h"



template <int dim>
void
test()
{
  unsigned int myid = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 1)
    deallog << "hyper_cube" << std::endl;

  parallel::distributed::Triangulation<dim> tr(MPI_COMM_WORLD);

  GridGenerator::hyper_cube(tr);
  tr.refine_global(3);
  DoFHandler<dim> dofh(tr);

  static const FE_Q<dim> fe(2);
  dofh.distribute_dofs(fe);

  const std::vector<types::global_dof_index>
    n_locally_owned_dofs_per_processor =
      Utilities::MPI::all_gather(MPI_COMM_WORLD, dofh.n_locally_owned_dofs());
  if (myid == 1)
    {
      deallog << "dofh.n_dofs() " << n_locally_owned_dofs_per_processor
              << std::endl;
      deallog << "dofh.n_locally_owned_dofs() " << dofh.n_locally_owned_dofs()
              << std::endl;

      const IndexSet set = DoFTools::dof_indices_with_subdomain_association(
        dofh, tr.locally_owned_subdomain());

      deallog << set.n_elements() << std::endl;
      for (unsigned int i = 0; i < set.n_elements(); ++i)
        deallog << "   " << set.nth_index_in_set(i) << std::endl;
    }
}


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  unsigned int myid = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);


  deallog.push(Utilities::int_to_string(myid));

  if (myid == 1)
    {
      initlog();

      deallog.push("2d");
      test<2>();
      deallog.pop();
    }
  else
    test<2>();
}
