// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2008 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



// like kelly_crash_01 (which has since been renamed circular_01 because it
// continued to fail, though for other reasons) but much reduced. it turned
// out that cell->face(f)->at_boundary() and cell->at_boundary(f) did not
// always return the same thing, although they of course should. as a result,
// the KellyErrorEstimator forgot to work on certain faces
//
// the problem turned out to be that we were setting a boundary indicator for
// an interior face. while cell->face(f)->at_boundary() checks for a boundary
// indicator != 255, cell->at_boundary(f) checks whether the cell has no
// neighbor in position f. The latter was false (there was a neighbor) but the
// former true, thus the discrepancy.
//
// the problem was fixed by fixing the code

#include <deal.II/base/utilities.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include "../tests.h"



void
test()
{
  const unsigned int dim = 3;

  static const Point<3> vertices_1[] = {
    // points on the lower surface
    Point<dim>(0, 0, -4),
    Point<dim>(std::cos(0 * numbers::PI / 6),
               std::sin(0 * numbers::PI / 6),
               -4),
    Point<dim>(std::cos(2 * numbers::PI / 6),
               std::sin(2 * numbers::PI / 6),
               -4),
    Point<dim>(std::cos(4 * numbers::PI / 6),
               std::sin(4 * numbers::PI / 6),
               -4),
    Point<dim>(std::cos(6 * numbers::PI / 6),
               std::sin(6 * numbers::PI / 6),
               -4),
    Point<dim>(std::cos(8 * numbers::PI / 6),
               std::sin(8 * numbers::PI / 6),
               -4),
    Point<dim>(std::cos(10 * numbers::PI / 6),
               std::sin(10 * numbers::PI / 6),
               -4),

    // same points on the top
    // of the stem, with
    // indentation in the middle
    Point<dim>(0, 0, 4 - std::sqrt(2.) / 2),
    Point<dim>(std::cos(0 * numbers::PI / 6), std::sin(0 * numbers::PI / 6), 4),
    Point<dim>(std::cos(2 * numbers::PI / 6), std::sin(2 * numbers::PI / 6), 4),
    Point<dim>(std::cos(4 * numbers::PI / 6), std::sin(4 * numbers::PI / 6), 4),
    Point<dim>(std::cos(6 * numbers::PI / 6), std::sin(6 * numbers::PI / 6), 4),
    Point<dim>(std::cos(8 * numbers::PI / 6), std::sin(8 * numbers::PI / 6), 4),
    Point<dim>(std::cos(10 * numbers::PI / 6),
               std::sin(10 * numbers::PI / 6),
               4),

    // point at top of chevron
    Point<dim>(0, 0, 4 + std::sqrt(2.) / 2),

    // points at the top of the
    // first extension
    // points 15-18
    Point<dim>(0, 0, 7) + Point<dim>(std::cos(2 * numbers::PI / 6),
                                     std::sin(2 * numbers::PI / 6),
                                     0) *
                            4.0,
    Point<dim>(std::cos(0 * numbers::PI / 6),
               std::sin(0 * numbers::PI / 6),
               7) +
      Point<dim>(std::cos(2 * numbers::PI / 6),
                 std::sin(2 * numbers::PI / 6),
                 0) *
        4.0,
    Point<dim>(std::cos(2 * numbers::PI / 6),
               std::sin(2 * numbers::PI / 6),
               7) +
      Point<dim>(std::cos(2 * numbers::PI / 6),
                 std::sin(2 * numbers::PI / 6),
                 0) *
        4.0,
    Point<dim>(std::cos(4 * numbers::PI / 6),
               std::sin(4 * numbers::PI / 6),
               7) +
      Point<dim>(std::cos(2 * numbers::PI / 6),
                 std::sin(2 * numbers::PI / 6),
                 0) *
        4.0,

    // points at the top of the
    // second extension
    // points 19-22
    Point<dim>(0, 0, 7) + Point<dim>(std::cos(6 * numbers::PI / 6),
                                     std::sin(6 * numbers::PI / 6),
                                     0) *
                            4.0,
    Point<dim>(std::cos(4 * numbers::PI / 6),
               std::sin(4 * numbers::PI / 6),
               7) +
      Point<dim>(std::cos(6 * numbers::PI / 6),
                 std::sin(6 * numbers::PI / 6),
                 0) *
        4.0,
    Point<dim>(std::cos(6 * numbers::PI / 6),
               std::sin(6 * numbers::PI / 6),
               7) +
      Point<dim>(std::cos(6 * numbers::PI / 6),
                 std::sin(6 * numbers::PI / 6),
                 0) *
        4.0,
    Point<dim>(std::cos(8 * numbers::PI / 6),
               std::sin(8 * numbers::PI / 6),
               7) +
      Point<dim>(std::cos(6 * numbers::PI / 6),
                 std::sin(6 * numbers::PI / 6),
                 0) *
        4.0,

    // points at the top of the
    // third extension
    // points 23-26
    Point<dim>(0, 0, 7) + Point<dim>(std::cos(10 * numbers::PI / 6),
                                     std::sin(10 * numbers::PI / 6),
                                     0) *
                            4.0,
    Point<dim>(std::cos(8 * numbers::PI / 6),
               std::sin(8 * numbers::PI / 6),
               7) +
      Point<dim>(std::cos(10 * numbers::PI / 6),
                 std::sin(10 * numbers::PI / 6),
                 0) *
        4.0,
    Point<dim>(std::cos(10 * numbers::PI / 6),
               std::sin(10 * numbers::PI / 6),
               7) +
      Point<dim>(std::cos(10 * numbers::PI / 6),
                 std::sin(10 * numbers::PI / 6),
                 0) *
        4.0,
    Point<dim>(std::cos(0 * numbers::PI / 6),
               std::sin(0 * numbers::PI / 6),
               7) +
      Point<dim>(std::cos(10 * numbers::PI / 6),
                 std::sin(10 * numbers::PI / 6),
                 0) *
        4.0,

  };

  const unsigned int n_vertices = sizeof(vertices_1) / sizeof(vertices_1[0]);
  const std::vector<Point<dim>> vertices(&vertices_1[0],
                                         &vertices_1[n_vertices]);
  static const int cell_vertices[][GeometryInfo<dim>::vertices_per_cell] = {
    // the three cells in the stem
    {0, 2, 4, 3, 7, 9, 11, 10},
    {6, 0, 5, 4, 13, 7, 12, 11},
    {6, 1, 0, 2, 13, 8, 7, 9},
    // the chevron at the center
    {13, 8, 7, 9, 12, 14, 11, 10},
    // first extension
    {14, 8, 10, 9, 15, 16, 18, 17},
    // second extension
    {11, 12, 10, 14, 21, 22, 20, 19},
    // third extension
    {12, 13, 14, 8, 24, 25, 23, 26},
  };
  const unsigned int n_cells = sizeof(cell_vertices) / sizeof(cell_vertices[0]);

  std::vector<CellData<dim>> cells(n_cells, CellData<dim>());
  for (unsigned int i = 0; i < n_cells; ++i)
    {
      for (const unsigned int j : GeometryInfo<dim>::vertex_indices())
        cells[i].vertices[j] = cell_vertices[i][j];
      cells[i].material_id = 0;
    }

  Triangulation<3> triangulation;
  triangulation.create_triangulation(vertices, cells, SubCellData());

  for (Triangulation<dim>::active_cell_iterator cell =
         triangulation.begin_active();
       cell != triangulation.end();
       ++cell)
    for (const unsigned int f : GeometryInfo<dim>::face_indices())
      if ((cell->face(f)->center()[2] != -4) &&
          (cell->face(f)->center()[2] != 7) && (cell->face(f)->at_boundary()))
        cell->face(f)->set_boundary_id(1);

  triangulation.refine_global(1);

  for (Triangulation<dim>::active_cell_iterator cell =
         triangulation.begin_active();
       cell != triangulation.end();
       ++cell)
    for (const unsigned int face_no : GeometryInfo<dim>::face_indices())
      AssertThrow(cell->at_boundary(face_no) ==
                    cell->face(face_no)->at_boundary(),
                  ExcInternalError());

  deallog << "OK" << std::endl;
}


int
main()
{
  initlog();
  deallog.get_file_stream().precision(3);

  test();
  return 0;
}
