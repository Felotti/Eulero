#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/parameter_handler.h>

#include<deal.II/base/thread_local_storage.h>
#include<deal.II/base/revision.h>

#include<deal.II/grid/manifold_lib.h>
#include<deal.II/grid/tria_iterator.h>
#include<deal.II/grid/tria_accessor.h>
#include<deal.II/base/aligned_vector.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include<deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>
#include<deal.II/grid/grid_out.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include<deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <array>
#include "eulerproblem.h"


namespace Euler_DG{
using namespace dealii;


/// constructor of Postprocessor class
template<int dim>
EulerProblem<dim>::Postprocessor::Postprocessor(Parameters::Data_Storage &parameters_in, int a_)
    : do_schlieren_plot(dim==2),parameters(parameters_in),a(a_){}


/*! For the main evaluation of the field variables, we first check that the
* lengths of the arrays equal the expected values. Then we loop over all evaluation
*  points and fill the respective information: First we fill the primal solution
* variables of density $\rho$, momentum $\rho \mathbf{u}$ and energy $E$,
* then we compute the derived velocity $\mathbf u$, the pressure $p$, the
* speed of sound $c=\sqrt{\gamma p / \rho}$, as well as the Schlieren plot
* showing $s = |\nabla \rho|^2$ in case it is enabled.*/
template <int dim>
void EulerProblem<dim>::Postprocessor::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<Vector<double>> &               computed_quantities) const
{
    const unsigned int n_evaluation_points = inputs.solution_values.size();

    if (do_schlieren_plot == true)
    Assert(inputs.solution_gradients.size() == n_evaluation_points,
           ExcInternalError());

    Assert(computed_quantities.size() == n_evaluation_points,
           ExcInternalError());
    Assert(inputs.solution_values[0].size() == dim + 2, ExcInternalError());
    Assert(computed_quantities[0].size() ==
           dim + 2 + (do_schlieren_plot == true ? 1 : 0),
           ExcInternalError());

    for (unsigned int q = 0; q < n_evaluation_points; ++q)
    {
        Tensor<1, dim + 2> solution;
        for (unsigned int d = 0; d < dim + 2; ++d)
            solution[d] = inputs.solution_values[q](d);

        const double         density  = solution[0];
        const Tensor<1, dim> velocity = euler_velocity<dim>(solution);
        const double         pressure = euler_pressure<dim>(solution,parameters);

        for (unsigned int d = 0; d < dim; ++d)
            computed_quantities[q](d) = velocity[d];

            computed_quantities[q](dim)     = pressure;
            computed_quantities[q](dim + 1) = std::sqrt(parameters.gamma * pressure / density);

        if (do_schlieren_plot == true)
            computed_quantities[q](dim + 2) =
                    inputs.solution_gradients[q][0] * inputs.solution_gradients[q][0];
    }
}


template <int dim>
std::vector<std::string> EulerProblem<dim>::Postprocessor::get_names() const
{
    if (a==0)
    { std::vector<std::string> names;
        for (unsigned int d = 0; d < dim; ++d)
            names.emplace_back("velocity");
        names.emplace_back("pressure");
        names.emplace_back("speed_of_sound");

        if (do_schlieren_plot == true)
            names.emplace_back("schlieren_plot");
        return names;
}
    else if(a==1)
    {
        std::vector<std::string> names_Q0;
        for (unsigned int d = 0; d < dim; ++d)
            names_Q0.emplace_back("velocity_Q0");
        names_Q0.emplace_back("pressure_Q0");
        names_Q0.emplace_back("speed_of_sound_Q0");

        if (do_schlieren_plot == true)
            names_Q0.emplace_back("schlieren_plot_Q0");
        return names_Q0;

    }


}

/// For the interpretation of quantities, we have scalar density, energy,
/// pressure, speed of sound, and the Schlieren plot, and vectors for the
/// momentum and the velocity.
template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
EulerProblem<dim>::Postprocessor::get_data_component_interpretation() const
{
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
            interpretation;
    for (unsigned int d = 0; d < dim; ++d)
        interpretation.push_back(
                DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    if (do_schlieren_plot == true)
        interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);

    return interpretation;
}

/// With respect to the necessary update flags, we only need the values for
/// all quantities but the Schlieren plot, which is based on the density
/// gradient.
template <int dim>
UpdateFlags EulerProblem<dim>::Postprocessor::get_needed_update_flags() const
{
    if (do_schlieren_plot == true)
        return update_values | update_gradients;
    else
        return update_values;
}


/// The constructor for this class is unsurprising: We set up a parallel
/// triangulation based on the `MPI_COMM_WORLD` communicator, a vector finite
/// element with `dim+2` components for density, momentum, and energy, a
/// high-order mapping of the same degree as the underlying finite element,
/// initialize the time and time step to zero, and finite element usefull for the TVB limiter.
template <int dim>
EulerProblem<dim>::EulerProblem(Parameters::Data_Storage  &parameters_in)
        : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
#ifdef DEAL_II_WITH_P4EST
        , triangulation(MPI_COMM_WORLD)
#endif
        , parameters(parameters_in)
        , fe(FE_DGQ<dim>(parameters.fe_degree),dim+2)
        , fe_Q0(FE_DGQ<dim>(parameters.fe_degree_Q0), dim+2)
        , mapping()
        , mapping_Q0()
        , dof_handler(triangulation)
        , dof_handler_Q0(triangulation)
        , quadrature(fe.degree + 1)
        , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
        , euler_operator(timer,parameters_in)
        , euler_operator_Q0(timer,parameters_in)
        , time(0)
        , time_step(0)
        , fe_cell (FE_DGQ<dim>(0))
        , dh_cell (triangulation)
       {
    dof_handlers.clear();
    dof_handlers_Q0.clear();
    quadratures.clear();
    quadratures_Q0.clear();
}



/// given a cell iterator, return the cell numbe of a given cell
template <typename ITERATOR>
unsigned int cell_number (const ITERATOR &cell)
{
    return cell->user_index();
}

/*!
* \brief EulerProblem::make_grid
*  As a mesh, this program implements different options, depending on the
* global variable `testcase`.
* if testcase == 1, then grid is channel with hole
* if testcase == 2, then grid is backward step
* if testcase == 3, then grid is forward step
* if testcase == 4, then grid is a rectangle
* if testcase == 5, then grid is a half cylinder (inner shell is sphere while outer shell is ellipse)
* if testcase == 6, then grid is a rectangle
* if testcase == 7, then grid is a cube
*
* For each testcase we impose also the boundary condition on the ghost cells.
*/
template <int dim>
void EulerProblem<dim>::make_grid()
{
    switch (parameters.testcase)
    {
    case 1:  // channel with hole
   {
        GridGenerator::channel_with_cylinder(
                triangulation, 0.03, 1, 0, true);
        triangulation.refine_global(2);


        euler_operator.set_inflow_boundary(
                0, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,1));
        euler_operator.set_subsonic_outflow_boundary(
                1, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,1));
        euler_operator.set_wall_boundary(2);
        euler_operator.set_wall_boundary(3);

        euler_operator_Q0.set_inflow_boundary(
                0, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,1));
        euler_operator_Q0.set_subsonic_outflow_boundary(
                1, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,1));
        euler_operator_Q0.set_wall_boundary(2);
        euler_operator_Q0.set_wall_boundary(3);

        if (dim == 3){
            euler_operator.set_body_force(
                    std::make_unique<Functions::ConstantFunction<dim>>(
                            std::vector<double>({0., 0., -0.2})));
            euler_operator_Q0.set_body_force(
                    std::make_unique<Functions::ConstantFunction<dim>>(
                            std::vector<double>({0., 0., -0.2})));
            }
        break;
        }
    case 2 : //backward facing step test case
    {
        const double L1 = 1;
        const double L2 = 12;
        const double H = 11;
        const double h = 6;
        const double X1_coordinate_inflow = 0.;
        const double Y1_bottom = 0.;
        const double X1_coordinate_outflow = L2 + L1;

        std::vector<unsigned int > repetitions = {210,178};
        Point<dim> left_bottom = {0,0};
        Point<dim> right_top = {L1+L2,H};
        std::vector<int> cells_to_remove = {17,97};
        GridGenerator::subdivided_hyper_L(triangulation, repetitions, left_bottom, right_top, cells_to_remove);
        //set boundary ID's
        for(auto &face : triangulation.active_face_iterators())
        {
           if(face->at_boundary())
           {
                //left boundary has ID = 2
                if(face->center()[0] = X1_coordinate_inflow )
                   face->set_boundary_id(2);

                //right boundary has ID = 3
                if(face->center()[0] == X1_coordinate_outflow)
                   face->set_boundary_id(3);

                //bottom boundary has ID = 0
                if(face->center()[1] == Y1_bottom)
                   face->set_boundary_id(0);

                //top boundary has ID = 1
                if(face->center()[1] == H)
                   face->set_boundary_id(1);

                //L boundary has ID = 4
                 if(face->center()[1] == h || face->center()[0] == L1 )
                    face->set_boundary_id(4);

            }
        }
        euler_operator.set_inflow_boundary(
            2, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,1));
        euler_operator.set_supersonic_outflow_boundary(3);
        euler_operator.set_supersonic_outflow_boundary(0);
        euler_operator.set_supersonic_outflow_boundary(1);
        euler_operator.set_wall_boundary(4);

        euler_operator_Q0.set_inflow_boundary(
            2, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,1));
        euler_operator_Q0.set_supersonic_outflow_boundary(3);
        euler_operator_Q0.set_supersonic_outflow_boundary(0);
        euler_operator_Q0.set_supersonic_outflow_boundary(1);
        euler_operator_Q0.set_wall_boundary(4);

        if (dim == 3){
            euler_operator.set_body_force(
                        std::make_unique<Functions::ConstantFunction<dim>>(
                                std::vector<double>({0., 0., -0.2})));
            euler_operator_Q0.set_body_force(
                        std::make_unique<Functions::ConstantFunction<dim>>(
                                std::vector<double>({0., 0., -0.2})));
            }
          break;
        }
    case 3 : //forward facing step test case
    {

        const double height_step = 0.2;
        const double pos_step = 0.6;  // step positioned at x=0.6
        const int height = 1;
        const int X1_coordinate_inflow = 0;
        const int X1_coordinate_outflow = 3;
        const int Y1_bottom = 0;

        std::vector<unsigned int > repetitions = {50,10};
        Point<dim> left_bottom = {0,0};
        Point<dim> right_top = {3,1};
        std::vector<int> cells_to_remove = {-40,2};
        GridGenerator::subdivided_hyper_L(triangulation, repetitions, left_bottom, right_top, cells_to_remove);
        triangulation.refine_global(2);
        //set boundary ID's
        for(auto &face : triangulation.active_face_iterators())
        {
           if(face->at_boundary())
           {
                //left boundary has ID = 2
                if(face->center()[0] == X1_coordinate_inflow )
                   face->set_boundary_id(2);

                //right boundary has ID = 3
                if(face->center()[0] == X1_coordinate_outflow)
                   face->set_boundary_id(3);

                //bottom boundary has ID = 0
                if(face->center()[1] == Y1_bottom)
                   face->set_boundary_id(0);

                //top boundary has ID = 1
                if(face->center()[1] == height)
                   face->set_boundary_id(1);

                //obstacle boundary has ID = 4
                if((face->center()[0] == pos_step) || (face->center()[1] == height_step))
                   face->set_boundary_id(4);
            }
        }
        euler_operator.set_inflow_boundary(
                2, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,1));
        euler_operator.set_supersonic_outflow_boundary(3);
        euler_operator.set_wall_boundary(1);
        euler_operator.set_wall_boundary(4);
        euler_operator.set_wall_boundary(0);

        euler_operator_Q0.set_inflow_boundary(
                2, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,1));
        euler_operator_Q0.set_supersonic_outflow_boundary(3);
        euler_operator_Q0.set_wall_boundary(1);
        euler_operator_Q0.set_wall_boundary(4);
        euler_operator_Q0.set_wall_boundary(0);



        if (dim == 3){
            euler_operator.set_body_force(
                    std::make_unique<Functions::ConstantFunction<dim>>(
                            std::vector<double>({0., 0., -0.2})));
        euler_operator_Q0.set_body_force(
                    std::make_unique<Functions::ConstantFunction<dim>>(
                            std::vector<double>({0., 0., -0.2})));
        }
        break;
    }
    case 4 : //sod-shock test case
    {

        GridGenerator::subdivided_hyper_rectangle(triangulation,{160,13},Point<dim>(-0.5,0),Point<dim>(0.5,0.2));

        for(auto &face : triangulation.active_face_iterators())
        {
            if(face->at_boundary())
            {
                //left boundary has ID = 0
                if(face->center()[0] == -0.5 )
                   {  face->set_boundary_id(0); }

                //right boundary has ID = 1
                if(face->center()[0] == 0.5)
                   {  face->set_boundary_id(1); }

                //bottom boundary has ID = 2
                if(face->center()[1] == 0)
                   {  face->set_boundary_id(2); }

                //top boundary has ID = 3
                if(face->center()[1] == 0.2)
                   {  face->set_boundary_id(3); }

            }
        }
        euler_operator.set_inflow_boundary(
                  0, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,1));
        euler_operator.set_supersonic_outflow_boundary(1);

        euler_operator.set_wall_boundary(2);
        euler_operator.set_wall_boundary(3);

        euler_operator_Q0.set_inflow_boundary(
                0, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,1));
        euler_operator_Q0.set_supersonic_outflow_boundary(1);


        euler_operator_Q0.set_wall_boundary(2);
        euler_operator_Q0.set_wall_boundary(3);

        if (dim == 3){
            euler_operator_Q0.set_body_force(
                    std::make_unique<Functions::ConstantFunction<dim>>(
                            std::vector<double>({0., 0., -0.2})));
            euler_operator_Q0.set_body_force(
                    std::make_unique<Functions::ConstantFunction<dim>>(
                            std::vector<double>({0., 0., -0.2})));
        }

        break;
    }
    case 5:  //supersonic flow past a circular cylinder test case
    {
        const double inner_radius =0.5;
        const double outer_radius = 2.;
        const double eccentricity = 0.8;
        Triangulation<dim> grid;
        Point<dim> center;
        center[0]=0.0;
        center[1]=0.0;
        const std::array<double, 5> angle{{0, numbers::PI_4, numbers::PI_2, 3.0 * numbers::PI_4, numbers::PI}};
        std::vector<Point<2>> points;
        // Create an elliptical manifold to use push_forward()
        Tensor<1, 2> axis;
        axis[0] = 1.0;
        axis[1] = 0.0;
        PolarManifold<2> manip(center);
        EllipticalManifold<2, 2> mani(center, axis, eccentricity);

        for (auto &j : angle)
        {
            points.push_back(manip.push_forward(Point<2>(inner_radius, j)));
        }
        for (auto &j : angle)
        {
            points.push_back(mani.push_forward(Point<2>(outer_radius*eccentricity, j))+center);
        }
        std::array<int, 6> dup{{1, 2, 3, 6, 7, 8}};
        for (int &i : dup)
        {
            Point<2> pt = points[i];
            pt[1]      = -pt[1];
            points.push_back(pt);
        }
        unsigned int          cell[][4] = {{5, 6, 0, 1},
                                  {6, 7, 1, 2},
                                  {8, 3, 7, 2},
                                  {9, 4, 8, 3},
                                  {5, 0, 13, 10},
                                  {13, 10, 14, 11},
                                  {15, 14, 12, 11},
                                  {9, 15, 4, 12}};

        std::vector<CellData<2>> cells(8, CellData<2>());
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j)
                cells[i].vertices[j] = cell[i][j];
        SubCellData scdata;
        grid.create_triangulation(points, cells, scdata);

        grid.reset_all_manifolds();
        for(auto &cell : grid.active_cell_iterators())
        {
            for(unsigned int v = 0; v<GeometryInfo<2>::faces_per_cell; ++v)
            {
                if(std::fabs(center.distance(cell->face(v)->center()))<=0.5 && cell->face(v)->at_boundary())
                   cell->face(v)->set_manifold_id(1);
                else if(std::abs(center.distance(cell->face(v)->center()))<=2.5 && cell->face(v)->at_boundary())
                   cell->face(v)->set_manifold_id(0);
                //else
                //  cell->face(v)->set_manifold_id(1);
            }
        }

        grid.set_manifold(1,SphericalManifold<2>(center));
        grid.set_manifold(0, EllipticalManifold<2, 2>(center, axis, eccentricity));
        grid.refine_global(4);
        GridTools::rotate(numbers::PI_2,grid);

      //  build_simple_hyper_shell(cyl, center, inner_radius,outer_radius, 0.8);

        std::set<typename Triangulation<dim>::active_cell_iterator> cells_to_remove;
        for(const auto &cell : grid.active_cell_iterators())
        {
            for(unsigned int v = 0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
            {
                if((cell->vertex(v)[0]>0.01))
                    cells_to_remove.insert(cell);
            }
        }

        GridGenerator::create_triangulation_with_removed_cells(grid,cells_to_remove,triangulation);

        // outer boundary has ID = 1 inflow
        // inner boundary has ID = 0 wall
        // parts of the boundary where x=0 has ID = 2 outflow
        for(auto &face : triangulation.active_face_iterators())
        {
                if(face->at_boundary())
                {
                    if(center.distance(face->center())<=inner_radius)
                        face->set_boundary_id(0);
                    else if(std::fabs(face->center()[0])<1e-4)
                        face->set_boundary_id(2);
                    else
                        face->set_boundary_id(1);
                }
        }

        euler_operator.set_inflow_boundary(
                1, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,1));
        euler_operator.set_supersonic_outflow_boundary(2);
        euler_operator.set_wall_boundary(0);


        euler_operator_Q0.set_inflow_boundary(
                1, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,1));
        euler_operator_Q0.set_supersonic_outflow_boundary(2);
        euler_operator_Q0.set_wall_boundary(0);

        if (dim == 3)
            {
            euler_operator.set_body_force(
                    std::make_unique<Functions::ConstantFunction<dim>>(
                            std::vector<double>({0., 0., -0.2})));
            euler_operator_Q0.set_body_force(
                    std::make_unique<Functions::ConstantFunction<dim>>(
                            std::vector<double>({0., 0., -0.2})));
            }

        break;
    }
    case 6:  //double mach reflection test case
    {
        double length = 3.2;
        double height = 1.0;
        double wall_position = 1./6.;
        Triangulation<dim> tria1,tria2,tria3;
        tria3.set_mesh_smoothing(triangulation.get_mesh_smoothing());

        GridGenerator::subdivided_hyper_rectangle(tria1,{18,6},Point<dim>(wall_position,0.),Point<dim>(length,height));
        GridGenerator::subdivided_hyper_rectangle(tria2,{1,6},Point<dim>(0.,0.),Point<dim>(wall_position,height));
        GridGenerator::merge_triangulations(tria1,tria2,tria3);
        triangulation.copy_triangulation(tria3);
        triangulation.refine_global(3);

        for(auto cell :  triangulation.active_cell_iterators())
        {
            for(unsigned int j=0; j< GeometryInfo<dim>::faces_per_cell; ++j )
            {
                if(cell->face(j)->at_boundary())
                {
                    if(cell->face(j)->center()[0] == 0. )
                       cell->face(j)->set_boundary_id(0);

                    if(cell->face(j)->center()[0] >length - 1.e-6)
                        cell->face(j)->set_boundary_id(1);

                    if(cell->face(j)->center()[1] == height)
                       cell->face(j)->set_boundary_id(2);

                    //bottom boundary x<=1/6 has ID = 3
                    if(cell->face(j)->center()[0] <= wall_position && cell->face(j)->center()[1] <1.e-6)
                        cell->face(j)->set_boundary_id(3);

                    //bottom boundary x>=1/6 has ID = 4
                    if(cell->face(j)->center()[0] > wall_position && cell->face(j)->center()[1]<1.e-6)
                        cell->face(j)->set_boundary_id(4);


                }
            }
        }

        euler_operator.set_inflow_boundary(
                0, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,0));
        euler_operator.set_inflow_boundary(
                2, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,1));
        euler_operator.set_supersonic_outflow_boundary(1);
        euler_operator.set_inflow_boundary(
                3, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,0));
        euler_operator.set_wall_boundary(4);

        euler_operator_Q0.set_inflow_boundary(
                0, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,0));
        euler_operator_Q0.set_supersonic_outflow_boundary(1);
        euler_operator_Q0.set_inflow_boundary(
                2, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,1));
        euler_operator_Q0.set_inflow_boundary(
                3, std::make_unique<EquationData::BoundaryData<dim>>(0,parameters,0));
        euler_operator_Q0.set_wall_boundary(4);

        if (dim == 3){
            euler_operator_Q0.set_body_force(
                    std::make_unique<Functions::ConstantFunction<dim>>(
                            std::vector<double>({0., 0., -0.2})));
            euler_operator_Q0.set_body_force(
                    std::make_unique<Functions::ConstantFunction<dim>>(
                            std::vector<double>({0., 0., -0.2})));
        }

        break;

    }
    case 7:  //2D Riemann problem
    {
        GridGenerator::subdivided_hyper_cube(triangulation,120,0.,1.);

        for(auto &face : triangulation.active_face_iterators())
        {
            if(face->at_boundary())
            {
                //left-up boundary has ID = 0
                if(face->center()[0] <=0.5 && face->center()[1] >0.5 )
                   face->set_boundary_id(0);

                //right-up boundary has ID = 1
                if(face->center()[0] > 0.5 && face->center()[0] > 0.5)
                   face->set_boundary_id(1);

                //bottom-left boundary has ID = 2
                if(face->center()[0] <= 0.5 && face->center()[1]<=0.5)
                   face->set_boundary_id(2);

                //bottom-right boundary has ID = 3
                if(face->center()[0] >0.5 && face->center()[1]<=0.5)
                   face->set_boundary_id(3);

            }
        }
        euler_operator.set_wall_boundary(0);
        euler_operator.set_wall_boundary(1);
        euler_operator.set_wall_boundary(2);
        euler_operator.set_wall_boundary(3);
        euler_operator_Q0.set_wall_boundary(0);
        euler_operator_Q0.set_wall_boundary(1);
        euler_operator_Q0.set_wall_boundary(2);
        euler_operator_Q0.set_wall_boundary(3);

        if (dim == 3){
            euler_operator_Q0.set_body_force(
                    std::make_unique<Functions::ConstantFunction<dim>>(
                            std::vector<double>({0., 0., -0.2})));
            euler_operator_Q0.set_body_force(
                    std::make_unique<Functions::ConstantFunction<dim>>(
                            std::vector<double>({0., 0., -0.2})));
        }

        break;
    }
        default:

        Assert(false, ExcNotImplemented());
    }

}

/*!
*  \brief EulerProblem::make_dofs
* We call `make_dofs` every time we compute a refine of the mesh.
* With respect to quadrature, we want to select two different
* ways of computing the underlying integrals: The first is a flexible one,
* based on a template parameter `n_points_1d`. More accurate
* integration is necessary to avoid the aliasing problem due to the
* variable coefficients in the Euler operator. The second less accurate
* quadrature formula is a tight one based on `fe_degree+1` and needed for
* the inverse mass matrix. While that formula provides an exact inverse
* only on affine element shapes and not on deformed elements, it enables
* the fast inversion of the mass matrix by tensor product techniques,
* necessary to ensure optimal computational efficiency overall.
*
* For each cell, find neighbourig cell. This is needed for limiter*/
template<int dim>
void EulerProblem<dim>::make_dofs()
{

    dof_handler.distribute_dofs(fe);
    dof_handler_Q0.distribute_dofs(fe_Q0);

    dof_handlers.push_back(&dof_handler);
    dof_handlers_Q0.push_back(&dof_handler_Q0);

    quadratures.push_back(QGauss<1>(parameters.n_q_points_1d));
    quadratures_Q0.push_back(QGauss<1>(parameters.n_q_points_1d_Q0));
    quadratures.push_back(QGauss<1>(parameters.fe_degree+1));
    quadratures_Q0.push_back(QGauss<1>(parameters.fe_degree_Q0+1));

    euler_operator.reinit(mapping, dof_handlers,quadratures);
    euler_operator.initialize_vector(solution);

    euler_operator_Q0.reinit(mapping_Q0, dof_handlers_Q0,quadratures_Q0);
    euler_operator_Q0.initialize_vector(solution_Q0);

    tmp_solution.reinit(solution);
    tmp_solution_Q0.reinit(solution_Q0);
    sol_aux.reinit(solution);
    
    cell_average.resize(triangulation.n_active_cells(),
                        Vector<double>(dim+2));


    unsigned int index=0;
    for (typename Triangulation<dim>::active_cell_iterator cell=triangulation.begin_active();
        cell!=triangulation.end(); ++cell, ++index)
             cell->set_user_index(index);

    dh_cell.clear();
    dh_cell.distribute_dofs (fe_cell);
    shock_indicator.reinit (dh_cell.n_dofs());
    jump_indicator.reinit (dh_cell.n_dofs());

    cell_degree.resize(triangulation.n_active_cells());
    re_update.resize(triangulation.n_active_cells());
    for(unsigned int i=0; i<triangulation.n_active_cells(); ++i)
    {
        cell_degree[i] = fe.degree;
        re_update[i] = true;
    }

    lcell.resize(triangulation.n_cells());
    rcell.resize(triangulation.n_cells());
    bcell.resize(triangulation.n_cells());
    tcell.resize(triangulation.n_cells());

    typename DoFHandler<dim>::active_cell_iterator
        cell = dh_cell.begin_active(),
        endc = dh_cell.end();
    for (; cell!=endc; ++cell)
    {
        unsigned int c = cell_number(cell);
        lcell[c] = endc;
        rcell[c] = endc;
        bcell[c] = endc;
        tcell[c] = endc;
        double dx = cell->diameter() / std::sqrt(1.0*dim);

        for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
         if (! cell->at_boundary(face_no))
         {
            const typename DoFHandler<dim>::cell_iterator
               neighbor = cell->neighbor(face_no);
            Assert(neighbor->level() == cell->level() || neighbor->level() == cell->level()-1,
                   ExcInternalError());
            Tensor<1,dim> dr = neighbor->center() - cell->center();
            if(dr[0] < -0.5*dx)
               lcell[c] = neighbor;
            else if(dr[0] > 0.5*dx)
               rcell[c] = neighbor;
            else if(dr[1] < -0.5*dx)
               bcell[c] = neighbor;
            else if(dr[1] > 0.5*dx)
               tcell[c] = neighbor;

         }
   }
  /*  double length = 3.2;
    double height = 1.0;
    double wall_position = 1./6.;
    for (typename Triangulation<dim>::active_cell_iterator cell=triangulation.begin_active();
         cell!=triangulation.end(); ++cell)
    {
        for(unsigned int j=0; j< GeometryInfo<dim>::faces_per_cell; ++j )
        {
         if(cell->face(j)->at_boundary())
         {
            if(cell->face(j)->center()[0] <1.0e-6 )
               cell->face(j)->set_boundary_id(0);

            if(cell->face(j)->center()[0] >length - 1.e-6)
               cell->face(j)->set_boundary_id(1);

            if(std::fabs(cell->face(j)->center()[1] - height)<1e-6)
               cell->face(j)->set_boundary_id(2);

            //bottom boundary x<=1/6 has ID = 3
            if(cell->face(j)->center()[0] <= wall_position && cell->face(j)->center()[1] <1.e-6)
               cell->face(j)->set_boundary_id(3);

            //bottom boundary x>=1/6 has ID = 4
            if(cell->face(j)->center()[0] > wall_position && cell->face(j)->center()[1]<1.e-6)
               cell->face(j)->set_boundary_id(4);


    }
        }
    }

    euler_operator.set_inflow_boundary(
        0, std::make_unique<EquationData::BoundaryData<dim>>(time,parameters,0));
    euler_operator.set_inflow_boundary(
        2, std::make_unique<EquationData::BoundaryData<dim>>(time,parameters,1));
    euler_operator.set_supersonic_outflow_boundary(1);
    euler_operator.set_inflow_boundary(
        3, std::make_unique<EquationData::BoundaryData<dim>>(time,parameters,0));
    euler_operator.set_wall_boundary(4);

    euler_operator_Q0.set_inflow_boundary(
        0, std::make_unique<EquationData::BoundaryData<dim>>(time,parameters,0));
    euler_operator_Q0.set_supersonic_outflow_boundary(1);
    euler_operator_Q0.set_inflow_boundary(
        2, std::make_unique<EquationData::BoundaryData<dim>>(time,parameters,1));
    euler_operator_Q0.set_inflow_boundary(
        3, std::make_unique<EquationData::BoundaryData<dim>>(time,parameters,0));
    euler_operator_Q0.set_wall_boundary(4);*/
}


/// @sect14{Compute refinement indicator and refine grid}
  /*!
 * \brief EulerProblem::adapt_mesh
 * This function take care of the adaptive mesh refinement.
 * the three tasks this function performs is to first find out wich cells to
 * refine, then to actually do the refinement and eventually transfer the solution
 * vectors between the two different grids. The first task is simply
 * achieved by computing the refinements indicator, as the gradient of the density
 * $\eta_K = \log\left(1+|\nabla\rho(x_K)|\right)$ where $x_K$ is the center of the cell or
 * as the gradient of the pressure $\eta = |\nabla p |$.
 * The second task is to loop over all cells and mark those that
 * we think should be refined. Then we need to transfer the various solution
 * vectors from the old to the new grid while we do refinement. The SolutionTransfer
 * class is our friend here.
 */
template<int dim>
void EulerProblem<dim>::adapt_mesh()
  {
    Vector<double> estimated_error_per_cell(triangulation.n_active_cells());
   if(parameters.refine_indicator == "gradient_density")
   {

        const QMidpoint<dim>  quadrature_formula;
        const UpdateFlags update_flags = dealii::update_gradients;
        FEValues<dim> fe_v (mapping, dof_handler.get_fe(),
                           quadrature_formula, update_flags);


        std::vector<std::vector<dealii::Tensor<1,dim> > >   dU(1, std::vector<dealii::Tensor<1,dim> >(dim+2));

        typename DoFHandler<dim>::active_cell_iterator  cell1 = dof_handler.begin_active(), endc1 = dof_handler.end();
        for (unsigned int cell_no=0; cell1!=endc1; ++cell1, ++cell_no)
        {
         fe_v.reinit(cell1);
         fe_v.get_function_gradients(solution, dU);

         estimated_error_per_cell(cell_no) = std::log(1+std::sqrt(dU[0][0] * dU[0][0]));
        }
   }
   else if ( parameters.refine_indicator == "gradient_pressure")
   {
        Vector<float> estimated_error(triangulation.n_active_cells());
        QGauss<dim>   quadrature_formula(fe.degree+1);
        const unsigned int n_q_points = quadrature_formula.size();
        LinearAlgebra::distributed::Vector<Number> current_solution(solution);
        FEValues<dim> fe_values (mapping, fe,
                                quadrature_formula, update_values | update_JxW_values);
        std::vector<double> density_values(n_q_points), energy_values(n_q_points);
        std::vector< Tensor<1,dim> > momentum_values(n_q_points);

        const unsigned int density_component = 0;
        const unsigned int energy_component = dim+1;
        const unsigned int momentum_component = 1;
        const FEValuesExtractors::Scalar density  (density_component);
        const FEValuesExtractors::Scalar energy   (energy_component);
        const FEValuesExtractors::Vector momentum (momentum_component);
        for(const auto& cell: dof_handler.active_cell_iterators())
        {

         fe_values.reinit (cell);

         fe_values[density].get_function_values(solution, density_values);
         fe_values[momentum].get_function_values(solution, momentum_values);
         fe_values[energy].get_function_values(solution, energy_values);

         std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
         cell->get_dof_indices(dof_indices);

         for(unsigned int idx = 0; idx < dof_indices.size(); ++idx) {
            unsigned int comp_i = fe.system_to_component_index(idx).first;
            if(comp_i == density_component){
               current_solution(dof_indices[idx]) =(1.4-1.0)*(energy_values[0] -
                                                                   0.5*momentum_values[0].norm_square()/density_values[0]);

            }
            if(comp_i == momentum_component){
               current_solution(dof_indices[idx]) = solution(dof_indices[idx]);

            }
           if(comp_i == energy_component){
               current_solution(dof_indices[idx]) = solution(dof_indices[idx]);

            }

         }

        }
        DerivativeApproximation::approximate_gradient(mapping,
                                                      dof_handler,
                                                      current_solution,
                                                      estimated_error);

        for(const auto &cell : triangulation.active_cell_iterators())
        {
         estimated_error_per_cell[cell->active_cell_index()] = estimated_error[cell->active_cell_index()];
        }

   }

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
   if(parameters.testcase==3){

        const double radius = 0.05;
        const Point<dim> corner(0.6, 0.2);
        for (const auto &cell : triangulation.active_cell_iterators())
        {
         cell->clear_coarsen_flag();
         cell->clear_refine_flag();
         Tensor<1,dim> dr = cell->center() - corner;
         if(dr.norm() < radius)
            cell->set_refine_flag();
        }
   }
    for (unsigned int cell_no=0; cell!=endc; ++cell, ++cell_no)
    {
       cell->clear_coarsen_flag();
       cell->clear_refine_flag();

       if ((cell->level() < parameters.max_loc_refinements) &&
           (std::fabs(estimated_error_per_cell(cell_no)) > parameters.min_loc_refinements))
         cell->set_refine_flag();
       else
         if ((cell->level() > 0) &&
             (std::fabs(estimated_error_per_cell(cell_no)) < 0.75*parameters.min_loc_refinements))
            cell->set_coarsen_flag();
    }


#ifdef DEAL_II_WITH_P4EST
  paralel::distriuted::SolutionTransfer<dim,LinearAlgebra::distributed::Vector<Number>> sol_trans(*reinterpret_cast<const DofHandler<dim>*>(&dof_handler));
  triangulation.prepare_coarsening_and_refinement();
  sol_trans.prepare_for_coarsening_and_refinement(solution);
  triangulation.execute_coarsening_and_refinement();
  make_dofs();
  sol_trans.interpolate(solution);

#else
    SolutionTransfer<dim,LinearAlgebra::distributed::Vector<Number>> sol_trans(dof_handler);
    SolutionTransfer<dim,LinearAlgebra::distributed::Vector<Number>> sol_trans_Q0(dof_handler_Q0);

    triangulation.prepare_coarsening_and_refinement();

    LinearAlgebra::distributed::Vector<Number> xsol(solution);
    LinearAlgebra::distributed::Vector<Number> xsol_Q0(solution_Q0);

    sol_trans.prepare_for_coarsening_and_refinement(xsol);
    sol_trans_Q0.prepare_for_coarsening_and_refinement(xsol_Q0);


    triangulation.execute_coarsening_and_refinement();
    make_dofs();
    sol_trans.interpolate(xsol,solution);
    sol_trans_Q0.interpolate(xsol_Q0,solution_Q0);

#endif

}

/// @sect41{Limiters and related functions}



/// Compute cell average solution
template <int dim>
void EulerProblem<dim>::compute_cell_average (LinearAlgebra::distributed::Vector<double> & current_solution)
{
  QGauss<dim>   quadrature_formula(fe.degree+1);
  const unsigned int n_q_points = quadrature_formula.size();

  FEValues<dim> fe_values (mapping, fe,
                           quadrature_formula,
                           update_values | update_JxW_values);
  std::vector<Vector<double> > solution_values(n_q_points,
                                               Vector<double>(dim+2));

  typename DoFHandler<dim>::active_cell_iterator
     cell = dof_handler.begin_active(),
     endc = dof_handler.end();

  for (; cell!=endc; ++cell)
  {
     unsigned int cell_no = cell_number(cell);
     if(re_update[cell_no])
     {
        fe_values.reinit (cell);
        fe_values.get_function_values (current_solution, solution_values);

        cell_average[cell_no] = 0.0;

        for (unsigned int q=0; q<n_q_points; ++q)
           for(unsigned int c=0; c<dim+2; ++c)
              cell_average[cell_no][c] += solution_values[q][c] * fe_values.JxW(q);

        cell_average[cell_no] /= cell->measure();
     }
  }

}

/// if cell is activem return cell average.
/// if cell is not active, return area average of child cells.
template<int dim>
void EulerProblem<dim>::get_cell_average(const typename dealii::DoFHandler<dim>::cell_iterator& cell,
                       dealii::Vector<double>& avg)
 {

    if(cell->active())
    {
       unsigned int cell_no = cell_number(cell);
       for(unsigned int c=0; c<dim+2; ++c)
          avg(c) = cell_average[cell_no][c];
    }
    else
    {  // compute average solution on child cells
       auto child_cells =
          dealii::GridTools::get_active_child_cells< dealii::DoFHandler<dim> > (cell);
       avg = 0;
       double measure = 0;
       for(unsigned int i=0; i<child_cells.size(); ++i)
       {
          unsigned int child_cell_no = cell_number(child_cells[i]);
          for(unsigned int c=0; c<dim+2; ++c)
             avg(c) += cell_average[child_cell_no][c] * child_cells[i]->measure();
          measure += child_cells[i]->measure();
       }
       avg /= measure;
    }
 }


/// Compute shock indicator - KXRCF indicator
template <int dim>
void EulerProblem<dim>::compute_shock_indicator(LinearAlgebra::distributed::Vector<double> & current_solution)
{

    const unsigned int density_component = 0;
       const unsigned int energy_component = dim+1;

       QGauss<dim-1> quadrature(fe.degree + 1);
       FEFaceValues<dim> fe_face_values (mapping, fe, quadrature,
                                         update_values | update_normal_vectors);
       FEFaceValues<dim> fe_face_values_nbr (mapping, fe, quadrature,
                                             update_values);
       FESubfaceValues<dim> fe_subface_values (mapping, fe, quadrature,
                                               update_values | update_normal_vectors);
       FESubfaceValues<dim> fe_subface_values_nbr (mapping, fe, quadrature,
                                                   update_values);

       unsigned int n_q_points = quadrature.size();
       std::vector<double> face_values(n_q_points), face_values_nbr(n_q_points);

       // select indicator variable
       unsigned int component;

             component = density_component;

           //  component = energy_component;

       const FEValuesExtractors::Scalar variable (component);

       typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();

       double jump_ind_min = 1.0e20;
       double jump_ind_max = 0.0;
       double jump_ind_avg = 0.0;

       for(; cell != endc; ++cell)
       {
          unsigned int c = cell_number(cell);
         // shock_indicator[c] = 1e20;
          double& cell_shock_ind = shock_indicator (c);
          double& cell_jump_ind = jump_indicator (c);

          cell_shock_ind = 0;
          cell_jump_ind = 0;
          double inflow_measure = 0;

          // velocity based on cell average. we use this to determine inflow/outflow
          // parts of the cell boundary.
          Point<dim> vel;
          for(unsigned int i=0; i<dim; ++i)
             vel(i) = cell_average[c][i+1] / cell_average[c][density_component];

          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
             if (cell->at_boundary(f) == false)
             {
                if ((cell->neighbor(f)->level() == cell->level()) &&
                    (cell->neighbor(f)->has_children() == false))
                {
                   fe_face_values.reinit(cell, f);
                   fe_face_values_nbr.reinit(cell->neighbor(f), cell->neighbor_of_neighbor(f));
                   fe_face_values[variable].get_function_values(current_solution, face_values);
                   fe_face_values_nbr[variable].get_function_values(current_solution, face_values_nbr);
                   for(unsigned int q=0; q<n_q_points; ++q)
                   {
                      int inflow_status = (vel * fe_face_values.normal_vector(q) < 0);
                      cell_shock_ind += inflow_status *
                                        (face_values[q] - face_values_nbr[q]) *
                                        fe_face_values.JxW(q);
                      cell_jump_ind += std::pow(face_values[q] - face_values_nbr[q], 2) *
                                       fe_face_values.JxW(q);
                      inflow_measure += inflow_status * fe_face_values.JxW(q);
                   }

                }
                else if ((cell->neighbor(f)->level() == cell->level()) &&
                         (cell->neighbor(f)->has_children() == true))
                {
                   for (unsigned int subface=0; subface<cell->face(f)->n_children(); ++subface)
                   {
                      fe_subface_values.reinit (cell, f, subface);
                      fe_face_values_nbr.reinit (cell->neighbor_child_on_subface (f, subface),
                                                 cell->neighbor_of_neighbor(f));
                      fe_subface_values[variable].get_function_values(current_solution, face_values);
                      fe_face_values_nbr[variable].get_function_values(current_solution, face_values_nbr);
                      for(unsigned int q=0; q<n_q_points; ++q)
                      {
                         int inflow_status = (vel * fe_subface_values.normal_vector(q) < 0);
                         cell_shock_ind += inflow_status *
                                           (face_values[q] - face_values_nbr[q]) *
                                           fe_subface_values.JxW(q);
                         cell_jump_ind += std::pow(face_values[q] - face_values_nbr[q], 2) *
                                          fe_face_values.JxW(q);
                         inflow_measure += inflow_status * fe_subface_values.JxW(q);
                      }
                   }
                }
                else if (cell->neighbor_is_coarser(f))
                {
                   fe_face_values.reinit(cell, f);
                   fe_subface_values_nbr.reinit (cell->neighbor(f),
                                                 cell->neighbor_of_coarser_neighbor(f).first,
                                                 cell->neighbor_of_coarser_neighbor(f).second);
                   fe_face_values[variable].get_function_values(current_solution, face_values);
                   fe_subface_values_nbr[variable].get_function_values(current_solution, face_values_nbr);
                   for(unsigned int q=0; q<n_q_points; ++q)
                   {
                      int inflow_status = (vel * fe_face_values.normal_vector(q) < 0);
                      cell_shock_ind += inflow_status *
                                        (face_values[q] - face_values_nbr[q]) *
                                        fe_face_values.JxW(q);
                      cell_jump_ind += std::pow(face_values[q] - face_values_nbr[q], 2) *
                                       fe_face_values.JxW(q);
                      inflow_measure += inflow_status * fe_face_values.JxW(q);
                   }
                }
             }
             else
             {
                // Boundary face
                // We dont do anything here since we assume solution is constant near
                // boundary.
             }

          // normalized shock indicator
          double cell_norm = cell_average[c][component];
          double denominator = std::pow(cell->diameter(), 0.5*(fe.degree+1)) *
                               inflow_measure *
                               cell_norm;
          if (denominator > 1.0e-12)
          {
               cell_shock_ind = std::fabs(cell_shock_ind) / denominator;
               cell_shock_ind = (cell_shock_ind > 1.0) ? 1.0 : 0.0;
            }
          else
            cell_shock_ind = 0;

          double dx = cell->diameter() / std::sqrt(1.0*dim);

          cell_jump_ind = std::sqrt( cell_jump_ind / (4.0*dx) ) * cell->diameter();
          jump_ind_min = std::min(jump_ind_min, cell_jump_ind);
          jump_ind_max = std::max(jump_ind_max, cell_jump_ind);
          jump_ind_avg += cell_jump_ind;
       }

      jump_ind_avg /= triangulation.n_active_cells();



}

/// TVB version of minmod limiter. If Mdx2=0 then it is TVD limiter.
double minmod (const double& a,
             const double& b,
             const double& c,
             const double& Mdx2)
{
 double aa = std::fabs(a);
 if(aa < Mdx2) return a;

 if(a*b > 0 && b*c > 0)
 {
    double s = (a > 0) ? 1.0 : -1.0;
    return s * std::min(aa, std::min(std::fabs(b), std::fabs(c)));
 }
 else
    return 0;
}


/// TVB version of van Albada limiter.
double vanAlbada (const double& c,
              const double& a,
              const double &b,
                 const double& Mdx2,
              const double& eps)
{
 double cc = std::fabs(c);
 if(cc < Mdx2) return c;

 double inv = 1./((a*a)+(b*b)+2*(eps*eps));
 return (((a*a + eps*eps)*b) + ((b*b + eps*eps)*a))*inv;
}

/// Apply the TVB limiter using or minmod function or van Albada function
template <int dim>
LinearAlgebra::distributed::Vector<Number> EulerProblem<dim>::apply_limiter_TVB(LinearAlgebra::distributed::Vector<Number> & current_solution)
{

 const unsigned int n_components = dim+2;

 QGauss<dim> qrule (fe.degree + 1);
 FEValues<dim> fe_values_grad (mapping, fe, qrule, update_gradients | update_JxW_values);

 // NOTE: We get multiple sets of same support points since fe is an FESystem
 Quadrature<dim> qsupport (fe.get_unit_support_points());
 FEValues<dim>   fe_values (mapping, fe, qsupport, update_quadrature_points);

 Vector<double> dfx (n_components);
 Vector<double> dbx (n_components);
 Vector<double> Dx  (n_components);

 Vector<double> dfy (n_components);
 Vector<double> dby (n_components);
 Vector<double> Dy  (n_components);

 Vector<double> Dx_new (n_components);
 Vector<double> Dy_new (n_components);
 Vector<double> avg_nbr (n_components);

 std::vector<unsigned int> dof_indices (fe.dofs_per_cell);
 std::vector< std::vector< Tensor<1,dim> > > grad (qrule.size(),
                                                   std::vector< Tensor<1,dim> >(n_components));

 typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end(),
    endc0 = dh_cell.end();

 const double beta = parameters.beta;

 for(; cell != endc; ++cell)
 {
    const unsigned int c = cell_number(cell);
    if(shock_indicator[c] > 1.0)
    {
       const double dx = cell->diameter() / std::sqrt(1.0*dim);
       const double Mdx2 = parameters.M * dx * dx;

       // Compute average gradient in cell
       fe_values_grad.reinit(cell);
       fe_values_grad.get_function_gradients(current_solution, grad);
       Tensor<1,dim> avg_grad;

       for(unsigned int i=0; i<n_components; ++i)
       {
          avg_grad = 0;
          for(unsigned int q=0; q<qrule.size(); ++q)
             avg_grad += grad[q][i] * fe_values_grad.JxW(q);
          avg_grad /= cell->measure();
          Dx(i) = dx * avg_grad[0];
          Dy(i) = dx * avg_grad[1];
       }

       // Backward difference of cell averages
       dbx = Dx;
       if(lcell[c] != endc0)
       {
          get_cell_average (lcell[c], avg_nbr);
          for(unsigned int i=0; i<n_components; ++i)
             dbx(i) = cell_average[c][i] - avg_nbr(i);
       }

       // Forward difference of cell averages
       dfx = Dx;
       if(rcell[c] != endc0)
       {
          get_cell_average (rcell[c], avg_nbr);
          for(unsigned int i=0; i<n_components; ++i)
             dfx(i) = avg_nbr(i) - cell_average[c][i];
       }

       // Backward difference of cell averages
       dby = Dy;
       if(bcell[c] != endc0)
       {
          get_cell_average (bcell[c], avg_nbr);
          for(unsigned int i=0; i<n_components; ++i)
             dby(i) = cell_average[c][i] - avg_nbr(i);
       }

       // Forward difference of cell averages
       dfy = Dy;
       if(tcell[c] != endc0)
       {
          get_cell_average (tcell[c], avg_nbr);
          for(unsigned int i=0; i<n_components; ++i)
             dfy(i) = avg_nbr(i) - cell_average[c][i];
       }

       // Apply minmod or van Albada limiter
       double change_x = 0;
       double change_y = 0;
       for(unsigned int i=0; i<n_components; ++i)
       {
          if (parameters.function_limiter == "minmod")
          {
          Dx_new(i) = minmod(Dx(i), beta*dbx(i), beta*dfx(i), Mdx2);
          Dy_new(i) = minmod(Dy(i), beta*dby(i), beta*dfy(i), Mdx2);
          }
          else if(parameters.function_limiter == "vanAlbada")
          {
          const double eps = 1e-8;
          Dx_new(i) = vanAlbada(Dx(i),dbx(i), dfx(i), Mdx2, eps);
          Dy_new(i) = vanAlbada(Dy(i),dby(i), dfy(i), Mdx2, eps);
          }
          change_x += std::fabs(Dx_new(i) - Dx(i));
          change_y += std::fabs(Dy_new(i) - Dy(i));
       }
       change_x /= n_components;
       change_y /= n_components;

       // If limiter is active, reduce polynomial to linear
       if(change_x + change_y > 1.0e-10)
       {
          Dx_new /= dx;
          Dy_new /= dx;

          cell->get_dof_indices(dof_indices);
          fe_values.reinit (cell);
          const std::vector<Point<dim> >& p = fe_values.get_quadrature_points();
          for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
          {
             unsigned int comp_i = fe.system_to_component_index(i).first;
             Tensor<1,dim> dr = p[i] - cell->center();
             current_solution(dof_indices[i]) = cell_average[c][comp_i]
                                                + dr[0] * Dx_new(comp_i)
                                                + dr[1] * Dy_new(comp_i);
          }
       }
    }
 }
 return current_solution;
}

/// Apply posititivy preserving limiter
template <int dim>
LinearAlgebra::distributed::Vector<Number> EulerProblem<dim>::apply_positivity_limiter ( LinearAlgebra::distributed::Vector<Number> & current_solution)
{

 const double gas_gamma = parameters.gamma;
 const unsigned int density_component = 0;
 const unsigned int energy_component = dim+1;

 // Find mininimum density and pressure in the whole grid
 const double eps = 1.0e-13;
 {
    for (unsigned int c=0; c<triangulation.n_active_cells(); ++c)
    {
       double eps1 = cell_average[c][density_component];
       double pressure = EulerEquations<dim>::template compute_pressure<double> (cell_average[c]);
       eps1 = std::min(eps1, pressure);
       if(eps1 < eps)
       {
          //std::cout << "\n Negative state at position " << cell0->center() << "\n\n";
          AssertThrow(false, ExcMessage("Fatal: Negative states"));
       }
    }
 }

 // Need 2N - 3 >= degree for the quadrature to be exact.
 // Choose same order as used for assembly process.
 unsigned int N = (fe.degree+3)%2==0 ? (fe.degree+3)/2 : (fe.degree+4)/2;
 Quadrature<dim> quadrature_x (QGaussLobatto<1>(N), QGauss<1>(fe.degree+1));
 Quadrature<dim> quadrature_y (QGauss<1>(fe.degree+1), QGaussLobatto<1>(N));
 FEValues<dim> fe_values_x (mapping, fe, quadrature_x, update_values);
 FEValues<dim> fe_values_y (mapping, fe, quadrature_y, update_values);

 const unsigned int n_q_points = quadrature_x.size();
 std::vector<double> density_values(n_q_points), energy_values(n_q_points);
 std::vector< Tensor<1,dim> > momentum_values(n_q_points);
 std::vector<unsigned int> local_dof_indices (fe.dofs_per_cell);

 const FEValuesExtractors::Scalar density  (density_component);
 const FEValuesExtractors::Scalar energy   (energy_component);
 const FEValuesExtractors::Vector momentum (1);

 typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

 for(; cell != endc; ++cell)
 {
    unsigned int c = cell_number(cell);
    fe_values_x.reinit(cell);
    fe_values_y.reinit(cell);

    // First limit density
    // find minimum density at GLL points
    double rho_min = 1.0e20;

    fe_values_x[density].get_function_values(current_solution, density_values);
    for(unsigned int q=0; q<n_q_points; ++q)
       rho_min = std::min(rho_min, density_values[q]);

    fe_values_y[density].get_function_values(current_solution, density_values);
    for(unsigned int q=0; q<n_q_points; ++q)
       rho_min = std::min(rho_min, density_values[q]);

    double density_average = cell_average[c][density_component];
    double rat = std::fabs(density_average - eps) /
                 (std::fabs(density_average - rho_min) + 1.0e-13);
    double theta1 = std::min(rat, 1.0);

    if(theta1 < 1.0)
    {
       cell->get_dof_indices (local_dof_indices);

       for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
       {
          unsigned int comp_i = fe.system_to_component_index(i).first;
          if(comp_i == density_component)
             current_solution(local_dof_indices[i]) =
                theta1           * current_solution(local_dof_indices[i])
                + (1.0 - theta1) * density_average;
       }
    }

    // now limit pressure
    double energy_average = cell_average[c][energy_component];
    Tensor<1,dim> momentum_average;
    for(unsigned int i=0; i<dim; ++i)
       momentum_average[i] = cell_average[c][i+1];

    double theta2 = 1.0;
    for(int d=0; d<dim; ++d)
    {
       if(d==0)
       {
          fe_values_x[density].get_function_values(current_solution, density_values);
          fe_values_x[momentum].get_function_values(current_solution, momentum_values);
          fe_values_x[energy].get_function_values(current_solution, energy_values);
       }
       else
       {
          fe_values_y[density].get_function_values(current_solution, density_values);
          fe_values_y[momentum].get_function_values(current_solution, momentum_values);
          fe_values_y[energy].get_function_values(current_solution, energy_values);
       }

       for(unsigned int q=0; q<n_q_points; ++q)
       {
          double pressure = (gas_gamma-1.0)*(energy_values[q] -
                                             0.5*momentum_values[q].norm_square()/density_values[q]);
          if(pressure < eps)
          {
             double drho = density_values[q] - density_average;
             Tensor<1,dim> dm = momentum_values[q] - momentum_average;
             double dE = energy_values[q] - energy_average;
             double a1 = 2.0*drho*dE - dm*dm;
             double b1 = 2.0*drho*(energy_average - eps/(gas_gamma-1.0))
                       + 2.0*density_average*dE
                       - 2.0*momentum_average*dm;
             double c1 = 2.0*density_average*energy_average
                       - momentum_average*momentum_average
                       - 2.0*eps*density_average/(gas_gamma-1.0);
             // Divide by a1 to avoid round-off error
             b1 /= a1; c1 /= a1;
             double D = std::sqrt( std::fabs(b1*b1 - 4.0*c1) );
             double t1 = 0.5*(-b1 - D);
             double t2 = 0.5*(-b1 + D);
             double t;
             if(t1 > -1.0e-12 && t1 < 1.0 + 1.0e-12)
                t = t1;
             else if(t2 > -1.0e-12 && t2 < 1.0 + 1.0e-12)
                t = t2;
             else
             {
                std::cout << "Problem in positivity limiter\n";
                std::cout << "\t a1, b1, c1 = " << a1 << " " << b1 << " " << c1 << "\n";
                std::cout << "\t t1, t2 = " << t1 << " " << t2 << "\n";
                std::cout << "\t eps, rho_min = " << eps << " " << rho_min << "\n";
                std::cout << "\t theta1 = " << theta1 << "\n";
                std::cout << "\t pressure = " << pressure << "\n";
                exit(0);
             }
             // t should strictly lie in [0,1]
             t = std::min(1.0, t);
             t = std::max(0.0, t);
             // Need t < 1.0. If t==1 upto machine precision
             // then we are suffering from round off error.
             // In this case we take the cell average value, t=0.
             if(std::fabs(1.0-t) < 1.0e-14) t = 0.0;
             theta2 = std::min(theta2, t);
          }
       }
    }

    if(theta2 < 1.0)
    {
       if(!(theta1<1.0)) // local_dof_indices has not been computed before
          cell->get_dof_indices (local_dof_indices);

       for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
       {
          unsigned int comp_i = fe.system_to_component_index(i).first;
          current_solution(local_dof_indices[i]) =
             theta2            * current_solution(local_dof_indices[i])
             + (1.0 - theta2)  * cell_average[c][comp_i];
       }
    }
 }
 return current_solution;
}

/// Apply filtering monotonization approach
template< int dim>
LinearAlgebra::distributed::Vector<Number> EulerProblem<dim>::apply_filter(LinearAlgebra::distributed::Vector<Number> &    sol_H,
                                                                     LinearAlgebra::distributed::Vector<Number> &    sol_M)
{


    QGauss<dim>   quadrature_formula(fe.degree+1);

     FEValues<dim> fe_values (mapping, fe,
                             quadrature_formula,
                             update_values);

    const unsigned int n_q_points = quadrature_formula.size();
    std::vector<double> density_values(n_q_points), energy_values(n_q_points);
    std::vector< Tensor<1,dim> > momentum_values(n_q_points);

    const unsigned int density_component = 0;
    const unsigned int energy_component = dim+1;
    const unsigned int momentum_component = 1;
    const FEValuesExtractors::Scalar density  (density_component);
    const FEValuesExtractors::Scalar energy   (energy_component);
    const FEValuesExtractors::Vector momentum (momentum_component);


    FETools::interpolate(dof_handler_Q0, sol_M, dof_handler, sol_aux);


   for(const auto& cell: dof_handler.active_cell_iterators()) {

       if(cell->is_locally_owned()) {

           fe_values.reinit (cell);

           fe_values[density].get_function_values(sol_H, density_values);
           fe_values[momentum].get_function_values(sol_H, momentum_values);
           fe_values[energy].get_function_values(sol_H, energy_values);

           std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
           cell->get_dof_indices(dof_indices);
           for(unsigned int idx = 0; idx < dof_indices.size(); ++idx) {
           unsigned int comp_i = fe.system_to_component_index(idx).first;
           if(comp_i == density_component){
               sol_H(dof_indices[idx]) -= sol_aux(dof_indices[idx]);
               sol_H(dof_indices[idx]) /= (parameters.beta_density*sol_aux(dof_indices[idx]));
               sol_H(dof_indices[idx])  = sgn(sol_H(dof_indices[idx]))*
                                             std::max(1.0 - std::abs(std::abs(sol_H(dof_indices[idx])) - 1.0), 0.0);
               sol_H(dof_indices[idx]) *= (parameters.beta_density*sol_aux(dof_indices[idx]));
               sol_H(dof_indices[idx]) += sol_aux(dof_indices[idx]);
           }
           //unsigned int comp_i = fe.system_to_component_index(idx).first;
         else  if(comp_i == momentum_component){
                   sol_H(dof_indices[idx]) -= sol_aux(dof_indices[idx]);
                   sol_H(dof_indices[idx]) /= (parameters.beta_momentum*sol_aux(dof_indices[idx])+ 1e-15);
                   sol_H(dof_indices[idx])  = sgn(sol_H(dof_indices[idx]))*
                                                 std::max(1.0 - std::abs(std::abs(sol_H(dof_indices[idx])) - 1.0), 0.0);
                   sol_H(dof_indices[idx]) *= (parameters.beta_momentum*sol_aux(dof_indices[idx])+ 1e-15);
                   sol_H(dof_indices[idx]) += sol_aux(dof_indices[idx]);
               }
          // unsigned int comp_i = fe.system_to_component_index(idx).first;
         else  if(comp_i == energy_component){
               sol_H(dof_indices[idx]) -= sol_aux(dof_indices[idx]);
               sol_H(dof_indices[idx]) /= (parameters.beta_energy*sol_aux(dof_indices[idx]));
               sol_H(dof_indices[idx])  = sgn(sol_H(dof_indices[idx]))*
                                             std::max(1.0 - std::abs(std::abs(sol_H(dof_indices[idx])) - 1.0), 0.0);
               sol_H(dof_indices[idx]) *= (parameters.beta_energy*sol_aux(dof_indices[idx]));
               sol_H(dof_indices[idx]) += sol_aux(dof_indices[idx]);
           }
       }

}
   }
   return sol_H;
}

/// Let us move to the function that does an entire stage of a Runge--Kutta
/// update. It calls EulerOperator::apply() followed by some updates
/// to the vectors. Rather than performing these
/// steps through the vector interfaces, we here present an alternative
/// strategy that is faster on cache-based architectures.
/// At each step, we apply the limiter in order to control the solution
template <int dim>
void EulerProblem<dim>::update(const double    current_time,
                              const double    time_step,
                               LinearAlgebra::distributed::Vector<Number> &    solution_np,
                              LinearAlgebra::distributed::Vector<Number> &    tmp_sol_n,
                               LinearAlgebra::distributed::Vector<Number> &    solution_np_Q0,
                               LinearAlgebra::distributed::Vector<Number> &    tmp_sol_n_Q0) {

    LinearAlgebra::distributed::Vector<Number> vec_tmp1, vec_tmp1_Q0;
    LinearAlgebra::distributed::Vector<Number> vec_tmp2, vec_tmp2_Q0;
    LinearAlgebra::distributed::Vector<Number> vec_tmp3, vec_tmp3_Q0;

    if (!vec_tmp1.partitioners_are_globally_compatible(*tmp_sol_n.get_partitioner())) {
        vec_tmp1.reinit(solution_np);
        vec_tmp2.reinit(solution_np);
        vec_tmp3.reinit(solution_np);
    }
    if (!vec_tmp1_Q0.partitioners_are_globally_compatible(*tmp_sol_n_Q0.get_partitioner())) {
        vec_tmp1_Q0.reinit(solution_np_Q0);
        vec_tmp2_Q0.reinit(solution_np_Q0);
        vec_tmp3_Q0.reinit(solution_np_Q0);
    }
   // reifacciomesh(current_time);

    //stage 1 DENSITY
    euler_operator_Q0.apply(current_time, tmp_solution_Q0, vec_tmp1_Q0);
    solution_np_Q0 = tmp_sol_n_Q0;
    solution_np_Q0.add(time_step/6.,vec_tmp1_Q0);
    vec_tmp2_Q0 = tmp_sol_n_Q0;
    vec_tmp2_Q0.add(time_step,vec_tmp1_Q0);
    euler_operator.apply(current_time, tmp_solution, vec_tmp1);
    solution_np = tmp_sol_n;
    solution_np.add(time_step/6.,vec_tmp1);
    vec_tmp2 = tmp_sol_n;
    vec_tmp2.add(time_step,vec_tmp1);
if(parameters.type == "TVB")
{
    compute_cell_average(vec_tmp2);
    compute_shock_indicator(vec_tmp2);
    vec_tmp2 = apply_limiter_TVB(vec_tmp2);

  if(parameters.positivity)
      vec_tmp2 = apply_positivity_limiter(vec_tmp2);
}else if (parameters.type == "filter")
{
     vec_tmp2 = apply_filter(vec_tmp2,vec_tmp2_Q0);
}

    //stage 2
    euler_operator_Q0.apply(current_time,vec_tmp2_Q0,vec_tmp3_Q0);
    solution_np_Q0.add(time_step/6.,vec_tmp3_Q0);
    vec_tmp2_Q0 = tmp_sol_n_Q0;
    vec_tmp2_Q0.add(time_step/4.,vec_tmp1_Q0);
    vec_tmp2_Q0.add(time_step/4.,vec_tmp3_Q0);
    euler_operator.apply(current_time,vec_tmp2,vec_tmp3);
    solution_np.add(time_step/6.,vec_tmp3);
    vec_tmp2 = tmp_sol_n;
    vec_tmp2.add(time_step/4.,vec_tmp1);
    vec_tmp2.add(time_step/4.,vec_tmp3);
if(parameters.type == "TVB")
{
  compute_cell_average(vec_tmp2);
   compute_shock_indicator(vec_tmp2);
   vec_tmp2 = apply_limiter_TVB(vec_tmp2);

    if(parameters.positivity)
        vec_tmp2 = apply_positivity_limiter(vec_tmp2);
}
else if (parameters.type == "filter")
{
    vec_tmp2 = apply_filter(vec_tmp2,vec_tmp2_Q0);
}

   //PASSO 3
    euler_operator_Q0.apply(current_time,vec_tmp2_Q0,vec_tmp1_Q0);
    solution_np_Q0.add(2.*time_step/3.,vec_tmp1_Q0);
    euler_operator.apply(current_time,vec_tmp2,vec_tmp1);
    solution_np.add(2.*time_step/3.,vec_tmp1);
    if(parameters.type == "TVB")
{
   compute_cell_average(solution_np);
   compute_shock_indicator(solution_np);

   solution_np = apply_limiter_TVB(solution_np);

   if(parameters.positivity)
        solution_np = apply_positivity_limiter(solution_np);
}
    else if (parameters.type == "filter")
{
     solution_np = apply_filter(solution_np,solution_np_Q0);
}

}



/*! We let the postprocessor defined above control most of the
* output, except for the primal field that we write directly. For the
* analytical solution test case, we also perform another projection of the
* analytical solution and print the difference between that field and the
* numerical solution. Once we have defined all quantities to be written, we
* build the patches for output. We create a
* high-order VTK output by setting the appropriate flag, which enables us
* to visualize fields of high polynomial degrees. Finally, we call the
* `DataOutInterface::write_vtu_in_parallel()` function to write the result
* to the given file name. This function uses special MPI parallel write
* facilities, which are typically more optimized for parallel file systems
* than the standard library's `std::ofstream` variants used in most other
* tutorial programs. A particularly nice feature of the
* `write_vtu_in_parallel()` function is the fact that it can combine output
* from all MPI ranks into a single file, making it unnecessary to have a
* central record of all such files (namely, the "pvtu" file).
*
* For parallel programs, it is often instructive to look at the partitioning
* of cells among processors. To this end, one can pass a vector of numbers
* to DataOut::add_data_vector() that contains as many entries as the
* current processor has active cells; these numbers should then be the
* rank of the processor that owns each of these cells. Such a vector
* could, for example, be obtained from
* GridTools::get_subdomain_association(). On the other hand, on each MPI
* process, DataOut will only read those entries that correspond to locally
* owned cells, and these of course all have the same value: namely, the rank
* of the current process. What is in the remaining entries of the vector
* doesn't actually matter, and so we can just get away with a cheap trick: We
* just fill *all* values of the vector we give to DataOut::add_data_vector()
* with the rank of the current MPI process. The key is that on each process,
* only the entries corresponding to the locally owned cells will be read,
* ignoring the (wrong) values in other entries. The fact that every process
* submits a vector in which the correct subset of entries is correct is all
* that is necessary.*/
template <int dim>
void EulerProblem<dim>::output_results(const unsigned int result_number)
{
    const std::array<double, 3> errors =
            euler_operator.compute_errors(EquationData::InitialData<dim>(parameters), solution);
    const std::string quantity_name = parameters.testcase == 0 ? "error" : "norm";
    const std::string filename_grid =
            "./results/grid_" + Utilities::int_to_string(result_number, 3) + ".svg";
    std::ofstream out(filename_grid);
    GridOut grid_out;
    grid_out.write_svg(triangulation,out);

    pcout << "Time:" << std::setw(8) << std::setprecision(3) << time
          << ", dt: " << std::setw(8) << std::setprecision(2) << time_step
          << ", " << quantity_name << " rho: " << std::setprecision(4)
          << std::setw(10) << errors[0] << ", rho * u: " << std::setprecision(4)
          << std::setw(10) << errors[1] << ", energy:" << std::setprecision(4)
          << std::setw(10) << errors[2] << std::endl;


        TimerOutput::Scope t(timer, "output");

        Postprocessor postprocessor(parameters,0);
        Postprocessor postprocessor_Q0(parameters,1);
        DataOut<dim>  data_out;

        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;
        data_out.set_flags(flags);

        data_out.attach_dof_handler(dof_handler);

            std::vector<std::string> names;
            names.emplace_back("density");
            for (unsigned int d = 0; d < dim; ++d)
                names.emplace_back("momentum");
            names.emplace_back("energy");

            std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
            interpretation.push_back(DataComponentInterpretation::component_is_scalar);
            for (unsigned int d = 0; d < dim; ++d)
                interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
            interpretation.push_back(DataComponentInterpretation::component_is_scalar);
            solution.update_ghost_values();
            data_out.add_data_vector(dof_handler, solution, names, interpretation);
        //  if(parameters.testcase == 4 && dim ==2)
          // {
             LinearAlgebra::distributed::Vector<Number> sol_exact(solution);
             euler_operator.project(EquationData::ExactSolution<dim>(time,parameters), sol_exact);

             std::vector<std::string> names_exact;
             names_exact.emplace_back("density_analytical");
             for (unsigned int d = 0; d < dim; ++d)
                 names_exact.emplace_back("velocity_analytical");
             names_exact.emplace_back("pressure_analytical");

             std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation_exact;
             interpretation_exact.push_back(DataComponentInterpretation::component_is_scalar);
             for (unsigned int d = 0; d < dim; ++d)
                 interpretation_exact.push_back(DataComponentInterpretation::component_is_part_of_vector);
             interpretation_exact.push_back(DataComponentInterpretation::component_is_scalar);
            sol_exact.update_ghost_values();
             data_out.add_data_vector(dof_handler, sol_exact, names_exact, interpretation_exact);
           // }

          //  data_out.build_patches(mapping,fe.degree,DataOut<dim>::curved_inner_cells);
            std::vector<std::string> names_Q0;
            names_Q0.emplace_back("density_Q0");
            for (unsigned int d = 0; d < dim; ++d)
                names_Q0.emplace_back("momentum_Q0");
            names_Q0.emplace_back("energy_Q0");

            std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation_Q0;
            interpretation_Q0.push_back(DataComponentInterpretation::component_is_scalar);
            for (unsigned int d = 0; d < dim; ++d)
                interpretation_Q0.push_back(DataComponentInterpretation::component_is_part_of_vector);
            interpretation_Q0.push_back(DataComponentInterpretation::component_is_scalar);
            solution_Q0.update_ghost_values();
            data_out.add_data_vector(dof_handler_Q0, solution_Q0, names_Q0, interpretation_Q0);
            LinearAlgebra::distributed::Vector<Number> sol0(solution);
               FETools::interpolate(dof_handler_Q0, solution_Q0, dof_handler, sol0);
        data_out.add_data_vector(solution, postprocessor);

        data_out.add_data_vector(sol0, postprocessor_Q0);


        Vector<double> mpi_owner(triangulation.n_active_cells());
        mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        data_out.add_data_vector(mpi_owner, "owner");
data_out.build_patches();
        const std::string filename =
                "./results/solution_" + Utilities::int_to_string(result_number, 3) + ".vtu";
        data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);

}

/*! The EulerProblem::run() function puts all pieces together. It starts off
* by calling the function that creates the mesh and sets up data structures,
* Before we start the time loop, we compute the time step size by the
* `EulerOperator::compute_cell_transport_speed()` function. For reasons of
* comparison, we compare the result obtained there with the minimal mesh
* size and print them to screen. For velocities and speeds of sound close
* to unity as in this tutorial program, the predicted effective mesh size
* will be close, but they could vary if scaling were different.
*
* Now we are ready to start the time loop, which we run until the time
* has reached the desired end time. Every 5 time steps, we compute a new
* estimate for the time step -- since the solution is nonlinear, it is
* most effective to adapt the value during the course of the
* simulation. In case the Courant number was chosen too aggressively, the
* simulation will typically blow up with time step NaN, so that is easy
* to detect here. One thing to note is that roundoff errors might
* propagate to the leading digits due to an interaction of slightly
* different time step selections that in turn lead to slightly different
* solutions. To decrease this sensitivity, it is common practice to round
* or truncate the time step size to a few digits, e.g. 3 in this case. In
* case the current time is near the prescribed 'tick' value for output
* (e.g. 0.02), we also write the output. After the end of the time loop,
* we summarize the computation by printing some statistics, which is
*  mostly done by the TimerOutput::print_wall_time_statistics() function.*/
template <int dim>
void EulerProblem<dim>::run()
{
    {
        const unsigned int n_vect_number = VectorizedArray<Number>::size();
        const unsigned int n_vect_bits   = 8 * sizeof(Number) * n_vect_number;

        pcout << "Running with "
              << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
              << " MPI processes" << std::endl;
        pcout << "Vectorization over " << n_vect_number << " "
              << (std::is_same<Number, double>::value ? "doubles" : "floats")
              << " = " << n_vect_bits << " bits ("
              << Utilities::System::get_current_vectorization_level() << ")"
              << std::endl;
    }
    const double courant_number = 0.15 / std::pow(parameters.fe_degree, 1.5);

    unsigned int timestep_number = 0;
    unsigned int pre_refinement_step = 0;

    make_grid();

    make_dofs();


    euler_operator_Q0.project(EquationData::InitialData<dim>(parameters),solution_Q0);
    euler_operator.project(EquationData::InitialData<dim>(parameters), solution);


    double min_vertex_distance = std::numeric_limits<double>::max();
    for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
            min_vertex_distance =
                    std::min(min_vertex_distance, cell->minimum_vertex_distance());
    min_vertex_distance =
            Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);

    time_step =courant_number * parameters.n_stages/euler_operator.compute_cell_transport_speed(solution);
    pcout << "Time step size: " << time_step
          << ", minimal h: " << min_vertex_distance
          << ", initial transport scaling: "
          << 1. / euler_operator.compute_cell_transport_speed(solution)
          << std::endl
          << std::endl;
    output_results(0);

    while (time < parameters.final_time - 1e-12)
    {
        ++timestep_number;

        tmp_solution.swap(solution);
        tmp_solution_Q0.swap(solution_Q0);

       if (timestep_number % 5 == 0)
            time_step =
                   courant_number * parameters.n_stages /
                    Utilities::truncate_to_n_digits(
                            euler_operator.compute_cell_transport_speed(solution), 3);

        {
            TimerOutput::Scope t(timer, "rk time stepping total");
            update(time,
                   time_step,
                   solution,
                   tmp_solution,
                   solution_Q0,
                   tmp_solution_Q0);


        }
    time += time_step;

if(parameters.refine && (timestep_number==1))
{
 adapt_mesh();
++pre_refinement_step;
}
else if(parameters.refine && (timestep_number>0) && (timestep_number % 50 == 0))
{
 adapt_mesh();
}

if (static_cast<int>(time / parameters.output_tick) !=
        static_cast<int>((time - time_step) / parameters.output_tick) ||
    time >= parameters.final_time - 1e-12)
 output_results(
     static_cast<unsigned int>(std::round(time / parameters.output_tick)));


    }

    timer.print_wall_time_statistics(MPI_COMM_WORLD);
    pcout << std::endl;
}
}

//To handle linking erros
template class Euler_DG::EulerProblem<2>;
