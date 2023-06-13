#ifndef EQUATIONDATA_H
#define EQUATIONDATA_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/base/quadrature_lib.h>

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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include<deal.II/numerics/vector_tools.h>
#include <deal.II/matrix_free/operators.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <array>


using namespace dealii;

/// @sect3{Equation data}
namespace EquationData{

/// We now define a class with the exact solution for the test case 4.
/// We define an analytic solution of Sod-Shock tube problem that we try to
/// reproduce with our discretization
template <int dim>
class ExactSolution : public dealii::Function<dim>
{
public:
    ExactSolution(const double time, Parameters::Data_Storage &parameters_in);

    virtual double value(const dealii::Point<dim> & p,
                         const unsigned int component = 0) const override;
private :
    Parameters::Data_Storage &parameters;
};


/*! We define a class for the boundary condition. Given that
 * the Euler equations are a problem with $d+2$ equations in $d$ dimensions,
 * we need to tell the Function base class about the correct number of
 * components.
 */
template <int dim>
class BoundaryData : public dealii::Function<dim>
{
public:
    BoundaryData(const double time,Parameters::Data_Storage &parameters_in,const unsigned int a);

    unsigned int a;

    virtual double value(const dealii::Point<dim> & p,
                         const unsigned int component = 0) const override;
private :
    Parameters::Data_Storage &parameters;
};

/** We define a class for the initial condition. Given that
 * the Euler equations are a problem with $d+2$ equations in $d$ dimensions,
 * we need to tell the Function base class about the correct number of
 * components.
 */
template <int dim>
class InitialData : public dealii::Function<dim>
{
public:
    InitialData(Parameters::Data_Storage &parameters_in);

    virtual double value(const dealii::Point<dim> & p,
                         const unsigned int component = 0) const override;
private:
    Parameters::Data_Storage &parameters;
};
}
#endif // EQUATIONDATA_H
