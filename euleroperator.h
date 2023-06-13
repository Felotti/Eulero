#ifndef EULEROPERATOR_H
#define EULEROPERATOR_H

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

#include"parameters.h"
#include"equationdata.h"
#include"operations.h"
namespace Euler_DG{

using Number = double;

/*! This class implements the evaluators for the Euler problem. Since the present
 * operator is non-linear and does not require a matrix interface (to be
 * handed over to preconditioners), we only implement an `apply`
 * function as well as the combination of `apply` with the required vector
 * updates for the Runge--Kutta time integrator. Furthermore, we have added 3 additional
 * functions involving matrix-free routines, namely one to compute an
 * estimate of the time step scaling (that is combined with the Courant
 * number for the actual time step size) based on the velocity and speed of
 * sound in the elements, one for the projection of solutions (specializing
 * VectorTools::project() for the DG case), and one to compute the errors
 * against a possible analytical solution or norms against some background
 * state.
 *
 * We provide a few functions to allow a user
 * to pass in various forms of boundary conditions on different parts of the
 * domain boundary marked by types::boundary_id variables, as well as
 * possible body forces. */
template <int dim, int degree, int n_points_1d>
class EulerOperator
{
public:
    static constexpr unsigned int n_quadrature_points_1d = n_points_1d;

    EulerOperator(dealii::TimerOutput &timer_output,Parameters::Data_Storage &parameters_in);

    void reinit(const dealii::MappingQ1<dim>& mapping,
                std::vector<const DoFHandler<dim> *> &dof_handler,
                const std::vector<QGauss<1>> & quadratures);

    void set_inflow_boundary(const dealii::types::boundary_id boundary_id,
                             std::unique_ptr<dealii::Function<dim>> inflow_function);

    void set_subsonic_outflow_boundary(const dealii::types::boundary_id boundary_id,
                                       std::unique_ptr<dealii::Function<dim>> outflow_energy);

    void set_supersonic_outflow_boundary(const dealii::types::boundary_id boundary_id);

    void set_wall_boundary(const dealii::types::boundary_id boundary_id);

    void set_body_force(std::unique_ptr<dealii::Function<dim>> body_force);

    // Standard evaluation routine
    void apply(const double current_time, const dealii::LinearAlgebra::distributed::Vector<Number> &src,
               dealii::LinearAlgebra::distributed::Vector<Number>& dst) const;

    void project(const dealii::Function<dim>& function, dealii::LinearAlgebra::distributed::Vector<Number>& solution) const;

    std::array<double, 3> compute_errors(const dealii::Function<dim>& function, const dealii::LinearAlgebra::distributed::Vector<Number>& solution) const;

    double compute_cell_transport_speed(const dealii::LinearAlgebra::distributed::Vector<Number>& solution) const;

    void initialize_vector(dealii::LinearAlgebra::distributed::Vector<Number>& vector) const;


private:
    Parameters::Data_Storage & parameters;
    dealii::MatrixFree<dim, Number> data;
    dealii::TimerOutput &timer;

    std::map<dealii::types::boundary_id, std::unique_ptr<dealii::Function<dim>>> inflow_boundaries;
    std::map<dealii::types::boundary_id, std::unique_ptr<dealii::Function<dim>>> subsonic_outflow_boundaries;
    std::set<dealii::types::boundary_id> supersonic_outflow_boundaries;
    std::set<dealii::types::boundary_id>   wall_boundaries;
    std::unique_ptr<dealii::Function<dim>> body_force;

    void local_apply_inverse_mass_matrix(const dealii::MatrixFree<dim, Number>& data,
                                         dealii::LinearAlgebra::distributed::Vector<Number>& dst,
                                         const dealii::LinearAlgebra::distributed::Vector<Number> &src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const;

    void local_apply_cell(const dealii::MatrixFree<dim, Number>& data,
                          dealii::LinearAlgebra::distributed::Vector<Number>& dst,
                          const dealii::LinearAlgebra::distributed::Vector<Number> &src,
                          const std::pair<unsigned int, unsigned int>& cell_range) const;

    void local_apply_face(const dealii::MatrixFree<dim, Number>& data,
                          dealii::LinearAlgebra::distributed::Vector<Number>& dst,
                          const dealii::LinearAlgebra::distributed::Vector<Number>& src,
                          const std::pair<unsigned int, unsigned int>& face_range) const;

    void local_apply_boundary_face(const dealii::MatrixFree<dim, Number>& data,
                                   dealii::LinearAlgebra::distributed::Vector<Number>& dst,
                                   const dealii::LinearAlgebra::distributed::Vector<Number>& src,
                                   const std::pair<unsigned int, unsigned int>& face_range) const;

};

}

#endif // EULEROPERATOR_H
