#ifndef __EULERPROBLEM_H__
#define __EULERPROBLEM_H__


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

#include <vector>

#include"parameters.h"
#include"equationdata.h"
#include"euleroperator.h"
#include"operations.h"

namespace Euler_DG{


using namespace dealii;
using Number = double;

/// @sect{The EulerProblem class}

  /*! This class combines the EulerOperator class with the time integrator and
  * the usual global data structures such as FiniteElement and DoFHandler, to
  * actually run the simulations of the Euler problem.
  *
  * The member variables are a triangulation, a finite element, a mapping,
  * and a DoFHandler to describe the degrees of freedom.
  * In this class, we implemente the member functions usefull for
  * refinement and compute limiter as TVB or filter limiter.
  * In the update funcion we implement all the stages for the SSP Runge-Kutta
  * in which at each stage we do the limiter of the solution.
  * In addition, we keep an instance of the
  * EulerOperator described above around, which will do all heavy lifting in
  * terms of integrals, and some parameters for time integration like the
  * current time or the time step size.
  * Furthermore, we use a PostProcessor instance to write some additional
  * information to the output file. The interface of the DataPostprocessor class
  * is intuitive, requiring us to provide information about what needs to be evaluated
  * (typically only the values of the solution, except for the Schlieren plot
  * that we only enable in 2D where it makes sense), and the names of what
  * gets evaluated. Note that it would also be possible to extract most
  * information by calculator tools within visualization programs such as
  * ParaView, but it is so much more convenient to do it already when writing
  * the output. */
template <int dim>
class EulerProblem
{

public:

    EulerProblem(Parameters::Data_Storage  &parameters_in);

    void run();

private:
    void make_grid();
    void adapt_mesh();
    void make_dofs();

    void compute_cell_average (LinearAlgebra::distributed::Vector<double> & current_solution);
    void get_cell_average(const typename dealii::DoFHandler<dim>::cell_iterator& cell,
                          dealii::Vector<double>& avg);
    void compute_shock_indicator(LinearAlgebra::distributed::Vector<double> & current_solution);

    LinearAlgebra::distributed::Vector<Number> apply_limiter_TVB (LinearAlgebra::distributed::Vector<Number> & solution);
    LinearAlgebra::distributed::Vector<Number> apply_positivity_limiter (LinearAlgebra::distributed::Vector<Number> & current_solution);
    LinearAlgebra::distributed::Vector<Number> apply_filter(LinearAlgebra::distributed::Vector<Number> &    sol_H,
                                                            LinearAlgebra::distributed::Vector<Number> &    sol_M);

    void update(const double  current_time,
                const double    time_step,
                LinearAlgebra::distributed::Vector<Number> &    solution_np,
                LinearAlgebra::distributed::Vector<Number> &    tmp_sol_n,
                LinearAlgebra::distributed::Vector<Number> &    solution_np_Q0,
                LinearAlgebra::distributed::Vector<Number> &    tmp_sol_n_Q0);

    Parameters::Data_Storage &parameters;

    void output_results(const unsigned int result_number);

    LinearAlgebra::distributed::Vector<Number> solution, tmp_solution, sol_aux;
    LinearAlgebra::distributed::Vector<Number> solution_Q0, tmp_solution_Q0;

    ConditionalOStream pcout;

#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> triangulation;
#else
    Triangulation<dim> triangulation;
#endif

    FESystem<dim>        fe;
    FESystem<dim>        fe_Q0;

    // Iterators to neighbouring cells
    std::vector<typename dealii::DoFHandler<dim>::cell_iterator> lcell, rcell, bcell, tcell;
    const dealii::FE_DGQ<dim>    fe_cell;
    dealii::DoFHandler<dim>      dh_cell;
    Vector<double> shock_indicator;
    Vector<double> jump_indicator;

    const MappingQ1<dim> mapping, mapping_Q0;

    DoFHandler<dim>      dof_handler,dof_handler_Q0;
    std::vector<const DoFHandler<dim>*> dof_handlers;
    std::vector<const DoFHandler<dim>*> dof_handlers_Q0;

    Vector<double>  estimated_indicator_per_cell;

    const QGauss<dim> quadrature;
    std::vector<QGauss<1>> quadratures;
    std::vector<QGauss<1>> quadratures_Q0;

    std::vector<unsigned int> cell_degree;
    std::vector<bool> re_update;

    std::vector< dealii::Vector<double> >  cell_average;
    TimerOutput timer;

    EulerOperator<2, 2,5> euler_operator;
    EulerOperator<2, 0,1> euler_operator_Q0;

    double time, time_step;

    class Postprocessor : public DataPostprocessor<dim>
    {
    public:
        Postprocessor(Parameters::Data_Storage &parameters, int a);
        Parameters::Data_Storage &parameters;
        virtual void evaluate_vector_field(
                const DataPostprocessorInputs::Vector<dim> &inputs,
                std::vector<Vector<double>> &computed_quantities) const override;

        virtual std::vector<std::string> get_names() const override;
        int a ;  /*!< int value for plot different solution */
        virtual std::vector<
                DataComponentInterpretation::DataComponentInterpretation>
        get_data_component_interpretation() const override;

        virtual UpdateFlags get_needed_update_flags() const override;

    private:
        const bool do_schlieren_plot;
    };
};

template <int dim>
struct EulerEquations
{
    // First dim components correspond to momentum
    static const unsigned int n_components             = dim + 2;
    static const unsigned int density_component        = 0;
    static const unsigned int energy_component         = dim+1;

    //---------------------------------------------------------------------------
    // Compute kinetic energy from conserved variables
    //---------------------------------------------------------------------------
    template <typename number, typename InputVector>
    static
        number
        compute_kinetic_energy (const InputVector &W)
    {
        number kinetic_energy = 0;
        for (unsigned int d=0; d<dim; ++d)
            kinetic_energy += *(W.begin()+d) *
                              *(W.begin()+d);
        kinetic_energy *= 0.5/(*(W.begin() + density_component));

        return kinetic_energy;
    }

    //---------------------------------------------------------------------------
    // Compute pressure from conserved variables
    //---------------------------------------------------------------------------
    template <typename number, typename InputVector>
    static
        number
        compute_pressure (const InputVector &W)
    {
        return ((1.4-1.0) *
                (*(W.begin() + energy_component) -
                 compute_kinetic_energy<number>(W)));
    }
};
}
#endif // EULERPROBLEM_H
