#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__


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
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include<deal.II/numerics/vector_tools.h>
#include <deal.II/base/parameter_handler.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <array>


using namespace dealii;


/*! We set up a class that holds all
* parameters that control the execution of the program.*/
namespace Parameters
{


  /*! We select a detail of the spatial discretization, namely the
  * numerical flux (Riemann solver) at the faces between cells. For this
  * program, we have implemented a modified variant of the Lax--Friedrichs
  * flux, the Harten--Lax--van Leer (HLL) flux, the HLLC flux, the Roe flux and
  * the SLAU flux.*/
struct Solver
   {
    enum EulerNumericalFlux
    {
        lax_friedrichs,
        harten_lax_vanleer,
        hllc_centered,
        HLLC,
        SLAU,
        roe,
    };

   EulerNumericalFlux numerical_flux_type;


   static void declare_parameters (dealii::ParameterHandler &prm);
   void parse_parameters (dealii::ParameterHandler &prm);
   };



  /*! We collect all parameters that control the execution of the program.
  * Besides the dimension and polynomial degree we want to run with, we
  * also specify a number of points in the Gaussian quadrature formula we
  * want to use for the nonlinear terms in the Euler equations.
  * Furthermore, we specify parameters for the limiter problem (TVB and filter).
  * We specify the parameters for the h-refinements.
  * Depending on the test case, we also change the final time up to which
  * we run the simulation, and a variable `output_tick` that specifies
  * in which intervals we want to write output
*/
struct Data_Storage : public Solver
{

    Data_Storage();

    void read_data(const std::string& filename);

    int testcase;  /*!< Number of specific testcase. Choices are : 1) Channel with hole; 2) backward facing step 3) forward facing step
                    * 4) sod-Shock problem; 5) supersonic flow past a circular cylinder test; 6) double-mach reflection problem; 7) 2D Riemann problem */
    int n_stages; /*!< number of stages in SSP runge kutta */
    int fe_degree; /*!< polynomial degree */
    int fe_degree_Q0; /*!< polynomial degree for Q0 solution */
    int n_q_points_1d; /*!< number of points in the Gaussian quadrature formula */
    int n_q_points_1d_Q0; /*!< number of points in the Gaussian quadrature formula when polynomial degree is zero*/

    int max_loc_refinements; /*!< number of maximum local refinements */
    int min_loc_refinements; /*!< number of minimum local refinements */

    double gamma; /*!< adiabatic constant dependent on the type of the gas */
    double beta_density; /*!< tolerance for density component used in filter technique */
    double beta_momentum; /*!< tolerance for momentum component used in filter technique*/
    double beta_energy; /*!< tolerance for energy component used in filter technique*/
    double beta; /*!< TVB limiter parameter */
    double M; /*!< TVB parameter */
    bool positivity; /*!< whether to use positivity limiter */
    std::string function_limiter; /*!< type of slope limiter function : minmod function or van Albada function */
    std::string type; /*!< type of limiter : filter technique or TVB limiter */
    bool refine; /*!< true do refine, false not do refine */
    std::string refine_indicator; /*!< do refine with density indicator or with pressure indicator */

    double final_time; /*!< The final time of the simulation */
    double output_tick; /*!< This indicates between how many time steps we print the solution*/
protected:
      ParameterHandler prm;

};
}
#endif // PARAMETERS_H
