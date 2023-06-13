#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>

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

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>
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


#include"parameters.h"

#include"eulerproblem.h"

using namespace dealii;


namespace Euler_DG{
using Number = double;

 constexpr unsigned int dimension            = 2;

}


/// The main() function is not surprising and follows what was done in all
/// previous MPI programs: As we run an MPI program, we need to call `MPI_Init()`
/// and `MPI_Finalize()`, which we do through the
/// Utilities::MPI::MPI_InitFinalize data structure. Note that we run the program
/// only with MPI, and set the thread count to 1.
/// argv[1] can be inputfile_sod.prm, inputfile_DMR.prm, inputfile_fstep.prm,
/// inputfile_bstep.prm, inputfile_cylinder.prm, inputfile_2Driemann.prm
int main(int argc, char **argv)
{

    using namespace Euler_DG;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    try
       {

        Parameters::Data_Storage data;
        data.read_data(argv[1]);

        EulerProblem<dimension> euler_problem(data);
        euler_problem.run();

       }
     catch (std::exception &exc)
       {
         std::cerr << std::endl
                   << std::endl
                   << "----------------------------------------------------"
                   << std::endl;
         std::cerr << "Exception on processing: " << std::endl
                   << exc.what() << std::endl
                   << "Aborting!" << std::endl
                   << "----------------------------------------------------"
                   << std::endl;

         return 1;
       }
     catch (...)
       {
         std::cerr << std::endl
                   << std::endl
                   << "----------------------------------------------------"
                   << std::endl;
         std::cerr << "Unknown exception!" << std::endl
                   << "Aborting!" << std::endl
                   << "----------------------------------------------------"
                   << std::endl;
         return 1;
       }
    return 0;
}
