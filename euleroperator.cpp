#include "euleroperator.h"

namespace Euler_DG{
using namespace dealii;

/// constructor
template <int dim, int degree, int n_points_1d>
EulerOperator<dim, degree, n_points_1d>::EulerOperator(dealii::TimerOutput &timer, Parameters::Data_Storage &parameters_in)
    : timer(timer), parameters(parameters_in)
{}

/*! For the initialization of the Euler operator, we set up the MatrixFree
* variable contained in the class. This can be done given a mapping to
* describe possible curved boundaries as well as a DoFHandler object
* describing the degrees of freedom. Since we use a discontinuous Galerkin
* discretization in this tutorial program where no constraints are imposed
* strongly on the solution field, we do not need to pass in an
* AffineConstraints object and rather use a dummy for the
* construction.*/
template <int dim, int degree, int n_points_1d>
void EulerOperator<dim, degree, n_points_1d>::reinit(const dealii::MappingQ1<dim> &   mapping,
                                                     std::vector<const DoFHandler<dim> *> &dof_handlers,
                                                     const std::vector<QGauss<1>>& quadratures)
{

    AffineConstraints<double>      dummy;

   std::vector<const AffineConstraints<double> *> constraints(dof_handlers.size(),&dummy);

    typename MatrixFree<dim, Number>::AdditionalData additional_data;

    additional_data.mapping_update_flags = (update_gradients | update_JxW_values | update_quadrature_points | update_values);
    additional_data.mapping_update_flags_inner_faces = (update_JxW_values | update_quadrature_points | update_normal_vectors | update_values);
    additional_data.mapping_update_flags_boundary_faces = (update_JxW_values | update_quadrature_points | update_normal_vectors | update_values);
    additional_data.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::none;

    data.reinit(mapping, dof_handlers, constraints, quadratures, additional_data);


}

template <int dim, int degree, int n_points_1d>
void EulerOperator<dim, degree, n_points_1d>::initialize_vector(
        LinearAlgebra::distributed::Vector<Number> &vector) const
{
    data.initialize_dof_vector(vector);
}

/*! The subsequent member functions are the ones that must be called from
* outside to specify the various types of boundaries. For an inflow boundary,
* we must specify all components in terms of density $\rho$, momentum $\rho
* \mathbf{u}$ and energy $E$. Given this information, we then store the
* function alongside the respective boundary id in a map member variable of
* this class. Likewise, we proceed for the subsonic outflow boundaries (where
* we request a function as well, which we use to retrieve the energy), and
* the supersonic outflow condition and for wall (no-penetration) boundaries
* where we impose zero normal velocity. For the present
* DG code where boundary conditions are solely applied as part of the weak
* form (during time integration), the call to set the boundary conditions
* can appear both before or after the `reinit()` call to this class.
*
* The checks added in each of the four function are used to
* ensure that boundary conditions are mutually exclusive on the various
* parts of the boundary, i.e., that a user does not accidentally designate a
* boundary as both an inflow and say a subsonic outflow boundary.*/
template <int dim, int degree, int n_points_1d>
void EulerOperator<dim, degree, n_points_1d>::set_inflow_boundary(
        const types::boundary_id       boundary_id,
        std::unique_ptr<Function<dim>> inflow_function)
{
    AssertThrow(subsonic_outflow_boundaries.find(boundary_id) == subsonic_outflow_boundaries.end() &&
                wall_boundaries.find(boundary_id) == wall_boundaries.end() &&
                supersonic_outflow_boundaries.find(boundary_id) == supersonic_outflow_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as inflow"));
    AssertThrow(inflow_function->n_components == dim + 2,
                ExcMessage("Expected function with dim+2 components"));

    inflow_boundaries[boundary_id] = std::move(inflow_function);
}


template <int dim, int degree, int n_points_1d>
void EulerOperator<dim, degree, n_points_1d>::set_subsonic_outflow_boundary(
        const types::boundary_id       boundary_id,
        std::unique_ptr<Function<dim>> outflow_function)
{
    AssertThrow(inflow_boundaries.find(boundary_id) == inflow_boundaries.end() &&
                wall_boundaries.find(boundary_id) == wall_boundaries.end() &&
                supersonic_outflow_boundaries.find(boundary_id) == supersonic_outflow_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as subsonic outflow"));
    AssertThrow(outflow_function->n_components == dim + 2,
                ExcMessage("Expected function with dim+2 components"));

    subsonic_outflow_boundaries[boundary_id] = std::move(outflow_function);
}

template <int dim, int degree, int n_points_1d>
void EulerOperator<dim, degree, n_points_1d>::set_supersonic_outflow_boundary(
        const types::boundary_id       boundary_id)
{
    AssertThrow(inflow_boundaries.find(boundary_id) == inflow_boundaries.end() &&
                wall_boundaries.find(boundary_id) == wall_boundaries.end() &&
                subsonic_outflow_boundaries.find(boundary_id) == subsonic_outflow_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as supersonic outflow"));

    supersonic_outflow_boundaries.insert(boundary_id);
}

template <int dim, int degree, int n_points_1d>
void EulerOperator<dim, degree, n_points_1d>::set_wall_boundary(
        const types::boundary_id boundary_id)
{
    AssertThrow(inflow_boundaries.find(boundary_id) == inflow_boundaries.end() &&
                subsonic_outflow_boundaries.find(boundary_id) == subsonic_outflow_boundaries.end() &&
                supersonic_outflow_boundaries.find(boundary_id) == supersonic_outflow_boundaries.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as wall boundary"));

    wall_boundaries.insert(boundary_id);
}


template <int dim, int degree, int n_points_1d>
void EulerOperator<dim, degree, n_points_1d>::set_body_force(
        std::unique_ptr<Function<dim>> body_force)
{
    AssertDimension(body_force->n_components, dim);

    this->body_force = std::move(body_force);
}

/// @sect{Local evaluators}

/*! Now we proceed to the local evaluators for the Euler problem.
* We use an FEEvaluation with a non-standard number of quadrature
* points. Whereas we previously always set the number of quadrature points
* to equal the polynomial degree plus one, we now set the number quadrature
* points as a separate variable (e.g. the polynomial degree plus two)
* to more accurately handle nonlinear terms. Since
* the evaluator is fed with the appropriate loop lengths via the template
* argument and keeps the number of quadrature points in the whole cell in
* the variable FEEvaluation::n_q_points, we now automatically operate on
* the more accurate formula without further changes.
*
* We are evaluating a multi-component system. The matrix-free framework
* provides several ways to handle the multi-component case. Here we use an FEEvaluation
* object with multiple components embedded into it, specified by the fourth
* template argument `dim + 2` for the components in the Euler system. As a
* consequence, the return type of FEEvaluation::get_value() is not a scalar
* any more (that would return a VectorizedArray type, collecting data from
* several elements), but a Tensor of `dim+2` components. The functionality
* is otherwise similar to the scalar case; it is handled by a template
* specialization of a base class, called FEEvaluationAccess. An alternative
* variant would have been to use several FEEvaluation objects, a scalar one
* for the density, a vector-valued one with `dim` components for the
* momentum, and another scalar evaluator for the energy. To ensure that
* those components point to the correct part of the solution, the
* constructor of FEEvaluation takes three optional integer arguments after
* the required MatrixFree field, namely the number of the DoFHandler for
* multi-DoFHandler systems (taking the first by default), the number of the
* quadrature point in case there are multiple Quadrature objects (see more
* below), and as a third argument the component within a vector system. As
* we have a single vector for all components, we would go with the third
* argument, and set it to `0` for the density, `1` for the vector-valued
* momentum, and `dim+1` for the energy slot. FEEvaluation then picks the
* appropriate subrange of the solution vector during
* FEEvaluationBase::read_dof_values() and
* FEEvaluation::distributed_local_to_global() or the more compact
* FEEvaluation::gather_evaluate() and FEEvaluation::integrate_scatter()
* calls.
*
* When it comes to the evaluation of the body force vector, we distinguish
* between two cases for efficiency reasons: In case we have a constant
* function (derived from Functions::ConstantFunction), we can precompute
* the value outside the loop over quadrature points and simply use the
* value everywhere. For a more general function, we instead need to call
* the `evaluate_function()` method we provided above; this path is more
* expensive because we need to access the memory associated with the
* quadrature point data.
*
* Since we have implemented all physics for the Euler equations in the
* separate `euler_flux()` function, all we have to do here is to call this function
* given the current solution evaluated at quadrature points, returned by
* `phi.get_value(q)`, and tell the FEEvaluation object to queue the flux
* for testing it by the gradients of the shape functions (which is a Tensor
* of outer `dim+2` components, each holding a tensor of `dim` components
* for the $x,y,z$ component of the Euler flux). One final thing worth
* mentioning is the order in which we queue the data for testing by the
* value of the test function, `phi.submit_value()`, in case we are given an
* external function: We must do this after calling `phi.get_value(q)`,
* because `get_value()` (reading the solution) and `submit_value()`
* (queuing the value for multiplication by the test function and summation
* over quadrature points) access the same underlying data field. Here it
* would be easy to achieve also without temporary variable `w_q` since
* there is no mixing between values and gradients. For more complicated
* setups, one has to first copy out e.g. both the value and gradient at a
* quadrature point and then queue results again by
* FEEvaluationBase::submit_value() and FEEvaluationBase::submit_gradient().
*
* As a final note, we mention that we do not use the first MatrixFree
* argument of this function, which is a call-back from MatrixFree::loop().
* The interfaces imposes the present list of arguments, but since we are in
* a member function where the MatrixFree object is already available as the
* `data` variable, we stick with that to avoid confusion.*/
template <int dim, int degree, int n_points_1d>
void EulerOperator<dim, degree, n_points_1d>::local_apply_cell(
        const MatrixFree<dim, Number> &,
        LinearAlgebra::distributed::Vector<Number> &      dst,
        const LinearAlgebra::distributed::Vector<Number> &src,
        const std::pair<unsigned int, unsigned int> &     cell_range) const
{
    FEEvaluation<dim, degree, n_points_1d, dim + 2, Number> phi(data);

    Tensor<1, dim, VectorizedArray<Number>> constant_body_force;
    const Functions::ConstantFunction<dim> *constant_function =
            dynamic_cast<Functions::ConstantFunction<dim> *>(body_force.get());

    if (constant_function)
        constant_body_force = evaluate_function<dim, Number, dim>(
                    *constant_function, Point<dim, VectorizedArray<Number>>());

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
            const auto w_q = phi.get_value(q);
            phi.submit_gradient(euler_flux<dim>(w_q,parameters), q);
            if (body_force.get() != nullptr)
            {
                const Tensor<1, dim, VectorizedArray<Number>> force =
                        constant_function ? constant_body_force :
                                            evaluate_function<dim, Number, dim>(
                            *body_force, phi.quadrature_point(q));

                Tensor<1, dim + 2, VectorizedArray<Number>> forcing;
                for (unsigned int d = 0; d < dim; ++d)
                    forcing[d + 1] = w_q[0] * force[d];
                for (unsigned int d = 0; d < dim; ++d)
                    forcing[dim + 1] += force[d] * w_q[d + 1];

                phi.submit_value(forcing, q);
            }
        }

        phi.integrate_scatter(((body_force.get() != nullptr) ?
                                   EvaluationFlags::values :
                                   EvaluationFlags::nothing) |
                              EvaluationFlags::gradients,
                              dst);
    }
}

/*! The next function concerns the computation of integrals on interior
* faces, where we need evaluators from both cells adjacent to the face. We
* associate the variable `phi_m` with the solution component $\mathbf{w}^-$
* and the variable `phi_p` with the solution component $\mathbf{w}^+$. We
* distinguish the two sides in the constructor of FEFaceEvaluation by the
* second argument, with `true` for the interior side and `false` for the
* exterior side, with interior and exterior denoting the orientation with
* respect to the normal vector.
*
* Note that the calls FEFaceEvaluation::gather_evaluate() and
* FEFaceEvaluation::integrate_scatter() combine the access to the vectors
* and the sum factorization parts. This combined operation not only saves a
* line of code, but also contains an important optimization: Given that we
* use a nodal basis in terms of the Lagrange polynomials in the points of
* the Gauss-Lobatto quadrature formula, only $(p+1)^{d-1}$ out of the
* $(p+1)^d$ basis functions evaluate to non-zero on each face. Thus, the
* evaluator only accesses the necessary data in the vector and skips the
* parts which are multiplied by zero. If we had first read the vector, we
* would have needed to load all data from the vector, as the call in
* isolation would not know what data is required in subsequent
* operations. If the subsequent FEFaceEvaluation::evaluate() call requests
* values and derivatives, indeed all $(p+1)^d$ vector entries for each
* component are needed, as the normal derivative is nonzero for all basis
* functions.
*
* The arguments to the evaluators as well as the procedure is similar to
* the cell evaluation. We again use the more accurate (over-)integration
* scheme. At the quadrature points, we then go to our
* free-standing function for the numerical flux. It receives the solution
* evaluated at quadrature points from both sides (i.e., $\mathbf{w}^-$ and
* $\mathbf{w}^+$), as well as the normal vector onto the minus side. As
* explained above, the numerical flux is already multiplied by the normal
* vector from the minus side. We need to switch the sign because the
* boundary term comes with a minus sign in the weak form derived in the
* introduction. The flux is then queued for testing both on the minus sign
* and on the plus sign, with switched sign as the normal vector from the
* plus side is exactly opposed to the one from the minus side.*/
template <int dim, int degree, int n_points_1d>
void EulerOperator<dim, degree, n_points_1d>::local_apply_face(
        const MatrixFree<dim, Number> &,
        LinearAlgebra::distributed::Vector<Number> &      dst,
        const LinearAlgebra::distributed::Vector<Number> &src,
        const std::pair<unsigned int, unsigned int> &     face_range) const
{
    FEFaceEvaluation<dim, degree, n_points_1d, dim + 2, Number> phi_m(data,
                                                                      true);
    FEFaceEvaluation<dim, degree, n_points_1d, dim + 2, Number> phi_p(data,
                                                                      false);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
    {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, EvaluationFlags::values);

        phi_m.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
        {
            const auto numerical_flux =
                    euler_numerical_flux<dim>(phi_m.get_value(q),
                                              phi_p.get_value(q),
                                              phi_m.get_normal_vector(q),parameters);
            phi_m.submit_value(-numerical_flux, q);
            phi_p.submit_value(numerical_flux, q);
        }

        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
}

/*! For faces located at the boundary, we need to impose the appropriate
* boundary conditions. In this tutorial program, we implement fifth cases as
* mentioned above. The discontinuous Galerkin
* method imposes boundary conditions not as constraints, but only
* weakly. Thus, the various conditions are imposed by finding an appropriate
* <i>exterior</i> quantity $\mathbf{w}^+$ that is then handed to the
* numerical flux function also used for the interior faces. In essence,
* we "pretend" a state on the outside of the domain in such a way that
* if that were reality, the solution of the PDE would satisfy the boundary
* conditions we want.
*
* For wall boundaries, we need to impose a no-normal-flux condition on the
* momentum variable, whereas we use a Neumann condition for the density and
* energy with $\rho^+ = \rho^-$ and $E^+ = E^-$. To achieve the no-normal
* flux condition, we set the exterior values to the interior values and
* subtract two times the velocity in wall-normal direction, i.e., in the
* direction of the normal vector.
*
* For inflow boundaries, we simply set the given Dirichlet data
* $\mathbf{w}_\mathrm{D}$ as a boundary value. An alternative would have been
* to use $\mathbf{w}^+ = -\mathbf{w}^- + 2 \mathbf{w}_\mathrm{D}$, the
* so-called mirror principle.
*
* The imposition of outflow is essentially a Neumann condition, i.e.,
* setting $\mathbf{w}^+ = \mathbf{w}^-$.
*
* In the implementation below, we check for the various types
* of boundaries at the level of quadrature points. Of course, we could also
* have moved the decision out of the quadrature point loop and treat entire
* faces as of the same kind, which avoids some map/set lookups in the inner
* loop over quadrature points. However, the loss of efficiency is hardly
* noticeable, so we opt for the simpler code here. Also note that the final
* `else` clause will catch the case when some part of the boundary was not
* assigned any boundary condition via `EulerOperator::set_..._boundary(...)`.*/
template <int dim, int degree, int n_points_1d>
void EulerOperator<dim, degree, n_points_1d>::local_apply_boundary_face(
        const MatrixFree<dim, Number> & ,
        LinearAlgebra::distributed::Vector<Number> &      dst,
        const LinearAlgebra::distributed::Vector<Number> &src,
        const std::pair<unsigned int, unsigned int> &     face_range) const
{
    FEFaceEvaluation<dim, degree, n_points_1d, dim + 2, Number> phi(data, true);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
    {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
            const auto w_m    = phi.get_value(q);
            const auto normal = phi.get_normal_vector(q);

            auto rho_u_dot_n = w_m[1] * normal[0];
            for (unsigned int d = 1; d < dim; ++d)
                rho_u_dot_n += w_m[1 + d] * normal[d];

            bool at_outflow = false;

            Tensor<1, dim + 2, VectorizedArray<Number>> w_p;
            const auto boundary_id = data.get_boundary_id(face);
            if (wall_boundaries.find(boundary_id) != wall_boundaries.end())
            {
                w_p[0] = w_m[0];
                for (unsigned int d = 0; d < dim; ++d)
                    w_p[d + 1] = w_m[d + 1] - 2. * rho_u_dot_n * normal[d];
                w_p[dim + 1] = w_m[dim + 1];
            }
            else if (inflow_boundaries.find(boundary_id) !=
                     inflow_boundaries.end())
                w_p =
                        evaluate_function(*inflow_boundaries.find(boundary_id)->second,
                                          phi.quadrature_point(q));
            else if (subsonic_outflow_boundaries.find(boundary_id) !=
                     subsonic_outflow_boundaries.end())
            {
                w_p          = w_m;
                w_p[dim + 1] = evaluate_function(
                            *subsonic_outflow_boundaries.find(boundary_id)->second,
                            phi.quadrature_point(q),
                            dim + 1);
                at_outflow = true;
            }
            else if (supersonic_outflow_boundaries.find(boundary_id) != supersonic_outflow_boundaries.end())
            {
                w_p = w_m;
                at_outflow = true;
            }
            else
                AssertThrow(false,
                            ExcMessage("Unknown boundary id, did "
                                       "you set a boundary condition for "
                                       "this part of the domain boundary?"));

            auto flux = euler_numerical_flux<dim>(w_m, w_p, normal,parameters);

            if (at_outflow)
                for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
                {
                    if (rho_u_dot_n[v] < -1e-12)
                        for (unsigned int d = 0; d < dim; ++d)
                            flux[d + 1][v] = 0.;
                }

            phi.submit_value(-flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
    }
}

/*! The next function implements the inverse mass matrix operation. It does similar
* operations as the forward evaluation of the mass matrix, except with a
* different interpolation matrix, representing the inverse $S^{-1}$
* factors. These represent a change of basis from the specified basis (in
* this case, the Lagrange basis in the points of the Gauss--Lobatto
* quadrature formula) to the Lagrange basis in the points of the Gauss
* quadrature formula. In the latter basis, we can apply the inverse of the
* point-wise `JxW` factor, i.e., the quadrature weight times the
* determinant of the Jacobian of the mapping from reference to real
* coordinates. Once this is done, the basis is changed back to the nodal
* Gauss-Lobatto basis again. All of these operations are done by the
* `apply()` function below. What we need to provide is the local fields to
* operate on (which we extract from the global vector by an FEEvaluation
* object) and write the results back to the destination vector of the mass
* matrix operation.
*
* One thing to note is that we added two integer arguments (that are
* optional) to the constructor of FEEvaluation, the first being 0
* (selecting among the DoFHandler in multi-DoFHandler systems; here, we
* only have one) and the second being 1 to make the quadrature formula
* selection. As we use the quadrature formula 0 for the over-integration of
* nonlinear terms, we use the formula 1 with the default $p+1$ (or
* `fe_degree+1` in terms of the variable name) points for the mass
* matrix. This leads to square contributions to the mass matrix and ensures
* exact integration, as explained in the introduction.*/
template <int dim, int degree, int n_points_1d>
void EulerOperator<dim, degree, n_points_1d>::local_apply_inverse_mass_matrix(
        const MatrixFree<dim, Number> & ,
        LinearAlgebra::distributed::Vector<Number> &      dst,
        const LinearAlgebra::distributed::Vector<Number> &src,
        const std::pair<unsigned int, unsigned int> &     cell_range) const
{
    FEEvaluation<dim, degree, degree + 1, dim + 2, Number> phi(data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim + 2, Number>
            inverse(phi);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
        phi.reinit(cell);
        phi.read_dof_values(src);

        inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

        phi.set_dof_values(dst);
    }
}

/// @sect4{The apply() and related functions}

  /*! We now come to the function which implements the evaluation of the Euler
  * operator as a whole, i.e., $\mathcal M^{-1} \mathcal L(t, \mathbf{w})$,
  * calling into the local evaluators presented above. The steps should be
  * clear from the previous code. One thing to note is that we need to adjust
  * the time in the functions we have associated with the various parts of
  * the boundary, in order to be consistent with the equation in case the
  * boundary data is time-dependent. Then, we call MatrixFree::loop() to
  * perform the cell and face integrals, including the necessary ghost data
  * exchange in the `src` vector. The seventh argument to the function,
  * `true`, specifies that we want to zero the `dst` vector as part of the
  * loop, before we start accumulating integrals into it. This variant is
  * preferred over explicitly calling `dst = 0.;` before the loop as the
  * zeroing operation is done on a subrange of the vector in parts that are
  * written by the integrals nearby. This enhances data locality and allows
  * for caching, saving one roundtrip of vector data to main memory and
  * enhancing performance. The last two arguments to the loop determine which
  * data is exchanged: Since we only access the values of the shape functions
  * one faces, typical of first-order hyperbolic problems, and since we have
  * a nodal basis with nodes at the reference element surface, we only need
  * to exchange those parts. This again saves precious memory bandwidth.
  *
  * Once the spatial operator $\mathcal L$ is applied, we need to make a
  * second round and apply the inverse mass matrix. Here, we call
  * MatrixFree::cell_loop() since only cell integrals appear. The cell loop
  * is cheaper than the full loop as access only goes to the degrees of
  * freedom associated with the locally owned cells, which is simply the
  * locally owned degrees of freedom for DG discretizations. Thus, no ghost
  * exchange is needed here.
  *
  * Around all these functions, we put timer scopes to record the
  * computational time for statistics about the contributions of the various
  * parts.*/
template <int dim, int degree, int n_points_1d>
void EulerOperator<dim, degree, n_points_1d>::apply(
        const double                                      current_time,
        const LinearAlgebra::distributed::Vector<Number> &src,
        LinearAlgebra::distributed::Vector<Number> &      dst) const
{
    {
        TimerOutput::Scope t(timer, "apply - integrals");

        for (auto &i : inflow_boundaries)
            i.second->set_time(current_time);
        for (auto &i : subsonic_outflow_boundaries)
            i.second->set_time(current_time);


        data.loop(&EulerOperator::local_apply_cell,
                  &EulerOperator::local_apply_face,
                  &EulerOperator::local_apply_boundary_face,
                  this,
                  dst,
                  src,
                  true,
                  MatrixFree<dim, Number>::DataAccessOnFaces::values,
                  MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

    {
        TimerOutput::Scope t(timer, "apply - inverse mass");

        data.cell_loop(&EulerOperator::local_apply_inverse_mass_matrix,
                       this,
                       dst,
                       dst);
    }

}


/*! Having discussed the implementation of the functions that deal with
* advancing the solution by one time step, let us now move to functions
* that implement other, ancillary operations. Specifically, these are
* functions that compute projections, evaluate errors, and compute the speed
* of information transport on a cell.
*
* The first of these functions is essentially equivalent to
* VectorTools::project(), just much faster because it is specialized for DG
* elements where there is no need to set up and solve a linear system, as
* each element has independent basis functions. The reason why we show the
* code here, besides a small speedup of this non-critical operation, is that
* it shows additional functionality provided by
* MatrixFreeOperators::CellwiseInverseMassMatrix.
*
* The projection operation works as follows: If we denote the matrix of
* shape functions evaluated at quadrature points by $S$, the projection on
* cell $K$ is an operation of the form $\underbrace{S J^K S^\mathrm
* T}_{\mathcal M^K} \mathbf{w}^K = S J^K
* \tilde{\mathbf{w}}(\mathbf{x}_q)_{q=1:n_q}$, where $J^K$ is the diagonal
* matrix containing the determinant of the Jacobian times the quadrature
* weight (JxW), $\mathcal M^K$ is the cell-wise mass matrix, and
* $\tilde{\mathbf{w}}(\mathbf{x}_q)_{q=1:n_q}$ is the evaluation of the
* field to be projected onto quadrature points. (In reality the matrix $S$
* has additional structure through the tensor product, as explained in the
* introduction.) This system can now equivalently be written as
* $\mathbf{w}^K = \left(S J^K S^\mathrm T\right)^{-1} S J^K
* \tilde{\mathbf{w}}(\mathbf{x}_q)_{q=1:n_q} = S^{-\mathrm T}
* \left(J^K\right)^{-1} S^{-1} S J^K
* \tilde{\mathbf{w}}(\mathbf{x}_q)_{q=1:n_q}$. Now, the term $S^{-1} S$ and
* then $\left(J^K\right)^{-1} J^K$ cancel, resulting in the final
* expression $\mathbf{w}^K = S^{-\mathrm T}
* \tilde{\mathbf{w}}(\mathbf{x}_q)_{q=1:n_q}$.
* This operation is implemented by
* MatrixFreeOperators::CellwiseInverseMassMatrix::transform_from_q_points_to_basis().
* The name is derived from the fact that this projection is simply
* the multiplication by $S^{-\mathrm T}$, a basis change from the
* nodal basis in the points of the Gaussian quadrature to the given finite
* element basis. Note that we call FEEvaluation::set_dof_values() to write
* the result into the vector, overwriting previous content, rather than
* accumulating the results as typical in integration tasks -- we can do
* this because every vector entry has contributions from only a single
* cell for discontinuous Galerkin discretizations.*/
template <int dim, int degree, int n_points_1d>
void EulerOperator<dim, degree, n_points_1d>::project(
        const Function<dim> &                       function,
        LinearAlgebra::distributed::Vector<Number> &solution) const
{
    FEEvaluation<dim, degree, degree + 1, dim + 2, Number> phi(data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim + 2, Number>
            inverse(phi);
    solution.zero_out_ghost_values();
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
    {
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_dof_value(evaluate_function(function,
                                                   phi.quadrature_point(q)),
                                 q);
        inverse.transform_from_q_points_to_basis(dim + 2,
                                                 phi.begin_dof_values(),
                                                 phi.begin_dof_values());
        phi.set_dof_values(solution);
    }
}

/*! The next function again repeats functionality also provided by the
* deal.II library, namely VectorTools::integrate_difference(). We here show
* the explicit code to highlight how the vectorization across several cells
* works and how to accumulate results via that interface: Recall that each
* <i>lane</i> of the vectorized array holds data from a different cell.
*/
template <int dim, int degree, int n_points_1d>
std::array<double, 3> EulerOperator<dim, degree, n_points_1d>::compute_errors(
        const Function<dim> &                             function,
        const LinearAlgebra::distributed::Vector<Number> &solution) const
{
    TimerOutput::Scope t(timer, "compute errors");
    double             errors_squared[3] = {};
    FEEvaluation<dim, degree, n_points_1d, dim + 2, Number> phi(data, 0, 0);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
    {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);
        VectorizedArray<Number> local_errors_squared[3] = {};
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
            const auto error =
                    evaluate_function(function, phi.quadrature_point(q)) -
                    phi.get_value(q);
            const auto JxW = phi.JxW(q);

            local_errors_squared[0] += error[0] * error[0] * JxW;
            for (unsigned int d = 0; d < dim; ++d)
                local_errors_squared[1] += (error[d + 1] * error[d + 1]) * JxW;
            local_errors_squared[2] += (error[dim + 1] * error[dim + 1]) * JxW;
        }
        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell);
             ++v)
            for (unsigned int d = 0; d < 3; ++d)
                errors_squared[d] += local_errors_squared[d][v];
    }

    Utilities::MPI::sum(errors_squared, MPI_COMM_WORLD, errors_squared);

    std::array<double, 3> errors;
    for (unsigned int d = 0; d < 3; ++d)
        errors[d] = std::sqrt(errors_squared[d]);

    return errors;
}

/*! This final function of the EulerOperator class is used to estimate the
* transport speed, scaled by the mesh size, that is relevant for setting
* the time step size in the explicit time integrator. In the Euler
* equations, there are two speeds of transport, namely the convective
* velocity $\mathbf{u}$ and the propagation of sound waves with sound
* speed $c = \sqrt{\gamma p/\rho}$ relative to the medium moving at
* velocity $\mathbf u$.
*
* In the formula for the time step size, we are interested not by
* these absolute speeds, but by the amount of time it takes for
* information to cross a single cell. For information transported along with
* the medium, $\mathbf u$ is scaled by the mesh size,
* so an estimate of the maximal velocity can be obtained by computing
* $\|J^{-\mathrm T} \mathbf{u}\|_\infty$, where $J$ is the Jacobian of the
* transformation from real to the reference domain. Note that
* FEEvaluationBase::inverse_jacobian() returns the inverse and transpose
* Jacobian, representing the metric term from real to reference
* coordinates, so we do not need to transpose it again. We store this limit
* in the variable `convective_limit` in the code below.
*
* The sound propagation is isotropic, so we need to take mesh sizes in any
* direction into account. The appropriate mesh size scaling is then given
* by the minimal singular value of $J$ or, equivalently, the maximal
* singular value of $J^{-1}$. Note that one could approximate this quantity
* by the minimal distance between vertices of a cell when ignoring curved
* cells. The speed of convergence of this method depends
* on the ratio of the largest to the next largest eigenvalue and the
* initial guess, which is the vector of all ones.*/
template <int dim, int degree, int n_points_1d>
double EulerOperator<dim, degree, n_points_1d>::compute_cell_transport_speed(
        const LinearAlgebra::distributed::Vector<Number> &solution) const
{
    TimerOutput::Scope t(timer, "compute transport speed");
    Number             max_transport = 0;
    FEEvaluation<dim, degree, degree + 1, dim + 2, Number> phi(data, 0, 1);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
    {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);
        VectorizedArray<Number> local_max = 0.;
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
            const auto solution = phi.get_value(q);
            const auto velocity = euler_velocity<dim>(solution);
            const auto pressure = euler_pressure<dim>(solution,parameters);

            const auto inverse_jacobian = phi.inverse_jacobian(q);
            const auto convective_speed = inverse_jacobian * velocity;
            VectorizedArray<Number> convective_limit = 0.;
            for (unsigned int d = 0; d < dim; ++d)
                convective_limit =
                        std::max(convective_limit, std::abs(convective_speed[d]));

            const auto speed_of_sound =
                    std::sqrt(parameters.gamma * pressure * (1. / solution[0]));

            Tensor<1, dim, VectorizedArray<Number>> eigenvector;
            for (unsigned int d = 0; d < dim; ++d)
                eigenvector[d] = 1.;
            for (unsigned int i = 0; i < 5; ++i)
            {
                eigenvector = transpose(inverse_jacobian) *
                        (inverse_jacobian * eigenvector);
                VectorizedArray<Number> eigenvector_norm = 0.;
                for (unsigned int d = 0; d < dim; ++d)
                    eigenvector_norm =
                            std::max(eigenvector_norm, std::abs(eigenvector[d]));
                eigenvector /= eigenvector_norm;
            }
            const auto jac_times_ev   = inverse_jacobian * eigenvector;
            const auto max_eigenvalue = std::sqrt(
                        (jac_times_ev * jac_times_ev) / (eigenvector * eigenvector));
            local_max =
                    std::max(local_max,
                             max_eigenvalue * speed_of_sound + convective_limit);
        }

        // Similarly to the previous function, we must make sure to accumulate
        // speed only on the valid cells of a cell batch.
        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell);
             ++v)
            for (unsigned int d = 0; d < 3; ++d)
                max_transport = std::max(max_transport, local_max[v]);
    }

    max_transport = Utilities::MPI::max(max_transport, MPI_COMM_WORLD);

    return max_transport;
}

}

template class Euler_DG::EulerOperator<2,2,5>;
template class Euler_DG::EulerOperator<2,0,1>;

