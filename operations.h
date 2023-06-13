#ifndef OPERATIONS_H
#define OPERATIONS_H

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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <array>
#include <deal.II/base/parameter_handler.h>

#include"equationdata.h"
#include"parameters.h"

namespace Euler_DG{
using namespace dealii;
using Number =double;

/** In the following functions, we implement the various problem-specific
* operators pertaining to the Euler equations. Each function acts on the
* vector of conserved variables \f$[\rho, \rho\mathbf{u}, E]\f$ that we hold in
* the solution vectors, and computes various derived quantities.
*/

/*! First out is the computation of the velocity, that we derive from the
* momentum variable \f$\rho \mathbf{u}\f$ by division by \f$\rho\f$. One thing to
* note here is that we decorate all those functions with the keyword
* `DEAL_II_ALWAYS_INLINE`.
*/
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
Tensor<1, dim, Number>
euler_velocity(const Tensor<1, dim + 2, Number> &conserved_variables)
{
    const Number inverse_density = Number(1.) / conserved_variables[0];

    Tensor<1, dim, Number> velocity;
    for (unsigned int d = 0; d < dim; ++d)
        velocity[d] = conserved_variables[1 + d] * inverse_density;

    return velocity;
}

/*! The next function computes the pressure from the vector of conserved
* variables, using the formula \f$p = (\gamma - 1) \left(E - \frac 12 \rho
* \mathbf{u}\cdot \mathbf{u}\right)\f$.
* */
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
Number
euler_pressure(const Tensor<1, dim + 2, Number> &conserved_variables,Parameters::Data_Storage &par)
{
    const Tensor<1, dim, Number> velocity =
            euler_velocity<dim>(conserved_variables);

    Number rho_u_dot_u = conserved_variables[1] * velocity[0];
    for (unsigned int d = 1; d < dim; ++d)
        rho_u_dot_u += conserved_variables[1 + d] * velocity[d];

    return (par.gamma - 1.) * (conserved_variables[dim + 1] - 0.5 * rho_u_dot_u);
}

/*! Here is the definition of the Euler flux function, i.e., the definition
 * of the actual equation. Given the velocity and pressure (that the
 * compiler optimization will make sure are done only once), this is
 * straight-forward given the equation stated in the introduction.
 * */
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
Tensor<1, dim + 2, Tensor<1, dim, Number>>
euler_flux(const Tensor<1, dim + 2, Number> &conserved_variables,Parameters::Data_Storage &par)
{
    const Tensor<1, dim, Number> velocity =
            euler_velocity<dim>(conserved_variables);
    const Number pressure = euler_pressure<dim>(conserved_variables,par);

    Tensor<1, dim + 2, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; ++d)
    {
        flux[0][d] = conserved_variables[1 + d];
        for (unsigned int e = 0; e < dim; ++e)
            flux[e + 1][d] = conserved_variables[e + 1] * velocity[d];
        flux[d + 1][d] += pressure;
        flux[dim + 1][d] =
                velocity[d] * (conserved_variables[dim + 1] + pressure);
    }

    return flux;
}

/*! This next function is a helper to simplify the implementation of the
 * numerical flux, implementing the action of a tensor of tensors (with
 * non-standard outer dimension of size dim + 2, so the standard overloads
 * provided by deal.II's tensor classes do not apply here) with another
 * tensor of the same inner dimension, i.e., a matrix-vector product.*/
template <int n_components, int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
Tensor<1, n_components, Number>
operator*(const Tensor<1, n_components, Tensor<1, dim, Number>> &matrix,
          const Tensor<1, dim, Number> &                         vector)
{
    Tensor<1, n_components, Number> result;
    for (unsigned int d = 0; d < n_components; ++d)
        result[d] = matrix[d] * vector;
    return result;
}

/*! Function that returns the sign of a given value.*/
template <typename Number>
inline DEAL_II_ALWAYS_INLINE //
Number sgn(Number & val)
{
    return (Number(0.) < val) - (val < Number(0.));
}

/*! This function implements the numerical flux (Riemann solver). It gets the
* state from the two sides of an interface and the normal vector, oriented
* from the side of the solution \f$\mathbf{w}^- \f$ towards the solution
* \f$\mathbf{w}^+ \f$.
* In this and the following functions, we use variable suffixes `_m` and
* `_p` to indicate quantities derived from \f$\mathbf{w}^- \f$ and \f$\mathbf{w}^+ \f$,
* i.e., values "here" and "there" relative to the current cell when looking
* at a neighbor cell.
*/
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
Tensor<1, dim + 2, Number>
euler_numerical_flux(const Tensor<1, dim + 2, Number> &u_m,
                     const Tensor<1, dim + 2, Number> &u_p,
                     const Tensor<1, dim, Number> &    normal,
                     Parameters::Data_Storage &par)
{
    double gamma=par.gamma;

    const auto velocity_m = euler_velocity<dim>(u_m);
    const auto velocity_p = euler_velocity<dim>(u_p);

    const auto pressure_m = euler_pressure<dim>(u_m,par);
    const auto pressure_p = euler_pressure<dim>(u_p,par);

    const auto flux_m = euler_flux<dim>(u_m,par);
    const auto flux_p = euler_flux<dim>(u_p,par);

    const auto density_m = u_m[0];
    const auto density_p = u_p[0];

    const Number enthalpy_m = (u_m[dim+1] + pressure_m)/density_m;
    const Number enthalpy_p = (u_p[dim+1] + pressure_p)/density_p;


    switch (par.numerical_flux_type)
    {

    case Parameters::Solver::EulerNumericalFlux::lax_friedrichs:
        {
            const auto lambda =
                    0.5 * std::sqrt(std::max(velocity_p.norm_square() +
                                            gamma* pressure_p * (1. / u_p[0]),
                                             velocity_m.norm_square() +
                                            gamma * pressure_m * (1. / u_m[0])));

            return 0.5 * (flux_m * normal + flux_p * normal) +
                   0.5 * lambda * (u_m - u_p);
        }

    case Parameters::Solver::EulerNumericalFlux::harten_lax_vanleer:
        {

            const auto avg_velocity_normal = ((std::sqrt(std::abs(density_m))*velocity_m + \
                                             std::sqrt(std::abs(density_p))*velocity_p)*normal)/ \
                                             (std::sqrt(std::abs(density_m))+std::sqrt(std::abs(density_p)));

            const auto avg_enthalpy = ((std::sqrt(std::abs(density_m))*enthalpy_m + \
                                       std::sqrt(std::abs(density_p))*enthalpy_p))/ \
                                       (std::sqrt(std::abs(density_m))+std::sqrt(std::abs(density_p)));
            const auto avg_c = std::sqrt(std::abs(((gamma-1)*(avg_enthalpy-0.5*avg_velocity_normal*avg_velocity_normal))));
            const Number s_pos =
                    std::max(Number(), avg_velocity_normal + avg_c);
            const Number s_neg =
                    std::min(Number(), avg_velocity_normal - avg_c);
            const Number inverse_s = Number(1.) / (s_pos - s_neg);

            return inverse_s *
                   ((s_pos * (flux_m * normal) - s_neg * (flux_p * normal)) -
                    s_pos * s_neg * (u_m - u_p));
        }

    case Parameters::Solver::EulerNumericalFlux::roe :{

        Number density_m_sqrt = std::sqrt(density_m);
        Number density_p_sqrt = std::sqrt(density_p);
        Number fact_m = density_m_sqrt / (density_m_sqrt + density_p_sqrt);
        Number fact_p = 1.0 - fact_m;

        Tensor<1, dim, Number> velocity, dv;
        Number v2_m = 0.0, v2_p = 0.0;
        Number v_m_normal = 0., v_p_normal = 0.0;
        Number vel_normal = 0.0, v2 = 0.0;
        Number v_dot_dv = 0.0;

        for(unsigned int d=0; d<dim; ++d)
         {

            v2_m       += velocity_m[d] * velocity_m[d];
            v2_p       += velocity_p[d] * velocity_p[d];
            v_m_normal += velocity_m[d] * normal[d];
            v_p_normal += velocity_p[d] * normal[d];

            velocity[d] = velocity_m[d] * fact_m + velocity_p[d] * fact_p;
            vel_normal += velocity[d] * normal[d];
            v2         += velocity[d] * velocity[d];
            dv[d]      = velocity_p[d] - velocity_m[d];
            v_dot_dv   += velocity[d] * dv[d];
         }

        //Pressure
        Number p_m = (gamma - 1)*(u_m[dim+1] - (0.5*density_m*v2_m));
        Number p_p = (gamma - 1)*(u_p[dim+1] - (0.5*density_p*v2_p));

        Number h_m = gamma * p_m / density_m / (gamma - 1) + 0.5 * v2_m;
        Number h_p = gamma * p_p / density_p / (gamma - 1) + 0.5 * v2_p;

        Number density = density_m_sqrt * density_p_sqrt;
        Number h = h_m * fact_m + h_p * fact_p;
        Number c = std::sqrt((gamma-1)*(h-0.5*v2));
        Number d_density = density_p - density_m;
        Number d_p = p_p - p_m;
        Number d_vn = v_p_normal - v_m_normal;

        Number a1 = (d_p - density * c * d_vn ) / (2.0 * c * c);
        Number a2 = d_density - d_p / (c*c);
        Number a3 = (d_p + density * c * d_vn) / (2.0 * c * c);

        Number l1 = std::fabs(vel_normal[0] - c[0]);
        Number l2 = std::fabs(vel_normal[0]);
        Number l3 = std::fabs(vel_normal[0] + c[0]);

        //entropy fix
        Number delta = 0.1*c;
        if(l1[0]<delta[0]) l1 = 0.5 * (l1*l1/delta + delta);
        if(l3[0]<delta[0]) l3 = 0.5 * (l3*l3/delta + delta);

       Tensor<1, dim + 2, Number> Dflux;
       Dflux[0] = l1 * a1 + l2 * a2 + l3 * a3;
       Dflux[dim +1] = l1*a1*(h-c*vel_normal) + l2*a2*0.5*v2 + \
               l2*density*(v_dot_dv - vel_normal* d_vn) + l3*a3*(h + c*vel_normal);

       Tensor<1, dim + 2, Number> flux_roe;
       flux_roe[0] = 0.5 * (density_m * v_m_normal + density_p  *v_p_normal - Dflux[0]);
       flux_roe[dim+1] = 0.5 * (density_m*h_m*v_m_normal + density_p*h_p*v_p_normal - Dflux[dim+1]);

       Number p_avg = 0.5*(p_p + p_m);
       for(unsigned int d = 1; d<=dim; ++d)
       {
           Dflux[d] = (velocity[d-1] - normal[d-1] * c)*l1*a1 + velocity[d-1]*l2*a2  + \
                   (dv[d-1]-normal[d-1]*d_vn)*l2*density + (velocity[d-1] + normal[d-1]*c)*l3*a3;
           flux_roe[d] = normal[d-1]*p_avg + 0.5*(u_m[d]*v_m_normal + u_p[d]*v_p_normal) - 0.5*Dflux[d];
       }
    return flux_roe;
   }

    case Parameters::Solver::EulerNumericalFlux::HLLC :
    {
        Number density_m_sqrt = std::sqrt(density_m);
        Number density_p_sqrt = std::sqrt(density_p);
        Number fact_m = density_m_sqrt / (density_m_sqrt + density_p_sqrt);
        Number fact_p = 1.0 - fact_m;

        Tensor<1, dim, Number> velocity;
        Number v2_m = 0.0, v2_p = 0.0;
        Number v_m_normal = 0., v_p_normal = 0.0;
        Number vel_normal = 0.0, v2 = 0.0;
        for(unsigned int d=0; d<dim; ++d)
         {

            v2_m       += velocity_m[d] * velocity_m[d];
            v2_p       += velocity_p[d] * velocity_p[d];
            v_m_normal += velocity_m[d] * normal[d];
            v_p_normal += velocity_p[d] * normal[d];

            velocity[d] = velocity_m[d] * fact_m + velocity_p[d] * fact_p;
            vel_normal += velocity[d] * normal[d];
            v2         += velocity[d] * velocity[d];
         }

         //pressure
         Number p_m = (gamma-1) * (u_m[dim+1] - 0.5 * density_m * v2_m);
         Number p_p = (gamma-1) * (u_p[dim+1] - 0.5 * density_p * v2_p);

         // enthalpy
         Number h_m = (u_m[dim+1] + p_m) / density_m;
         Number h_p = (u_p[dim+1] + p_p) / density_p;

         // sound speed
         Number c_m = std::sqrt(gamma * p_m / density_m);
         Number c_p = std::sqrt(gamma * p_p / density_p);

         // energy per unit mass
         Number e_m = u_m[dim+1] / density_m;
         Number e_p = u_p[dim+1] / density_p;

         // roe average
         Number h = h_m * fact_m + h_p * fact_p;
         Number c = std::sqrt( (gamma-1.0) * (h - 0.5*v2) );

         // speed of sound at l and r
         Number s_m = std::min(vel_normal-c, v_m_normal-c_m);
         Number s_p = std::max(vel_normal+c, v_p_normal+c_p);

         // speed of contact  //segno -pm+pp prima cera p_m - p_p
         Number s_star = (p_p - p_m - density_m * v_m_normal * (s_m-v_m_normal) + \
                       density_p * v_p_normal * (s_p-v_p_normal)) /(density_p*(s_p-v_p_normal) - density_m*(s_m-v_m_normal));

         // Pressure at right and left (Pressure_j=Pressure_i) side of contact surface
         Number pStar = density_p * (v_p_normal-s_p)*(v_p_normal-s_star) + p_p;

         Tensor<1, dim + 2, Number> flux_hllc;
         if (s_star[0] >= 0.0) {
            if (s_m[0] > 0.0)
            {
               flux_hllc[0] = density_m*v_m_normal;
               for (unsigned int d = 1; d < dim+1; ++d)
                  flux_hllc[d] = density_m*velocity_m[d-1]*v_m_normal + p_m*normal[d-1];
               flux_hllc[dim+1] = e_m*density_m*v_m_normal + p_m*v_m_normal;
            }
            else
            {
               Number invSMmSs = Number(1.0)/(s_m-s_star);
               Number sMmuM = s_m-v_m_normal;
               Number rhoSM = density_m*sMmuM*invSMmSs;
               Tensor<1, dim, Number> rhouSM;
               for (unsigned int d = 0; d < dim; ++d)
                  rhouSM[d] = (density_m*velocity_m[d]*sMmuM+(pStar-p_m)*normal[d])*invSMmSs;
               Number eSM = (sMmuM*e_m*density_m-p_m*v_m_normal+pStar*s_star)*invSMmSs;

               flux_hllc[0] = rhoSM*s_star;
               for (unsigned int d = 1; d < dim+1; ++d)
                  flux_hllc[d] = rhouSM[d-1]*s_star + pStar*normal[d-1];
               flux_hllc[dim+1] = (eSM+pStar)*s_star;
            }
         }
         else
         {
            if (s_p[0] >= 0.0)
            {
               Number invSPmSs = Number(1.0)/(s_p-s_star);
               Number sPmuP = s_p-v_p_normal;
               Number rhoSP = density_p*sPmuP*invSPmSs;
               Tensor<1, dim, Number> rhouSP;
               for (unsigned int d = 0; d < dim; ++d)
                  rhouSP[d] = (density_p*velocity_p[d]*sPmuP+(pStar-p_p)*normal[d])*invSPmSs;
               Number eSP = (sPmuP*e_p*density_p-p_p*v_p_normal+pStar*s_star)*invSPmSs;

               flux_hllc[0] = rhoSP*s_star;
               for (unsigned int d = 1; d < dim+1; ++d)
                  flux_hllc[d] = rhouSP[d-1]*s_star + pStar*normal[d-1];
               flux_hllc[dim+1] = (eSP+pStar)*s_star;
            }
            else
            {
               flux_hllc[0] = density_p*v_p_normal;
               for (unsigned int d = 1; d < dim+1; ++d)
                  flux_hllc[d] = density_p*velocity_p[d-1]*v_p_normal + p_p*normal[d-1];
               flux_hllc[dim+1] = e_p*density_p*v_p_normal + p_p*v_p_normal;
            }
         }

     return flux_hllc;
    }

    case Parameters::Solver::EulerNumericalFlux::hllc_centered:
    {
        Tensor<1, dim + 2, Number> flux_hllc;

        const Number avg_velocity_normal = ((std::sqrt(std::abs(density_m)) * velocity_m * normal + \
                                            std::sqrt(std::abs(density_p)) * velocity_p * normal)) / \
                                          (std::sqrt(std::abs(density_m)) + std::sqrt(std::abs(density_p)));
        const Number avg_enthalpy =
                ((std::sqrt(std::abs(density_m)) * enthalpy_m + std::sqrt(std::abs(density_p)) * enthalpy_p)) / \
                            (std::sqrt(std::abs(density_m)) + std::sqrt(std::abs(density_p)));
        const Number avg_c = std::sqrt(
                             std::abs(((gamma - 1) * (avg_enthalpy - 0.5 * avg_velocity_normal * avg_velocity_normal))));
        const Number s_p =  avg_velocity_normal + avg_c;
        const Number s_m =  avg_velocity_normal - avg_c;

        const Number inverse1 = Number(1.)/(density_m*(s_m-velocity_m[0])-density_p*(s_p-velocity_p[0]));
        const Number s_star = (pressure_p-pressure_m + density_m*velocity_m[0]*(s_m-velocity_m[0])-density_p*velocity_p[0]*(s_p-velocity_p[0]))*inverse1;



        Tensor<1, dim + 2, Number> u_star_m;
        Tensor<1, dim + 2, Number> u_star_p;

        Number inverse_m = Number(1.)/(s_m-s_star);
        const Number mult_m = density_m*((s_m-velocity_m[0])*inverse_m);
        u_star_m[0]=mult_m;
        u_star_m[1]=mult_m*s_star;
        u_star_m[2]=mult_m*velocity_m[1];
        Number inverse2_m = Number(1.)/(density_m*(s_m-velocity_m[0]));
        u_star_m[dim+1]=mult_m*((u_m[dim+1]/density_m)+((s_star-velocity_m[0])*(s_star+pressure_m*inverse2_m)));

        Number inverse_p = Number(1.)/(s_p-s_star);
        const Number mult_p = density_p*((s_p-velocity_p[0])*inverse_p);
        u_star_p[0]=mult_p;
        u_star_p[1]=mult_p*s_star;
        u_star_p[2]=mult_p*velocity_p[1];
        Number inverse2_p = Number(1.)/(density_p*(s_p-velocity_p[0]));
        u_star_p[dim+1]=mult_p*((u_p[dim+1]/density_p)+((s_star-velocity_p[0])*(s_star+pressure_p*inverse2_p)));

        if (s_m[0] >= 0.0)
            flux_hllc = flux_m*normal;

        else if (s_p[0]<=0.0)
            flux_hllc = flux_p * normal;

        else {
            flux_hllc = 0.5*(flux_m*normal+flux_p*normal)+0.5*(s_m*(u_star_m-u_m)+std::abs(s_star)*(u_star_m-u_star_p)+s_p*(u_star_p-u_p));


        }
        return flux_hllc;

    }

    case Parameters::Solver::EulerNumericalFlux::SLAU :
   {
     Tensor<1, dim + 2, Number> flux_SLAU;

       const Number c_plus = std::sqrt(gamma*pressure_m/density_m);
       const Number c_minus = std::sqrt(gamma*pressure_p/density_p);
       const Number c_bar = 0.5*(c_plus+c_minus);

       const Number V_n_bar =(density_m*std::abs(velocity_m*normal)+ \
                            density_p*std::abs(velocity_p*normal))/(density_m+density_p);

       Number v2_m = (velocity_m[0]*velocity_m[0])+(velocity_m[1]*velocity_m[1]);
       Number v2_p = (velocity_p[0]*velocity_p[0])+(velocity_p[1]*velocity_p[1]);
       const Number M_hat = std::min(Number(1.0),std::sqrt(0.5*(v2_m+v2_p))/c_bar);
       Number chi = (1.0-M_hat)*(1.0-M_hat);

       Number p_bar =0.5*(pressure_m+pressure_p);

       Number M_m = (velocity_m*normal)/c_bar;
       Number M_p = (velocity_p*normal)/c_bar;
       Number p_mplus, p_pminus;
       if(std::abs(M_m[0])<1.) {
           p_mplus =0.25*((M_m+1.)*(M_m+1.)*(2.-M_m));
       }
       else {
           p_mplus = 0.5 *(M_m+std::abs(M_m))/M_m;
       }
       if(std::abs(M_p[0])<1.) {
           p_pminus =0.25*((M_p-1.)*(M_p-1.)*(2.+M_p));
       }
       else {
           p_pminus = 0.5 *(M_p-std::abs(M_p))/M_p;
       }


       Number delta_beta = p_mplus-p_pminus;
       Number p_csi = (1.-chi)*(p_mplus+p_pminus-1.);
       Number p_m_minus_p = pressure_m-pressure_p;
       Number p_tilde = (p_bar + 0.5*delta_beta*p_m_minus_p)+p_csi*p_bar;
       Number g = -std::max(std::min(M_m,Number(0.)),Number(-1.))*std::min(std::max(M_p,Number(0.)),Number(1.));


       Number m_hat = (density_m*velocity_m*normal) + (density_p*velocity_p*normal)-(std::abs(V_n_bar)*(density_p-density_m));
       Number m_dot = 0.5*m_hat*(1.-g)-(chi/(2.*c_bar))*(pressure_p-pressure_m);

       Number m_dot_abs = std::abs(m_dot);
       flux_SLAU[0]= (0.5*(m_dot+m_dot_abs)) + (0.5*(m_dot-m_dot_abs));
       flux_SLAU[1] =(0.5*(m_dot+m_dot_abs)*velocity_m[0])+(0.5*(m_dot-m_dot_abs)*velocity_p[0]) + p_tilde*normal[0];
       flux_SLAU[2] =(0.5*(m_dot+m_dot_abs)*velocity_m[1])+(0.5*(m_dot-m_dot_abs)*velocity_p[1]) + p_tilde*normal[1];
       flux_SLAU[dim+1] = (0.5*(m_dot+m_dot_abs)*enthalpy_m)+(0.5*(m_dot-m_dot_abs)*enthalpy_p);

       return flux_SLAU;

   }

    default:
        {
            Assert(false, ExcNotImplemented());
            return {};
        }
    }
}


/*! This and the next function are helper functions to provide compact
* evaluation calls as multiple points get batched together via a
* VectorizedArray argument. This function is used for the subsonic outflow boundary
* conditions where we need to set the energy component to a prescribed value. The next one
* requests the solution on all components and is used for inflow boundaries
* where all components of the solution are set.
*/
template <int dim, typename Number>
VectorizedArray<Number>
evaluate_function(const Function<dim> &                      function,
                  const Point<dim, VectorizedArray<Number>> &p_vectorized,
                  const unsigned int                         component)
{
    VectorizedArray<Number> result;
    for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
    {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
            p[d] = p_vectorized[d][v];
        result[v] = function.value(p, component);
    }
    return result;
}


template <int dim, typename Number, int n_components = dim + 2>
Tensor<1, n_components, VectorizedArray<Number>>
evaluate_function(const Function<dim> &                      function,
                  const Point<dim, VectorizedArray<Number>> &p_vectorized)
{
    AssertDimension(function.n_components, n_components);
    Tensor<1, n_components, VectorizedArray<Number>> result;
    for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
    {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
            p[d] = p_vectorized[d][v];
        for (unsigned int d = 0; d < n_components; ++d)
            result[d][v] = function.value(p, d);
    }
    return result;
}


}
#endif // OPERATIONS_H
