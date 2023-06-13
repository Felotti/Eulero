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

namespace EquationData{

///constructor of ExactSolution
template<int dim>
ExactSolution<dim>::ExactSolution(const double time,Parameters::Data_Storage &parameters_in): dealii::Function<dim>(dim + 2, time), parameters(parameters_in)
{}

///constructor of BoundaryData
template<int dim>
BoundaryData<dim>::BoundaryData(const double time,Parameters::Data_Storage &parameters_in, const unsigned int a):
    dealii::Function<dim>(dim + 2, time), parameters(parameters_in), a(a)
{}

///constructor of InitialData
template<int dim>
InitialData<dim>::InitialData(Parameters::Data_Storage &parameters_in):
    dealii::Function<dim>(dim + 2), parameters(parameters_in)
{}


template <int dim>
double ExactSolution<dim>::value(const dealii::Point<dim> & x,
                                const unsigned int component) const
{
    double t = this->get_time();
    switch (parameters.testcase) {
    case 4: {
        double x_0       =0.0;

        double rho_r     =0.125; // right states
        double p_r       =0.1;
        double u_r       =0.0;
        double rho_l     =1.0; // left states
        double p_l       =1.0;
        double u_l       =0.0;

        double mu = 0.4082; // std::sqrt((parameters.gamma-1.)/(parameters.gamma + 1));

        // Sound speed
        double cs_l       =1.18321595661992318149; //std::sqrt(parameters.gamma*p_l/rho_l);

        double p_post = 0.30313017805064701449;
        double v_post = 0.92745262004895057117;//2*(std::sqrt(parameters.gamma)/(parameters.gamma-1))*(1-std::pow(p_post,(parameters.gamma-1)/(2*parameters.gamma)));
        double rho_post = 0.26557371170530713611;//rho_r*(((p_post/p_r)+(mu*mu))/(1+mu*mu*(p_post/p_r)));
        double v_shock = 1.75215573203017838111;// v_post*((rho_post/rho_r)/((rho_post/rho_r)-1));
        double rho_middle = 0.42631942817849538541;//rho_l * std::pow(p_post/p_l,1/parameters.gamma) ;

          // Key Positions
         double cs_2 =0.99772543261013335592;//cs_l - ((parameters.gamma -1)/2)*v_post;
         double x_4 = 0.356;//x_0 + (v_shock * 0.2);//0.3277;//x_0 + v_shock * t;      // position of shock
         double x_3 = 0.181;//x_0 + (v_post * 0.2);//0.1758;//679;//x_0 + v_post * t;          // position of contact discontinuity
         double x_2 = x_0 + ((v_post - cs_2) * 0.2);; // foot of rarefaction wave
         double x_1 = x_0 - (cs_l * 0.2);  // head of rarefaction wave

            if (x[0] < x_1) {
                if (component == 0)
                    return rho_l;
                else if (component == 1)
                    return   u_l;
                else if (component == dim+1)
                    return p_l;
                else
                    return 0.0;
            }
            else if (x[0] <x_2)
            {
              // rarefaction wave
              double u =(1-(mu*mu))*((-(x_0-x[0])/0.2)+cs_l );
             double factor = (mu*mu*((x_0-x[0])/0.2))+((1-(mu*mu))*cs_l);
              double rho = rho_l * std::pow( factor/cs_l, 2/(parameters.gamma-1) );
               if (component == 0)
                  return rho;
              else if (component == 1)
                  return  u;
              else if (component == dim+1)
              {
                  double p =p_l * std::pow(rho/rho_l,parameters.gamma);
                  return p;
              }
              else
                  return 0.0;

            }
            else if (x[0]<= x_3)
            {
                if (component == 0)
                    return rho_middle;
                else if (component == 1)
                    return   v_post;
                else if (component == dim+1)
                    return p_post;
                else
                    return 0.0;

            }
            else if (x[0] <= x_4)
            {
                if (component == 0)
                    return rho_post;
                else if (component == 1)
                    return  v_post;
                else if (component == dim+1)
                    return p_post;
                else
                    return 0.0;

            }
            else
            {
                if (component == 0)
                    return rho_r;
                else if (component == 1)
                    return  u_r;
                else if (component == dim+1)
                    return p_r;
                else
                    return 0.0;

            }


    }

    default:
    Assert(false, ExcNotImplemented());
        return 0.;

    }

    }

template <int dim>
double InitialData<dim>::value(const dealii::Point<dim> & x,
                                const unsigned int component) const
{

    switch (parameters.testcase) {
    case 1: { //channel with hole
        if (component == 0)
            return 1.;
        else if (component == 1)
            return 0.4;
        else if (component == dim + 1)
            return 3.097857142857143;
        else
            return 0.;
    }
    case 2: { //b-step
        if (component == 0)
            return 7.0406*(x[0]<=0.5) + 1.4*(x[0]>0.5);
        else if (component == 1)
            return 28.72565*(x[0]<=0.5) + 0.0*(x[0]>0.5);
        else if (component == dim + 1)
            return 133.74782*(x[0]<=0.5) + 2.5*(x[0]>0.5);
        else
            return 0.;
    }
    case 3: { //f-step
        if (component == 0)
            return 1.4;
        else if (component == 1)
            return 4.2;
        else if (component == dim + 1)
            return 8.8;
        else
            return 0.;
    }
    case 4: { //sod shock tube

        if (component == 0)
            return 1.0*(x[0]<=0.0) + 0.125*(x[0] > 0.0);
        else if (component == 1)
            return 0.;
        else if (component == dim + 1)
            return 2.5*(x[0]<=0.0) + 0.25*(x[0] > 0.0);
        else
            return 0.;

    }
    case 5: { //cylinder
        if (component == 0)
            return 1.4;
        else if (component == 1)
            return 3.0*1.4;
        else if (component == dim + 1)
        {   double gamma = 1.4;
            double pressure = 1.0;
            double energy = (pressure/(gamma-1)) + ((1.4*3.0*3.0)*0.5);
            return energy;
        }
        else
            return 0.;
        }
    case 6: { //double mach reflection

           if (component == 0)
               return 8.0*(x[0]<(1.0/6.0+(x[1]/sqrt(3)))) + 1.4*(x[0]>=(1.0/6.0+(x[1]/sqrt(3))));
           else if (component == 1)
               return 57.1576766498*(x[0]<(1.0/6.0+(x[1]/sqrt(3)))) + 0.0*(x[0]>=(1.0/6.0+(x[1]/sqrt(3))));
           else if (component == 2)
               return -33.0*(x[0]<(1.0/6.0+(x[1]/sqrt(3)))) + 0.0*(x[0]>=(1.0/6.0+(x[1]/sqrt(3))));
           else {
               double pressure = 116.5;
               const double gamma = 1.4;
               double energy_post_shock = pressure / (gamma- 1.) +
                       0.5 * 8.0 * ( 8.25 * 8.25);
               return  energy_post_shock*(x[0]<(1.0/6.0+(x[1]/sqrt(3)))) + 2.5*(x[0]>=(1.0/6.0+(x[1]/sqrt(3))));
           }



    }
    case 7: { //2D Riemann problem
        double rho = 1.1*(x[0]>0.5)*(x[1]>0.5)+0.5065*(x[0]<=0.5)*(x[1]>0.5)+1.1*(x[0]<=0.5)*(x[1]<=0.5)+0.5065*(x[0]>0.5)*(x[1]<=0.5);
        double u = 0.0*(x[0]>0.5)*(x[1]>0.5)+0.8939*(x[0]<=0.5)*(x[1]>0.5)+0.8939*(x[0]<=0.5)*(x[1]<=0.5)+0.0*(x[0]>0.5)*(x[1]<=0.5);
        double v = 0.0*(x[0]>0.5)*(x[1]>0.5)+0.0*(x[0]<=0.5)*(x[1]>0.5)+0.8939*(x[0]<=0.5)*(x[1]<=0.5)+0.8939*(x[0]>0.5)*(x[1]<=0.5);
        double pressure = 1.*(x[0]>0.5)*(x[1]>0.5)+0.35*(x[0]<=0.5)*(x[1]>0.5)+1.1*(x[0]<=0.5)*(x[1]<=0.5)+0.35*(x[0]>0.5)*(x[1]<=0.5);
        if (component == 0)
            return rho;
        else if (component == 1)
            return rho*u;  //rho*u
        else if (component == dim + 1)
       {
            double gamma = 1.4;
            double energy = (pressure/(gamma-1)) + (rho*(u*u + v*v)*0.5);
            return energy;
        }
         else
            return rho*v;

    }
    default:
        Assert(false, ExcNotImplemented());
            return 0.;
    }
}

template <int dim>
double BoundaryData<dim>::value(const dealii::Point<dim> & x,
                                const unsigned int component) const
{
    const double t = this->get_time();

    switch (parameters.testcase) {
    case 1: { //channel with hole
        if (component == 0)
            return 1.;
        else if (component == 1)
            return 0.4;
        else if (component == dim + 1)
            return 3.097857142857143;
        else
            return 0.;
    }
    case 2: { //b-step
        if (component == 0)
            return 7.0406;
        else if (component == 1)
            return 28.72565;  // u = 4.08
        else if (component == dim + 1)
            return 133.74782;  // P = 30.059
        else
            return 0.;
    }
    case 3: { //f-step
        if (component == 0)
            return 1.4;
        else if (component == 1)
            return 3*1.4;  // u = 3
        else if (component == dim + 1)
            return 8.8; // P = 1
        else
            return 0.;
    }
    case 4: { //sod shock tube

        if (component == 0)
            return 1.0;
        else if (component == 1)
            return 0.;
        else if (component == dim + 1)
            return 2.5;
        else
            return 0.;

    }
    case 5: { //cylinder
        if (component == 0)
            return 1.4;
        else if (component == 1)
            return 3.0*1.4;
        else if (component == dim + 1)
        {   double gamma = 1.4;
            double pressure = 1.0;
            double energy = (pressure/(gamma-1)) + ((1.4*3.0*3.0)*0.5);
            return energy;
        }
        else
            return 0.;
        }
    case 6: { //double mach reflection
    if (a==0){ //left
           if (component == 0)
               return 8.0; //density
           else if (component == 1)
               return 57.1576766498;  //density * 8.25 * cos(pi/6)
           else if (component == 2)
               return -33.0;          //-density * 8.25 * sin(pi/6)
           else {
               double pressure = 116.5;
               const double gamma = 1.4;
               double energy_post_shock = pressure / (gamma- 1.) +
                       0.5 * 8.0 * ( 8.25 * 8.25);
               return  energy_post_shock;
}

    }
           else if (a == 1){ // top
               if (component == 0)
               return 8.0*(x[0]<(1.0/6.0+((1+20*t)/sqrt(3)))) + 1.4*(x[0]>=(1.0/6.0+((1+20*t)/sqrt(3))));
               else if (component == 1)
               return 57.1576766498*(x[0]<(1.0/6.0+((1+20*t)/sqrt(3)))) + 0.0*(x[0]>=(1.0/6.0+((1+20*t)/sqrt(3))));
               else if (component == 2)
               return -33.0*(x[0]<(1.0/6.0+((1+20*t)/sqrt(3)))) + 0.0*(x[0]>=(1.0/6.0+((1+20*t)/sqrt(3))));
               else {
                   double pressure = 116.5;
                   const double gamma = 1.4;
                   double energy_post_shock = pressure / (gamma- 1.) +
                           0.5 * 8.0 * (8.25 * 8.25);
                   return  energy_post_shock*(x[0]<(1.0/6.0+((1+20*t)/sqrt(3)))) + 2.5*(x[0]>=(1.0/6.0+((1+20*t)/sqrt(3))));
           }
    }



    }
    case 7: { //2D Riemann problem
        double rho;
        double u;
        double v;
        double p;
        if(a==0)
        {
            rho = 1.1 ;
            u = 0.0;
            v = 0.0;
            p =1.;
        }
        else if(a==1)
        {
            rho = 0.5065 ;
            u = 0.8939;
            v = 0.0;
            p =0.35;
        }
        else if ( a==2)
        {
            rho = 1.1 ;
            u = 0.8939;
            v = 0.8939;
            p =1.1;
        }
        else //a==3
        {
            rho = 0.5065 ;
            u = 0.0;
            v = 0.8939;
            p =0.35;
        }
        if (component == 0)
            return rho;
        else if (component == 1)
            return rho*u;
        else if (component == dim + 1)
       {
            double gamma = 1.4;
            double energy = (p/(gamma-1)) + (rho*(u*u + v*v)*0.5);
            return energy;
        }
         else
            return rho*v;

    }
    default:
        Assert(false, ExcNotImplemented());
            return 0.;
    }
}
}

template class EquationData::BoundaryData<2>;
template class EquationData::InitialData<2>;
template class EquationData::ExactSolution<2>;

