#include"parameters.h"

using namespace dealii;


namespace Parameters {

void Solver::declare_parameters (ParameterHandler &prm)
   {
      prm.enter_subsection("Param");
      {
          prm.declare_entry("EulerNumericalFlux", "harten_lax_vanleer",
                            Patterns::Selection("lax_friedrichs|harten_lax_vanleer|SLAU|hllc_centered|HLLC|roe"),
                            "Choices are <lax_friedrichs|harten_lax_vanleer|hllc_centered|SLAU|HLLC|roe");

      }
      prm.leave_subsection();
   }


void Solver::parse_parameters (ParameterHandler &prm)
   {
      prm.enter_subsection("Param");
      {
         const std::string flux = prm.get("EulerNumericalFlux");
         if(flux == "lax_friedrichs")
            numerical_flux_type = lax_friedrichs;
         else if(flux == "harten_lax_vanleer")
            numerical_flux_type = harten_lax_vanleer;
         else if(flux == "hllc_centered")
            numerical_flux_type = hllc_centered;
         else if(flux == "HLLC")
            numerical_flux_type = HLLC;
         else if(flux == "SLAU")
            numerical_flux_type = SLAU;
         else if(flux == "roe")
            numerical_flux_type = roe;
         else
            AssertThrow (false, ExcNotImplemented());


      }
      prm.leave_subsection();
   }


Data_Storage::Data_Storage(){

    prm.enter_subsection("Physical data");
    {
        prm.declare_entry("testcase",
                          "4",
                          Patterns::Integer(),
                          "Choices are : 1|2|3|4|5|6|7");
        prm.declare_entry("fe_degree",
                          "5",
                          Patterns::Integer(),
                          " polynomial degree ");
        prm.declare_entry("fe_degree_Q0",
                          "0",
                          Patterns::Integer(),
                          "polynomial degree for Q0 solution ");
        prm.declare_entry("n_q_points_1d",
                          "7",
                          Patterns::Integer(),
                          " number of points in the Gaussian quadrature formula");
         prm.declare_entry("n_q_points_1d_Q0",
                          "2",
                          Patterns::Integer(),
                          " number of points in the Gaussian quadrature formula when polynomial degree is zero ");
        prm.declare_entry("final_time",
                           "0.2",
                           Patterns::Double(),
                           " The final time of the simulation. ");
        prm.declare_entry("gamma",
                         "1.4",
                         Patterns::Double(),
                          " adiabatic constant dependent on the type of the gas ");

   }
     prm.leave_subsection();

    prm.enter_subsection("Space discretization");
    {
      prm.declare_entry("max_loc_refinements",
                        "4",
                         Patterns::Integer(1,5),
                         " The number of maximum local refinements. ");
      prm.declare_entry("min_loc_refinements",
                        "2",
                         Patterns::Integer(0,5),
                         " The number of minimum local refinements. ");
    }
    prm.leave_subsection();
    prm.enter_subsection("Limiter");
    {
        prm.declare_entry("beta_density",
                          "12.",
                          Patterns::Double(0.0),
                          " tolerance for density component used in filter technique");
        prm.declare_entry("beta_momentum",
                          "20.",
                          Patterns::Double(0.0),
                          " tolerance for momentum component used in filter technique ");
        prm.declare_entry("beta_energy",
                          "20.",
                          Patterns::Double(0.0),
                          "tolerance for energycomponent used in filter technique");
        prm.declare_entry("beta",
                          "1.",
                          Patterns::Double(1.0,2.0),
                          " TVB limiter parameter ");
        prm.declare_entry("M",
                          "0.0",
                          Patterns::Double(),
                          " TVB parameter ");
        prm.declare_entry("positivity",
                          "false",
                          Patterns::Bool(),
                          " whether to use positivity limiter ");
        prm.declare_entry("type",
                          "TVB",
                          Patterns::Selection("TVB|filter"),
                          " type of limiter : filter technique or TVB limiter");
        prm.declare_entry("function_limiter",
                          "minmod",
                          Patterns::Selection("minmod|vanAlbada"),
                          " type of slope limiter function : minmod function or vanAlbada function ");
  }
    prm.leave_subsection();
  prm.declare_entry("output_tick",
                      "0.005",
                      Patterns::Double(0,1.),
                      " This indicates between how many time steps we print the solution. ");
  prm.declare_entry("n_stages",
                      "3",
                      Patterns::Integer(2,4),
                      " number of stages in SSP runge kutta ");
  prm.declare_entry("refine",
                    "true",
                    Patterns::Bool(),
                    "true do refine, false not do refine");
  prm.declare_entry("refine_indicator",
                    "gradient_density",
                    Patterns::Selection("gradient_density|gradient_pressure"),
                    " do refine with density indicator or do refine with pressure indicator");

    Parameters::Solver::declare_parameters(prm);
}


/// Function to read all declared parameters
void Data_Storage::read_data(const std::string& filename)
{
    std::ifstream file(filename);
   AssertThrow(file, ExcFileNotOpen(filename));

  prm.parse_input(file);

  prm.enter_subsection("Physical data");
  {
      testcase =prm.get_integer("testcase");
    fe_degree = prm.get_integer("fe_degree");
    fe_degree_Q0    = prm.get_integer("fe_degree_Q0");
    n_q_points_1d = prm.get_integer("n_q_points_1d");
    n_q_points_1d_Q0 = prm.get_integer("n_q_points_1d_Q0");
    final_time   = prm.get_double("final_time");
    gamma      = prm.get_double("gamma");
  }
  prm.leave_subsection();

  prm.enter_subsection("Space discretization");
  {
    max_loc_refinements = prm.get_integer("max_loc_refinements");
    min_loc_refinements = prm.get_integer("min_loc_refinements");

  }
  prm.leave_subsection();

  prm.enter_subsection("Limiter");
  {
      beta_density = prm.get_double("beta_density");
      beta_momentum = prm.get_double("beta_momentum");
      beta_energy = prm.get_double("beta_energy");
      beta = prm.get_double("beta");
      positivity = prm.get_bool("positivity");
      type = prm.get("type");
      M = prm.get_double("M");
      function_limiter = prm.get("function_limiter");
  }
  prm.leave_subsection();
  output_tick = prm.get_double("output_tick");
  refine = prm.get_bool("refine");
  refine_indicator = prm.get("refine_indicator");
  n_stages= prm.get_integer("n_stages");
 Parameters::Solver::parse_parameters(prm);


}
}

