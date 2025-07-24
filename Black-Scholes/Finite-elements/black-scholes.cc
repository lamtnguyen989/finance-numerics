#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/numerics/derivative_approximation.h>

#include <iostream>
#include <fstream>

using namespace dealii;

/* ------------------------------------------------------------------------------
    Black-Scholes paramaters
--------------------------------------------------------------------------------- */
struct Black_Scholes_param
{
    double r = 0.1;         // Risk-free rate
    double sigma = 0.01;    // Volatility
    double S_min = 50;      // Minimum asset price
    double S_max = 150;     // Maximum asset price
    double K = 100;         // Strike price
    double t_0 = 0;         // Initial time
    double T = 1.0;         // Expiry time

    unsigned int n_price_cells = 100;
    unsigned int n_time_steps = 150;
};

/* ------------------------------------------------------------------------------
    Conditions of the problem (Vanilla European call options here)
--------------------------------------------------------------------------------- */
/*
template <int dim>
class TerminalCondition : public Function<dim>
{

};

template <int dim>
class MaxPriceCondition : public Function<dim>
{

};

template <int dim>
class MinPriceCondition : public Function<dim>
{

};
*/

/* ------------------------------------------------------------------------------
    Solver object
--------------------------------------------------------------------------------- */
template <int dim>
class BlackScholes
{
    public:
        BlackScholes(Black_Scholes_param parameters);
        void run();
    private:
        Black_Scholes_param param;
};

template <int dim>
BlackScholes<dim>::BlackScholes(Black_Scholes_param parameters)
    : param(parameters)
{}


template <int dim>
void BlackScholes<dim>::run()
{

}

/* ------------------------------------------------------------------------------
    main()
--------------------------------------------------------------------------------- */
int main()
{
    try
    {
        Black_Scholes_param parameters;
        BlackScholes<1> solver(parameters);
        solver.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
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
        std::cerr << std::endl << std::endl
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