#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>


#include <iostream>
#include <fstream>

using namespace dealii;

/* ------------------------------------------------------------------------------
    Black-Scholes paramaters
--------------------------------------------------------------------------------- */
struct Black_Scholes_parameters
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

template <int dim>
class TerminalCondition : public Function<dim> 
{
    public:
        TerminalCondition(Black_Scholes_parameters &parameters)
            : Function<dim> ()
            , params(parameters)
        {}

        virtual double value(const Point<dim> p, const unsigned int /*component*/) const override
        {
            const double S = p[0];
            return std::max(S - params.K, 0.0);
        }

    private:
        Black_Scholes_parameters params;
};
/*
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
        BlackScholes(Black_Scholes_parameters &parameters, const unsigned int deg);
        void run();
    private:
        void make_and_setup_system();
        void applying_terminal_condition();
        void assemble_system_matrices();

        Black_Scholes_parameters params;
        const unsigned int degree;

        FE_Q<dim>                   fe;
        Triangulation<dim>          triangulation;
        DoFHandler<dim>             dof_handler;


        AffineConstraints<double>   constraints;
        SparseMatrix<double>        mass_matrix;
        SparseMatrix<double>        stiffness_matrix;
        SparseMatrix<double>        system_matrix;
        DynamicSparsityPattern      dsp;
        SparsityPattern             sparsity_pattern;

        Vector<double>              solution;
        Vector<double>              old_solution;
        Vector<double>              rhs;
};

template <int dim>
BlackScholes<dim>::BlackScholes(Black_Scholes_parameters &parameters, const unsigned int deg)
    : params(parameters)
    , degree(deg)
    , fe(deg)
    , dof_handler(triangulation)
{}

template <int dim>
void BlackScholes<dim>::make_and_setup_system()
{
    // Making the 1d spatial grid and distributes DoFs
    GridGenerator::subdivided_hyper_cube(triangulation, params.n_price_cells, params.S_min, params.S_max);
    dof_handler.distribute_dofs(fe);

    // Make and copy Sparsity pattern (with initializing the constraints)
    constraints.clear();
    constraints.close();
    unsigned int n_dofs = dof_handler.n_dofs();  // Note that we are not doing anything funky here so the number of DoFs should be globally constant
    dsp.reinit(n_dofs, n_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    sparsity_pattern.copy_from(dsp);

    // (Re)initialize matrices
    mass_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    // (Re)initialize solution containers
    solution.reinit(n_dofs);
    old_solution.reinit(n_dofs);
    rhs.reinit(n_dofs);

    // Output discretization info
    std::cout << "Price range: [" << params.S_min << ", " << params.S_max << "]" << std::endl;
    std::cout << "Discretized with: " << params.n_price_cells << std::endl;
    std::cout << "Actual active cells number: " << triangulation.n_active_cells() << std::endl;
    std::cout << "Number of DoFs: " << triangulation.n_active_cells() << std::endl;
}

template <int dim>
void BlackScholes<dim>::applying_terminal_condition()
{

}

template <int dim>
void BlackScholes<dim>::assemble_system_matrices()
{
    
}


template <int dim>
void BlackScholes<dim>::run()
{
    make_and_setup_system();
}

/* ------------------------------------------------------------------------------
    main()
--------------------------------------------------------------------------------- */
int main()
{
    try
    {
        Black_Scholes_parameters parameters;
        const unsigned int degree = 1;

        BlackScholes<1> solver(parameters, degree);
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