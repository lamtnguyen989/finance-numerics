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
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>


#include <iostream>
#include <fstream>
#include <string>

#define EPSILON 1e-14

using namespace dealii;

/* ------------------------------------------------------------------------------
    Black-Scholes paramaters
--------------------------------------------------------------------------------- */
struct Black_Scholes_parameters
{
    // Standard parameters
    double r = 0.1;         // Risk-free rate
    double sigma = 0.01;    // Volatility
    double S_min = 50;      // Minimum asset price
    double S_max = 150;     // Maximum asset price
    double K = 100;         // Strike price
    double t_0 = 0;         // Initial time
    double T = 0.25;        // Expiry time

    // Transformation parameters
    double x_min = std::log(S_min/K);
    double x_max = std::log(S_max/K);
    double tau_0 = 0;
    double tau_final = std::pow(sigma,2)/2.0 * (T - t_0);
    double alpha = 1.0/2.0 - r/std::pow(sigma,2);
    double beta = std::pow(alpha, 2) + alpha*(2*r/std::pow(sigma,2) - 1) - 2*r/std::pow(sigma,2);

    // Discretization parameters
    unsigned int n_price_cells = 100;
    unsigned int n_time_steps = 300;
};

/* ------------------------------------------------------------------------------
    Conditions of the problem (Vanilla European call options here)

    ** Computed under transformation to Heat equation: x = ln(S/K) and \tau = (\sigma^2/2)(T-t)
    ** This is used in conjuction with Ansatz V(S,t) = e^{\alpha*x + beta*\tau}u(x,\tau)
--------------------------------------------------------------------------------- */

// Original Terminal condition: V(S,T) = max(S-K, 0)
// Initial Condition under Heat equation transformation: u(x,0) = Ke^{-\alpha x}\max(e^x-1 , 0)
template <int dim>
class InitialCondition : public Function<dim> 
{
    public:
        InitialCondition(Black_Scholes_parameters &parameters)
            : Function<dim> ()
            , params(parameters)
        {}

        virtual double value(const Point<dim> &p, const unsigned int /*component*/) const override
        {
            const double x = p[0];
            return params.K * std::max(std::exp(x)-1.0, 0.0) * std::exp(-params.alpha*x);
        }

    private:
        Black_Scholes_parameters params;
};

// Original Max Price Condition (discount): V(S_max, t) = S_max - Ke^{-r(T-t)}
// Under Heat equation transformation: u(x_max, \tau) = e^{\alpha*x_max - \beta\tau}(S_max - Ke^{-r(2\tau/sigma^2)})
template <int dim>
class MaxPriceCondition : public Function<dim>
{
    public:
        MaxPriceCondition(Black_Scholes_parameters &parameters, double tau)
            : Function<dim> ()
            , params(parameters)
            , tau(tau)
        {}

        virtual double value(const Point<dim> &p, const unsigned int /*component*/) const override
        {
            double x_max = p[0];
            return std::exp(params.alpha*x_max - params.beta*tau) * (params.S_max - params.K*std::exp(-params.r*(2*tau/std::pow(params.sigma,2))));
        }
        
    private:
        Black_Scholes_parameters params;
        double tau;
};


// Min Price condition: V(S_min, t) = 0
// No change under transformation: u(x_min, \tau) = 0
template <int dim>
class MinPriceCondition : public Function<dim>
{
    public: 
        MinPriceCondition(Black_Scholes_parameters &parameters)
            : Function<dim> ()
            , params(parameters)
        {}

        virtual double value(const Point<dim> &p, const unsigned int /*component*/) const override
        {
            return 0.0 * p[0];  // Shutting up the compiler (hopefully this wouldn't add run time due to -O3)
        }

    private:
        Black_Scholes_parameters params;
};

/* ------------------------------------------------------------------------------
    Solver object
--------------------------------------------------------------------------------- */
template <int dim>
class BlackScholes
{
    public:
        BlackScholes(Black_Scholes_parameters &parameters, const unsigned int deg, const unsigned int refinement_cycles);
        void run();
    private:
        void setup_system();
        void apply_initial_condition();
        void assemble_mass_and_stiffness_matrices();

        void apply_boundary_conditions(double tau);
        void SDIRK_2_solve();
        void transform_solution();
        void output_timestep(double tau);

        // Solver paramters
        Black_Scholes_parameters params;
        unsigned int degree;
        unsigned int n_refinement_cycles;
        unsigned int cycle;

        // Finite element components
        FE_Q<dim>                   fe;
        Triangulation<dim>          triangulation;
        DoFHandler<dim>             dof_handler;

        // Constraints and matrices components
        AffineConstraints<double>   constraints;
        SparseMatrix<double>        mass_matrix;
        SparseMatrix<double>        stiffness_matrix;
        SparseMatrix<double>        system_matrix;
        DynamicSparsityPattern      dsp;
        SparsityPattern             sparsity_pattern;

        // Solution and RHS containers
        Vector<double>              solution;
        Vector<double>              old_solution;
        std::vector<Vector<double>> all_solutions;
        Vector<double>              rhs;
};

template <int dim>
BlackScholes<dim>::BlackScholes(Black_Scholes_parameters &parameters, const unsigned int deg, const unsigned int refinement_cycles)
    : params(parameters)
    , degree(deg)
    , n_refinement_cycles(refinement_cycles)
    , cycle(0)
    , fe(deg)
    , dof_handler(triangulation)
{}

template <int dim>
void BlackScholes<dim>::setup_system()
{
    // Distribute DoFs
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
    std::cout << "Discretized with: " << params.n_price_cells << " cells." << std::endl;
    std::cout << "Actual active cells number: " << triangulation.n_active_cells() << std::endl;
    std::cout << "Number of DoFs: " << n_dofs << std::endl;
}

template <int dim>
void BlackScholes<dim>::apply_initial_condition()
{
    // Interpolating terminal condition
    InitialCondition<dim> initial_condition(params);
    VectorTools::interpolate(dof_handler, initial_condition, solution);

    // Storing the interpolated condition
    all_solutions.push_back(solution);
    old_solution = solution;

    // Output
    //output_timestep(params.T);
}

template <int dim>
void BlackScholes<dim>::assemble_mass_and_stiffness_matrices()
{
    // Zero out the matrices
    mass_matrix = 0;
    stiffness_matrix = 0;

    // Initializing quadrature rules and Finite Element values
    QGauss<dim> quadrature(degree + 2);
    FEValues<dim> fe_values(fe, quadrature,
                    update_values | update_gradients | update_quadrature_points | update_JxW_values | update_hessians);
    
    // Looping through cells to assemble the matrices
    FullMatrix<double> cell_mass_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
    FullMatrix<double> cell_stiffness_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
    std::vector<unsigned int> local_dof_indices(fe.dofs_per_cell);
    
    for (auto cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell_mass_matrix = 0;
        cell_stiffness_matrix = 0;

        for (unsigned int q_index = 0; q_index < quadrature.size(); q_index++)
        {
            for (unsigned int i = 0; i< fe.dofs_per_cell; i++)
            {
                for (unsigned int j = 0; j < fe.dofs_per_cell; j++)
                {
                    /*  Mass matrix terms calculation */
                    cell_mass_matrix(i, j) += fe_values.shape_value(i,q_index) * fe_values.shape_value(j,q_index) * fe_values.JxW(q_index);
                    
                    /* Stiffness matrix */
                    cell_stiffness_matrix(i, j) += fe_values.shape_grad(i,q_index)[0] * fe_values.shape_grad(j,q_index)[0] * fe_values.JxW(q_index);
                }
            }
        }

        // Distribute the calculated result to the global system matrices
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_mass_matrix, local_dof_indices, mass_matrix);
        constraints.distribute_local_to_global(cell_stiffness_matrix, local_dof_indices, stiffness_matrix);
    }
    //std::cout << "Mass and Stiffness matrices assembly finished." << std::endl;
}

template <int dim>
void BlackScholes<dim>::apply_boundary_conditions(double tau)
{

}

template <int dim>
void BlackScholes<dim>::SDIRK_2_solve()
{

}

template <int dim>
void BlackScholes<dim>::transform_solution()
{

}

template <int dim>
void BlackScholes<dim>::output_timestep(double tau)
{

}


template <int dim>
void BlackScholes<dim>::run()
{
    // Setup Grid
    GridGenerator::subdivided_hyper_cube(triangulation, params.n_price_cells, params.x_min, params.x_max);

    setup_system();
    apply_initial_condition();
    assemble_mass_and_stiffness_matrices();
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
        const unsigned int refinement_cycles = 1;

        BlackScholes<1> Black_Scholes_solver(parameters, degree, refinement_cycles);
        Black_Scholes_solver.run();
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