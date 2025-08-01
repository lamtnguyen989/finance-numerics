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
    double r = 0.1;         // Risk-free rate
    double sigma = 0.01;    // Volatility
    double S_min = 50;      // Minimum asset price
    double S_max = 150;     // Maximum asset price
    double K = 100;         // Strike price
    double t_0 = 0;         // Initial time
    double T = 0.25;        // Expiry time

    unsigned int n_price_cells = 100;
    unsigned int n_time_steps = 300;
};

/* ------------------------------------------------------------------------------
    Conditions of the problem (Vanilla European call options here)
--------------------------------------------------------------------------------- */

// Terminal condition: u(S, T) = max(S-K, 0)
template <int dim>
class TerminalCondition : public Function<dim> 
{
    public:
        TerminalCondition(Black_Scholes_parameters &parameters)
            : Function<dim> ()
            , params(parameters)
        {}

        virtual double value(const Point<dim> &p, const unsigned int /*component*/) const override
        {
            const double S = p[0];
            return std::max(S - params.K, 0.0);
        }

    private:
        Black_Scholes_parameters params;
};

// Max Price Condition (discount): u(S_max, t) = S_max - Ke^{-r(T-t)}
template <int dim>
class MaxPriceCondition : public Function<dim>
{
    public:
        MaxPriceCondition(Black_Scholes_parameters &parameters, double t)
            : Function<dim> ()
            , params(parameters)
            , t(t)
        {}

        virtual double value(const Point<dim> &p, const unsigned int /*component*/) const override
        {
            double S_max = p[0];
            return S_max - params.K * std::exp(-params.r * (params.T - t));
        }
        
    private:
        Black_Scholes_parameters params;
        double t;
};


// Min Price condition: u(S_min, t) = 0
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
        void applying_terminal_condition();
        void assemble_mass_and_stiffness_matrices();
        void apply_boundary_conditions(double t);
        void SDIRK_2_solve();
        void output_timestep(double t);
        void output_time_evolution();

        double compute_stabilization_parameter(double cell_size, double S);

        Black_Scholes_parameters params;
        unsigned int degree;
        unsigned int n_refinement_cycles;
        unsigned int cycle;

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
void BlackScholes<dim>::applying_terminal_condition()
{
    // Interpolating terminal condition
    TerminalCondition<dim> terminal_condition(params);
    VectorTools::interpolate(dof_handler, terminal_condition, solution);

    // Storing the interpolated condition
    all_solutions.push_back(solution);
    old_solution = solution;

    // Output
    //output_timestep(params.T);
}

template <int dim>
double BlackScholes<dim>::compute_stabilization_parameter(double cell_size, double S)
{
    // Computing Peclet number
    double Pe = (cell_size * params.r * S) / (std::pow(params.sigma*S, 2));

    // Stabilization parameter that is "optimal" for Convection-Diffusion equation
    // Note that we have Convection-Diffusion-Reaction in Black-Scholes that we have but for start, this will do

    return (cell_size / (2 * params.r * S)) * (1.0/std::tanh(Pe) + (1.0/Pe));
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

        double cell_size = cell->diameter();

        for (unsigned int q_index = 0; q_index < quadrature.size(); q_index++)
        {
            double S = fe_values.quadrature_point(q_index)[0];
            double eta = compute_stabilization_parameter(cell_size, S);

            for (unsigned int i = 0; i< fe.dofs_per_cell; i++)
            {
                for (unsigned int j = 0; j < fe.dofs_per_cell; j++)
                {
                    /*  Mass matrix terms calculation */
                    cell_mass_matrix(i, j) += fe_values.shape_value(i,q_index) * fe_values.shape_value(j,q_index) * fe_values.JxW(q_index);
                    cell_mass_matrix(i, j) += eta * params.r * S * fe_values.shape_grad(i, q_index)[0] * fe_values.shape_value(j,q_index) * fe_values.JxW(q_index);
                    
                    /* Stiffness matrix */
                    // Diffusion term
                    cell_stiffness_matrix(i, j) += (std::pow(S * params.sigma, 2) )/ 2.0
                                                * fe_values.shape_grad(i,q_index)[0] * fe_values.shape_grad(j,q_index)[0];
                    cell_stiffness_matrix(i, j) += eta*params.r*std::pow(S,3) * fe_values.shape_hessian(j,q_index)[0][0] * fe_values.shape_grad(i, q_index)[0];

                    // Convection term
                    cell_stiffness_matrix(i, j) += params.r*S * fe_values.shape_value(i, q_index) * fe_values.shape_grad(j,q_index)[0];
                    cell_stiffness_matrix(i, j) += std::pow(params.r*S, 2) * eta * fe_values.shape_grad(j,q_index)[0];

                    // Reaction term
                    cell_stiffness_matrix(i, j) -= params.r * fe_values.shape_value(i, q_index) * fe_values.shape_value(j, q_index);
                    cell_stiffness_matrix(i, j) -= std::pow(params.r,2) * eta * S * fe_values.shape_grad(i, q_index)[0];
                    
                    // Differential weights terms multiplication to finish the integrand
                    cell_stiffness_matrix(i, j) *= fe_values.JxW(q_index);
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
void BlackScholes<dim>::apply_boundary_conditions(double t)
{
    std::map<unsigned int, double> boundary_values;
    
    // S_min boundary interpolation
    unsigned int boundary_id_min = 0;
    VectorTools::interpolate_boundary_values(dof_handler, boundary_id_min, MinPriceCondition<dim>(params), boundary_values);

    // S_max boundary interpolation
    unsigned int boundary_id_max = 1;
    VectorTools::interpolate_boundary_values(dof_handler, boundary_id_max, MaxPriceCondition<dim>(params, t), boundary_values);

    // Applying BCs to system_matrix and RHS
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, rhs);
}

template <int dim>
void BlackScholes<dim>::SDIRK_2_solve()
{
    /*
        Coefficient DiffEq system (for mass matrix M and stiffness matrix K): M(du/d_tau) = Ku
        This will be solved with 2-stage SDIRK method which uses the constant \gamma = 1-1/sqrt(2)
        
        Based on Butcher's table (with the fact du/d\tau = du/dt), the system matrix is: A = M - \gamma*d_tau*K
        The stages B_1 and B_2 are calculated from:
            A * B_1 = M * u_{n}
            A * B_2 = K * (u_{n} + \gamma*d_\tau*B_1)
        Update:
            u_{n+1} = u_{n} + d_tau * [(1-\gamma)*B_1 + \gamma * B_2]
    */

    // Time-step size
    double d_tau = (params.T - params.t_0) / (params.n_time_steps);

    // L-stabe constant for SDIRK-2
    double gamma = 1.0 - (std::sqrt(2) / 2.0);

    // Assemble mass and stiffness matrix
    assemble_mass_and_stiffness_matrices();

    // Initialize (direct) solver
    SparseDirectUMFPACK solver;

    // Initialize stages solution vectors
    Vector<double> stage_1(solution.size());
    Vector<double> stage_2(solution.size());

    for (unsigned int n = 0; n < params.n_time_steps; n++)
    {
        // Extracting times
        double current_time = params.T - n*d_tau;
        double next_time = current_time - d_tau;

        // Stage 1
        stiffness_matrix.vmult(rhs, old_solution);
        system_matrix.copy_from(mass_matrix);
        system_matrix.add(-d_tau*gamma, stiffness_matrix);
        solver.initialize(system_matrix);
        apply_boundary_conditions(next_time);
        solver.vmult(stage_1, rhs);

        // Stage 2
        stiffness_matrix.vmult(rhs, old_solution);
        rhs.add(gamma*d_tau, stage_1);
        system_matrix.copy_from(mass_matrix);
        system_matrix.add(-d_tau*gamma, stiffness_matrix);
        apply_boundary_conditions(next_time);
        solver.initialize(system_matrix);
        solver.vmult(stage_2, rhs);

        // Update solution
        solution = old_solution;
        solution.add((1-gamma)*d_tau, stage_1);
        solution.add(gamma*d_tau, stage_2);

        apply_boundary_conditions(next_time);

        // Storing solution
        all_solutions.push_back(solution);
        old_solution = solution;

        // Output last timestep
        if (std::abs(next_time - params.t_0) < 1e-8) {output_timestep(std::abs(next_time));}
    }
    std::cout << "Finished SDIRK-2 time-stepping" << std::endl;
}

template <int dim>
void BlackScholes<dim>::output_timestep(double t)
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "V");
    data_out.build_patches();

    std::string status = "t=" + std::to_string(t) + "_cycle_" + std::to_string(cycle+1);

    std::ofstream output_vtu(status + ".vtu");
    data_out.write_vtu(output_vtu);

    std::ofstream output_plot(status + ".gnuplot");
    data_out.write_gnuplot(output_plot);
}

template <int dim>
void BlackScholes<dim>::output_time_evolution() 
{
    // Create a 2D time-evolution grid in (S,t)
    Triangulation<dim+1> spacetime_triangulation;
    std::vector<unsigned int> repetitions = {triangulation.n_active_cells(), params.n_time_steps};
    Point<dim+1> bottom_left(params.S_min, params.t_0);
    Point<dim+1> top_right(params.S_max, params.T);
    GridGenerator::subdivided_hyper_rectangle(spacetime_triangulation, repetitions, bottom_left, top_right);

    // Distribute DoFs on the evolution grid
    FE_Q<dim+1> spacetime_fe(degree);
    DoFHandler<dim+1> spacetime_dof_handler(spacetime_triangulation);
    spacetime_dof_handler.distribute_dofs(spacetime_fe);

    // Initialize solution vector
    Vector<double> spacetime_solution(spacetime_dof_handler.n_dofs());
    double dt = (params.T - params.t_0) / params.n_time_steps;

    // Interpolating the solution on the grid
    std::vector<Point<dim+1>> support_points(spacetime_dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(MappingQ1<dim+1>(), spacetime_dof_handler, support_points);
    for (unsigned int i = 0; i < spacetime_dof_handler.n_dofs(); ++i)
    {
        // Extracting (S,t) coordinates of the support point
        const Point<dim+1>& point = support_points[i];
        double S = point[0];
        double t = point[1]; 
        
        // Find time-based index within stored-solutions to interpolate
        int time_index = std::round((t - params.t_0) / dt);
        time_index = std::max(0, std::min(time_index, static_cast<int>(all_solutions.size() - 1)));
        
        // Interpolate
        Vector<double> interpolated_value(1);
        VectorTools::point_value(dof_handler, all_solutions[time_index], Point<dim>(S), interpolated_value);
        spacetime_solution(i) = interpolated_value(0);
    }


    // Output the spacetime solution
    DataOut<dim+1> data_out;
    data_out.attach_dof_handler(spacetime_dof_handler);
    data_out.add_data_vector(spacetime_solution, "V");
    data_out.build_patches();

    std::string file = "Black-Scholes-evolution";
    
    std::ofstream output_vtu(file + ".vtu");
    data_out.write_vtu(output_vtu);

    std::ofstream output_plot(file + ".gnuplot");
    data_out.write_gnuplot(output_plot);
    
    std::cout << "Time evolution solution written to: " << file << std::endl;
}

template <int dim>
void BlackScholes<dim>::run()
{
    for (; cycle < n_refinement_cycles; cycle++)
    {
        if (cycle == 0) // Making the 1d spatial grid on first cycle
        {
            GridGenerator::subdivided_hyper_cube(triangulation, params.n_price_cells, params.S_min, params.S_max);
        }
        else    // Refine for subsequent cycles
        {
            Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
            KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(degree + 1), {}, solution,estimated_error_per_cell);
            GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.1, 0.01); 

            triangulation.execute_coarsening_and_refinement();
        }

        setup_system();
        applying_terminal_condition();

        SDIRK_2_solve();

        // Some test-values
        std::vector<double> prices = {75, 100, 125, 150};
        Vector<double> values(1);
        for (unsigned int i = 0; i < prices.size(); i++)
        {
            VectorTools::point_value(dof_handler, solution, Point<dim>(prices[i]), values);
            std::cout << "Option value at " << prices[i] << ": " << values[0] << std::endl;
        }
        if (cycle == 0) {output_time_evolution();}
        std::cout << std::endl;
    }
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
        const unsigned int refinement_cycles = 2;

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