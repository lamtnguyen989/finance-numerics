#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
//#include <unistd.h>

/* Macros */
using Complex = Kokkos::complex<double>;
using exec_space = Kokkos::DefaultExecutionSpace;
#define i Complex(0.0, 1.0) // Can't use `using` to device code conflict
#define PI 3.141592653589793
#define EPSILON 5e-15
#define HUGE 1e6
#define square(x) (x*x)


// ------------------------------------------------------------------------------------ //
/* Heston model parameters */
struct HestonParameters
{
    double v0;       // Initial variance
    double kappa;    // Mean-reversing variance process factor
    double theta;    // Long-term variance
    double rho;      // Correlation
    double sigma;    // Vol of vol
};

// ------------------------------------------------------------------------------------ //
/* Parameter bound for calibrating */
struct ParameterBounds 
{
    /* Parameters bounds */
    double v0_min, v0_max;
    double kappa_min, kappa_max;
    double theta_min, theta_max;
    double rho_min, rho_max;
    double sigma_min, sigma_max;

    /* Generation bounds (If I want to incorporate later) */

    /* Hard-setting default bounds */
    ParameterBounds()
        : v0_min(0.01) , v0_max(0.1)
        , kappa_min(0.5) , kappa_max(10.0)
        , theta_min(0.05) , theta_max(0.8)
        , rho_min(-0.999) , rho_max(0.999)
        , sigma_min(0.05) , sigma_max(0.8)
    {}
};

// ------------------------------------------------------------------------------------ //
/* Differential Evolution configurations */
struct Diff_EV_config
{
    unsigned int population_size;   // The total population size
    double crossover_prob;          // Cross-over probability
    double weight;                  // Differential weight
    unsigned int n_gen;             // Max iteration
    double tolerance;               // Conergence threshold

    Diff_EV_config(unsigned int NP, double CR, double w, unsigned int max_gen, double tol)
        : population_size(NP) , crossover_prob(CR) , weight(w) 
        , n_gen(max_gen) , tolerance(tol)
        {}

    Diff_EV_config()    // Wikipedia suggested parameters for empty constructor
        : population_size(50) , crossover_prob(0.9) , weight(0.8)
        , n_gen(100) , tolerance(EPSILON)
        {}
};

// ------------------------------------------------------------------------------------ //
/* FFT solver object */
class Heston_FFT
{
    public:
        /* Constructors */
        Heston_FFT(double S_0, double r, double t, HestonParameters heston_params)
            : S_0(S_0), r(r), t(t)
            , params(heston_params)
            , alpha(1.5)
        {}

        Heston_FFT(double S_0, double r, double t, HestonParameters heston_params, double alpha)
            : S_0(S_0), r(r), t(t)
            , params(heston_params)
            , alpha(alpha)
        {}

        /* Pricing methods */
        Kokkos::View<double*> call_prices(Kokkos::View<double*> strikes, unsigned int grid_points, bool print) const;
        Kokkos::View<double*> put_prices(Kokkos::View<double*> strikes, unsigned int grid_points, bool print) const;
        Kokkos::View<double*> black_scholes_call(Kokkos::View<double*> strikes, double sigma, bool print);

        /* Modify model methods */
        KOKKOS_INLINE_FUNCTION void update_option_condition(double updated_S0, double updated_r, double updated_T) { S_0 = updated_S0; r = updated_r;  t = updated_T;}
        KOKKOS_INLINE_FUNCTION void update_parameters(HestonParameters updatedParams) const { params = updatedParams;}
        KOKKOS_INLINE_FUNCTION HestonParameters get_params() {return params;}

        /* Implied volatility */
        Kokkos::View<double*> implied_volatility(Kokkos::View<double*> call_prices, Kokkos::View<double*> K, unsigned int max_iter, double epsilon, bool print);
        Kokkos::View<double*> market_vega(Kokkos::View<double*> call_prices, Kokkos::View<double*> K, unsigned int max_iter, double epsilon, bool print);

        /* Loss functions for calibration */
        double price_vega_weighted_loss(Kokkos::View<double*> call_prices, Kokkos::View<double*> K);
        double price_sq_loss(Kokkos::View<double*> call_prices, Kokkos::View<double*> K);
        double implied_vol_sq_loss(Kokkos::View<double*> call_prices, Kokkos::View<double*> K);

        /* Calibration and optimization */
        HestonParameters diff_EV_calibration(Kokkos::View<double*> prices, Kokkos::View<double*> K, 
                                            ParameterBounds bounds, Diff_EV_config config,
                                            unsigned int seed);


    private:
        /* Data fields */
        double S_0;                         // Initial price
        double r;                           // Risk-free rate
        double t;                           // Expiration time (In hindsight, it should be named T)
        mutable HestonParameters params;    // Parameters
        double alpha;                       // Dampening factor within FFT

        /* FFT pricing helpers */
        KOKKOS_INLINE_FUNCTION Complex heston_characteristic(Complex u) const;
        KOKKOS_INLINE_FUNCTION Complex damped_call(Complex v) const;

        /* Black-Scholes formula related stuff for implied volatility */
        KOKKOS_INLINE_FUNCTION double std_normal_cdf(double x) const    {return 0.5 * (1 + Kokkos::erf(x * 0.70710678118654752));}
        KOKKOS_INLINE_FUNCTION double std_normal_dist(double x) const   {return 0.39894228040143268 * Kokkos::exp(-0.5*x*x);}
        KOKKOS_INLINE_FUNCTION double _black_scholes(double S, double K, double sigma, double tau) const;
        KOKKOS_INLINE_FUNCTION double _vega(double S, double K, double sigma, double tau) const;
};


HestonParameters Heston_FFT::diff_EV_calibration(Kokkos::View<double*> call_prices, Kokkos::View<double*> K, 
                                    ParameterBounds bounds=ParameterBounds(), Diff_EV_config config=Diff_EV_config(),
                                    unsigned int seed=12345)
{
    // Variables
    unsigned int population_size = config.population_size; 
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    Kokkos::View<HestonParameters*> population("population", population_size);
    Kokkos::View<double*> losses("calibration_losses", population_size);

    // Initialize population
    Kokkos::parallel_for("init_population", population_size, 
        KOKKOS_CLASS_LAMBDA(unsigned int k) {

            // Initialize random state
            Kokkos::Random_XorShift64_Pool<>::generator_type generator = rand_pool.get_state();

            // Initialize population randomly
            population(k).v0 = generator.drand()*(bounds.v0_max - bounds.v0_min);
            population(k).kappa = generator.drand()*(bounds.kappa_max - bounds.kappa_min);
            population(k).theta = generator.drand()*(bounds.theta_max - bounds.theta_min);
            population(k).rho = generator.drand()*(bounds.rho_max - bounds.rho_min);
            population(k).sigma = generator.drand()*(bounds.sigma_max - bounds.sigma_min);

            // Free the random state 
            rand_pool.free_state(generator);
        });
    Kokkos::fence();
    
    // Evaluate inital population performance (serially due to `warning #20011-D`)
    for (unsigned int k = 0; k < population_size; k++)
    {
        this->update_parameters(population(k));
        this->price_sq_loss(call_prices, K);
    }

    // Mutation
    Kokkos::parallel_for("mutation", population_size,
        KOKKOS_CLASS_LAMBDA(unsigned int k) {

            // Initialize random state
            Kokkos::Random_XorShift64_Pool<>::generator_type generator = rand_pool.get_state();

            // Pick 3 distinct random candidate indices (wasting compute cycles for now but just making sure the algorithm is correct)
            int a, b, c;
            do {a = static_cast<unsigned int>(generator.drand()*population_size);} while (a == k);
            do {b = static_cast<unsigned int>(generator.drand()*population_size);} while (a == k || a == b);
            do {c = static_cast<unsigned int>(generator.drand()*population_size);} while (a == k || a == b || b == c);

            // Random dimensionality index (5 Heston parameters)
            unsigned int R = static_cast<unsigned int>(generator.drand() * 5);

            // Mutation loop
            HestonParameters potential;
            for (unsigned int j = 0; j < 5; j++) {

                // Random floating point to compare againt the crossover probablity
                double r_j = generator.drand(0,1);

                if (r_j < config.crossover_prob || j == R) {
                    switch (j) {
                        case 0: potential.v0 = population(a).v0 + config.weight*(population(b).v0 - population(c).v0); break;
                        case 1: potential.kappa = population(a).kappa + config.weight*(population(b).kappa - population(c).kappa); break;
                        case 2: potential.theta = population(a).theta + config.weight*(population(b).theta - population(c).theta); break;
                        case 3: potential.rho = population(a).rho + config.weight*(population(b).rho - population(c).rho); break;
                        case 4: potential.sigma = population(a).sigma + config.weight*(population(b).sigma - population(c).sigma); break;
                    }
                } else {
                    switch (j) {
                        case 0: potential.v0 = population(k).v0; break;
                        case 1: potential.kappa = population(k).kappa; break;
                        case 2: potential.theta = population(k).theta; break;
                        case 3: potential.rho = population(k).rho; break;
                        case 4: potential.sigma = population(k).sigma; break;
                    }
                }
            }

            // Compare current parameter with the candidate one
            

            // Free the random state 
            rand_pool.free_state(generator);
        });

    // TODO
    return HestonParameters();
}


/* Heston Characteristic functions */
KOKKOS_INLINE_FUNCTION Complex Heston_FFT::heston_characteristic(Complex u) const
{
    // A bunch of repeated constants in the calculation
    Complex xi = params.kappa - i*params.rho*params.sigma*u;
    Complex d = Kokkos::sqrt(square(xi) + square(params.sigma)*(square(u) + i*u));
    if (Kokkos::real(d) < 0.0) { d = -d;}
    //Complex g_1 = (xi + d) / (xi - d);
    Complex g_2 = (xi - d) / (xi + d);

    // Safe guard the fraction within the exponential terms 
    Complex log_arg_frac = Kokkos::abs(1.0 - g_2) > EPSILON 
                            ? (1.0 - g_2*Kokkos::exp(-d*t)) / (1.0 - g_2) 
                            : Complex(EPSILON, EPSILON);

    Complex last_term_frac = Kokkos::abs(1.0 - g_2*Kokkos::exp(-d*t)) > EPSILON
                            ? (1.0 - Kokkos::exp(-d*t)) / (1.0 - g_2*Kokkos::exp(-d*t))
                            : Complex(EPSILON, EPSILON);

    // Exponent
    Complex exponent = i*u*(Kokkos::log(S_0) + r*t)
                    + (params.kappa*params.theta)/square(params.sigma) * ((xi -d)*t - 2.0*Kokkos::log(log_arg_frac))
                    + (params.v0/square(params.sigma))*(xi-d)*(last_term_frac);

    return Kokkos::exp(exponent);
}


/* Damped call price function */
KOKKOS_INLINE_FUNCTION Complex Heston_FFT::damped_call(Complex v) const
{
    Complex phi = heston_characteristic((v - i*(alpha + 1.0)));
    return (Kokkos::exp(-r*t) * phi) / (alpha*alpha + alpha - v*v +i*(2.0*alpha + 1.0)*v);
}


/* Call price computation */
Kokkos::View<double*> Heston_FFT::call_prices(Kokkos::View<double*> strikes, unsigned int grid_points=8192, bool print=false) const
{
    // FFT setup
    double eta = 0.2;   // Step-size in damped Fourier space
    double lambda = (2.0*PI) / (grid_points*eta); // Transformed step size in log-price space
    double bound = 0.5*grid_points*lambda;

    // Compute input for the shifted phased inverse Fourier transform
    Kokkos::View<Complex*> x("Fourier input", grid_points);
    Kokkos::parallel_for("FFT_input", grid_points,
        KOKKOS_CLASS_LAMBDA(const unsigned int k) {
            
            // Compute damped call price
            double v_k = eta*k;
            Complex damped = this->damped_call(Complex(v_k, 0.0));

            // Compute the Simpson quadrature weights
            double w_k;
            if (k == 0 || k == grid_points-1)   { w_k = 1.0/3.0;}
            else if (k % 2 == 1)                { w_k = 4.0/3.0;}
            else                                { w_k = 2.0/3.0;}

            // Modify the damped call price to get the input
            x(k) = Kokkos::exp(-i*bound*v_k) * damped * eta * w_k; 
        }
    );

    // FFT
    Kokkos::View<Complex*> x_hat("x_hat", grid_points);
    KokkosFFT::fft(exec_space(), x, x_hat);

    // Undamped and compute the option prices
    Kokkos::View<double*> prices("prices", grid_points);
    Kokkos::parallel_for("extract_prices",  strikes.extent(0),
        KOKKOS_CLASS_LAMBDA(const unsigned int k) {

            // Find the Fourier grid price index
            double log_K = Kokkos::log(strikes(k));
            int index = static_cast<int>((log_K + bound) / lambda + 0.5);

            // Undamp to get the price at this index
            if ((index < grid_points) && (index >= 0))
                prices(k) = x_hat(index).real()*Kokkos::exp(-alpha*(lambda*index - bound))/PI ;
            else
                prices(k) = 0.0;
        });

    if (print) {
        Kokkos::printf("Strike \t\t Call Price\n");
        Kokkos::printf("-----------------------\n");
        Kokkos::parallel_for("print_result", strikes.extent(0),
            KOKKOS_LAMBDA(unsigned int k){
                Kokkos::printf("%.2lf \t\t %.2lf\n", strikes(k), prices(k));
        });
        Kokkos::fence();
        Kokkos::printf("\n");
    }

    return prices;
}

/* Put price computation from duality */
Kokkos::View<double*> Heston_FFT::put_prices(Kokkos::View<double*> strikes, unsigned int grid_points=8192, bool print=false) const
{
    // Compute call prices 
    Kokkos::View<double*> prices = call_prices(strikes, grid_points, false);

    // Apply parity
    Kokkos::parallel_for("apply_parity", strikes.extent(0),
        KOKKOS_CLASS_LAMBDA(unsigned int k) {
            prices(k) = prices(k) - S_0 + strikes(k) + Kokkos::exp(-r*t);
    });

    // Print if needed
    if (print) {
        Kokkos::printf("Strike \t\t Put Price\n");
        Kokkos::printf("-----------------------\n");
        Kokkos::parallel_for("print_result", strikes.extent(0),
            KOKKOS_LAMBDA(unsigned int k){
                Kokkos::printf("%.2lf \t\t %.2lf\n", strikes(k), prices(k));
        });
        Kokkos::fence();
        Kokkos::printf("\n");
    }

    return prices;
}

/* Black-Scholes related stuff (note that tau is the time to expiration) */
KOKKOS_INLINE_FUNCTION double Heston_FFT::_black_scholes(double S, double K, double sigma, double tau) const
{
    double d_plus = (Kokkos::log(S/K) + tau*(r + 0.5*square(sigma))) / (sigma * Kokkos::sqrt(tau));
    double d_minus = d_plus - sigma*Kokkos::sqrt(tau);
    return S*std_normal_cdf(d_plus) - std_normal_cdf(d_minus)*(K*Kokkos::exp(-r*tau));
}

/* */
Kokkos::View<double*> Heston_FFT::black_scholes_call(Kokkos::View<double*> strikes, double sigma, bool print=false)
{
    Kokkos::View<double*> calls("BS", strikes.extent(0));
    double S = S_0;

    Kokkos::parallel_for("b-s_calls", strikes.extent(0),
        KOKKOS_CLASS_LAMBDA(unsigned int k){
            calls(k) = _black_scholes(S, strikes(k), sigma, t);
        });

    if (print) {
        Kokkos::printf("Strike \t\t BS Call Price\n");
        Kokkos::printf("-----------------------\n");
        Kokkos::parallel_for("print_result", strikes.extent(0),
            KOKKOS_LAMBDA(unsigned int k){
                Kokkos::printf("%.2lf \t\t %.2lf\n", strikes(k), calls(k));
        });
        Kokkos::fence();
        Kokkos::printf("\n");
    }

    return calls;
}

/* Computing Implied volatility from minimization routine of B-S prices */
Kokkos::View<double*> Heston_FFT::implied_volatility(Kokkos::View<double*> call_prices, Kokkos::View<double*> strikes, 
                                                    unsigned int max_iter=10, double epsilon=1e-10, bool print=false)
{
    unsigned int num_options = call_prices.extent(0);
    Kokkos::View<double*> implied_vols("implied_vols", num_options);

    // Capture variables
    double S = S_0;

    Kokkos::parallel_for("compute_iv", num_options, 
        KOKKOS_CLASS_LAMBDA(unsigned int j) {
            double price_diff = HUGE;
            double current_iv = 0.1;
            for (unsigned int iter = 0; iter < max_iter; iter++) {
                
                // Compute the difference between current computed implied vol vs market price 
                price_diff = call_prices(j) - _black_scholes(S, strikes(j), current_iv, t);
                if (Kokkos::abs(price_diff) < epsilon) {break;}

                // Vega computation
                double vega = _vega(S, strikes(j), current_iv, t);
                if (vega < EPSILON) {vega = EPSILON;}

                // Newton update to the volatility
                current_iv += price_diff / vega;
            }

            implied_vols(j) = current_iv;
        });
    
    if (print) {
        Kokkos::printf("Strike \t\t Call Price   Implied Vol \n");
        Kokkos::printf("-----------------------\n");
        Kokkos::parallel_for("print_result", strikes.extent(0),
            KOKKOS_LAMBDA(unsigned int k){
                Kokkos::printf("%.2lf \t\t %.2lf \t\t %.2lf\n", strikes(k), call_prices(k), implied_vols(k));
        });
        Kokkos::fence();
        Kokkos::printf("\n");
    }

    return implied_vols;
}

/* Vega computation */
KOKKOS_INLINE_FUNCTION double Heston_FFT::_vega(double S, double K, double sigma, double tau) const
{
    double d_plus = (Kokkos::log(S/K) + tau*(r + 0.5*square(sigma))) / (sigma * Kokkos::sqrt(tau));
    return S*std_normal_dist(d_plus)*Kokkos::sqrt(tau);
}

Kokkos::View<double*> Heston_FFT::market_vega(Kokkos::View<double*> call_prices, Kokkos::View<double*> K, 
                                            unsigned int max_iter=10, double epsilon=1e-10, bool print=false)
{
    // Capturing variables
    double S = S_0;
    double tau = t;
    unsigned int num_prices = call_prices.extent(0);

    // Note that we can compute vega from the implied vols and these calculations are independent
    Kokkos::View<double*> iv = implied_volatility(call_prices, K, max_iter, epsilon, false);

    Kokkos::View<double*> vega("market_vegas", num_prices);
    Kokkos::parallel_for("compute_market_vega", num_prices,
        KOKKOS_CLASS_LAMBDA(unsigned int k) {
            vega(k) = _vega(S, K(k), iv(k), tau);
        });

    return vega;
}

double Heston_FFT::price_vega_weighted_loss(Kokkos::View<double*> call_prices, Kokkos::View<double*> K)
{
    // Capture variables
    unsigned int num_prices = call_prices.extent(0);
    
    // Computing calls and vegas based on strikes
    Kokkos::View<double*> heston_calls = this->call_prices(K);
    Kokkos::View<double*> vega = market_vega(call_prices, K);

    // Reduce the vega-weighted loss
    double loss;
    Kokkos::parallel_reduce("weighted_vega_loss", num_prices,
        KOKKOS_LAMBDA(unsigned int k, double& local_loss) {
            double price_diff = call_prices(k) - heston_calls(k);
            local_loss += square(price_diff) / square(vega(k));
        }, loss);

    return loss;
}

double Heston_FFT::price_sq_loss(Kokkos::View<double*> call_prices, Kokkos::View<double*> K)
{
    // Capture variables
    Heston_FFT local_model = *this;
    unsigned int num_prices = call_prices.extent(0);

    // Computing calls based on strikes for squared loss
    Kokkos::View<double*> heston_calls = local_model.call_prices(K);

    // Reduce squared loss
    double loss;
    Kokkos::parallel_reduce("squared_vega_loss", num_prices,
        KOKKOS_LAMBDA(unsigned int k, double& local_loss) {
            local_loss += square(call_prices(k) - heston_calls(k));
        }, loss);

    return loss;
}

double Heston_FFT::implied_vol_sq_loss(Kokkos::View<double*> call_prices, Kokkos::View<double*> K)
{
    // Capture variables
    unsigned int num_prices = call_prices.extent(0);

    // Computing calls based on strikes for squared loss
    Kokkos::View<double*> heston_calls = this->call_prices(K);
    Kokkos::View<double*> heston_iv = implied_volatility(heston_calls, K);
    Kokkos::View<double*> market_iv = implied_volatility(call_prices, K);

    // Reduce squared loss
    double loss;
    Kokkos::parallel_reduce("squared_vega_loss", num_prices,
        KOKKOS_LAMBDA(unsigned int k, double& local_loss) {
            local_loss += square(heston_iv(k) - market_iv(k));
        }, loss);

    return loss;
}

// ------------------------------------------------------------------------------------ //
int main(int argc, char* argv[])
{
    // Setting up OpenMP backend for the cases when run on host only
    setenv("OMP_PROC_BIND", "spread", 1);
    setenv("OMP_PLACES", "threads", 1);

    // Everything will be done within the Kokkos environment
    Kokkos::initialize(argc, argv);
    {
        // Pricing parameters
        double r = 0.03;
        double S_0 = 100.0;
        double T = 1.0;
        unsigned int num_strikes = 10;

        // Initial Heston parameters
        HestonParameters hestonParams;
        hestonParams.v0 = 0.04;
        hestonParams.kappa = 2.0;
        hestonParams.theta = 0.04;
        hestonParams.sigma = 0.3;
        hestonParams.rho = -0.7;

        // FFT parameters
        double alpha = 2.0;
        unsigned int N = 8192;

        // Strikes and corresponding price parameters
        Kokkos::View<double*> strikes("strikes", num_strikes);
        Kokkos::parallel_for("fill_strikes", num_strikes,
            KOKKOS_LAMBDA(unsigned int k){
                strikes(k) = 80 + k*5.0;
        });
        
        // Computing the call and put prices
        Heston_FFT solver(S_0, r, T, hestonParams, alpha);
        Kokkos::View<double*> call_prices = solver.call_prices(strikes, N, true);
        Kokkos::View<double*> put_prices = solver.put_prices(strikes, N, true);


        // Implied vols
        double goal_vol = 0.13;
        Kokkos::View<double*> test_prices = solver.black_scholes_call(strikes, goal_vol, true);
        Kokkos::View<double*> iv = solver.implied_volatility(test_prices, strikes, 10, 1e-15, true);
        Kokkos::View<double*> iv_Heston = solver.implied_volatility(call_prices, strikes, 20, 1e-15, true);



    }
    Kokkos::finalize();

    return 0;
}