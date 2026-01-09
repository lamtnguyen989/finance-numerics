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
    public:
        double v0;       // Initial variance
        double kappa;    // Mean-reversing variance process factor
        double theta;    // Long-term variance
        double rho;      // Correlation
        double sigma;    // Vol of vol

        // Paramterized constructor
        KOKKOS_INLINE_FUNCTION HestonParameters(double v_0, double kappa, double theta, double rho, double sigma)
            : v0(v_0) , kappa(kappa), theta(theta), rho(rho), sigma(sigma) {}

        // Empty constructor for manually setting values later
        KOKKOS_INLINE_FUNCTION HestonParameters() 
            : v0(EPSILON) , kappa(EPSILON), theta(EPSILON), rho(0.0), sigma(EPSILON) 
            {}
};

// ------------------------------------------------------------------------------------ //
/* Particle swarm config */
struct PSOConfig
{
    /* PSO parameters */
    double w;
    double c_1;
    double c_2;
    double tolerance;
    unsigned int n_particles;
    unsigned int max_iter;

    /* Parameters bounds */
    double v0_min, v0_max;
    double kappa_min, kappa_max;
    double theta_min, theta_max;
    double rho_min, rho_max;
    double sigma_min, sigma_max;


    PSOConfig()
        : w(0.729843788128) 
        , c_1(1.49617976566), c_2(1.49617976566) 
        , tolerance(1e-8) 
        , n_particles(100), max_iter(200)
        , v0_min(0.001), v0_max(0.75)
        , kappa_min(0.1), kappa_max(16.0)
        , theta_min(0.001), theta_max(10.0)
        , rho_min(-0.999), rho_max(0.999)
        , sigma_min(0.01), sigma_max(4.0)
        {}
};

// ------------------------------------------------------------------------------------ //
/* Particle */
struct Particle 
{
    HestonParameters position;
    HestonParameters velocity;
    HestonParameters best_position;
    double loss;
    double best_loss;

    KOKKOS_INLINE_FUNCTION Particle() 
        : position(), velocity(), best_position(), 
          loss(HUGE), best_loss(HUGE) 
        {}

    KOKKOS_INLINE_FUNCTION Particle(HestonParameters init_pos, HestonParameters init_vel=HestonParameters(), double init_loss=HUGE)
        : position(init_pos) , velocity(init_vel) , best_position(init_pos) ,
        loss(init_loss) , best_loss(init_loss)
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
        Kokkos::View<double*> call_prices(Kokkos::View<double*> strikes, unsigned int grid_points, bool print);
        Kokkos::View<double*> put_prices(Kokkos::View<double*> strikes, unsigned int grid_points, bool print);
        Kokkos::View<double*> black_scholes_call(Kokkos::View<double*> strikes, double sigma, bool print);

        /* Modify model methods */
        KOKKOS_INLINE_FUNCTION void update_option_condition(double updated_S0, double updated_r, double updated_T) { S_0 = updated_S0; r = updated_r;  t = updated_T;}
        KOKKOS_INLINE_FUNCTION void update_parameters(HestonParameters updatedParams) { params = updatedParams;}

        /* Getters */
        KOKKOS_INLINE_FUNCTION HestonParameters get_params() {return this->params;}

        /* Implied volatility */
        Kokkos::View<double*> implied_volatility(Kokkos::View<double*> call_prices, Kokkos::View<double*> K, unsigned int max_iter, double epsilon, bool print);
        Kokkos::View<double*> market_vega(Kokkos::View<double*> call_prices, Kokkos::View<double*> K, unsigned int max_iter, double epsilon, bool print);

        /* Loss functions for calibration */
        double price_vega_weighted_loss(Kokkos::View<double*> call_prices, Kokkos::View<double*> K);
        double price_sq_loss(Kokkos::View<double*> call_prices, Kokkos::View<double*> K);
        double iv_sq_loss(Kokkos::View<double*> call_prices, Kokkos::View<double*> K);

        /* Calibration with particle swarm */
        HestonParameters calibrate_PSO(Kokkos::View<double*> call_prices, 
                                                Kokkos::View<double*> K,
                                                PSOConfig config,
                                                unsigned int seed);


    private:
        /* Data fields */
        double S_0;                 // Initial price
        double r;                   // Risk-free rate
        double t;                   // Expiration time (In hindsight, it should be named T)
        HestonParameters params;    // Parameters
        double alpha;               // Dampening factor within FFT

        /* Kokkos functions */
        KOKKOS_INLINE_FUNCTION Complex heston_characteristic(Complex u);
        KOKKOS_INLINE_FUNCTION Complex damped_call(Complex v);

        /* Black-Scholes formula related stuff for implied volatility */
        KOKKOS_INLINE_FUNCTION double std_normal_cdf(double x) {return 0.5 * (1 + Kokkos::erf(x * 0.70710678118654752));}
        KOKKOS_INLINE_FUNCTION double std_normal_dist(double x) {return 0.39894228040143268 * Kokkos::exp(-0.5*x*x);}
        KOKKOS_INLINE_FUNCTION double _black_scholes(double S, double K, double sigma, double tau);
        KOKKOS_INLINE_FUNCTION double _vega(double S, double K, double sigma, double tau);

        /* Particle swarm */
};

/* Heston Characteristic functions */
KOKKOS_INLINE_FUNCTION Complex Heston_FFT::heston_characteristic(Complex u)
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
                            : Complex(EPSILON, 0.0);

    Complex last_term_frac = Kokkos::abs(1.0 - g_2*Kokkos::exp(-d*t)) > EPSILON
                            ? (1.0 - Kokkos::exp(-d*t)) / (1.0 - g_2*Kokkos::exp(-d*t))
                            : Complex(EPSILON, 0.0);

    // Exponent
    Complex exponent = i*u*(Kokkos::log(S_0) + r*t)
                    + (params.kappa*params.theta)/square(params.sigma) * ((xi -d)*t - 2.0*Kokkos::log(log_arg_frac))
                    + (params.v0/square(params.sigma))*(xi-d)*(last_term_frac);

    return Kokkos::exp(exponent);
}


/* Damped call price function */
KOKKOS_INLINE_FUNCTION Complex Heston_FFT::damped_call(Complex v)
{
    Complex phi = heston_characteristic((v - i*(alpha + 1.0)));
    return (Kokkos::exp(-r*t) * phi) / (alpha*alpha + alpha - v*v +i*(2.0*alpha + 1.0)*v);
}


/* Call price computation */
Kokkos::View<double*> Heston_FFT::call_prices(Kokkos::View<double*> strikes, unsigned int grid_points=8192, bool print=false)
{
    // FFT setup
    double eta = 0.2;   // Step-size in damped Fourier space
    double lambda = (2.0*PI) / (grid_points*eta); // Transformed step size in log-price space
    double bound = 0.5*grid_points*lambda;

    // Compute input for the shifted phased inverse Fourier transform
    Kokkos::View<Complex*> x("Fourier input", grid_points);
    Kokkos::parallel_for("FFT_input", grid_points,
        KOKKOS_LAMBDA(const unsigned int k) {
            
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
        KOKKOS_LAMBDA(const unsigned int k) {

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
Kokkos::View<double*> Heston_FFT::put_prices(Kokkos::View<double*> strikes, unsigned int grid_points=8192, bool print=false)
{
    // Compute call prices 
    Kokkos::View<double*> prices = call_prices(strikes, grid_points, false);

    // Apply parity
    Kokkos::parallel_for("apply_parity", strikes.extent(0),
        KOKKOS_LAMBDA(unsigned int k) {
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
KOKKOS_INLINE_FUNCTION double Heston_FFT::_black_scholes(double S, double K, double sigma, double tau)
{
    double d_plus = (Kokkos::log(S/K) + tau*(r + 0.5*square(sigma))) / (sigma * Kokkos::sqrt(tau));
    double d_minus = d_plus - sigma*Kokkos::sqrt(tau);
    return S*std_normal_cdf(d_plus) - std_normal_cdf(d_minus)*(K*Kokkos::exp(-r*tau));
}

KOKKOS_INLINE_FUNCTION double Heston_FFT::_vega(double S, double K, double sigma, double tau)
{
    double d_plus = (Kokkos::log(S/K) + tau*(r + 0.5*square(sigma))) / (sigma * Kokkos::sqrt(tau));
    return S*std_normal_dist(d_plus)*Kokkos::sqrt(tau);
}

Kokkos::View<double*> Heston_FFT::black_scholes_call(Kokkos::View<double*> strikes, double sigma, bool print=false)
{
    Kokkos::View<double*> calls("BS", strikes.extent(0));
    double S = S_0;

    Kokkos::parallel_for("b-s_calls", strikes.extent(0),
        KOKKOS_LAMBDA(unsigned int k){
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
        KOKKOS_LAMBDA(unsigned int j) {

            double price_diff = HUGE;    // Essentially error term
            double current_iv = 0.1;    // Initial iv guess

            for (unsigned int iter = 0; iter < max_iter; iter++) {
                
                // Compute the difference between current computed implied vol vs market price 
                price_diff = _black_scholes(S, strikes(j), current_iv, t) - call_prices(j);
                if (Kokkos::abs(price_diff) < epsilon) {break;}

                // Vega computation
                double vega = _vega(S, strikes(j), current_iv, t);
                if (vega < EPSILON) {vega = EPSILON;}

                // Newton update to the volatility
                current_iv -= price_diff / vega;
            }

            implied_vols(j) = current_iv;
        });
    
    if (print) {
        Kokkos::printf("Strike \t\t Call Price   Implied Vol \n");
        Kokkos::printf("-----------------------\n");
        Kokkos::parallel_for("print_result", strikes.extent(0),
            KOKKOS_LAMBDA(unsigned int k){
                Kokkos::printf("%.2lf \t\t %.2lf \t\t %.6lf\n", strikes(k), call_prices(k), implied_vols(k));
        });
        Kokkos::fence();
        Kokkos::printf("\n");
    }

    return implied_vols;
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
        KOKKOS_LAMBDA(unsigned int k) {
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
    unsigned int num_prices = call_prices.extent(0);

    // Computing calls based on strikes for squared loss
    Kokkos::View<double*> heston_calls = this->call_prices(K);

    // Reduce squared loss
    double loss;
    Kokkos::parallel_reduce("squared_vega_loss", num_prices,
        KOKKOS_LAMBDA(unsigned int k, double& local_loss) {
            local_loss += square(call_prices(k) - heston_calls(k));
        }, loss);

    return loss;
}

double Heston_FFT::iv_sq_loss(Kokkos::View<double*> call_prices, Kokkos::View<double*> K)
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

HestonParameters Heston_FFT::calibrate_PSO(Kokkos::View<double*> call_prices, Kokkos::View<double*> K, PSOConfig config, unsigned int seed=12345)
{
    // Metadata
    double global_best_loss = HUGE;
    Particle global_best_particle;
/*
    unsigned int n_particles = config.n_particles;
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);

    // Initializing paticles
    Kokkos::View<Particle*> particles("particles", n_particles);
    Kokkos::parallel_for("initializing_particles", n_particles,
        KOKKOS_LAMBDA(unsigned int k) {

            // Grab the random state
            Kokkos::Random_XorShift64_Pool<>::generator_type generator = rand_pool.get_state();

            // Initalize positions for particles
            HestonParameters initial_position = HestonParameters(
                                                config.v0_min + generator.drand(0.0, 1.0)*(config.v0_max - config.v0_min), 
                                                config.kappa_min + generator.drand(0.0, 1.0)*(config.kappa_max - config.kappa_min), 
                                                config.theta_min + generator.drand(0.0, 1.0)*(config.theta_max - config.theta_min),
                                                config.rho_min + generator.drand(0.0, 1.0)*(config.rho_max - config.rho_min),
                                                config.sigma_min + generator.drand(0.0, 1.0)*(config.sigma_max - config.sigma_min)
                                            );

            // Zero init velocity and HUGE loss (a.k.a default values)
            particles(k) = Particle(initial_position);

            // Release random state
            rand_pool.free_state(generator);
        });

    // Swarming
    for (unsigned int iter = 0; iter < config.max_iter; iter++) {

        // Evaluate (right now it has to be done on host)
        Kokkos::View<Particle*>::HostMirror h_particles = Kokkos::create_mirror_view(particles);
        for (unsigned int k = 0; k < n_particles; k++) {

            // Temporary update parameters to price
            HestonParameters original = this->get_params();
            this->update_parameters(h_particles(k).position);

            // Calculate loss
            double loss = this->price_sq_loss(call_prices, K);

            // Restore params
            this->update_parameters(original);

            // Update (host-side) particle loss at this index
            h_particles(k).loss = loss;
            if (h_particles(k).best_loss > loss) {
                h_particles(k).best_loss = loss;
                h_particles(k).best_position = h_particles(k).position;
            }

            // Update global best if this loss is less then global loss
            if (loss < global_best_loss) {
                global_best_loss = loss;
                global_best_particle.best_position = h_particles(k).best_position;
            }
        }
        // Copy the view back to device to parallelize the update algorithm
        Kokkos::deep_copy(particles, h_particles);

        // Update 
        Kokkos::parallel_for("update_particles", n_particles, 
            KOKKOS_LAMBDA(unsigned int k) {
                
                // Grab the random state
                Kokkos::Random_XorShift64_Pool<>::generator_type generator = rand_pool.get_state();

                // Coordinate-wise update the particle states (why C++ does not have a way to loop through all struct data fields grrrr...)
                double r1_v0 = generator.drand(0.0, 1.0);
                double r2_v0 = generator.drand(0.0, 1.0);
                particles(k).velocity.v0 = config.w * particles(k).velocity.v0
                                        + config.c_1 * r1_v0 * (particles(k).best_position.v0 - particles(k).position.v0)
                                        + config.c_2 * r2_v0 * (global_best_particle.position.v0 - particles(k).position.v0);

                double r1_kappa = generator.drand(0.0, 1.0);
                double r2_kappa = generator.drand(0.0, 1.0);
                particles(k).velocity.kappa = config.w * particles(k).velocity.kappa
                                        + config.c_1 * r1_kappa * (particles(k).best_position.kappa - particles(k).position.kappa)
                                        + config.c_2 * r2_kappa * (global_best_particle.position.kappa - particles(k).position.kappa);

                double r1_theta = generator.drand(0.0, 1.0);
                double r2_theta = generator.drand(0.0, 1.0);
                particles(k).velocity.theta = config.w * particles(k).velocity.theta
                                        + config.c_1 * r1_theta * (particles(k).best_position.theta - particles(k).position.theta)
                                        + config.c_2 * r2_theta * (global_best_particle.position.theta - particles(k).position.theta);

                double r1_rho = generator.drand(0.0, 1.0);
                double r2_rho = generator.drand(0.0, 1.0);
                particles(k).velocity.rho = config.w * particles(k).velocity.rho
                                        + config.c_1 * r1_rho * (particles(k).best_position.rho - particles(k).position.rho)
                                        + config.c_2 * r2_rho * (global_best_particle.position.rho - particles(k).position.rho);

                double r1_sigma = generator.drand(0.0, 1.0);
                double r2_sigma = generator.drand(0.0, 1.0);
                particles(k).velocity.sigma = config.w * particles(k).velocity.sigma
                                        + config.c_1 * r1_sigma * (particles(k).best_position.sigma - particles(k).position.sigma)
                                        + config.c_2 * r2_sigma * (global_best_particle.position.sigma - particles(k).position.sigma);

                // Release random state
                rand_pool.free_state(generator);

                // Update position while still enforce bounds
                double v0_pos = particles(k).position.v0 + particles(k).velocity.v0;
                particles(k).position.v0 = Kokkos::max(config.v0_min, Kokkos::min(config.v0_max, v0_pos));
                
                double kappa_pos = particles(k).position.kappa + particles(k).velocity.kappa;
                particles(k).position.kappa = Kokkos::max(config.kappa_min, Kokkos::min(config.kappa_max, kappa_pos));

                double theta_pos = particles(k).position.theta + particles(k).velocity.theta;
                particles(k).position.theta = Kokkos::max(config.theta_min, Kokkos::min(config.theta_max, theta_pos));

                double rho_pos = particles(k).position.rho + particles(k).velocity.rho;
                particles(k).position.rho = Kokkos::max(config.rho_min, Kokkos::min(config.rho_max, rho_pos));

                double sigma_pos = particles(k).position.sigma + particles(k).velocity.sigma;
                particles(k).position.sigma = Kokkos::max(config.sigma_min, Kokkos::min(config.sigma_max, sigma_pos));

            });
        Kokkos::fence();
    }
*/
    return global_best_particle.best_position;
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
        double goal_vol = 0.13431432;
        Kokkos::View<double*> test_prices = solver.black_scholes_call(strikes, goal_vol, true);
        Kokkos::View<double*> iv = solver.implied_volatility(test_prices, strikes, 10, 1e-15, true);
        Kokkos::View<double*> iv_Heston = solver.implied_volatility(call_prices, strikes, 20, 1e-15, true);

        // Calibrate
        //HestonParameters cal_param = solver.calibrate_PSO(test_prices, strikes, PSOConfig());
        //solver.update_parameters(cal_param);
        //Kokkos::View<double*> cal_heston_iv = solver.implied_volatility(solver.call_prices(strikes), strikes, 20, 1e-10, true);
    }
    Kokkos::finalize();

    return 0;
}