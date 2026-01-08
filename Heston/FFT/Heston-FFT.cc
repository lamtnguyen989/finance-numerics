#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/* Macros */
using Complex = Kokkos::complex<double>;
using exec_space = Kokkos::DefaultExecutionSpace;
#define PI 3.141592653589793
#define EPSILON 5e-15
#define i Complex(0.0, 1.0) // It has to be defined like this due to device code conflict


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
        void update_option_condition(double updated_S0, double updated_r, double updated_T) { S_0 = updated_S0; r = updated_r;  t = updated_T;}
        void update_parameters(HestonParameters updatedParams) { params = updatedParams;}

        /* Implied volatility methods */
        Kokkos::View<double*> implied_volatility(Kokkos::View<double*> call_prices, Kokkos::View<double*> K, unsigned int max_iter, double epsilon, bool print);

        /* Calibration and optimization */
        HestonParameters iv_calibrate();

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
};

/* Heston Characteristic functions */
KOKKOS_INLINE_FUNCTION Complex Heston_FFT::heston_characteristic(Complex u)
{
    // A bunch of repeated constants in the calculation
    Complex xi = params.kappa - i*params.rho*params.sigma*u;
    Complex d = Kokkos::sqrt(xi*xi + params.sigma*params.sigma*(u*u + i*u));
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
                    + (params.kappa*params.theta)/(params.sigma*params.sigma) * ((xi -d)*t - 2.0*Kokkos::log(log_arg_frac))
                    + (params.v0/(params.sigma*params.sigma))*(xi-d)*(last_term_frac);

    return Kokkos::exp(exponent);
}


/* Damped call price function */
KOKKOS_INLINE_FUNCTION Complex Heston_FFT::damped_call(Complex v)
{
    Complex phi = heston_characteristic((v - i*(alpha + 1.0)));
    return (Kokkos::exp(-r*t) * phi) / (alpha*alpha + alpha - v*v +i*(2.0*alpha + 1.0)*v);
}


/* Call price computation */
Kokkos::View<double*> Heston_FFT::call_prices(Kokkos::View<double*> strikes, unsigned int grid_points, bool print=false)
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
Kokkos::View<double*> Heston_FFT::put_prices(Kokkos::View<double*> strikes, unsigned int grid_points, bool print=false)
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
    double d_plus = (Kokkos::log(S/K) + tau*(r + 0.5*sigma*sigma)) / (sigma * Kokkos::sqrt(tau));
    double d_minus = d_plus - sigma*Kokkos::sqrt(tau);
    return S*std_normal_cdf(d_plus) - std_normal_cdf(d_minus)*(K*Kokkos::exp(-r*tau));
}

KOKKOS_INLINE_FUNCTION double Heston_FFT::_vega(double S, double K, double sigma, double tau)
{
    double d_plus = (Kokkos::log(S/K) + tau*(r + 0.5*sigma*sigma)) / (sigma * Kokkos::sqrt(tau));
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
            unsigned int iter = 0;
            double price_diff = 1e5;
            double current_iv = 0.1;
            while (iter < max_iter) {
                
                // Compute the difference between current computed implied vol vs market price 
                price_diff = call_prices(j) - _black_scholes(S, strikes(j), current_iv, t);
                if (Kokkos::abs(price_diff) < epsilon) {break;}

                // Vega computation
                double vega = _vega(S, strikes(j), current_iv, t);
                if (vega < EPSILON) {vega = EPSILON;}

                // Newton update to the volatility
                current_iv += price_diff / vega;

                // Make sure we are moving along
                iter++;
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