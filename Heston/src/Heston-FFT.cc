#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/* Macros */
using Complex = Kokkos::complex<double>;
using exec_space = Kokkos::DefaultExecutionSpace;
#define PI 3.141592653589793
#define EPSILON 5e-15
#define i Complex(0.0, 1.0) // It has to be defined due to device code conflict


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

        /* Public methods */
        Kokkos::View<double*> call_prices(Kokkos::View<double*> strikes, unsigned int grid_points, bool print);
        Kokkos::View<double*> put_prices(Kokkos::View<double*> strikes, unsigned int grid_points, bool print);
        void modify_option_condition(double updated_S0, double updated_r, double updated_T) { S_0 = updated_S0; r = updated_r;  t = updated_T;}

    private:
        /* Data fields */
        double S_0;
        double r;
        double t;
        HestonParameters params;
        double alpha;

        /* Kokkos functions */
        KOKKOS_INLINE_FUNCTION Complex heston_characteristic(Complex u);
        KOKKOS_INLINE_FUNCTION Complex damped_call(Complex v);
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

    // Compute input for the shifted inverse Fourier transform
    Kokkos::View<Complex*> x("Fourier input", grid_points);
    Kokkos::parallel_for("FFT_input", grid_points,
        KOKKOS_LAMBDA(const unsigned int k) {
            
            // Compute damped call price
            double v_k = eta*k;
            Complex damped = this->damped_call(Complex(v_k, 0.0));

            // Modify the damped call price to get the input
            x(k) = Kokkos::exp(-i*bound*v_k) * damped * eta; 
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
        
    }

    return prices;
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
        double r = 0.01;
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
        
        // Solving the call prices
        Heston_FFT solver(S_0, r, T, hestonParams, alpha);
        Kokkos::View<double*> call_prices = solver.call_prices(strikes, N, true);
        Kokkos::View<double*> put_prices = solver.put_prices(strikes, N, true);


    }
    Kokkos::finalize();
}
