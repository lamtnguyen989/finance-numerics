#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/* Macros */
using Complex = Kokkos::complex<double>;
using exec_space = Kokkos::DefaultExecutionSpace;
#define PI 3.141592653589793
#define EPSILON 5e-15
#define i Complex(0.0, 1.0) // It has to be defined due to device code conflict

/* Function declearations */

/* Heston model parameters */
struct HestonParameters
{
    double v0;       // Initial variance
    double kappa;    // Mean-reversing variance process factor
    double theta;    // Long-term variance
    double rho;      // Correlation
    double sigma;    // Vol of vol
};

/*
class Heston_FFT    // Leave this here for now to see if I need to do OOP
{
    public:

        KOKKOS_INLINE_FUNCTION void hello()
        {
            Kokkos::printf("Hello there!\n");
        }

};
*/


KOKKOS_INLINE_FUNCTION Complex heston_characteristic(Complex u, double r, double t, double S_0, HestonParameters params)
{
    // A bunch of repeated constants in the calculation
    Complex xi = params.kappa - i*params.rho*params.sigma*u;
    Complex d = Kokkos::sqrt(xi*xi + params.sigma*params.sigma*(u*u + i*u));
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
                    + (params.kappa*params.theta)/(params.sigma*params.sigma) * (xi -d)*t - 2.0*Kokkos::log(log_arg_frac)
                    + (params.v0/(params.sigma*params.sigma))*(xi-d)*(last_term_frac);

    return Kokkos::exp(exponent);
}

KOKKOS_INLINE_FUNCTION Complex damped_call(Complex v, double r, double t, double S_0, HestonParameters params, double alpha)
{
    Complex phi = heston_characteristic((v - i*(alpha + 1.0)), r, t, S_0, params);
    return (Kokkos::exp(-r*t) * phi) / (alpha*alpha + alpha - v*v +i*(2.0*alpha + 1.0)*v);
}

void fft_call_prices(
    Kokkos::View<double*> strikes,
    Kokkos::View<double*> prices,
    unsigned int N,
    double S_0,
    double r,
    double t,
    double alpha,
    HestonParameters params
){
    // FFT setup
    double eta = 0.2;   // Step-size in damped Fourier space
    double lambda = (2.0*PI) / (N*eta); // Transformed step size in log-price space
    double bound = 0.5*N*lambda;

    // Compute input for the Fourier transform
    Kokkos::View<Complex*> x("Fourier input", N);
    Kokkos::parallel_for("FFT_input", N,
        KOKKOS_LAMBDA(const unsigned int k) {
            
            // Compute damped call price
            double v_k = eta*k;
            Complex damped = damped_call(Complex(v_k, 0.0), r, t, S_0, params, alpha);

            // Modify the damped call price to get the input
            x(k) = Kokkos::exp(-i*bound*v_k) * damped * eta; 
        }
    );

    // FFT
    Kokkos::View<Complex*> x_hat("x_hat", N);
    KokkosFFT::fft(exec_space(), x, x_hat);

    // Extract the option prices
    Kokkos::parallel_for("extract_prices", strikes.extent(0),
        KOKKOS_LAMBDA(const unsigned int k) {

            // Find the Fourier grid price index
            double log_K = Kokkos::log(strikes(k));
            int index = static_cast<int>((log_K + bound) / lambda + 0.5);

            // Compute the price at this index
            if ((index < N) && (index >= 0))
                prices(k) = x_hat(index).real()*Kokkos::exp(-alpha*(lambda*index - bound))/PI ;
            else
                prices(k) = 0.0;
        }); 
}


/* main() */
int main(int argc, char* argv[])
{
    // Setting up OpenMP backend for the cases where I run on host only
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
        hestonParams.theta = 0.0225;
        hestonParams.sigma = 0.25;
        hestonParams.rho = -0.25;

        // FFT parameters
        double alpha = 2.0;
        unsigned int N = 8192;

        // Strikes and corresponding price parameters
        Kokkos::View<double*> strikes("strikes", num_strikes);
        Kokkos::parallel_for("fill_strikes", num_strikes,
            KOKKOS_LAMBDA(unsigned int k){
                strikes(k) = 80 + k*5.0;
        });
        Kokkos::View<double*> prices("prices", num_strikes);

        // Compute the FFT call prices
        fft_call_prices(strikes, prices, N, S_0, r, T, alpha, hestonParams);

        // Output
        Kokkos::printf("Strike \t\t Price\n");
        Kokkos::printf("-----------------------\n");
        Kokkos::parallel_for("print_result", num_strikes,
            KOKKOS_LAMBDA(unsigned int k){
                Kokkos::printf("%.2lf \t\t %.2lf\n", strikes(k), prices(k));
        });
    }
    Kokkos::finalize();
}
