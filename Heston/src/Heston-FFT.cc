#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>
//#include <Kokkos_Random.hpp>

/* Macros */
using Complex = Kokkos::complex<double>;
using exec_space = Kokkos::DefaultExecutionSpace;
#define EPSILON 5e-15
const Complex i = Complex(0.0, 1.0);

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

    // Safe guard the fraction within the 2nd and 3rd exponential terms 
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


/* main() */
int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        Complex a(1.0, 2.0);
        Complex scalar(4.0, 0.0);
        Complex m = scalar*a;
        Complex s = 4.0*a;

        Kokkos::printf("Complex mult: %lf + %lfi\n", m.real(), m.imag());
        Kokkos::printf("Double mult: %lf + %lfi\n", s.real(), s.imag());
    }
    Kokkos::finalize();
}
