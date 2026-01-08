#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>


/* Macros */
using Complex = Kokkos::complex<double>;
using exec_space = Kokkos::DefaultExecutionSpace;
#define PI 3.141592653589793

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

/* Heston Monte Carlo model */
class Heston_MC
{
    public:
        /* Constructors */
        Heston_MC(double S_0, double r, double t, HestonParameters heston_params)
            : S_0(S_0), r(r), t(t)
            , params(heston_params)
        {}

    private:
        /* Data fields */
        double S_0;
        double r;
        double t;
        HestonParameters params;
};

int main(int argc, char *argv[])
{
    // Setting up OpenMP backend for the cases when run on host only
    setenv("OMP_PROC_BIND", "spread", 1);
    setenv("OMP_PLACES", "threads", 1);
    
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


        // Solver
        Heston_MC solver(S_0, r, T, hestonParams);
    }
    Kokkos::finalize();
}