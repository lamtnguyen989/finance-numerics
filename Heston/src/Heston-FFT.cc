#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>
//#include <Kokkos_Random.hpp>

/* Macros */
using Complex = Kokkos::complex<float>;
using exec_space = Kokkos::DefaultExecutionSpace;

/* Function declearations */

/* Heston model parameters */
struct HestonParameters
{
    float v0;       // Initial variance
    float kappa;    // Mean-reversing variance process factor
    float theta;    // Long-term variance
    float rho;      // Correlation
    float sigma;    // Vol of vol
};

KOKKOS_INLINE_FUNCTION heston_log_characteristic()
{
    
}

/* main() */
int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {

    }
    Kokkos::finalize();
}
