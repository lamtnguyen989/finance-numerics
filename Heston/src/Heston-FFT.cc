#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

using Complex = Kokkos::complex<float>;
using exec_space = Kokkos::DefaultExecutionSpace;

/* Heston model parameters */
struct HestonParameters
{
    float v0;
    float kappa;
    float theta;
    float rho;
    float sigma;
}


/* main() */
int main(int argc, char* argv[])
{
    Kokkos::initalize(argc, argv);
    {

    }
    Kokkos::finalize();
}