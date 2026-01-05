#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>
//#include <Kokkos_Random.hpp>

using Complex = Kokkos::complex<float>;
using exec_space = Kokkos::DefaultExecutionSpace;

/* Heston model parameters */
struct HestonParameters
{
    float v0;       // Initial variance
    float kappa;    // Mean-reversing variance process factor
    float theta;    // Long-term variance
    float rho;      // Correlation
    float sigma;    // Vol of vol
};

/* Heston model FFT solver */
class Heston_FFT
{
    public:
        Heston_FFT() {}
    
    private:

};


/* main() */
int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {

    }
    Kokkos::finalize();
}
