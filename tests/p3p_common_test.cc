#include "PoseLib/solvers/p3p_common.h"
#include "test.h"

using namespace poselib;

namespace {
bool test_pivoted_conic_factorization() {
    // The first coordinate vanishes for the coplanar P1P2L pencil.
    // Exercise every permutation, including zero first row/column.
    for (int k = 0; k < 3; ++k) {
        Eigen::Vector3d p = Eigen::Vector3d::Zero(), q = Eigen::Vector3d::Zero();
        p((k + 1) % 3) = 1;
        p((k + 2) % 3) = 2;
        q((k + 1) % 3) = 3;
        q((k + 2) % 3) = -1;
        const Eigen::Matrix3d C = (p * q.transpose() + q * p.transpose()) * 0.5;
        const auto factors = compute_pq(C, ConicFactorStrategy::MAX_ABS);
        REQUIRE(factors[0].allFinite() && factors[1].allFinite());
        Eigen::Matrix3d reconstructed =
            (factors[0] * factors[1].transpose() + factors[1] * factors[0].transpose()) * 0.5;
        const double scale = reconstructed.cwiseProduct(C).sum() / C.squaredNorm();
        REQUIRE(std::abs(scale) > 0);
        REQUIRE_SMALL((reconstructed / scale - C).norm(), 1e-13);
    }
    return true;
}
} // namespace

std::vector<Test> register_p3p_common_test() { return {TEST(test_pivoted_conic_factorization)}; }
