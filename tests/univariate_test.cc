#include "PoseLib/misc/univariate.h"
#include "test.h"

using namespace poselib;

namespace {
bool test_projective_quadratic() {
    // Zero roots, repeated roots, linear polynomials, and roots at infinity.
    for (double scale : {1e-200, 1.0, 1e200}) {
        for (const Eigen::Vector3d &coeff :
             {Eigen::Vector3d(1, -3, 2), Eigen::Vector3d(1, 0, -1), Eigen::Vector3d(1, -1, 0),
              Eigen::Vector3d(0, 1, -2), Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(1, 2, 1), Eigen::Vector3d(1, 0, 0),
              Eigen::Vector3d(0, 0, 1)}) {
            Eigen::Vector2d roots[2];
            const int count =
                univariate::solve_quadratic_real(scale * coeff(0), scale * coeff(1), scale * coeff(2), roots);
            const double disc = coeff(1) * coeff(1) - 4 * coeff(0) * coeff(2);
            REQUIRE_EQ(count, disc == 0.0 ? 1 : 2);
            for (int i = 0; i < count; ++i) {
                REQUIRE(roots[i].allFinite());
                REQUIRE_SMALL(roots[i].norm() - 1.0, 1e-14);
                const double u = roots[i](0), v = roots[i](1);
                REQUIRE_SMALL(coeff(0) * u * u + coeff(1) * u * v + coeff(2) * v * v, 1e-14);
            }
            if (count == 2)
                REQUIRE(std::abs(roots[0](0) * roots[1](1) - roots[0](1) * roots[1](0)) > 0.1);
        }
    }
    Eigen::Vector2d roots[2];
    REQUIRE_EQ(univariate::solve_quadratic_real(0, 0, 0, roots), 0);
    REQUIRE_EQ(univariate::solve_quadratic_real(1, 0, 1, roots), 0);
    REQUIRE_EQ(univariate::solve_quadratic_real(1, std::numeric_limits<double>::quiet_NaN(), 1, roots), 0);
    double affine[2];
    REQUIRE_EQ(univariate::solve_quadratic_real(1, -2, 1, affine), 2);
    REQUIRE_EQ(affine[0], 1.0);
    REQUIRE_EQ(affine[1], 1.0);
    return true;
}

} // namespace

std::vector<Test> register_univariate_test() { return {TEST(test_projective_quadratic)}; }
