#include "PoseLib/misc/decompositions.h"
#include "test.h"

using namespace poselib;

namespace {
bool test_adjugate() {
    const Eigen::Matrix3d H = (Eigen::Matrix3d() << 1, 2, 3, 0, 4, 5, 1, 0, 6).finished();
    REQUIRE_SMALL((H * adjugate(H) - H.determinant() * Eigen::Matrix3d::Identity()).norm(), 1e-13);
    return true;
}
} // namespace

std::vector<Test> register_decompositions_test() { return {TEST(test_adjugate)}; }
