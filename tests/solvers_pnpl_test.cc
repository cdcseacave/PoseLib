#include "PoseLib/solvers/p1p2ll.h"
#include "PoseLib/solvers/p2p1ll.h"
#include "test.h"
#include "test_rng.h"

using namespace poselib;

namespace {

// Generate from world geometry, independently of the solver's elimination.
bool check_solvers_pnpl(int points, double thickness, bool incident, bool change_coordinates) {
    auto rng = test_rng::make_rng("solvers_pnpl");
    for (int sample = 0; sample < 300; ++sample) {
        const Eigen::Matrix3d R =
            Eigen::AngleAxisd(rng.uniform(-3.14, 3.14), test_rng::symmetric_vec3(rng).normalized()).toRotationMatrix();
        const Eigen::Vector3d t(0.2, -0.1, 5.0);
        std::vector<Eigen::Vector3d> xp, Xp, l, X, V;
        for (int i = 0; i < points; ++i) {
            Eigen::Vector3d P = test_rng::symmetric_vec3(rng);
            P.z() *= thickness;
            Xp.push_back(P);
            xp.push_back((R * P + t).normalized());
        }
        for (int i = 0; i < 3 - points; ++i) {
            Eigen::Vector3d P = test_rng::symmetric_vec3(rng);
            Eigen::Vector3d v = test_rng::symmetric_vec3(rng);
            P.z() *= thickness;
            v.z() *= thickness;
            v.normalize();
            X.push_back(P);
            V.push_back(v);
            l.push_back((R * P + t).cross(R * v).normalized());
        }
        if (incident) {
            Eigen::Vector3d P = R * Xp[0] + t;
            P -= l[0].dot(P) * l[0];
            Xp[0] = R.transpose() * (P - t);
            xp[0] = P.normalized();
        }
        CameraPose expected(R, t);
        if (change_coordinates) {
            const double scale = sample % 2 == 0 ? 1e-3 : 1e3;
            const Eigen::Matrix3d Q = Eigen::AngleAxisd(1.7, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
            const Eigen::Vector3d shift(3, -2, 1);
            for (auto &P : Xp)
                P = scale * (Q * P + shift);
            for (size_t i = 0; i < X.size(); ++i) {
                X[i] = scale * (Q * (X[i] + 2.0 * V[i]) + shift);
                V[i] = (i == 0 ? -3.0 : 7.0) * Q * V[i];
                l[i] *= i == 0 ? -2.0 : 4.0;
            }
            const Eigen::Matrix3d rotated = R * Q.transpose();
            expected = CameraPose(rotated, scale * (t - rotated * shift));
        }
        CameraPoseVector solutions(9);
        const auto solve = points == 2 ? p2p1ll : p1p2ll;
        const int count = solve(xp, Xp, l, X, V, &solutions);
        REQUIRE_EQ(count, solutions.size());
        REQUIRE(count > 0 && count <= (points == 2 ? 2 : 4));
        double error = std::numeric_limits<double>::infinity();
        for (const CameraPose &pose : solutions) {
            REQUIRE(pose.q.allFinite() && pose.t.allFinite());
            REQUIRE_SMALL(pose.q.squaredNorm() - 1.0, 1e-10);
            const Eigen::Matrix3d rotation = pose.R();
            REQUIRE_SMALL((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm(), 1e-10);
            REQUIRE_SMALL(rotation.determinant() - 1.0, 1e-10);
            for (size_t i = 0; i < xp.size(); ++i)
                REQUIRE(xp[i].dot(pose.apply(Xp[i])) > 0.0);
            for (size_t i = 0; i < xp.size(); ++i)
                REQUIRE_SMALL(xp[i].cross(pose.apply(Xp[i]).normalized()).norm(), 1e-7);
            for (size_t i = 0; i < l.size(); ++i) {
                REQUIRE_SMALL(l[i].normalized().dot(pose.apply(X[i]).normalized()), 1e-7);
                REQUIRE_SMALL(l[i].normalized().dot(pose.rotate(V[i]).normalized()), 1e-7);
            }
            error = std::min(error, (rotation - expected.R()).norm() +
                                        (pose.t - expected.t).norm() / std::max(1.0, expected.t.norm()));
        }
        REQUIRE_SMALL_M(error, 1e-7, test_rng::case_id("solvers_pnpl", sample));
        // Verify the complete returned set, not only the ground-truth root,
        // after reordering matches and changing the world-line anchors.
        if (sample < 20) {
            if (points == 2) {
                std::swap(xp[0], xp[1]);
                std::swap(Xp[0], Xp[1]);
            } else {
                std::swap(l[0], l[1]);
                std::swap(X[0], X[1]);
                std::swap(V[0], V[1]);
            }
            for (size_t i = 0; i < X.size(); ++i)
                X[i] += 0.3 * V[i];
            CameraPoseVector reordered;
            REQUIRE_EQ(solve(xp, Xp, l, X, V, &reordered), count);
            for (const auto &pose : solutions) {
                double distance = std::numeric_limits<double>::infinity();
                for (const auto &other : reordered)
                    distance = std::min(distance, (pose.R() - other.R()).norm() +
                                                      (pose.t - other.t).norm() / std::max(1.0, pose.t.norm()));
                REQUIRE_SMALL(distance, 1e-7);
            }
        }
    }
    return true;
}

bool test_p2p1ll_generic() { return check_solvers_pnpl(2, 1.0, false, false); }
bool test_p1p2ll_generic() { return check_solvers_pnpl(1, 1.0, false, false); }
bool test_p2p1ll_coplanar() { return check_solvers_pnpl(2, 0.0, false, false); }
bool test_p1p2ll_coplanar() { return check_solvers_pnpl(1, 0.0, false, false); }
bool test_p2p1ll_near_coplanar() { return check_solvers_pnpl(2, 1e-8, false, false); }
bool test_p1p2ll_near_coplanar() { return check_solvers_pnpl(1, 1e-8, false, false); }
bool test_p2p1ll_incident() { return check_solvers_pnpl(2, 1.0, true, false); }
bool test_p1p2ll_incident() { return check_solvers_pnpl(1, 1.0, true, false); }
bool test_p2p1ll_coordinates() { return check_solvers_pnpl(2, 1.0, false, true); }
bool test_p1p2ll_coordinates() { return check_solvers_pnpl(1, 1.0, false, true); }

bool test_solvers_pnpl_axis_aligned() {
    for (int points : {2, 1}) {
        for (bool parallel : {false, true}) {
            for (double angle : {0.0, 3.14159265358979323846}) {
                const Eigen::Matrix3d R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()).toRotationMatrix();
                std::vector<Eigen::Vector3d> Xp{{1, 2, 4}, {-1, 1, 3}}, X{{2, -1, 4}, {-2, 2, 6}};
                std::vector<Eigen::Vector3d> V{{0, 0, 1}, {1, 0, 0}}, xp, l;
                if (parallel)
                    V[1] = V[0];
                Xp.resize(points);
                X.resize(3 - points);
                V.resize(3 - points);
                for (const auto &P : Xp)
                    xp.push_back((R * P).normalized());
                for (size_t i = 0; i < X.size(); ++i)
                    l.push_back((R * X[i]).cross(R * V[i]).normalized());
                CameraPoseVector solutions;
                (points == 2 ? p2p1ll : p1p2ll)(xp, Xp, l, X, V, &solutions);
                double error = std::numeric_limits<double>::infinity();
                bool positive = false, negative = false;
                for (const auto &pose : solutions) {
                    REQUIRE(pose.q.allFinite() && pose.t.allFinite());
                    error = std::min(error, (pose.R() - R).norm() + pose.t.norm());
                    positive |= xp[0].dot(pose.apply(Xp[0])) > 0.0;
                    negative |= xp[0].dot(pose.apply(Xp[0])) < 0.0;
                }
                REQUIRE_SMALL(error, 1e-10);
                REQUIRE(positive && !negative);
                // Detected rank failures must clear a previously populated output.
                if (points == 2)
                    Xp[1] = Xp[0];
                else
                    l[1] = l[0];
                REQUIRE_EQ((points == 2 ? p2p1ll : p1p2ll)(xp, Xp, l, X, V, &solutions), 0);
                REQUIRE(solutions.empty());
            }
        }
    }
    return true;
}

bool test_solvers_pnpl_unrelated_observations() {
    auto rng = test_rng::make_rng("unrelated");
    for (int points : {2, 1}) {
        for (int sample = 0; sample < 1000; ++sample) {
            std::vector<Eigen::Vector3d> Xp, xp, X, V, l;
            for (int i = 0; i < points; ++i) {
                Xp.push_back(test_rng::symmetric_vec3(rng));
                xp.push_back(test_rng::symmetric_vec3(rng).normalized());
            }
            for (int i = 0; i < 3 - points; ++i) {
                X.push_back(test_rng::symmetric_vec3(rng));
                V.push_back(test_rng::symmetric_vec3(rng).normalized());
                l.push_back(test_rng::symmetric_vec3(rng).normalized());
            }
            CameraPoseVector solutions;
            const int count = (points == 2 ? p2p1ll : p1p2ll)(xp, Xp, l, X, V, &solutions);
            REQUIRE_EQ(count, solutions.size());
            REQUIRE(count <= (points == 2 ? 2 : 4));
            for (const auto &pose : solutions) {
                REQUIRE(pose.q.allFinite() && pose.t.allFinite());
                for (size_t i = 0; i < xp.size(); ++i)
                    REQUIRE(xp[i].dot(pose.apply(Xp[i])) > 0.0);
                for (size_t i = 0; i < xp.size(); ++i)
                    REQUIRE_SMALL(xp[i].cross(pose.apply(Xp[i]).normalized()).norm(), 1e-7);
                for (size_t i = 0; i < l.size(); ++i) {
                    REQUIRE_SMALL(l[i].dot(pose.apply(X[i]).normalized()), 1e-7);
                    REQUIRE_SMALL(l[i].dot(pose.rotate(V[i]).normalized()), 1e-7);
                }
            }
        }
    }
    return true;
}

} // namespace

std::vector<Test> register_solvers_pnpl_test() {
    return {TEST(test_solvers_pnpl_axis_aligned),
            TEST(test_solvers_pnpl_unrelated_observations),
            TEST(test_p2p1ll_generic),
            TEST(test_p1p2ll_generic),
            TEST(test_p2p1ll_coplanar),
            TEST(test_p1p2ll_coplanar),
            TEST(test_p2p1ll_near_coplanar),
            TEST(test_p1p2ll_near_coplanar),
            TEST(test_p2p1ll_incident),
            TEST(test_p1p2ll_incident),
            TEST(test_p2p1ll_coordinates),
            TEST(test_p1p2ll_coordinates)};
}
