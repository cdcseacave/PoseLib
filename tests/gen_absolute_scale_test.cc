#include "example_cameras.h"
#include "optim_test_utils.h"
#include "test.h"

#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust.h>
#include <PoseLib/robust/optim/generalized_absolute.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/lm_impl.h>
#include <PoseLib/robust/sampling.h>
#include <algorithm>

using namespace poselib;

//////////////////////////////
// Generalized absolute pose and scale

namespace test::gen_absolute_scale {

namespace {

// Fixed rig pose so the fixture is reproducible across runs.
CameraPose rig_pose() {
    const Eigen::Matrix3d R =
        (Eigen::AngleAxisd(-0.24, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(0.31, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(0.12, Eigen::Vector3d::UnitZ()))
            .toRotationMatrix();
    return CameraPose(R, Eigen::Vector3d(0.32, -0.15, 0.44));
}

// Rig with distinct camera centers. The scale is only observable through the parallax
// between them, so the centers are spread out w.r.t. the depth of the points.
std::vector<CameraPose> rig_extrinsics(size_t num_cams) {
    std::vector<CameraPose> camera_ext;
    for (size_t k = 0; k < num_cams; ++k) {
        const double centered = static_cast<double>(k) - 0.5 * static_cast<double>(num_cams - 1);
        const Eigen::Matrix3d R = Eigen::AngleAxisd(0.12 * centered, Eigen::Vector3d::UnitY()).toRotationMatrix();
        const Eigen::Vector3d center(0.45 * centered, 0.2 * (static_cast<double>(k % 2) - 0.5), 0.18 * centered);
        camera_ext.emplace_back(R, -R * center);
    }
    return camera_ext;
}

// Rig which only rotates around a single center, i.e. which cannot constrain the scale.
std::vector<CameraPose> rotating_rig_extrinsics(size_t num_cams) {
    std::vector<CameraPose> camera_ext;
    for (size_t k = 0; k < num_cams; ++k) {
        const Eigen::Matrix3d R =
            Eigen::AngleAxisd(0.9 * static_cast<double>(k), Eigen::Vector3d::UnitY()).toRotationMatrix();
        camera_ext.emplace_back(R, Eigen::Vector3d::Zero());
    }
    return camera_ext;
}

struct Scene {
    CameraPose pose;
    double scale;
    std::vector<CameraPose> camera_ext;
    std::vector<Camera> cameras;
    std::vector<std::vector<Point2D>> x;
    std::vector<std::vector<Point3D>> X;
};

// Builds a rig observing 3D points which live at another scale than the rig itself: the pose
// of the k:th rig camera is (Rk*R, Rk*t + scale*tk).
Scene setup_scene(const std::vector<CameraPose> &camera_ext, size_t num_pts, double scale, const Camera &camera,
                  const std::string &case_name, size_t case_index = 0) {
    test_rng::Rng rng = test_rng::make_rng(case_name, case_index);
    const size_t num_cams = camera_ext.size();

    Scene scene;
    scene.pose = rig_pose();
    scene.scale = scale;
    scene.camera_ext = camera_ext;
    scene.cameras.assign(num_cams, camera);
    scene.x.assign(num_cams, {});
    scene.X.assign(num_cams, {});

    const ScaledCameraPose scaled_pose(scene.pose, scale);
    for (size_t k = 0; k < num_cams; ++k) {
        const CameraPose full_pose = scaled_pose.camera_pose(camera_ext[k]);
        for (size_t i = 0; i < num_pts; ++i) {
            const Eigen::Vector2d xi = image_sample(camera, i, num_pts, rng);
            Eigen::Vector3d Xi;
            camera.unproject(xi, &Xi);
            Xi *= 3.5 + 0.45 * static_cast<double>(i % 7) + rng.uniform(0.0, 0.4);
            scene.x[k].push_back(xi);
            scene.X[k].push_back(full_pose.apply_inverse(Xi));
        }
    }
    return scene;
}

// Apply deterministic perturbations to all observations in the multi-camera fixture.
void add_multi_point_noise(std::vector<std::vector<Point2D>> &x, double scale, const std::string &case_name,
                           size_t case_index = 0) {
    test_rng::Rng rng = test_rng::make_rng(case_name, case_index);
    for (std::vector<Point2D> &points : x) {
        for (Point2D &point : points) {
            point += test_rng::symmetric_vec2(rng, scale);
        }
    }
}

// Perturb the observations and replace a fraction of them with random image points.
size_t add_noise_and_outliers(Scene *scene, double noise, double outlier_ratio, const std::string &case_name) {
    test_rng::Rng rng = test_rng::make_rng(case_name);
    size_t num_outliers = 0;
    for (size_t k = 0; k < scene->x.size(); ++k) {
        const Camera &camera = scene->cameras[k];
        for (size_t i = 0; i < scene->x[k].size(); ++i) {
            if (rng.uniform(0.0, 1.0) < outlier_ratio) {
                scene->x[k][i] = Eigen::Vector2d(rng.uniform(0.0, camera.width), rng.uniform(0.0, camera.height));
                num_outliers++;
            } else {
                scene->x[k][i] += test_rng::symmetric_vec2(rng, noise);
            }
        }
    }
    return num_outliers;
}

// Unit bearings of the observations, in the coordinate system of the observing rig camera.
std::vector<std::vector<Point3D>> scene_bearings(const Scene &scene) {
    std::vector<std::vector<Point3D>> bearings(scene.x.size());
    for (size_t k = 0; k < scene.x.size(); ++k) {
        for (const Point2D &xi : scene.x[k]) {
            Eigen::Vector3d bi;
            scene.cameras[k].unproject(xi, &bi);
            bearings[k].push_back(bi.normalized());
        }
    }
    return bearings;
}

double rotation_error(const CameraPose &pose, const CameraPose &pose_gt) { return (pose.R() - pose_gt.R()).norm(); }

double translation_error(const CameraPose &pose, const CameraPose &pose_gt) { return (pose.t - pose_gt.t).norm(); }

} // namespace

bool test_gen_absolute_scale_jacobian() {
    const size_t N = 10;
    const size_t Ncam = 4;

    for (size_t camera_idx = 0; camera_idx < example_cameras.size(); ++camera_idx) {
        const std::string &camera_str = example_cameras[camera_idx];
        log_test_case("camera", test_rng::case_id(camera_str, camera_idx));
        Camera camera;
        camera.initialize_from_txt(camera_str);

        Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_scale_jacobian", camera_idx);
        add_multi_point_noise(scene.x, 2e-4 * camera.max_dim(), "gen_absolute_scale_jacobian_noise", camera_idx);
        normalize_camera_points(scene.x, &scene.cameras);

        GeneralizedAbsolutePoseScaleRefiner<UniformWeightVectors, TestAccumulator> refiner(
            scene.x, scene.X, scene.camera_ext, scene.cameras);

        const ScaledCameraPose scaled_pose(scene.pose, scene.scale);
        const double delta = 1e-6;
        double jac_err = verify_jacobian<decltype(refiner), ScaledCameraPose>(refiner, scaled_pose, delta);
        REQUIRE_SMALL_M(jac_err, 1e-6, test_rng::case_id(camera_str, camera_idx));

        // Test that compute_residual and compute_jacobian are compatible
        TestAccumulator acc;
        acc.reset_residual();
        double r1 = refiner.compute_residual(acc, scaled_pose);
        acc.reset_jacobian();
        refiner.compute_jacobian(acc, scaled_pose);
        double r2 = 0.0;
        for (size_t i = 0; i < acc.rs.size(); ++i) {
            r2 += acc.weights[i] * acc.rs[i].squaredNorm();
        }
        REQUIRE_SMALL_M(std::abs(r1 - r2), 1e-8, test_rng::case_id(camera_str, camera_idx));
    }
    return true;
}

bool test_gen_absolute_scale_bearing_jacobian() {
    const size_t N = 10;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_scale_bearing_jacobian");
    add_multi_point_noise(scene.x, 5e-4, "gen_absolute_scale_bearing_jacobian_noise");
    const std::vector<std::vector<Point3D>> bearings = scene_bearings(scene);

    BearingGeneralizedAbsolutePoseScaleRefiner<UniformWeightVectors, TestAccumulator> refiner(bearings, scene.X,
                                                                                              scene.camera_ext);

    const ScaledCameraPose scaled_pose(scene.pose, scene.scale);
    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner), ScaledCameraPose>(refiner, scaled_pose, delta);
    REQUIRE_SMALL(jac_err, 1e-6);

    return true;
}

bool test_gen_absolute_scale_refinement() {
    const size_t N = 16;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_scale_refinement");
    add_multi_point_noise(scene.x, 1e-3, "gen_absolute_scale_refinement_noise");

    // Start away from the solution in all seven parameters
    ScaledCameraPose scaled_pose(scene.pose, scene.scale);
    scaled_pose.pose.q = quat_step_post(scaled_pose.pose.q, Eigen::Vector3d(0.02, -0.015, 0.01));
    scaled_pose.pose.t += Eigen::Vector3d(0.03, 0.02, -0.025);
    scaled_pose.scale *= 1.1;

    GeneralizedAbsolutePoseScaleRefiner refiner(scene.x, scene.X, scene.camera_ext, scene.cameras);
    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &scaled_pose, bundle_opt, print_iteration);
    log_bundle_stats(stats, "test_gen_absolute_scale_refinement");
    REQUIRE(check_bundle_cost_and_gradient(stats, 1e-6, "test_gen_absolute_scale_refinement"));

    REQUIRE_SMALL(rotation_error(scaled_pose.pose, scene.pose), 1e-2);
    REQUIRE_SMALL(translation_error(scaled_pose.pose, scene.pose), 1e-2);
    REQUIRE_SMALL(scaled_pose.scale - scene.scale, 1e-2);

    return true;
}

bool test_gen_absolute_pose_scale_ransac() {
    const size_t N = 30;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1200 800 800.0 800.0 600.0 400.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_pose_scale_ransac");
    const size_t num_outliers = add_noise_and_outliers(&scene, 0.5, 0.25, "gen_absolute_pose_scale_ransac_noise");

    AbsolutePoseOptions opt;
    opt.max_error = 2.0;
    opt.ransac.min_iterations = 100;
    opt.ransac.max_iterations = 1000;

    CameraPose pose;
    double scale = 1.0;
    std::vector<std::vector<char>> inliers;
    RansacStats stats = estimate_generalized_absolute_pose_scale(scene.x, scene.X, scene.camera_ext, scene.cameras, opt,
                                                                 &pose, &scale, &inliers);

    log_test_message("num_inliers=" + std::to_string(stats.num_inliers) +
                     ", num_outliers=" + std::to_string(num_outliers) + ", scale=" + std::to_string(scale));
    REQUIRE(stats.num_inliers + num_outliers >= Ncam * N - 2);
    REQUIRE_SMALL(rotation_error(pose, scene.pose), 1e-2);
    REQUIRE_SMALL(translation_error(pose, scene.pose), 2e-2);
    REQUIRE_SMALL(scale - scene.scale, 2e-2);

    return true;
}

bool test_gen_absolute_pose_scale_bearings_ransac() {
    const size_t N = 30;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1200 800 800.0 800.0 600.0 400.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_pose_scale_bearings_ransac");
    const size_t num_outliers =
        add_noise_and_outliers(&scene, 0.5, 0.25, "gen_absolute_pose_scale_bearings_ransac_noise");
    const std::vector<std::vector<Point3D>> bearings = scene_bearings(scene);

    AbsolutePoseOptions opt;
    // Angular threshold matching the 2 pixel threshold of the pixel-space estimator
    opt.max_error = 2.0 / camera.focal();
    opt.ransac.min_iterations = 100;
    opt.ransac.max_iterations = 1000;

    CameraPose pose;
    double scale = 1.0;
    std::vector<std::vector<char>> inliers;
    RansacStats stats = estimate_generalized_absolute_pose_scale_bearings(bearings, scene.X, scene.camera_ext, opt,
                                                                          &pose, &scale, &inliers);

    log_test_message("num_inliers=" + std::to_string(stats.num_inliers) +
                     ", num_outliers=" + std::to_string(num_outliers) + ", scale=" + std::to_string(scale));
    REQUIRE(stats.num_inliers + num_outliers >= Ncam * N - 2);
    REQUIRE_SMALL(rotation_error(pose, scene.pose), 1e-2);
    REQUIRE_SMALL(translation_error(pose, scene.pose), 2e-2);
    REQUIRE_SMALL(scale - scene.scale, 2e-2);

    return true;
}

bool test_gen_absolute_pose_scale_degenerate_rig() {
    const size_t N = 30;

    std::string camera_str = "0 PINHOLE 1200 800 800.0 800.0 600.0 400.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    AbsolutePoseOptions opt;
    opt.max_error = 2.0;
    opt.ransac.min_iterations = 100;
    opt.ransac.max_iterations = 1000;

    // A single camera cannot constrain the scale: scale * p is absorbed by the translation
    {
        Scene scene = setup_scene(rig_extrinsics(1), N, 2.4, camera, "gen_absolute_pose_scale_single_camera");

        CameraPose pose;
        double scale = 1.0;
        std::vector<std::vector<char>> inliers;
        RansacStats stats = estimate_generalized_absolute_pose_scale(scene.x, scene.X, scene.camera_ext, scene.cameras,
                                                                     opt, &pose, &scale, &inliers);

        REQUIRE_EQ_M(stats.num_inliers, size_t(0), std::string("single camera"));
        REQUIRE_EQ_M(scale, 1.0, std::string("single camera"));
    }

    // Neither can a rig which only rotates around a single center
    {
        Scene scene = setup_scene(rotating_rig_extrinsics(3), N, 2.4, camera, "gen_absolute_pose_scale_rotating_rig");

        CameraPose pose;
        double scale = 1.0;
        std::vector<std::vector<char>> inliers;
        RansacStats stats = estimate_generalized_absolute_pose_scale(scene.x, scene.X, scene.camera_ext, scene.cameras,
                                                                     opt, &pose, &scale, &inliers);

        REQUIRE_EQ_M(stats.num_inliers, size_t(0), std::string("rotating rig"));
        REQUIRE_EQ_M(scale, 1.0, std::string("rotating rig"));
    }

    // Two distinct centers are enough, even if one of them holds a single observation
    {
        Scene scene = setup_scene(rig_extrinsics(2), N, 2.4, camera, "gen_absolute_pose_scale_two_cameras");
        scene.x[1].resize(1);
        scene.X[1].resize(1);
        add_multi_point_noise(scene.x, 0.5, "gen_absolute_pose_scale_two_cameras_noise");

        CameraPose pose;
        double scale = 1.0;
        std::vector<std::vector<char>> inliers;
        RansacStats stats = estimate_generalized_absolute_pose_scale(scene.x, scene.X, scene.camera_ext, scene.cameras,
                                                                     opt, &pose, &scale, &inliers);

        REQUIRE(stats.num_inliers >= N);
        REQUIRE_SMALL(rotation_error(pose, scene.pose), 1e-2);
        REQUIRE_SMALL(scale - scene.scale, 5e-2);
    }

    return true;
}

bool test_draw_sample_distinct_centers() {
    // Two cameras with distinct centers, where all but one observation belong to the first
    const std::vector<size_t> num_pts_camera = {40, 1};
    const std::vector<size_t> center_group = {0, 1};
    const size_t sample_sz = 4;

    RNG_t rng = 0;
    std::vector<std::pair<size_t, size_t>> sample(sample_sz);
    for (size_t iter = 0; iter < 100; ++iter) {
        draw_sample_distinct_centers(sample_sz, num_pts_camera, center_group, &sample, rng);

        size_t num_second = 0;
        for (size_t k = 0; k < sample_sz; ++k) {
            REQUIRE(sample[k].second < num_pts_camera[sample[k].first]);
            if (center_group[sample[k].first] == 1) {
                num_second++;
            }
        }
        REQUIRE_EQ_M(num_second, size_t(1), test_rng::case_id("sample", iter));
    }

    // Cameras which share a center are grouped together
    std::vector<Point3D> camera_centers = {Point3D(0.0, 0.0, 0.0), Point3D(0.5, 0.0, 0.0), Point3D(0.0, 0.0, 0.0)};
    std::vector<size_t> groups;
    REQUIRE_EQ(group_camera_centers(camera_centers, &groups), size_t(2));
    REQUIRE_EQ(groups[0], groups[2]);
    REQUIRE(groups[0] != groups[1]);

    return true;
}

} // namespace test::gen_absolute_scale

using namespace test::gen_absolute_scale;
std::vector<Test> register_gen_absolute_scale_test() {
    return {TEST(test_gen_absolute_scale_jacobian),
            TEST(test_gen_absolute_scale_bearing_jacobian),
            TEST(test_gen_absolute_scale_refinement),
            TEST(test_gen_absolute_pose_scale_ransac),
            TEST(test_gen_absolute_pose_scale_bearings_ransac),
            TEST(test_gen_absolute_pose_scale_degenerate_rig),
            TEST(test_draw_sample_distinct_centers)};
}
