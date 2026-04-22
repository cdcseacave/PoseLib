// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "PoseLib/camera_pose.h"
#include "PoseLib/types.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Returns MSAC score of the reprojection error
double compute_msac_score(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                          double sq_threshold, size_t *inlier_count);
double compute_msac_score(const Image &image, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                          double sq_threshold, size_t *inlier_count);

double compute_msac_score(const CameraPose &pose, const std::vector<Line2D> &lines2D,
                          const std::vector<Line3D> &lines3D, double sq_threshold, size_t *inlier_count);
// MSAC score of the reprojection error on projected 3D points
double compute_msac_score(const CameraPose &pose, double focal, const std::vector<Point2D> &x,
                          const std::vector<Point3D> &X, double sq_threshold, size_t *inlier_count);

// Returns MSAC score of the Sampson error
double compute_sampson_msac_score(const CameraPose &pose, const std::vector<Point2D> &x1,
                                  const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count);
double compute_sampson_msac_score(const Eigen::Matrix3d &F, const std::vector<Point2D> &x1,
                                  const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count);

// Bearing-vector variants for central camera models (pinhole, spherical, fisheye, ...).
// The inputs are 3D unit (or near-unit) bearing vectors in camera space; for spherical
// cameras these preserve hemisphere information (sign(z)) that the 2D Point2D form loses.
// For pinhole bearings with z>0 these reduce algebraically to the 2D Point2D variants,
// so routing pinhole data through this path is zero-regression.
//
// compute_msac_score_bearing (absolute pose): residual is the chord distance squared
// between the observed unit bearing and the predicted bearing normalize(R*X + t).
// No cheirality check — works for full-sphere cameras.
//
// sq_threshold is the squared chord-distance threshold in unit-bearing coordinates.
// Callers should convert any pixel-domain threshold to an angular threshold in radians
// using their camera model/calibration, then convert that angle via 2*sin(angle/2)
// to obtain the corresponding chord-distance threshold.
double compute_msac_score_bearing(const CameraPose &pose, const std::vector<Point3D> &bearings,
                                  const std::vector<Point3D> &X, double sq_threshold, size_t *inlier_count);

// compute_sampson_msac_score_bearing (relative pose): Sampson-on-the-sphere.
// Algebraically identical to the 2D Sampson formula with (x.x, x.y, 1) replaced by
// (b.x, b.y, b.z); for pinhole bearings this reduces to compute_sampson_msac_score.
//
// Cheirality is enforced via the existing check_cheirality(pose, b1, b2) overload,
// which operates directly on unit bearings. sq_threshold is in squared Sampson-error
// units for this residual (approximately radians^2 in the small-error limit).
double compute_sampson_msac_score_bearing(const CameraPose &pose, const std::vector<Point3D> &bearings1,
                                          const std::vector<Point3D> &bearings2, double sq_threshold,
                                          size_t *inlier_count, bool check_cheirality_flag = true);

// Bearing-vector inlier selection for absolute pose (chord-distance threshold,
// no cheirality check — full-sphere). Threshold is sq_threshold in chord-distance
// units.
void get_inliers_abs_bearing(const CameraPose &pose, const std::vector<Point3D> &bearings,
                             const std::vector<Point3D> &X, double sq_threshold, std::vector<char> *inliers);

// Bearing-vector inlier selection for relative pose (Sampson-on-the-sphere, same
// (x,y)-subspace form as compute_sampson_msac_score_bearing). Returns the number
// of inliers. Optional cheirality check via check_cheirality(pose, b1, b2).
int get_inliers_rel_bearing(const CameraPose &pose, const std::vector<Point3D> &bearings1,
                            const std::vector<Point3D> &bearings2, double sq_threshold, std::vector<char> *inliers,
                            bool check_cheirality_flag = true);

// Returns MSAC score for the Tangent Sampson error (Terekhov and Larsson, CVPR 2023)
double compute_tangent_sampson_msac_score(const Eigen::Matrix3d &F, const std::vector<Point2D> &x1,
                                          const std::vector<Point2D> &x2, const Camera &cam1, const Camera &cam2,
                                          double sq_threshold, size_t *inlier_count);
// Returns MSAC score for the Tangent Sampson error (Terekhov and Larsson, CVPR 2023)
// with pre-computed unprojections and unprojection jacobians
double compute_tangent_sampson_msac_score(const CameraPose &pose, const std::vector<Point3D> &d1,
                                          const std::vector<Point3D> &d2,
                                          const std::vector<Eigen::Matrix<double, 3, 2>> &M1,
                                          const std::vector<Eigen::Matrix<double, 3, 2>> &M2, double sq_threshold,
                                          size_t *inlier_count);

// Returns MSAC score of transfer error for homography
double compute_homography_msac_score(const Eigen::Matrix3d &H, const std::vector<Point2D> &x1,
                                     const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count);

// Compute inliers for absolute pose estimation (using reprojection error and cheirality check)
void get_inliers(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                 double sq_threshold, std::vector<char> *inliers);
void get_inliers(const Image &image, const std::vector<Point2D> &x, const std::vector<Point3D> &X, double sq_threshold,
                 std::vector<char> *inliers);
void get_inliers(const CameraPose &pose, const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                 double sq_threshold, std::vector<char> *inliers);
// Compute inliers for relative pose with monodepth by using reprojection error
void get_inliers(const CameraPose &pose, double focal, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                 double sq_threshold, std::vector<char> *inliers);

// Compute inliers for relative pose estimation (using Sampson error)
int get_inliers(const CameraPose &pose, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                double sq_threshold, std::vector<char> *inliers);
int get_inliers(const Eigen::Matrix3d &E, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                double sq_threshold, std::vector<char> *inliers);

// Compute inliers for relative pose + distortion estimation using Tangent Sampson Error (Terekhov and Larsson, CVPR
// 2023)
int get_tangent_sampson_inliers(const Eigen::Matrix3d &F, const Camera &cam1, const Camera &cam2,
                                const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, double sq_threshold,
                                std::vector<char> *inliers);
int get_tangent_sampson_inliers(const CameraPose &pose, const std::vector<Point3D> &d1, const std::vector<Point3D> &d2,
                                const std::vector<Eigen::Matrix<double, 3, 2>> &M1,
                                const std::vector<Eigen::Matrix<double, 3, 2>> &M2, double sq_threshold,
                                std::vector<char> *inliers);

// inliers for homography
void get_homography_inliers(const Eigen::Matrix3d &H, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                            double sq_threshold, std::vector<char> *inliers);

// Helpers for the 1D radial camera model
double compute_msac_score_1D_radial(const CameraPose &pose, const std::vector<Point2D> &x,
                                    const std::vector<Point3D> &X, double sq_threshold, size_t *inlier_count);
void get_inliers_1D_radial(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                           double sq_threshold, std::vector<char> *inliers);

// Normalize points by shifting/scaling coordinate systems.
double normalize_points(std::vector<Eigen::Vector2d> &x1, std::vector<Eigen::Vector2d> &x2, Eigen::Matrix3d &T1,
                        Eigen::Matrix3d &T2, bool normalize_scale, bool normalize_centroid, bool shared_scale);

// Calculate whether F would yield real focals, assumes both pp at [0, 0]
bool calculate_RFC(const Eigen::Matrix3d &F);

} // namespace poselib