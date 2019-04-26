#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <deque>
#include "state.h"
#include "common_cpp/common.h"
#include "common_cpp/logger.h"
#include "common_cpp/measurement.h"
#include "geometry/quat.h"
#include "geometry/xform.h"
#include "geometry/support.h"

using namespace std;
using namespace Eigen;


namespace pbviekf
{


class EKF
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EKF();
  EKF(const string& filename, const string &name);
  ~EKF();

  void load(const string &filename, const string &name);
  void imuCallback(const common::Imud &z);
  void cameraCallback(const common::Imaged& z);
  void gpsCallback(const common::Gpsd &z);
  void mocapCallback(const common::Mocapd& z);
  void logTruth(const double& t, const Vector3d& p_t, const Vector3d& v_t, const quat::Quatd& q_t,
                const Vector3d& ba_t, const Vector3d& bg_t, const double& mu_t, const Vector3d &omegab_t, const MatrixXd& lm);

  void setVelocity(const Vector3d& v) { x_.v = v; }
  void setAttitude(const Vector4d& q) { x_.q.arr_ = q; }
  void setDrag(const double& mu) { x_.mu = mu; }
  const bool& getFilterUpdateStatus() const { return just_updated_filter_; }
  const Stated& getState() const { return x_; }
  const MatrixXd& getCov() const { return P_; }
  const Vector3d& getGlobalPosition() const
  {
    if (use_keyframe_reset_)
      return p_global_;
    else
      return x_.p;
  }
  quat::Quatd getGlobalAttitude() const
  {
    if (use_keyframe_reset_)
      return quat::Quatd(x_.q.roll(), x_.q.pitch(), q_yaw_global_.yaw());
    else
      return x_.q;
  }

private:
  void f(const Stated &x, const uVector& u, VectorXd &dx, const uVector& eta = uVector::Zero());
  void filterUpdate();
  void propagate(const double &t, const uVector &imu);
  void accelUpdate(const Vector2d &z);
  void cameraUpdate(const common::FeatVecd &tracked_feats);
  void gpsUpdate(const Vector6d& z);
  void mocapUpdate(const xform::Xformd& z);
  void measurementUpdate(const VectorXd& err, const MatrixXd &R, const Block<MatrixXd> &H, Block<MatrixXd> &K);
  void getPixMatches(const common::FeatVecd& tracked_feats);
  void removeFeatFromState(const int& idx);
  void addFeatToState(const common::FeatVecd& tracked_feats);
  void keyframeReset(const common::FeatVecd& tracked_feats);
  void analyticalFG(const Stated &x, const uVector& u, Block<MatrixXd> &F, Block<MatrixXd> &G);
  void numericalFG(const Stated &x, const uVector& u, Block<MatrixXd> &F, Block<MatrixXd> &G);
  void numericalN(const Stated &x, Block<MatrixXd>& N);
  void logEst();

  Matrix<double,2,3> Omega(const Vector2d& nu);
  Matrix<double,2,3> V(const Vector2d& nu);
  RowVector3d M(const Vector2d& nu);

  // Primary variables
  bool just_updated_filter_;
  bool enable_accel_update_;
  bool second_imu_received_;
  double update_rate_, last_filter_update_;
  bool use_drag_, use_partial_update_, use_keyframe_reset_;
  double rho0_;
  bool init_imu_bias_;
  State<double> x_;
  VectorXd xdot_, dxp_, dxm_, lambda_, dx_ones_;
  MatrixXd P_, F_, A_, Qx_, G_, B_, Lambda_, N_;
  Matrix3d P0_feat_, Qx_feat_;
  uMatrix Qu_;
  MatrixXd I_DOF_;
  VectorXd P_diag_;
  vector<Vector2d, aligned_allocator<Vector2d>> matched_feats_;

  int max_history_size_;
  common::Measurementsd all_measurements_;
  vector<common::Measurementd> new_measurements_;
  deque<State<double>> x_hist_;
  deque<MatrixXd> P_hist_;

  // Keyframe reset parameters
  bool initial_keyframe_;
  int kfr_min_matches_;
  double kfr_mean_pix_disparity_thresh_;
  common::FeatVecd kf_feats_;
  Vector3d p_global_;
  quat::Quatd q_yaw_global_;

  // Sensor parameters
  Vector2d h_acc_;
  MatrixXd H_acc_, K_acc_;
  Matrix2d R_acc_;
  Vector6d h_gps_;
  MatrixXd H_gps_, K_gps_;
  Matrix6d R_gps_;
  xform::Xformd h_mocap_;
  MatrixXd H_mocap_, K_mocap_;
  Matrix6d R_mocap_;
  VectorXd z_cam_, h_cam_;
  MatrixXd H_cam_, K_cam_;
  Matrix2d R_cam_;
  MatrixXd R_cam_big_;
  Vector3d p_ub_, p_um_, p_uc_;
  quat::Quatd q_ub_, q_um_, q_uc_;
  Matrix3d cam_matrix_;
  double fx_, fy_, u0_, v0_;
  Vector2d image_center_;

  // Logging
  common::Logger true_state_log_;
  common::Logger ekf_state_log_;
  common::Logger cov_log_;
};


} // namespace pbviekf
