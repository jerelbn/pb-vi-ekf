#include "quad_vi_ekf.h"


namespace qviekf
{


EKF::EKF()
  : last_filter_update_(-1e9), nfa_(0), second_imu_received_(false)
{}

EKF::EKF(const string& filename, const string& name)
  : last_filter_update_(-1e9), nfa_(0), second_imu_received_(false)
{
  load(filename, name);
}

EKF::~EKF() {}


void EKF::load(const string &filename, const std::string& name)
{
  // Resize arrays to the correct sizes
  common::get_yaml_node("ekf_update_rate", filename, update_rate_);
  common::get_yaml_node("ekf_max_history_size", filename, max_history_size_);
  common::get_yaml_node("ekf_num_features", filename, nfm_);
  common::get_yaml_node("ekf_use_drag", filename, use_drag_);
  common::get_yaml_node("ekf_use_partial_update", filename, use_partial_update_);
  common::get_yaml_node("ekf_use_keyframe_reset", filename, use_keyframe_reset_);
  if (use_drag_)
    nbs_ = 17;
  else
    nbs_ = 16;
  nbd_ = nbs_ - 1;
  num_states_ = nbs_ + 3 * nfm_;
  num_dof_ = nbd_ + 3 * nfm_;

  xdot_ = VectorXd::Zero(num_dof_);
  dxp_ = VectorXd::Zero(num_dof_);
  dxm_ = VectorXd::Zero(num_dof_);
  P_ = MatrixXd::Zero(num_dof_,num_dof_);
  F_ = MatrixXd::Zero(num_dof_,num_dof_);
  A_ = MatrixXd::Zero(num_dof_,num_dof_);
  Qx_ = MatrixXd::Zero(num_dof_,num_dof_);
  I_DOF_ = MatrixXd::Zero(num_dof_,num_dof_);
  N_ = MatrixXd::Zero(num_dof_,num_dof_);
  G_ = MatrixXd::Zero(num_dof_,NI);
  B_ = MatrixXd::Zero(num_dof_,NI);
  R_cam_big_ = MatrixXd::Zero(2*nfm_,2*nfm_);
  z_cam_ = VectorXd::Zero(2*nfm_);
  h_cam_ = VectorXd::Zero(2*nfm_);
  H_cam_ = MatrixXd::Zero(2*nfm_,num_dof_);
  K_cam_ = MatrixXd::Zero(num_dof_,2*nfm_);
  H_gps_ = MatrixXd::Zero(6,num_dof_);
  K_gps_ = MatrixXd::Zero(num_dof_,6);
  H_mocap_ = MatrixXd::Zero(6,num_dof_);
  K_mocap_ = MatrixXd::Zero(num_dof_,6);

  // Initializations
  Vector3d lambda_feat;
  VectorXd x0(17), P0_base(16), Qx_base(16), lambda_base(16);
  common::get_yaml_node("ekf_rho0", filename, rho0_);
  common::get_yaml_node("ekf_init_imu_bias", filename, init_imu_bias_);
  common::get_yaml_eigen("ekf_x0", filename, x0);
  common::get_yaml_eigen("ekf_P0", filename, P0_base);
  common::get_yaml_eigen("ekf_Qx", filename, Qx_base);
  common::get_yaml_eigen_diag("ekf_Qu", filename, Qu_);
  common::get_yaml_eigen_diag("ekf_P0_feat", filename, P0_feat_);
  common::get_yaml_eigen_diag("ekf_Qx_feat", filename, Qx_feat_);
  x_ = State<double>(0, uVector::Constant(NAN), nbs_, nfm_, nfa_, x0);
  P_.topLeftCorner(nbd_,nbd_) = P0_base.topRows(nbd_).asDiagonal();
  Qx_.topLeftCorner(nbd_,nbd_) = Qx_base.topRows(nbd_).asDiagonal();
  for (int i = 0; i < nfm_; ++i)
    Qx_.block<3,3>(nbd_+3*i,nbd_+3*i) = Qx_feat_;
  for (int i = 0; i < nfm_; ++i)
    H_cam_.block<2,2>(2*i,nbd_+3*i).setIdentity();
  I_DOF_.setIdentity();

  if (use_partial_update_)
  {
    common::get_yaml_eigen("ekf_lambda", filename, lambda_base);
    common::get_yaml_eigen("ekf_lambda_feat", filename, lambda_feat);
    dx_ones_ = VectorXd::Ones(num_dof_);
    lambda_ = VectorXd::Zero(num_dof_);
    Lambda_ = MatrixXd::Zero(num_dof_,num_dof_);
    lambda_.topRows(nbd_) = lambda_base.topRows(nbd_);
    for (int i = 0; i < nfm_; ++i)
      lambda_.segment<3>(nbd_+3*i) = lambda_feat;
    Lambda_ = dx_ones_ * lambda_.transpose() + lambda_*dx_ones_.transpose() - lambda_*lambda_.transpose();
  }

  if (use_keyframe_reset_)
  {
    common::get_yaml_node("ekf_kfr_mean_pix_disparity_thresh", filename, kfr_mean_pix_disparity_thresh_);
    common::get_yaml_node("ekf_kfr_min_matches", filename, kfr_min_matches_);
    initial_keyframe_ = true;
    p_global_ = x_.p;
    q_yaw_global_ = quat::Quatd(0, 0, x_.q.yaw());
    x_.p.setZero();
    x_.q = quat::Quatd(x_.q.roll(), x_.q.pitch(), 0);
  }

  // Load sensor parameters
  Vector4d q_ub, q_um, q_uc;
  common::get_yaml_eigen_diag("ekf_R_gps", filename, R_gps_);
  common::get_yaml_eigen_diag("ekf_R_mocap", filename, R_mocap_);
  common::get_yaml_eigen_diag("ekf_R_cam", filename, R_cam_);
  common::get_yaml_eigen("p_ub", filename, p_ub_);
  common::get_yaml_eigen("q_ub", filename, q_ub);
  common::get_yaml_eigen("p_um", filename, p_um_);
  common::get_yaml_eigen("q_um", filename, q_um);
  common::get_yaml_eigen("p_uc", filename, p_uc_);
  common::get_yaml_eigen("q_uc", filename, q_uc);
  common::get_yaml_eigen("camera_matrix", filename, cam_matrix_);
  q_ub_ = quat::Quatd(q_ub.normalized());
  q_um_ = quat::Quatd(q_um.normalized());
  q_uc_ = quat::Quatd(q_uc.normalized());
  for (int i = 0; i < nfm_; ++i)
    R_cam_big_.block<2,2>(2*i,2*i) = R_cam_;
  fx_ = cam_matrix_(0,0);
  fy_ = cam_matrix_(1,1);
  u0_ = cam_matrix_(0,2);
  v0_ = cam_matrix_(1,2);
  image_center_ << u0_, v0_;

  // Logging
  std::stringstream ss_t, ss_e, ss_c;
  ss_t << "/tmp/" << name << "_ekf_truth.log";
  ss_e << "/tmp/" << name << "_ekf_est.log";
  ss_c << "/tmp/" << name << "_ekf_cov.log";
  true_state_log_.open(ss_t.str());
  ekf_state_log_.open(ss_e.str());
  cov_log_.open(ss_c.str());
}


void EKF::run(const double &t, const sensors::Sensors &sensors, const Vector3d& vw, const vehicle::Stated& x_true, const MatrixXd& lm)
{
  // Initialize IMU close to the truth (simulate flat ground calibration)
  if (init_imu_bias_)
  {
    x_.ba = sensors.getAccelBias() + 0.01 * Vector3d::Random();
    x_.bg = sensors.getGyroBias() + 0.001 * Vector3d::Random();
    P_.block<3,3>(DBA,DBA) = 0.0001 * Matrix3d::Identity();
    P_.block<3,3>(DBG,DBG) = 0.000001 * Matrix3d::Identity();
    init_imu_bias_ = false;
  }

  // Run all sensor callbacks
  if (sensors.new_imu_meas_)
    imuCallback(sensors.imu_);
  if (sensors.new_camera_meas_)
    cameraCallback(sensors.image_);
  if (sensors.new_gps_meas_)
    gpsCallback(sensors.gps_);
  if (sensors.new_mocap_meas_)
    mocapCallback(sensors.mocap_);

  // Truth logging
  if (common::round2dec(t - last_filter_update_, 6) == 0 && second_imu_received_)
    logTruth(t, sensors, x_true, lm);
}


void EKF::imuCallback(const common::Imud& z)
{
  // Store the new measurement
  new_measurements_.emplace_back(common::Measurementd(common::IMU, z.t, z));

  // Filter initialization
  if (!second_imu_received_)
  {
    if (new_measurements_.size() == 1)
    {
      x_.t = z.t;
      x_.imu = z.vec();
      x_hist_.push_back(x_);
      P_hist_.push_back(P_);
    }
    if (new_measurements_.size() == 2)
      second_imu_received_ = true;
  }

  // Update the filter at desired update rate
  if (common::round2dec(z.t - last_filter_update_, 6) >= 1.0 / update_rate_ && second_imu_received_)
  {
    filterUpdate();
    last_filter_update_ = z.t;
  }
}


void EKF::cameraCallback(const common::Imaged &z)
{
  if (second_imu_received_)
    new_measurements_.emplace_back(common::Measurementd(common::IMAGE, z.t, z));
}


void EKF::gpsCallback(const common::Gpsd &z)
{
  if (second_imu_received_)
    new_measurements_.emplace_back(common::Measurementd(common::GPS, z.t, z));
}


void EKF::mocapCallback(const common::Mocapd &z)
{
  if (second_imu_received_)
    new_measurements_.emplace_back(common::Measurementd(common::MOCAP, z.t, z));
}


vehicle::Stated EKF::getState() const
{
  vehicle::Stated x;
  x.p = x_.p;
  x.v = x_.q.rotp(x_.v);
  x.q = x_.q;
  x.omega = xdot_.segment<3>(DQ);
  return x;
}


void EKF::f(const Stated &x, const uVector &u, VectorXd &dx, const uVector &eta)
{
  Vector3d accel = u.segment<3>(UA) - x.ba - eta.segment<3>(UA);
  Vector3d omega = u.segment<3>(UG) - x.bg - eta.segment<3>(UG);

  Vector3d omega_c = q_uc_.rotp(omega);
  Vector3d v_c = q_uc_.rotp(x.v + omega.cross(p_uc_));

  dx.setZero();
  dx.segment<3>(DP) = x.q.rota(x.v);
  if (use_drag_)
    dx.segment<3>(DV) = q_ub_.rota(common::e3 * common::e3.transpose() * q_ub_.rotp(accel)) + common::gravity * x.q.rotp(common::e3) -
                        q_ub_.rota(x.mu * (common::I_3x3 - common::e3 * common::e3.transpose()) * q_ub_.rotp(x.v)) - omega.cross(x.v);
  else
    dx.segment<3>(DV) = accel + common::gravity * x.q.rotp(common::e3) - omega.cross(x.v);
  dx.segment<3>(DQ) = omega;
  for (int i = 0; i < nfm_; ++i)
  {
    Vector2d pix = x.feats[i].pix;
    double rho = x.feats[i].rho;
    dx.segment<2>(nbd_+3*i) = Omega(pix) * omega_c + rho * V(pix) * v_c;
    dx(nbd_+3*i+2) = rho * M(pix) * omega_c + rho * rho * common::e3.dot(v_c);
  }
}


void EKF::filterUpdate()
{
  // Dump new measurements into sorted container for all measurements, while getting oldest time stamp
  double t_oldest = INFINITY;
  for (auto& nm : new_measurements_)
  {
    if (t_oldest > nm.t)
      t_oldest = nm.t;
    all_measurements_.emplace(nm);
  }
  new_measurements_.clear();

  // Rewind the state and covariance to just before the oldest measurement
  while (t_oldest < (x_hist_.end()-1)->t)
  {
    x_hist_.pop_back();
    P_hist_.pop_back();
  }
  x_ = *x_hist_.rbegin();
  P_ = *P_hist_.rbegin();

  // Get iterator to oldest measurement from new set
  auto nmit = --all_measurements_.end();
  while (nmit->t > t_oldest)
    --nmit;

  // Run the filter up throuth the latest measurements
  while (nmit != all_measurements_.end())
  {
    if (nmit->type == common::IMU)
      propagate(nmit->t, nmit->imu.vec());
    if (nmit->type == common::GPS)
    {
      if (fabs(nmit->t - x_.t) > 1e-5)
        propagate(nmit->t, x_.imu);
      gpsUpdate(nmit->gps.vec());
    }
    if (nmit->type == common::MOCAP)
    {
      if (fabs(nmit->t - x_.t) > 1e-5)
        propagate(nmit->t, x_.imu);
      mocapUpdate(nmit->mocap.transform);
    }
    if (nmit->type == common::IMAGE)
    {
      if (fabs(nmit->t - x_.t) > 1e-5)
        propagate(nmit->t, x_.imu);
      cameraUpdate(nmit->image.feats);
    }
    x_hist_.push_back(x_);
    P_hist_.push_back(P_);
    ++nmit;
  }

  // Drop states exceeding the max history size
  while (x_hist_.size() > max_history_size_)
  {
    x_hist_.pop_front();
    P_hist_.pop_front();
  }

  // Drop any measurements older than the state history
  while (x_hist_.begin()->t > all_measurements_.begin()->t)
    all_measurements_.erase(all_measurements_.begin());

  // Log current estimates
  logEst();
}


void EKF::propagate(const double &t, const uVector& imu)
{
  // Time step
  double dt = t - x_.t;

  // Propagate the covariance - guarantee positive-definite P with discrete propagation
  //    analyticalFG(x_, imu, F_, G_);
  numericalFG(x_, imu, F_, G_);
  A_ = I_DOF_ + F_ * dt + F_ * F_ * dt * dt / 2.0; // Approximate state transition matrix
  B_ = (I_DOF_ + F_ * dt / 2.0) * G_ * dt;
  P_.topLeftCorner(nbd_+3*x_.nfa,nbd_+3*x_.nfa) = A_.topLeftCorner(nbd_+3*x_.nfa,nbd_+3*x_.nfa) * P_.topLeftCorner(nbd_+3*x_.nfa,nbd_+3*x_.nfa) * A_.topLeftCorner(nbd_+3*x_.nfa,nbd_+3*x_.nfa).transpose() +
      B_.topRows(nbd_+3*x_.nfa) * Qu_ * B_.topRows(nbd_+3*x_.nfa).transpose() + Qx_.topLeftCorner(nbd_+3*x_.nfa,nbd_+3*x_.nfa);

  // Trapezoidal integration on the IMU input
  f(x_, 0.5*(imu+x_.imu), xdot_);
  x_ += xdot_ * dt;
  x_.t = t;
  x_.imu = imu;
}


void EKF::cameraUpdate(const common::FeatVecd &tracked_feats)
{
  // Collect measurement of each feature in the state and remove feature states that have lost tracking
  getPixMatches(tracked_feats);

  // Apply the update
  if (x_.nfa > 0)
  {
    // Build measurement vector and model
    z_cam_.setZero();
    h_cam_.setZero();
    for (int i = 0; i < x_.nfa; ++i)
    {
      z_cam_.segment<2>(2*i) = matched_feats_[i];
      h_cam_.segment<2>(2*i) = x_.feats[i].pix;
    }

    measurementUpdate(z_cam_-h_cam_, R_cam_big_, H_cam_, K_cam_);
  }

  // Fill state with new features if needed
  if (x_.nfa < nfm_)
    addFeatToState(tracked_feats);

  if (use_keyframe_reset_)
    keyframeReset(tracked_feats);
}


void EKF::gpsUpdate(const Vector6d& z)
{
  // Measurement model and matrix
  h_gps_.head<3>() = x_.p + x_.q.rota(q_ub_.rotp(-p_ub_));
  h_gps_.tail<3>() = x_.q.rota(x_.v);

  static Vector6d hp, hm;
  static Stated xp, xm;
  static double eps = 1e-5;
  for (int i = 0; i < H_gps_.cols(); ++i)
  {
    xp = x_ + I_DOF_.col(i) * eps;
    xm = x_ + I_DOF_.col(i) * -eps;

    hp.head<3>() = xp.p + xp.q.rota(q_ub_.rotp(-p_ub_));
    hp.tail<3>() = xp.q.rota(xp.v);
    hm.head<3>() = xm.p + xm.q.rota(q_ub_.rotp(-p_ub_));
    hm.tail<3>() = xm.q.rota(xm.v);

    H_gps_.col(i) = (hp - hm) / (2.0 * eps);
  }

  // Apply the update
  measurementUpdate(z-h_gps_, R_gps_, H_gps_, K_gps_);
}


void EKF::mocapUpdate(const xform::Xformd &z)
{
  // Measurement model and matrix
  h_mocap_.t_ = x_.p + x_.q.rota(p_um_);
  h_mocap_.q_ = x_.q * q_um_;

  static xform::Xformd hp, hm;
  static Stated xp, xm;
  static double eps = 1e-5;
  for (int i = 0; i < H_mocap_.cols(); ++i)
  {
    xp = x_ + I_DOF_.col(i) * eps;
    xm = x_ + I_DOF_.col(i) * -eps;

    hp.t_ = xp.p + xp.q.rota(p_um_);
    hp.q_ = xp.q * q_um_;
    hm.t_ = xm.p + xm.q.rota(p_um_);
    hm.q_ = xm.q * q_um_;

    H_mocap_.col(i) = (hp - hm) / (2.0 * eps);
  }

  // Apply the update
  measurementUpdate(z-h_mocap_, R_mocap_, H_mocap_, K_mocap_);
}


void EKF::measurementUpdate(const VectorXd &err, const MatrixXd& R, const MatrixXd &H, MatrixXd &K)
{
  K = P_ * H.transpose() * (R + H * P_ * H.transpose()).inverse();
  if (use_partial_update_)
  {
    x_ += lambda_.cwiseProduct(K * err);
    P_ += Lambda_.cwiseProduct((I_DOF_ - K * H) * P_ * (I_DOF_ - K * H).transpose() + K * R * K.transpose() - P_);
  }
  else
  {
    x_ += K * err;
    P_ = (I_DOF_ - K * H) * P_ * (I_DOF_ - K * H).transpose() + K * R * K.transpose();
  }
}


void EKF::getPixMatches(const common::FeatVecd &tracked_feats)
{
  int i = 0;
  matched_feats_.clear();
  while (i < x_.nfa)
  {
    // Look for an ID match to the current state feature
    bool id_match_found = false;
    for (auto& tf : tracked_feats)
    {
      if (x_.feats[i].id == tf.id)
      {
        matched_feats_.push_back(tf.pix);
        id_match_found = true;
        break;
      }
    }

    // Remove feature from state when no longer tracked
    if (!id_match_found)
      removeFeatFromState(i);
    else
      ++i;
  }

  if (matched_feats_.size() != x_.nfa)
    cerr << "Matched camera measurements does not match number of active features!" << endl;
}


void EKF::removeFeatFromState(const int &idx)
{
  // Shift and reset state and covariance
  int j;
  for (j = idx; j < x_.nfa-1; ++j)
  {
    x_.feats[j] = x_.feats[j+1];
    P_.row(nbd_+3*j) = P_.row(nbd_+3*(j+1));
    P_.row(nbd_+3*j+1) = P_.row(nbd_+3*(j+1)+1);
    P_.row(nbd_+3*j+2) = P_.row(nbd_+3*(j+1)+2);
    P_.col(nbd_+3*j) = P_.col(nbd_+3*(j+1));
    P_.col(nbd_+3*j+1) = P_.col(nbd_+3*(j+1)+1);
    P_.col(nbd_+3*j+2) = P_.col(nbd_+3*(j+1)+2);
  }
  x_.feats[j] = common::Featd();
  P_.row(nbd_+3*j).setZero();
  P_.row(nbd_+3*j+1).setZero();
  P_.row(nbd_+3*j+2).setZero();
  P_.col(nbd_+3*j).setZero();
  P_.col(nbd_+3*j+1).setZero();
  P_.col(nbd_+3*j+2).setZero();

  // Decrement the active feature counter
  --x_.nfa;
}


void EKF::addFeatToState(const common::FeatVecd &tracked_feats)
{
  // Sort features by proximity to image center
  auto sorted_feats = tracked_feats;
  sort(sorted_feats.begin(), sorted_feats.end(),
       [this](const common::Featd& f1, const common::Featd& f2)
       {return (f1.pix - image_center_).norm() < (f2.pix - image_center_).norm();});

  // Loop through features, adding the best ones to the state
  for (auto& f : sorted_feats)
  {
    // Check if current sorted feature ID is already in the state then add it to the state
    bool id_used = false;
    for (auto sf : x_.feats)
    {
      if (f.id == sf.id)
      {
        id_used = true;
        break;
      }
    }

    if (!id_used)
    {
      // Initialize feature state and corresponding covariance block
      x_.feats[x_.nfa].id = f.id;
      x_.feats[x_.nfa].pix = f.pix;
      x_.feats[x_.nfa].rho = rho0_;
      P_.block<3,3>(nbd_+3*x_.nfa,nbd_+3*x_.nfa) = P0_feat_;

      // Increment number of active features
      ++x_.nfa;
    }

    // Don't try adding more features than allowed
    if (x_.nfa == nfm_) break;
  }
}


void EKF::keyframeReset(const common::FeatVecd &tracked_feats)
{
  // Calculate mean pixel disparity between current features and keyframe
  double pix_disparity = 0;
  int match_count = 0;
  for (auto& tf : tracked_feats)
  {
    for (auto& kff : kf_feats_)
    {
      if (tf.id == kff.id)
      {
        pix_disparity = (match_count * pix_disparity + (tf.pix - kff.pix).norm()) / (match_count + 1);
        ++match_count;
      }
    }
  }

  if (initial_keyframe_ || pix_disparity >= kfr_mean_pix_disparity_thresh_ || match_count < kfr_min_matches_)
  {
    // Update keyframe tracked features, position, attitude
    kf_feats_ = tracked_feats;
    p_global_ += q_yaw_global_.rota(x_.p);
    q_yaw_global_ = q_yaw_global_ * quat::Quatd(0, 0, x_.q.yaw());

    if (initial_keyframe_)
    {
      initial_keyframe_ = false;
      return;
    }

    // Compute covariance update Jacobian
    numericalN(x_, N_);

    // Reset state and covariance
    x_.p.setZero();
    x_.q = quat::Quatd(x_.q.roll(), x_.q.pitch(), 0);
    P_ = N_ * P_ * N_.transpose();
  }
}


void EKF::analyticalFG(const Stated &x, const uVector &u, MatrixXd &F, MatrixXd &G)
{
  F.setZero();
  F.block<3,3>(DP,DV) = x.q.inverse().R();
  F.block<3,3>(DP,DQ) = -x.q.inverse().R() * common::skew(x.v);
  F.block<3,3>(DV,DV) = -common::skew(u.segment<3>(UG) - x.bg);
  F.block<3,3>(DV,DQ) = common::gravity * common::skew(x.q.rotp(common::e3));
  F.block<3,3>(DV,DBA) = -common::I_3x3;
  F.block<3,3>(DV,DBG) = -common::skew(x.v);
  F.block<3,3>(DQ,DQ) = -common::skew(u.segment<3>(UG) - x.bg);
  F.block<3,3>(DQ,DBG) = -common::I_3x3;

  G.setZero();
  G.block<3,3>(DV,UA) = -common::I_3x3;
  G.block<3,3>(DV,UG) = -common::skew(x.v);
  G.block<3,3>(DQ,UG) = -common::I_3x3;
}


void EKF::numericalFG(const Stated &x, const uVector &u, MatrixXd &F, MatrixXd &G)
{
  static const double eps(1e-5);
  static Matrix6d I6 = Matrix6d::Identity();
  static uVector eta = Vector6d::Ones();
  static Stated xp, xm;
  static uVector etap, etam;

  for (int i = 0; i < F.cols(); ++i)
  {
    xp = x + I_DOF_.col(i) * eps;
    xm = x + I_DOF_.col(i) * -eps;

    f(xp, u, dxp_);
    f(xm, u, dxm_);

    F.col(i) = (dxp_ - dxm_) / (2.0 * eps);
  }
  F.block<3,3>(DQ,DQ) = -common::skew(u.segment<3>(UG) - x.bg);

  for (int i = 0; i < G.cols(); ++i)
  {
    etap = eta + I6.col(i) * eps;
    etam = eta + I6.col(i) * -eps;

    f(x, u, dxp_, etap);
    f(x, u, dxm_, etam);

    G.col(i) = (dxp_ - dxm_) / (2.0 * eps);
  }
}


void EKF::numericalN(const Stated &x, MatrixXd &N)
{
  static const double eps(1e-5);
  static Stated xp, xm, x_plusp, x_plusm;
  for (int i = 0; i < N.cols(); ++i)
  {
    xp = x + I_DOF_.col(i) * eps;
    x_plusp = xp;
    x_plusp.p.setZero();
    x_plusp.q = quat::Quatd(xp.q.roll(), xp.q.pitch(), 0);

    xm = x + I_DOF_.col(i) * -eps;
    x_plusm = xm;
    x_plusm.p.setZero();
    x_plusm.q = quat::Quatd(xm.q.roll(), xm.q.pitch(), 0);

    N.col(i) = (x_plusp - x_plusm) / (2.0 * eps);
  }
}


void EKF::logTruth(const double &t, const sensors::Sensors &sensors, const vehicle::Stated& xb_true, const MatrixXd& lm)
{
  // This filter state is in the IMU frame but truth is in the body frame
  Vector3d p_iu = xb_true.p + xb_true.q.rota(q_ub_.rotp(-p_ub_));
  Vector3d v_ui = q_ub_.rota(xb_true.v + xb_true.omega.cross(q_ub_.rotp(-p_ub_)));
  quat::Quatd q_iu = xb_true.q * q_ub_.inverse();

  true_state_log_.log(t);
  true_state_log_.logMatrix(p_iu, v_ui, q_iu.euler(), sensors.getAccelBias(), sensors.getGyroBias());
  if(use_drag_)
    true_state_log_.log(xb_true.drag);

  // Compute true landmark pixel measurement
  for (int i = 0; i < nfm_; ++i)
  {
    if (i+1 <= x_.nfa)
    {
      // Find the inertial landmark that matches the current state feature label
      Vector3d lmi;
      for (int j = 0; j < lm.cols(); ++j)
      {
        if (j == x_.feats[i].id)
        {
          lmi = lm.col(j);
          break;
        }
      }

      // Calculate the true vector from camera to landmark in the camera frame
      quat::Quatd q_i2c = xb_true.q * q_ub_.inverse() * q_uc_;
      Vector3d p_i2c = xb_true.p + xb_true.q.rota(q_ub_.rotp(p_uc_ - p_ub_));
      Vector3d lmc = q_i2c.rotp(lmi - p_i2c);

      // Compute pixel position and inverse z-depth
      Vector2d pix;
      common::projToImg(pix, lmc, cam_matrix_);
      double rho = 1.0 / lmc(2);

      // Log the data
      true_state_log_.logMatrix(pix);
      true_state_log_.log(rho);
    }
    else
    {
      static const Vector2d a(NAN,NAN);
      static const double b(NAN);
      true_state_log_.logMatrix(a);
      true_state_log_.log(b);
    }
  }
}


void EKF::logEst()
{
  Vector3d p;
  quat::Quatd q;
  if (use_keyframe_reset_)
  {
    p = p_global_ + q_yaw_global_.rota(x_.p);
    q = quat::Quatd(x_.q.roll(), x_.q.pitch(), q_yaw_global_.yaw() + x_.q.yaw());
  }
  else
  {
    p = x_.p;
    q = x_.q;
  }

  ekf_state_log_.log(x_.t);
  ekf_state_log_.logMatrix(p, x_.v, q.euler(), x_.ba, x_.bg);
  if (use_drag_)
    ekf_state_log_.log(x_.mu);
  for (int i = 0; i < nfm_; ++i)
  {
    if (i+1 <= x_.nfa)
    {
      ekf_state_log_.logMatrix(x_.feats[i].pix);
      ekf_state_log_.log(x_.feats[i].rho);
    }
    else
    {
      static const Vector2d a(NAN,NAN);
      static const double b(NAN);
      ekf_state_log_.logMatrix(a);
      ekf_state_log_.log(b);
    }
  }

  P_diag_ = P_.diagonal();
  cov_log_.log(x_.t);
  cov_log_.logMatrix(P_diag_);
}


Matrix<double,2,3> EKF::Omega(const Vector2d &nu)
{
  double eps_x = nu(0) - u0_;
  double eps_y = nu(1) - v0_;

  Matrix<double,2,3> m;
  m(0,0) = eps_x * eps_y / fy_;
  m(0,1) = -(fx_ + eps_x * eps_x / fx_);
  m(0,2) = fx_ / fy_ * eps_y;
  m(1,0) = fy_ + eps_y * eps_y / fy_;
  m(1,1) = -eps_x * eps_y / fx_;
  m(1,2) = -fy_ / fx_ * eps_x;
  return m;
}


Matrix<double,2,3> EKF::V(const Vector2d &nu)
{
  double eps_x = nu(0) - u0_;
  double eps_y = nu(1) - v0_;

  Matrix<double,2,3> m;
  m.setZero();
  m(0,0) = -fx_;
  m(0,2) = eps_x;
  m(1,1) = -fy_;
  m(1,2) = eps_y;
  return m;
}


RowVector3d EKF::M(const Vector2d &nu)
{
  double eps_x = nu(0) - u0_;
  double eps_y = nu(1) - v0_;

  RowVector3d m;
  m(0) = eps_y / fy_;
  m(1) = -eps_x / fx_;
  m(2) = 0;
  return m;
}


} // namespace qviekf
