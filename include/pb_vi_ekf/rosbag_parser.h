#include <iostream>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseStamped.h>
#include <image_transport/image_transport.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include "common_cpp/common.h"
#include "common_cpp/progress_bar.h"
#include "common_cpp/measurement.h"
#include "geometry/quat.h"
#include "geometry/xform.h"
#include "feature_tracker/feature_tracker.h"
#include "pb_vi_ekf/ekf.h"

using namespace std;


namespace pbviekf
{


class RosbagParser
{
public:

  RosbagParser(const std::string& filename);
  ~RosbagParser();
  void parseBag();

private:

  void openBag();
  void imageUpdate(const double& t, const sensor_msgs::CompressedImageConstPtr& msg_ptr);
  string getBagName(const string& filename);

  string filename_= "";
  string bag_name_;
  string imu_topic_= "";
  string mocap_topic_= "";
  string image_topic_ = "";
  double start_time_ = 0;
  double duration_ = INFINITY;
  rosbag::Bag bag_;
  rosbag::View* view_;

  double end_time_ = INFINITY;
  double mocap_time_offset_;
  double camera_time_offset_;

  ros::Time bag_start_;
  ros::Time bag_end_;
  common::ProgressBar prog_;
  quat::Quatd q_mocap_to_NED_pos_, q_mocap_to_NED_att_;

  int next_imu_id_, next_mocap_id_, next_image_id_;
  tracker::FeatureTracker feature_tracker_;
  pbviekf::EKF ekf_;

};


} // namespace pbviekf
