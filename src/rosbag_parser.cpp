#include "pb_vi_ekf/rosbag_parser.h"

using namespace std;


int main(int argc, char* argv[])
{
  ros::init(argc, argv, "pb_vi_ekf_rosbag_parser");
  ros::Time::init();
  pbviekf::RosbagParser parser("../param/rosbag_param.yaml");
  parser.parseBag();
}


namespace pbviekf
{


RosbagParser::RosbagParser(const std::string& filename)
  : next_imu_id_(-1), next_mocap_id_(-1), next_image_id_(-1)
{
  common::get_yaml_node("bag_name", filename, filename_);
  common::get_yaml_node("imu_topic", filename, imu_topic_);
  common::get_yaml_node("mocap_topic", filename, mocap_topic_);
  common::get_yaml_node("image_topic", filename, image_topic_);
  common::get_yaml_node("start_time", filename, start_time_);
  common::get_yaml_node("duration", filename, duration_);

  Vector4d q_mocap_to_NED_pos, q_mocap_to_NED_att;
  common::get_yaml_eigen("q_mocap_to_NED_pos", filename, q_mocap_to_NED_pos);
  common::get_yaml_eigen("q_mocap_to_NED_att", filename, q_mocap_to_NED_att);
  q_mocap_to_NED_pos_ = Quatd(q_mocap_to_NED_pos);
  q_mocap_to_NED_att_ = Quatd(q_mocap_to_NED_att);

  // Get time offsets
  common::get_yaml_node("tm", "../param/param.yaml", mocap_time_offset_);
  common::get_yaml_node("tc", "../param/param.yaml", camera_time_offset_);

  openBag();

  feature_tracker_.load("../param/feat_track_param.yaml");
  ekf_.load("../param/param.yaml", bag_name_);
}


RosbagParser::~RosbagParser() {}


void RosbagParser::openBag()
{
  try
  {
    bag_.open(filename_.c_str(), rosbag::bagmode::Read);
  }
  catch(rosbag::BagIOException e)
  {
    ROS_ERROR("unable to load rosbag %s, %s", filename_.c_str(), e.what());
  }

  bag_name_ = getBagName(filename_);
  view_ = new rosbag::View(bag_);
  bag_start_ = view_->getBeginTime() + ros::Duration(start_time_);
  if (duration_ > 0)
    bag_end_ = bag_start_ + ros::Duration(duration_);
  else
    bag_end_ = view_->getEndTime();
  prog_.init(view_->size(), 20);
}


string RosbagParser::getBagName(const string &filename)
{
  string s;
  istringstream f(filename);
  while (getline(f, s, '/')) {} // grab last part of filename, including the extension
  f = istringstream(s);
  getline(f, s, '.'); // grab part before the period
  return s;
}


void RosbagParser::parseBag()
{

  for (rosbag::View::iterator it = view_->begin(); it != view_->end(); it++)
  {
    rosbag::MessageInstance m = *it;
    if (!ros::ok()) break; // break on Ctrl+C
    if (m.getTime() < bag_start_) continue; // skip messages before start time
    if (m.getTime() > bag_end_) break; // End bag after duration has passed

    // Cast datatype into proper format and call the appropriate callback
    string datatype = m.getDataType();

    if (datatype.compare("sensor_msgs/Imu") == 0)
    {
      if (!imu_topic_.empty() && imu_topic_.compare(m.getTopic()) != 0)
        continue;

      sensor_msgs::ImuConstPtr imu_msg(m.instantiate<sensor_msgs::Imu>());

      common::Imud imu;
      imu.accel << imu_msg->linear_acceleration.x,
                   imu_msg->linear_acceleration.y,
                   imu_msg->linear_acceleration.z;
      imu.gyro << imu_msg->angular_velocity.x,
                  imu_msg->angular_velocity.y,
                  imu_msg->angular_velocity.z;

      if ((imu.vec().array() != imu.vec().array()).any())
      {
        ROS_ERROR("NaNs detected in IMU data!, skipping measurement");
        continue;
      }

      imu.t = (imu_msg->header.stamp - bag_start_).toSec();
      imu.id = ++next_imu_id_;
      ekf_.imuCallback(imu);
    }

    else if (datatype.compare("geometry_msgs/PoseStamped") == 0)
    {
      if (!mocap_topic_.empty() && mocap_topic_.compare(m.getTopic()) != 0)
        continue;

      geometry_msgs::PoseStampedConstPtr pose_msg(m.instantiate<geometry_msgs::PoseStamped>());
      common::Mocapd mocap;
      mocap.transform.t_[0] = pose_msg->pose.position.x;
      mocap.transform.t_[1] = pose_msg->pose.position.y;
      mocap.transform.t_[2] = pose_msg->pose.position.z;
      mocap.transform.q_[0] = pose_msg->pose.orientation.w;
      mocap.transform.q_[1] = pose_msg->pose.orientation.x;
      mocap.transform.q_[2] = pose_msg->pose.orientation.y;
      mocap.transform.q_[3] = pose_msg->pose.orientation.z;

      // The mocap is a North, Up, East (NUE) reference frame, so we have to rotate the quaternion's
      // axis of rotation to NED by 90 deg. roll. Then we rotate that resulting quaternion by -90 deg.
      // in yaw because Leo thinks zero attitude is facing East, instead of North.
      mocap.transform.t_ = q_mocap_to_NED_pos_.rotp(mocap.transform.t_);
      mocap.transform.q_.arr_.segment<3>(1) = q_mocap_to_NED_pos_.rotp(mocap.transform.q_.arr_.segment<3>(1));
      mocap.transform.q_ = mocap.transform.q_ * q_mocap_to_NED_att_;

      if ((mocap.transform.arr_.array() != mocap.transform.arr_.array()).any())
      {
        ROS_ERROR("NaNs detected in mocap data!, skipping measurement");
        continue;
      }

      mocap.t = (pose_msg->header.stamp - bag_start_).toSec() + mocap_time_offset_;
      mocap.id = ++next_mocap_id_;
      ekf_.mocapCallback(mocap);
    }

    else if (datatype.compare("sensor_msgs/CompressedImage") == 0)
    {
      if (!image_topic_.empty() && image_topic_.compare(m.getTopic()) != 0)
        continue;

      // Convert message data into OpenCV type and create distortion mask on first image
      sensor_msgs::CompressedImageConstPtr msg(m.instantiate<sensor_msgs::CompressedImage>());
      cv_bridge::CvImagePtr cv_ptr;
      try
      {
        cv_ptr = cv_bridge::toCvCopy(msg, "mono8");
      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception %s", e.what());
      }

      // Track features in current image
      feature_tracker_.run(cv_ptr->image);

      common::Imaged image;
      image.t = (msg->header.stamp - bag_start_).toSec() + camera_time_offset_;
      image.id = ++next_image_id_;

      // Collect features in current image
      for (int i = 0; i < feature_tracker_.getFeatures().size(); ++i)
      {
        common::Featd feat;
        feat.pix = Vector2d{feature_tracker_.getFeatures()[i].x,feature_tracker_.getFeatures()[i].y};
        feat.id = feature_tracker_.getFeatureIds()[i];
        image.feats.push_back(feat);
      }

      ekf_.cameraCallback(image);
    }
  }
}


} // namespace pbviekf
