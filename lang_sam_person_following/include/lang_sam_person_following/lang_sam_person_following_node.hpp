#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <lang_sam_msgs/msg/track_array.hpp>

class FollowPersonNode : public rclcpp::Node {
public:
  FollowPersonNode();

private:
  void tracksCallback(const lang_sam_msgs::msg::TrackArray::SharedPtr msg);

  rclcpp::Subscription<lang_sam_msgs::msg::TrackArray>::SharedPtr tracks_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
};