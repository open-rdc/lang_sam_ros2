#include <rclcpp/rclcpp.hpp>
#include "lang_sam_person_following/lang_sam_person_following_node.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FollowPersonNode>());
  rclcpp::shutdown();
  return 0;
}