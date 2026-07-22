// ROS 토픽으로 신호등 렌즈 색을 실시간으로 바꾸는 Gazebo 모델 플러그인
//
// Gazebo 는 스폰된 모델의 <visual> 재질을 바꾸는 외부 API 를 제공하지 않는다.
// 기본 제공 LedPlugin 은 이 일을 하지만 SDF 에 적힌 시간표대로만 동작하고
// 런타임 제어 수단이 없다.
//
// 이 플러그인은 LedPlugin 이 내부에서 하는 것과 같은 일을 한다.
//   1) Link::VisualId() 로 렌즈 visual 의 숫자 ID 를 얻고
//   2) 그 ID 를 담은 gazebo::msgs::Visual 을 "~/visual" 토픽에 발행한다
// 다만 전환 시점을 SDF 가 아니라 ROS 토픽으로 받는다.
//
// 사용법: /traffic_light/color 에 "red" | "yellow" | "green" | "off" 발행

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gazebo/common/Plugin.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>

#include <gazebo_ros/node.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

namespace gazebo
{

class TrafficLightPlugin : public ModelPlugin
{
public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    this->model_ = model;

    // 렌즈 이름 -> (링크 이름, 켜졌을 때 색)
    this->lenses_ = {
      {"red", {"red_light_link", ignition::math::Color(1.0, 0.0, 0.0, 1.0)}},
      {"yellow", {"yellow_light_link", ignition::math::Color(1.0, 1.0, 0.0, 1.0)}},
      {"green", {"green_light_link", ignition::math::Color(0.0, 1.0, 0.0, 1.0)}},
    };

    // Gazebo transport - 렌즈 색을 실제로 바꾸는 통로
    this->gz_node_ = transport::NodePtr(new transport::Node());
    this->gz_node_->Init(model->GetWorld()->Name());
    this->visual_pub_ = this->gz_node_->Advertise<msgs::Visual>("~/visual");

    // ROS 쪽
    this->ros_node_ = gazebo_ros::Node::Get(sdf);

    std::string topic = "/traffic_light/color";
    if (sdf->HasElement("topic")) {
      topic = sdf->Get<std::string>("topic");
    }

    this->sub_ = this->ros_node_->create_subscription<std_msgs::msg::String>(
      topic, 10,
      [this](const std_msgs::msg::String::SharedPtr msg) { this->SetColor(msg->data); });

    std::string initial = sdf->HasElement("initial_color")
      ? sdf->Get<std::string>("initial_color")
      : "red";

    RCLCPP_INFO(
      this->ros_node_->get_logger(),
      "신호등 플러그인 준비됨 (토픽: %s, 초기색: %s)", topic.c_str(), initial.c_str());

    this->state_ = initial;

    // visual 갱신 메시지는 렌더러가 아직 준비되지 않았거나 큐가 밀리면
    // 조용히 유실된다. 그러면 두 불이 같이 켜지거나 다 꺼진 채로 굳는다.
    // 그래서 현재 상태를 주기적으로 다시 밀어넣어 스스로 복구되게 한다.
    this->timer_ = this->ros_node_->create_wall_timer(
      std::chrono::milliseconds(1000),
      [this]() { this->Apply(this->state_); });
  }

private:
  struct Lens
  {
    std::string link;
    ignition::math::Color color;
  };

  /// 토픽으로 색 변경 요청이 왔을 때
  void SetColor(const std::string & state)
  {
    if (this->lenses_.find(state) == this->lenses_.end()) {
      RCLCPP_WARN(this->ros_node_->get_logger(), "알 수 없는 색: %s", state.c_str());
      return;
    }
    this->state_ = state;
    this->Apply(state);
    RCLCPP_INFO(this->ros_node_->get_logger(), "신호등 -> %s", state.c_str());
  }

  /// 지정한 색만 켜고 나머지 렌즈는 끈다. (주기적으로도 호출된다)
  ///
  /// 순서가 중요하다. map 순서(green,red,yellow)대로 돌리면 어떤 색은 항상
  /// 뒤 차례라 반응이 한 박자 늦게 보인다. 끌 것을 먼저 다 끄고 켤 것을
  /// 마지막에 켜야 "겹침 없이 즉시 전환"으로 보인다.
  void Apply(const std::string & state)
  {
    for (const auto & entry : this->lenses_) {
      if (entry.first == state) {
        continue;
      }
      this->Paint(entry.second.link, entry.first, ignition::math::Color::Black, false);
    }

    auto it = this->lenses_.find(state);
    if (it != this->lenses_.end()) {
      this->Paint(it->second.link, it->first, it->second.color, true);
    }
  }

  /// 렌즈 하나를 켜거나 끈다.
  ///
  /// 중요: 필드 몇 개만 채운 메시지를 보내면 gzclient 가 무시한다.
  /// LedSetting 처럼 링크가 들고 있는 원본 visual 메시지를 통째로 복사한 뒤
  /// 색만 바꿔서 보내야 반영된다.
  void Paint(
    const std::string & link_name, const std::string & visual_name,
    const ignition::math::Color & color, bool on)
  {
    physics::LinkPtr link = this->model_->GetLink(link_name);
    if (!link) {
      RCLCPP_WARN(this->ros_node_->get_logger(), "링크를 찾을 수 없음: %s", link_name.c_str());
      return;
    }

    // 링크 하나에 <light name="red"> 와 <visual name="red"> 가 같은 이름으로
    // 들어있어서 visual 이 두 개씩 잡힌다. 이름으로 하나만 찾으면 나머지가
    // 계속 보이므로, 링크에 달린 visual 을 전부 같이 켜고 끈다.
    ignition::math::Color dim(color.R() * 0.15, color.G() * 0.15, color.B() * 0.15, 1.0);
    int count = 0;

    for (const auto & entry : link->Visuals()) {
      msgs::Visual msg = entry.second;
      msg.set_id(entry.first);
      msg.set_parent_name(link->GetScopedName());
      msg.set_parent_id(link->GetId());

      msgs::Set(msg.mutable_material()->mutable_emissive(), color);
      msgs::Set(msg.mutable_material()->mutable_ambient(), dim);
      msgs::Set(msg.mutable_material()->mutable_diffuse(), dim);

      // 색만으로는 꺼진 티가 안 나므로 꺼진 렌즈는 아예 안 보이게 만든다.
      msg.set_transparency(on ? 0.0 : 1.0);
      msg.set_visible(on);

      // 블로킹 발행(두 번째 인자 true)은 한 건씩 기다리느라 전환이 눈에 띄게
      // 느려진다. 비블로킹으로 즉시 쏘고, 유실은 위의 주기적 재적용이 메운다.
      this->visual_pub_->Publish(msg);
      ++count;
    }

    if (count == 0) {
      RCLCPP_WARN(
        this->ros_node_->get_logger(), "visual 을 찾을 수 없음: %s", link_name.c_str());
      return;
    }

    RCLCPP_DEBUG(
      this->ros_node_->get_logger(), "  %s %s (visual %d개)",
      visual_name.c_str(), on ? "ON" : "off", count);
  }

  physics::ModelPtr model_;
  std::map<std::string, Lens> lenses_;
  std::string state_{"red"};

  transport::NodePtr gz_node_;
  transport::PublisherPtr visual_pub_;

  gazebo_ros::Node::SharedPtr ros_node_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

GZ_REGISTER_MODEL_PLUGIN(TrafficLightPlugin)

}  // namespace gazebo
