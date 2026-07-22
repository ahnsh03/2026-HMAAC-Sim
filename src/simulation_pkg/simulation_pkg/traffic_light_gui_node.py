"""
신호등 색상 제어 GUI

- 수동 탭: 버튼으로 빨강 <-> 초록 전환
- 자동 탭: 지정한 간격으로 빨강 <-> 초록 을 계속 반복

동작 원리
---------
신호등 모델(traffic_ctrl)에 traffic_light_plugin 이 붙어 있다.
그 플러그인이 /traffic_light/color 토픽을 구독하고 있다가, 값이 오면
Gazebo 안에서 렌즈의 emissive 색을 직접 바꾼다.

즉 이 노드는 토픽에 "red" / "green" 을 던지기만 한다. 모델을 다시 띄우거나
움직이지 않으므로 신호등은 제자리에 그대로 있고 색만 바뀐다.

신호등 스폰도 이 노드가 담당하므로 별도의 load_..._traffic_light_node 는
같이 실행하지 않는다.
"""

import math
import threading
import time
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
from std_msgs.msg import String

from simulation_pkg import basic


ENTITY = "traffic_light_stand"
MODEL = "traffic_ctrl"
COLOR_TOPIC = "/traffic_light/color"
DEFAULT_INTERVAL = 10  # 자동 전환 기본 간격 (초)


class TrafficLightController(Node):
    def __init__(self):
        super().__init__("traffic_light_gui_node")

        self.state = "red"

        # 어느 위치에 띄울지 (hmobility_sim / mission_sim 이 서로 다르다)
        placement = self.declare_parameter("placement", "hmobility").value

        self.color_pub = self.create_publisher(String, COLOR_TOPIC, 10)
        self.spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")

        with open(basic.get_model(MODEL)) as f:
            self.sdf = f.read()

        if placement == "mission":
            x, y, z, roll, pitch, yaw = basic.traffic_light_stand()
        else:
            x, y, z, roll, pitch, yaw = basic.hmobility_traffic_light_stand()
        self.get_logger().info(f"신호등 위치: {placement} ({x:.2f}, {y:.2f})")
        self.pose = Pose()
        self.pose.position.x = float(x)
        self.pose.position.y = float(y)
        self.pose.position.z = float(z)
        self.pose.orientation.z = math.sin(yaw * 0.5)
        self.pose.orientation.w = math.cos(yaw * 0.5)

    # ------------------------------------------------------------------
    def spawn(self):
        """신호등을 한 번만 띄운다. 이후로는 다시 띄우지 않는다."""
        if not self.spawn_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn("/spawn_entity 서비스를 찾을 수 없습니다")
            return False

        req = SpawnEntity.Request()
        req.name = ENTITY
        req.xml = self.sdf
        req.initial_pose = self.pose

        future = self.spawn_cli.call_async(req)
        end = time.time() + 10.0
        while time.time() < end and not future.done():
            time.sleep(0.02)

        res = future.result() if future.done() else None
        if res is None or not res.success:
            self.get_logger().warn(
                f"신호등 스폰 실패: {getattr(res, 'status_message', '응답 없음')}"
            )
            return False

        self.get_logger().info("신호등 스폰 완료")
        return True

    # ------------------------------------------------------------------
    def set_color(self, state):
        """색만 바꾼다. 모델은 건드리지 않는다."""
        msg = String()
        msg.data = state
        self.color_pub.publish(msg)
        self.state = state
        self.get_logger().info(f"신호등 -> {state}")


class TrafficLightGUI:
    def __init__(self, node):
        self.node = node

        self.auto_job = None
        self.interval = DEFAULT_INTERVAL
        self.remaining = 0

        self.root = tk.Tk()
        self.root.title("신호등 제어")
        self.root.geometry("320x400")
        self.root.minsize(320, 400)

        # 내용물을 컨테이너에 담고 expand 로 채우면 창 세로 가운데에 놓인다
        body = tk.Frame(self.root)
        body.pack(expand=True)

        self.canvas = tk.Canvas(body, width=80, height=160, bg="#222222",
                                highlightthickness=0)
        self.canvas.pack()
        self.lamp_red = self.canvas.create_oval(20, 10, 60, 50, fill="#400000", outline="")
        self.lamp_green = self.canvas.create_oval(20, 105, 60, 145, fill="#004000", outline="")

        # 수동 / 자동 탭
        tabs = ttk.Notebook(body)
        tabs.pack(pady=(14, 6))

        manual = tk.Frame(tabs, padx=10, pady=14)
        auto = tk.Frame(tabs, padx=10, pady=14)
        tabs.add(manual, text="  수동  ")
        tabs.add(auto, text="  자동  ")
        tabs.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.tabs = tabs
        self.manual_tab = manual

        # --- 수동 탭
        tk.Button(manual, text="빨간불 <-> 초록불 전환", height=2, width=22,
                  command=self.toggle).pack()

        # --- 자동 탭
        row = tk.Frame(auto)
        row.pack()
        tk.Label(row, text="전환 간격").pack(side="left")
        self.sec = tk.Entry(row, width=5, justify="center")
        self.sec.insert(0, str(DEFAULT_INTERVAL))
        self.sec.pack(side="left", padx=6)
        tk.Label(row, text="초").pack(side="left")

        self.auto_btn = tk.Button(auto, text="자동 전환 시작", height=2, width=22,
                                  command=self.toggle_auto)
        self.auto_btn.pack(pady=(10, 0))

        self.status = tk.Label(body, text="신호등 준비 중...", fg="#666666")
        self.status.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ------------------------------------------------------------------
    def refresh(self):
        s = self.node.state
        self.canvas.itemconfig(self.lamp_red, fill="#ff0000" if s == "red" else "#400000")
        self.canvas.itemconfig(self.lamp_green, fill="#00ff00" if s == "green" else "#004000")

    def set_state(self, state):
        self.node.set_color(state)
        self.refresh()

    def toggle(self):
        self.set_state("green" if self.node.state == "red" else "red")

    # ------------------------------------------------------------------ 자동
    def on_tab_changed(self, _event=None):
        """수동 탭으로 넘어가면 자동 전환을 멈춘다 (둘이 싸우지 않도록)."""
        if self.tabs.select() == str(self.manual_tab):
            self.stop_auto()

    def toggle_auto(self):
        if self.auto_job is not None:
            self.stop_auto()
            return

        try:
            interval = float(self.sec.get())
        except ValueError:
            self.status.config(text="숫자를 입력하세요")
            return
        if interval < 1:
            self.status.config(text="1초 이상으로 입력하세요")
            return

        self.interval = interval
        self.auto_btn.config(text="자동 전환 정지")
        self.set_state("red")
        self.remaining = interval
        self.tick()

    def tick(self):
        """간격이 끝날 때마다 빨강 <-> 초록 을 계속 번갈아 바꾼다."""
        if self.remaining <= 0:
            self.toggle()
            self.remaining = self.interval

        nxt = "초록불" if self.node.state == "red" else "빨간불"
        self.status.config(text=f"{self.remaining:.0f}초 후 {nxt}")
        self.remaining -= 1
        self.auto_job = self.root.after(1000, self.tick)

    def stop_auto(self):
        if self.auto_job is not None:
            self.root.after_cancel(self.auto_job)
            self.auto_job = None
            self.auto_btn.config(text="자동 전환 시작")
            self.status.config(text="")

    # ------------------------------------------------------------------
    def close(self):
        self.stop_auto()
        self.root.quit()
        self.root.destroy()

    def startup(self):
        if self.node.spawn():
            self.status.config(text="")
        else:
            self.status.config(text="신호등 생성 실패 - 로그 확인")
        self.refresh()

    def run(self):
        # 트랙과 거의 같이 뜨도록 최대한 빨리 스폰을 시도한다.
        # spawn() 이 /spawn_entity 서비스가 준비될 때까지 기다리므로,
        # Gazebo 가 아직 로딩 중이어도 안전하게 대기했다가 뜬다.
        self.root.after(500, self.startup)
        self.root.mainloop()


def main():
    rclpy.init()
    node = TrafficLightController()

    # rclpy.spin 을 그대로 스레드에 넘기면 종료할 때 abort 가 난다.
    # executor 를 따로 두고 먼저 정리해야 깔끔하게 닫힌다.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    gui = TrafficLightGUI(node)
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        spin.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
