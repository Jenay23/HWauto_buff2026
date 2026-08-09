# buff_algorithm

RoboMaster 能量机关（Buff / 大符 / 小符）自动瞄准算法库。

从 Hello-World-Vision/autoaim2026
中抽离而来的纯 C++ 实现，**不依赖 ROS2**，只依赖 `Eigen3` 与 `OpenCV`，
方便集成到任意机器人软件框架中。

## 功能范围

覆盖能量机关自瞄链路中"检测结果之后"的部分：

- **selector**（目标筛选）：从多个检测到的扇叶中选出应该瞄准的目标
- **locate**（位姿解算）：对扇叶 4 个角点做平面 PnP 解算，处理镜像歧义的双解选择
- **tracker**（状态跟踪）：
  - 小符：固定角速度模型 + 卡尔曼滤波跟踪 roll/yaw
  - 大符：基于 RANSAC 的正弦转速模型拟合 + 卡尔曼滤波跟踪 yaw
  - 通用的跟踪状态机（LOST / CONVERGING / TRACKING / TEMP_LOST）
- **predictor**（位置预测）：根据跟踪到的运动模型，外推目标未来某一时刻的位置

不包含的部分（仍在主仓库中，与神经网络检测框架/具体机器人平台强相关）：

- 神经网络检测与传统视觉角点优化（refinator）
- ROS2 节点封装、消息定义、tf 坐标变换、可视化 marker
- 弹道解算 / 云台控制

## 依赖

- C++20
- Eigen3
- OpenCV (使用了 `cv::FileStorage` 读取参数、`cv::solvePnP`/`solvePnPGeneric` 做 PnP 解算)

## 目录结构

```
include/buff_algo/
  common/       # 公共数据类型 (BuffDetection/BuffState/Pose3f)、数学工具、EMA滤波器
  tracker/      # 卡尔曼滤波器、RANSAC正弦拟合、跟踪状态机、BuffTracker
  predictor/    # BuffPredictor 位置外推
  locate/       # BuffPnPSolver 位姿解算
  selector/     # BuffSelector 目标筛选
src/            # 对应的实现文件
config/         # 示例参数文件（YAML，cv::FileStorage 格式）
examples/       # 独立可运行的示例程序
```

## 编译

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./buff_algo_example
```

也可以直接把本目录作为子目录用 `add_subdirectory` 集成到你自己的 CMake 项目中，
链接 `buff_algorithm` 这个 target 即可。

## 快速上手

核心数据流：

```
BuffDetection (检测网络输出的4个角点)
    -> BuffSelector::select_buffs      // 筛选出应该打的那个目标
    -> BuffPnPSolver::solve_pnp        // 解算出相机坐标系下的位姿 Pose3f
    -> Buff(pose)                      // 构造观测量（换算出 R 字中心、roll角等）
    -> BuffTracker::push / update      // 跟踪滤波，得到 BuffState
    -> BuffPredictor::set_state / predict_position  // 预测提前量位置
```

具体用法可参考 `examples/example.cpp`，其中用构造的假数据模拟了一个绕
R 字中心匀速旋转的小符扇叶，完整跑通上述链路。

注意：`BuffPnPSolver`/`BuffSelector` 只解算相机坐标系下的位姿，
如果你的机器人云台/底盘会转动，需要在外部把相机坐标系下的位姿变换到
一个相对静止的参考系（例如底盘 yaw 轴坐标系）后再送入 `BuffTracker`，
否则跟踪器无法正确区分"目标运动"和"自身云台运动"。这部分坐标变换
与具体机器人平台强相关，不包含在本库中。

## 许可证

MIT License，见 [LICENSE](./LICENSE)。

本项目还包含第三方库 [termcolor](https://github.com/ikalnytskyi/termcolor)（BSD 3-Clause），
详见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。
