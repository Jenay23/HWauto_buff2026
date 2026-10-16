# bridge — 把 HWauto_buff2026 接到 Daedalus 模拟器，实现**自动打击**

**位置**：`HWauto_buff2026/bridge/`（就在 HW26 仓库根目录下）

**文档口径**：本文件已按 **2026-10-12 的当前代码**重写（分支 `ono`，HEAD `85b8bdf`）。
此前版本描述的双检测后端（sp25）**在代码里已经不存在**，见 §1.5.1；旧版 §7 里"改用
`detector: sp25`"一类建议是错的，已删除。

**一句话**：这是 HW26 的"后半截"——HW26 原本只到"相机系三维打击点"就停了（无云台角、无弹道、无开火）。本 bridge 补齐这段，并把它接到模拟器上，形成**全自动闭环**：

```text
识别 → PnP → 跟踪滤波 → 预测 → 弹道解算 → 云台 yaw/pitch → 开火判定
        全部自动，操作手不需要碰任何东西
```

**完全不需要 TensorRT / CUDA / GPU。**

**架构是"取百家之长"**：

| 环节 | 来源 | 为什么 |
|---|---|---|
| 检测 | **HW26 自带的 YOLO11s-pose**（转成 OpenVINO，`model/buff_ir.xml`） | 9 关键点，**分类 颜色 + 激活**（4 类：`b/r × inactive/active`）—— 大符双靶标策略需要这个信息；转盘中心 `keypoints[8]` 是**直接输出**，不用外推 |
| PnP | 本桥自实现（IPPE + 双解连续性消歧） | 物体点半边长按后端取值；HW26 的 `BUFF_POINTS = 0.114` 是它实车靶板的尺寸，用在模拟器上会让距离偏大 **10.2%**。所以**不调用 HW26 的 `BuffPnPSolver`**，目标检测和定位解算都在桥里完成，不需要改 HW26 源码 |
| **跟踪 / RANSAC / 预测** | **HW26 原装**（四态状态机 + `KF<1>` + RANSAC 正弦拟合） | 到 TRACKING 后它的正弦外推是准的 |
| **短时相位估计** | **桥自己的** `local_tracker`（`local_model` 0–6） | HW26 的 RANSAC 要 35 个内点，点亮窗口装不下；桥的局部拟合 6~8 个样本即可用（§1.7） |
| 云台角 / 弹道 / 开火 | 本桥补齐 | HW26 原本没有这一段 |
| 枪口外参 | **模拟器文档给的实测值** | `POWER_RUNE_INTERFACE.md:105` |

> **检测后端由模型的输出形状自动判定**，配置里只写一个 `model:` 路径：
> 只接受 `[1, 5040, 26]`（HW26，9 点）。**其它形状会启动即 FATAL 抛异常**，
> 不是静默回退 —— 所以不存在"开关与路径不匹配"的静默错误，但也不存在备用后端。

---

## 1. 自动打击是怎么实现的

| 环节 | 谁做的 | 说明 |
|---|---|---|
| 伺服跟瞄 | 本 bridge | 每帧发 `cmd/gimbal {yaw, pitch, roll:0, fire, control}`，模拟器云台以 18 rad/s 自动转向 |
| 开火 | 本 bridge | `control` 有效且到位后自动发 `fire: true` |
| 开火条件 | 本 bridge | `hypot(Δyaw, Δpitch) ≤ fire_err_rad` **连续 `fire_stable_frames` 帧** + `fire_cooldown` 冷却 |
| 失败保护 | 本 bridge | 无目标/图像过期（`max_image_age_ms`，取 `min(300, stale_ms)` = **200 ms**）→ `control: false`，云台保持不动，绝不在失去目标时开火 |
| 提前量 | 本 bridge | `lead = 实测单帧耗时 + predict_time + gimbal_lead_time + 相位滤波滞后`，飞行时间与仰角迭代 5 轮收敛 |

> ✅ **开火纪律已恢复**（`27886e0`）：`fire_err_rad: 0.08`（≈4.6°）连续 `fire_stable_frames: 2` 帧，
> 且 `fire_require_lock: 1`（要求 mode 7 模型 ready、rms ≤ `fire_rms_gate_deg`、样本 ≥ `fire_min_n`）
> 才发 `fire: true`。历史上一度放水到 `3.20` rad（≈183° = 取消对准检查），详见 §6.4。

---

## 1.5 检测器与几何：为什么这么选（都有实测依据）

### 1.5.1 后端：**只剩 HW26 一个**（历史结论已收敛）

**最新结论（2026-10-12，以代码为准）：只支持 HW26 自己的模型。**

- `DetectorKind` 枚举里只有 `kHw26`；构造函数只认 `[1,5040,26]`，其它形状在
  `hw26_pipeline.cpp:331-341` 直接 `throw`（`[hw26] FATAL: model output shape is not HW26 ...`）。
- sp25 的检测模块（`detector_sp25`）已在提交 `bfcfd70` 删除，全仓库搜不到 `kSp25`。
- `model/yolo11_buff_int8.xml`（sp25，6 点）**仍在磁盘上留档**，但把 `bridge.yaml` 的
  `model:` 换成它 = **启动即崩**，不是"换一行就能用"。

**为什么最后选 HW26 自己的模型**：
- 它**分类颜色 + 激活**（4 类），这是大符双靶标策略的必需信息；sp25 是单类模型，桥只能强制
  `color=配置值 / label=0`，两个过滤器都变成空转。
- R 中心是**直接输出** `keypoints[8]`；sp25 的是**外推** `kpt[4]+1.4*(kpt[5]-kpt[4])`。
- 大符时它天然给出 **2 个候选**（正好是两片点亮的靶标）。

**历史上那条"满画面乱飘"的记录**（`det=1 conf=0.90` 但位置在
`@484,339 → @704,375 → @503,167 → @0,216 → @992,25` 乱跳，且全标成 `c=2(RED,INACTIVE)`）
是在 **`fov_y = 45°` 的旧广角相机**下观察到的 —— 那时符在画面里只有现在的 **75%** 大。
**这个现象在 34.5° 下没有复现**（点亮的帧上稳定给出 2 个候选，`max_score ≈ 0.92`）。

### 1.5.2 物体点半边长：**两批物理点，两个值**

| 后端 | 四点是什么 | 半边长 | 状态 |
|---|---|---|---|
| sp25（已删除） | `kpt[0..3]` 直接给四角 | `0.127` | 代码路径不存在，仅留档 |
| **HW26（在用）** | `t/l/b/r` 是八边形相邻顶点的**中点** —— 另一批物理点 | **`blade_half_size_hw26 = 0.093`** | ✅ 当前在用 |

`Pipeline` 判定为 HW26 后会执行 `cfg.blade_half_size = cfg.blade_half_size_hw26`
（`hw26_pipeline.cpp:479`），所以**实际生效的就是 yaml 里那个值（当前 0.093）**。

- 用 0.127 时 `dis ≈ 9.4 m`，而真值 ≈ 6.4 m（偏大 45%）；
- 改用 0.088 后实测 `dis = 6.16 ~ 7.10 m`（中位 **6.53 m**），与真值吻合 —— 见 §6.1；
- **当前值 0.093**（`3b864d0`）：按新模拟器的距离真值反推 —— 小车在 `(±5,1,±5)` = **7.07 m**，
  而 0.088 实测 `dis ≈ 6.7 m`，所以尺寸要放大约 5%。
- **`reproj` 查不出这个错**：平面靶标 PnP 有尺度不可分辨性，重投影误差两种取值几乎相同。
  距离才是判据。

另有一条历史坑：HW26 的 letterbox 是**等比缩放后居中、灰边 (114) 填充**；sp25 是**贴左上角黑边**。
两者不通用，混用会让关键点整体偏移 → PnP 全错。现在只走 HW26 一条路径。

### 1.5.3 几何验证（**四角点/sp25 口径**的历史数据，供对照）

拿 ④ 运行日志里记录的 `kpt`（模型在模拟器上的真实输出）+ `buff_dis`（记录的真值），
喂进本桥的 PnP，比较 `|符心|`：

| 物体点半边长 | 均值差 | 平均相对误差 | 最大 | 首解重投影 |
|---|---|---|---|---|
| `0.114`（HW26 原值） | 0.7045 m | **10.18%** | 21.3% | 0.16 px |
| **`0.127`（③ 值）** | **0.0126 m** | **0.18%** | 14.9% | 0.16 px |

> 重投影误差两者**完全相同**（0.16 px）——正好说明它无法分辨尺度。
> 这张表针对的是**四角点**那一批物理点，用 20313 条真实模拟器关键点反算。
> HW26 后端的点集不同，用的是 §1.5.2 的 `blade_half_size_hw26`（当前 `0.093`）。

### 1.5.4 从模拟器源码直接确定的补偿（不是猜的）

来自 `POWER_RUNE_INTERFACE.md:79-123` 与 `src/main.rs:816`：

| 补偿 | 值 | 本桥 |
|---|---|---|
| 相机系→云台系旋转 | `[0,0,1; -1,0,0; 0,-1,0]`（OpenCV X右Y下Z前 → X前Y左Z上） | ✅ 已做 |
| **相机旋转 = 云台旋转** | 模拟器明确"只使用 `CAM_DIRECTION` 的位置，不使用其预置旋转" | ✅ 所以外参旋转只有上面那一项 |
| **枪口平移** | `[0.07383, -0.00108, -0.02848] m`（前 73.83mm、右 1.08mm、下 28.48mm） | ✅ 已做（`use_muzzle_offset`） |
| 枪管固定上仰 20° | `pitch_offset = -0.34906585 rad` | ✅ `pitch_offset_deg: -20` |
| 弹速 / 重力 | `22.0 m/s` / `9.81` | ✅ |
| 线性阻尼 0.0089 | 7 m 内影响 <1 cm | ⬜ 忽略（无阻力抛物线够用） |
| 畸变 | 全 0 | ✅ 不用管 |
| 马格努斯 | 模拟器**没有** | ✅ 不建模 |
| 车体速度 | 弹丸继承车体平动速度 | ⬜ 保持车体静止即可 |

**不需要补偿的**（从源码确认）：畸变（0）、roll（0）、马格努斯（无）。

### 1.5.5 观测参考系：**默认 `face`**

`frame` 三个选项是在权衡 HW26 的一个隐含前提：`Buff::roll = atan2(Δz, Δy)` 与
`predict_position` 都只在观测系的 y-z 平面重建圆，要求**观测系的 x 轴落在符面平面内**。

- **`face`（★当前）**：符面系。`x = 符面法向`（在车体系里是常量），`y,z` 为面内基。
  同时满足两件事：`e = R·(0,0,−0.7)` 落在面内 → `e.x = 0` **精确成立**；且坐标系**不随云台转**。
- **`gimbal`**：模拟器的相机旋转 = 云台旋转，云台伺服时会把符压在画面中心，前提近似成立，
  但**云台系就是瞄准系** → 把观测挂在自己的输出上，会正反馈自激。
- **`chassis`**：静止系，但符在侧方时前提被破坏 —— 实测（HW26 自己的视频）偏差 **16.5~23.4 cm**，≥ 靶标尺寸。

---

## 1.6 晃动（相机被猛拽）的根因与修复

### 根因：观测坐标系是"我们自己输出的函数"

之前用 `frame: gimbal`（相机/云台系）。但**相机和枪管是固定在一起的**，所以
**云台系就是瞄准系**，而云台系每一帧都随我们的指令在转。于是：

```text
云台转 δ  →  符面在观测里的倾角跟着变  →  重建出的圆变了
        →  预测点移动  →  指令再推云台  →  δ 变大  →  正反馈
```

实测（回放闭环，`--closed-loop`）：`gimbal` 模式的指令每帧变化 **p50 = 12.0°**
且 79.3% 的帧 >5°，而 `face` 模式只有 **4.6°**。

> ⚠️ 诚实说明：`--closed-loop` 回放里虚拟云台转动**不会改变视频画面**，
> 所以它**无法**忠实复现真实闭环（跑到限位是回放假象）。它只能看出两种参考系
> 的**结构性差异**，不能当闭环稳定性的证据。真正的验证必须在模拟器上做。

### 修复：符面系（`frame: face`）

符面法向 `n` 从 PnP 直接取出（= 扇叶局部 x 轴，在车体系下），**符不动 → n 在车体系里是常量**。
实现上还把喂给跟踪器的扇叶姿态**强制让局部 z 轴落在面内**，使 `e.x` 严格为 0（不是近似）。

### 额外的安全网（对齐 ④ `sp25_buff/cpp_lib/buff_vision.cpp:190-200`）

| 参数 | 值 | 作用 |
|---|---|---|
| `max_slew_deg_s` | 40 | 限速：`clamp(raw - 当前姿态, ±slew*dt)`，与帧率无关 |
| `cmd_smooth_alpha` | 0.5 | 指令低通：`cmd = (1-a)*prev + a*raw` |
| `max_aim_step_px` | 400 | aim 像素瞬移拒帧：挡住 PnP 翻转/换扇叶那一帧（像素阈值要随焦距缩放：原 300 @fx=1303.686 → 400 @fx=1739.1） |
| `stale_ms` | 200 | 图像过期门，与 `max_image_age_ms` 取较小者 |

### 顺带修掉的一个真 bug

双解消歧的连续性判据原来在**云台系**里比较 `prev_pose`，而帧本身在变
→ 判据失效 → PnP 频繁选错解。现在统一到一个一致的变换下比较。

---

## 1.7 桥自己的局部相位 tracker（`local_model` 0–7）

文件：`src/local_tracker.hpp` / `local_tracker.cpp`。它**只估相位**（转盘中心由调用方给）。

### 为什么需要它

HW26 内部用**相邻两帧差分**算角速度 `v = droll/dt`，而相位测量噪声约 3°/帧、真实变化约 2.9°/帧
→ 信噪比约 1。按 `threshold=0.5` 算，一个点亮窗口（实测 25~80 帧）里只有十几个内点，
够不到 RANSAC 的 `min_inliers` → `st` 卡在 CONVERGING，`ctrl=0`，完全不瞄准。
桥的局部拟合**只要 6~8 个样本**（约 0.2~0.3 s）就能给预测点。

### 八档（`local_model`，改配置即可切换，不用重编译）

| 值 | 名字 | 做法 | 离线实测 / 备注 |
|---|---|---|---|
| 0 | `csine` | 官方约束模型 `θ₀ + sgn·[(2.090−a)t + p(1−cos ωt) + q sin ωt]`，`(a,ω)` 在官方边界内**网格搜索** + 每帧重解线性部分 | 干净数据 227/341/455 ms 误差 2.30/3.69/5.09°，多步最好 |
| 1 | `AR(p)` | 纯数据驱动自回归 + 递推多步 | 114 ms 误差 0.5~0.75°，短程最好，长程变差 |
| 2 | `EKF` | 状态 `[θ,a,ω,φ]`，过程用官方速度公式，观测 θ，带创新门 | **离线实测不好**（窗口不足一周期时 a/ω 不可辨识），留作对比 |
| 3 | `free` | 自由线性正弦 `c₀+c₁t+c₂sin ωt+c₃cos ωt`（ω 网格） | 能拟合但 `c₁` 自由 → 会漂，兜底 |
| 4 | `AR + 物理包络` | 同 1 的 AR，但**不查 rms**；安全性改由 `predict()` 保证：每步增量夹在 `\|d\| ≤ 2.090·dt`，且平均角速度落在 `[1.045, 1.310]` 之外就退回等速外推 | 脏数据下 AR 的 rms 必然大，用 rms 门就没法用，所以换物理包络把关 |
| 5 | `裸 AR`（⚠️ **当前是坏的**） | 设计：无门、≥2 样本即可用、AR 递推 + `\|d\| ≤ 2.090·dt`。**实际**：`push()` 没把它接进 `fitAR()`，`ar_c_` 永远为空 → 一直走"最后两点等速外推"，且每帧白跑一遍 65 点 `fitFree` 网格（§9.2） | 名义上最快就绪（2 样本），实际退化成等速外推；**目前不要用** |
| 6 | `查表 + 灭灯滑行` | 先攒 `fit_s = 1.0 s` 数据做**一次**网格搜索，之后冻结参数外推；灭灯时滑行 | 设计意图是"覆盖空窗"的档，但要先攒 1.0 s —— 当前节拍下既慢又覆盖不了（§6.2） |
| **7** | **★当前：物理约束 + 稳健 LM** | 官方速度模型 `θ₀ + sgn·[(2.090−a)t + (a/ω)(cos φ − cos(ωt+φ))]`，参数 `(θ₀,a,ω,φ)` 每帧用 **Levenberg–Marquardt + Huber 稳健权重**精修；`a∈[0.780,1.045]`、`ω∈[1.884,2.000]` 硬夹；窗口参数热启动；换扇区时**相位重参考**（平移 k·72° 折回原参考系，不丢窗） | 大符 0.3 s 预测中位 **1.4°**（`bc5e7ca`）；好窗 rms 0.5~8°、坏窗（解缠自转）139~286° → `local_rms_gate_deg: 15` 能干净分开；就绪只要 ≥8 样本 |

**就绪速度排序**（`ready()` 的最少样本数）：

| 档 | 7 | 5 | 0 | 2 | 1 / 4 / 3 | 6 |
|---|---|---|---|---|---|---|
| 最少样本 | 8 | **2** | 4 | 12 | 12 | ≈30（1.0 s） |
| @25fps | ≈0.32 s | ≈0.08 s | ≈0.16 s | ≈0.48 s | ≈0.48 s | 1.0 s |

> `min_n_ = 12`、`min_span_deg_ = 30°` 是 `local_tracker.hpp` 里写死的默认值，配置调不了；
> 能调的只有 rms 门（`local_rms_gate_deg`），且只对需要门的档（0/1/3/**7**）有效。
> mode 6 还有一条快路径：有命中快照时先拿上次的 `(a, w)` 试一次，成功就立刻锁定。

`ready()` 判据按档不同：0/3 要样本数+跨度+rms 门；1 要 AR 系数+样本+跨度+rms；
2 看 EKF；4 只要系数+样本+跨度（无 rms）；5 只要 ≥2 个样本；6 = 已锁定且未超时；
**7 = 物理拟合有效 + ≥8 样本 + 跨度 ≥30° + rms ≤ `local_rms_gate_deg`**。

窗口 `win_` 默认 20（≈0.6 s @30fps）；**`mode 6` 会把窗口强制到 ≥48**（否则 1.0 s 的填充要求装不下）。

### 相位滑动平均（`phase_ma_n`）

喂进去之前先做 `phase_ma_n`（当前 **6**）点滑动平均，噪声降 √N，代价是滞后 `(N−1)/2` 帧
（N=6 时约 85 ms）。这段滞后已由 `base_lead` 里的 `ma_lag` 补回。

### 点亮 / 灭灯语义（`mode 6` 专属）

- **短暂缺失**（灭灯 < `local_coast_s`）：`on_dark()` 走滑行分支，**保留参数**继续外推；
- **真正灭灯**（灭灯 ≥ `local_coast_s`）：**清空参数**并报 `lost`。
  依据：模拟器每次**新激活**都会重抽 `(a, w)`，旧参数跨激活没有意义。
- **命中判定**：连续 `hit_s = 2.6 s` 都没有进入空窗（一直没 `det==0`），判为"刚打中"，
  把当前参数**快照**下来，之后不从零重拟合。

> 这三个时间常数与实测节拍的关系见 §6.2 —— **当前它们都调反了**。

---

## 1.8 大符双靶标（`dual_target`）

大符同时高亮 2 片靶标。HW26 的 `BuffSelector` 结尾写死 `dst.push_back(infos[0])`，
只保留"离光轴最近"的 1 个；而两片都在转，谁更近会来回切换 → 相位每帧跳 72°
→ 超过 `big_buff_observer` 的 `switch_buff_angle(45.8°)` → 每帧都被判成"换片"丢弃
→ RANSAC 拿不到数据 → `st` 永远 CONVERGING。所以桥自己保留两个候选。

### 现在做什么

1. **组号按 wrap 之后的相位先后定**，不按置信度（按置信度会在 0.91↔0.89 互换时来回换）。
2. 两片的相位差 `Δ = wrap(B.roll − A.roll)`，**吸附到 72° 的整数倍**（`k = round(Δ/72°)`，`k=0` 强制为 1）。
3. **不融合两片**：直接喂 **A 片自己的相位**。理由：正弦拟合本来就是对整个窗口做最小二乘，
   自己会平均噪声；而融合依赖 Δ 恰好是 72/144°（实测残差还有几十度），反而把这个误差混进去。
4. `A.R_center` 保持原样，不做两片平均。

### 诊断行怎么读

```text
[grp]  nt=2 ndet=2 nsel=2 nk=2 p0a=1 sa=0 sb=1 conf=(0.90,0.87) g0=(527,407) g1=(815,464) ra=2.204 rb=-1.917 rca=6.71 rcb=6.51
[dual] droll=(4.1,4.3)deg pick=(0,1) dres=20.1deg chord=1.252m rcd=0.20m dlt=144 cand=(125,49)px | local n=6 rdy=0 rms=-1.0deg span=30deg w=0.00 LK=0 om=0.000 A=0.000 drop=342 lost=0 hit=0
```

| 字段 | 含义 / 该看什么 |
|---|---|
| `p0a` | 第 0 个 pose 是不是 A。**在跳 → A/B 身份翻了**，紫点（原绿点）其实也在跳 |
| `sa` / `sb` | 两片各自的来源索引。`p0a` 稳而它在跳 → 组号映射在变 |
| `droll` | 本帧两片 roll 与上一帧做**不依赖身份**最优配对后的两个残差。**某一对接近 180° → 那片发生了 IPPE 双解翻转** |
| `pick` | 两片各自选了第几个 IPPE 解。在 0/1 间来回跳就是它 |
| `dres` | Δ 吸附到 72° 整数倍**之后的残差** —— "72° 刚性固连"假设的直接检验，应趋近 0 |
| `chord` | 两片实测板心间距 |
| `rcd` | **不是"圆心分歧"，而是两片的面外间距** `\|t₀.x − t₁.x\|` 的度量（推导见 §1.10）。同一转盘应共面 → 应≈0 |
| `dlt` | 吸附后的 Δ，只应是 72 / 144 |
| `cand` | 两个候选到光轴的距离（像素） |
| `local ...` | 局部 tracker 状态：`n` 样本数 · `rdy` 是否可用 · `rms` 拟合残差 · `span` 相位跨度 · `LK` 是否锁定 · `om`/`A` 拟合出的 ω/a · `drop` 被跳变门丢掉的帧数 · `lost` 报丢失 · `hit` 命中确认 |

### 实测：几何不自洽（2026-10-12，45 条 `[dual]`）

| 量 | 期望 | 实测 |
|---|---|---|
| `rcd` | 几 cm（两片应当共面） | **0.01 – 0.60 m**，典型 0.1–0.5 m —— 即两片在符面法向上被拉开了 10~60 cm |
| `dres` | → 0 | **最大 29.9°**，中位约 6° |
| `dlt` | 常数（72 或 144） | **72 ↔ 144 来回跳**，每次翻转都伴随窗口重置（`n` 从 48 掉到 1–10） |
| `pick` | 稳定 | 28/45 是 `(0,0)`，其余 `(0,1)/(1,0)/(1,1)`，与翻转同现 |

翻转那一帧的 `droll=(51.2°, 119.6°)` → 教科书式的 IPPE 双解翻转。
**结论：两片的 PnP 解不满足"同一个刚性转盘"，每次重新捕获都会错一个槽位。**
详见 §6.3。

---

## 1.9 可视化标记

> HUD 文字（原来的 5 行）**已全部移除**（省 `putText` 的开销）；画面中心十字也已去掉。

| 标记 | 含义 |
|---|---|
| 🟣 **紫空心圆（r=14，线宽 3）** | **未来打击点**（弹丸飞达时刻靶标所在位置）的投影，判读看**圆环中心**。原为绿色实心 r=8，按用户要求改紫并放大、改空心（BGR 240,32,160）；代码里仍叫 `green_px_*`、"绿点" |
| 🟡 黄实心圆（r=6）+ 数字 `1` | 组 1 的 `keypoints[8]`（模型直接输出的旋转中心） |
| 🟣 品红实心圆（r=6）+ 数字 `2` | 组 2 的同上 |
| 🟠 橙实心圆（r=6）+ 数字 `1` | **组 1 的三维板心重投影**（看 PnP 准不准） |
| 🔵 蓝实心圆（r=6）+ 数字 `2` | 组 2 的三维板心重投影（`(255,80,0)`，与下面的蓝点区分） |
| ⚪ 青圈（r=10，空心） | R 中心 |

- **"枪管射线 ∩ 符平面"那个蓝点已按用户要求取消绘制**（数据仍计算，供日志调试）。
- **瞄准好不好的判据**：紫点（空心圆环）的**中心**是否稳定地贴在应该打的位置上（原来靠"绿蓝重合"判断，蓝点已不画）。
- 绘制节流用 `visualize_every`（**1 = 每帧都画，不要调大**，否则画面掉到 1/N）。

---

## 1.10 转盘圆心 C 的来源（`center_source`）

**为什么圆心最重要**：C 是"相位 → 打击点"的唯一基准 ——

```text
roll = atan2(t.z - C.z, C.y - t.y)                  // 相位完全由 C 定义
预测点 = (C.x,  C.y - R·cosθ,  C.z + R·sinθ)         // 预测命中点也完全由 C 定义
```

相位对 C 的敏感度是 `1/R ≈ 1.43 rad/m`：**C 差 10 cm → 相位差 8°**。所以 C 错了，roll、
预测点、云台角、开火全跟着错。

**为什么不能用 PnP 朝向反推**：HW26 的 `R_center = R·(0,0,-R) + t` 依赖靶板的**面内朝向**，
而 PnP 的朝向由 4 个物点 `t/l/b/r` 恢复 —— 这 4 个点构成**正方形**，绕法向每转 90° 自己重合，
所以面内朝向有 **4 重退化**。实测两片由此解出的圆心分歧稳定在 **0.8~1.6 m**。
位置 `t` 没有这个退化，所以 C 只能靠"**位置 + 半径 0.7 这个硬约束**"来定。

| `center_source` | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **0 = dual** | `\|t_A−C\| = \|t_B−C\| = 0.7` → C 在 `t_A t_B` 的中垂线上，距中点 `h = √(0.7²−(d/2)²)`；两支用"模型输出的转盘中心"投到图像里二选一 | 第一帧就有 | **另一片的位置会进到这一片的结算里** |
| **1 = single_history** ★ | 只用被选中那一片的**历史 t** 做"已知半径 0.7"的圆拟合 | 彻底切断互相污染；还能剔掉该片自己 IPPE 翻转的野值 | 要攒够 `center_hist_min` 个样本；之前自动回退到 0 |

### 为什么 `rcd` 其实是"两片的面外间距"

当 `mk_obs` 生效时（`z` 取 `(t−C)/r` 并把 x 分量清零后归一化），

```text
R_center = R·(0,0,−0.7) + t = t − 0.7·z = (t.x, C.y, C.z)
⇒ rcd = |R_center_A − R_center_B| = |t₀.x − t₁.x|
```

也就是两片在**符面法向（深度）方向**上的间距。同一转盘的两片应当共面，所以它应≈0；
实测最大 **0.60 m**（转盘半径才 0.7 m）。用 17:38 那版数据可验证：`chord=1.240 m`、
`dlt=72°` 但真实面内夹角约 102° → 面内弦 `2·0.7·sin51° ≈ 1.088` →
`Δx = √(1.240² − 1.088²) ≈ 0.60`，与那一行的 `rcd` 一致。

### `fit_fixed_radius_center` 的实现要点

把 `|t−C| = R` 展开成 `2t_y·C_y + 2t_z·C_z − k = |t|²`（`k = |C|²−R²`）→ 对
`(C_y, C_z, k)` 是**线性**最小二乘，不需要初值；样本 ≥3 后也没有"中垂线两支"的二义性。
但**只做一次线性最小二乘 + 硬残差门是不够的**：外点会把它整体拽偏，于是连正常点的残差
都超门，结果把好点也一起剔掉（实测 40 点里 5 个 0.3 m 的外点就只剩 14 个、圆心偏 0.4 m）。
所以这里用**线性解做初值 + 对真实残差 `|t−C|−R` 做 Cauchy 加权的鲁棒 Gauss-Newton**（IRLS）：

```text
s   = 1.4826 · median(|r|)                  // 尺度，自适应
w_i = 1 / (1 + (r_i/(3s))²)                 // Cauchy 权，外点被压下去
H ΔC = Σ w_i·u_i·r_i ,  u_i = (t_i−C)/|t_i−C|   // 2×2 正规方程
```

另外两道闸：**样本跨度 `span < center_hist_span_m`（弧太短，法方程病态）直接拒绝**；
**内点数 < `center_hist_min` 也拒绝**（宁可回退到 dual，也不给一个飞掉的圆心）。

自检：`./build/hw26_sim_bridge --config config/bridge.yaml --selftest-center`

```text
[center-selftest] R=0.700 真值圆心=(5.066,1.923,1.242)
  40帧/120°/5mm                ok=1 err=  0.0011 m n= 40 rms=  0.0054 m
  40帧/120°/1cm                ok=1 err=  0.0022 m n= 40 rms=  0.0107 m
  40帧/120°/5mm+5面内外点      ok=1 err=  0.0009 m n= 35 rms=  0.0052 m   ← 外点被剔掉
  8帧/40°/5mm                  ok=1 err=  0.0019 m n=  8 rms=  0.0064 m
  6帧(不足min_n)               ok=0                                     ← 拒绝
  40帧/0.5°(弧太短)            ok=0                                     ← 拒绝
[center-selftest] 6/6 项符合预期
```

> 注意：拟合只用**面内 `(y,z)`**，所以沿法向（深度）的偏移天然不影响结果 ——
> 这既是优点（IPPE 翻转的深度误差进不来），也意味着外点门只管**面内**野值。

诊断行（`center_source: 1` 时每 10 帧一行）：

```text
[ctr] src=1 n=200 rms=0.0314m            ← 单片（小符 / 大符只检出一片）
[dual] ... | ctr=1 n=35 rms=0.005m | ...  ← 双片时并进 [dual] 行
```

> `src` = 0 表示本帧用的是 dual（可能是 `center_source: 0`，也可能是历史还没攒够的回退）。
> 实测回放（`buff.mp4`，单片）能稳定收敛到 `n≈200`、`rms≈3 cm`。

> ⚠️ **限制**：历史存的是**车体系**坐标，前提是"转盘中心在车体系里静止"。如果车体移动，
> 旧样本就失效了，必须清空历史。当前研究里车是静止的，`center_hist_keep_s: 8.0` 只是限制
> 内存与陈旧度（8 s 后的新样本会把旧窗口顶掉，中心会重新收敛）。

---

## 2. 编译

```bash
cd /home/jno/Documents/GitHub/Vision_Buff_study/HWauto_buff2026/bridge
./build.sh
```

产物：`build/hw26_sim_bridge`

依赖（本机已具备）：OpenCV 4.5.4、Eigen3、OpenVINO 2024.6.0、**Zenoh C 1.9.0**
（`/usr/local/lib/libzenohc.so`，与模拟器 `Cargo.lock` 里的 zenoh 1.9.0 同版本）。
OpenVINO 装在别处：`./build.sh -DOPENVINO_ROOT=/your/openvino`。

编译对象（`CMakeLists.txt`）：桥的 `main.cpp` + `hw26_pipeline.cpp` + `local_tracker.cpp`，
加 HW26 原装的 `detector_common / buff_selector / buff_pnp_solver / buff_tracker /
tracker_status / kalman_filter / ransac_filter / buff_predictor`。
**故意不编译** HW26 的 `src/detector/buff_detector.cpp`（唯一与 TensorRT/CUDA 耦合的文件）。

模型已生成在 `model/`（38 MB，已 gitignore）。要重新生成：

```bash
./tools/convert_model.sh          # 读 ../buff.onnx，输出 model/buff_ir.xml
```

---

## 3. 运行方式

### 3.1 `--replay`：不需要模拟器，先看窗口 + 验证算法链

```bash
./build/hw26_sim_bridge --config config/bridge.yaml --replay ../buff.mp4
```

按 `q` 或 `ESC` 退出，会打印汇总。**回放模式假定相机静止**，所以建议这时把 config 里
`frame` 临时设成 `"gimbal"`（或加 `--frame gimbal`）。

实测（HW26 自带的 `buff.mp4`，400 帧）：

```text
有检出=400 (100.0%)  TRACKING=399  开火=1
平均 29.4 ms/帧 (34.1 FPS)   ← headless
平均 38.3 ms/帧 (26.1 FPS)   ← 开窗口（imshow 约 +9 ms）
```

> 回放里只有 1 次开火是正常的：回放**没有真实云台去跟随指令**，姿态固定为 0，
> 所以 `err` 只有偶尔才落进门。真实模拟器里云台会跟着转。

### 3.2 接模拟器（**真正的自动打击**）

**终端 1 —— 起模拟器：**
```bash
cd /home/jno/Documents/GitHub/Vision_Buff_study/at_vision_simulator
cargo run --release
```
进去后：**F3** 切第一人称 → **方向键**把云台对准能量机关 → **Space** 打一枪激活转盘
（必须先手动命中一次才会转；激活后桥就能自己接管了）。

**终端 2 —— 起桥：**
```bash
cd /home/jno/Documents/GitHub/Vision_Buff_study/HWauto_buff2026/bridge
./run.sh
```

看到这行就通了：

```text
[bridge] subscribed rt/gimbal_pose + rt/image_raw, publishing cmd/gimbal
[bridge] #30 age=28ms det=2 sel=2 solved=1 q=1 st=0 dis=6.53 roll=2.31 w=-1.05 lead=0.412 lat=29ms cmd=(3.21,-4.55) err=0.0312 ctrl=1 fire=0 nt=2 dlt=72 rc=6.45
```

- `st=1` = **TRACKING**（开始自动跟瞄）· `ctrl=1` = 正在伺服 · `fire=1` = 自动开火
- `nt=2 dlt=72` = 本帧融合了 2 片，吸附后的 Δ 是 72°
- `Ctrl-C` 退出并打印汇总

不想开火先试跟瞄：`./run.sh --no-fire`

### 3.3 先量模型的泛化性（**最重要的一步**）

模拟器跑着、云台对准符时：

```bash
./build/hw26_sim_bridge --config config/bridge.yaml --capture sim_frame.png
./build/hw26_sim_bridge --config config/bridge.yaml --selftest sim_frame.png
```

退出码 **0 = 有检出**（继续）；**4 = 无检出**（模型不泛化到模拟器渲染的符 → 见 §6.5）。

### 3.4 `--selftest-aim`（合成关键点，验算瞄准数学）

> ⚠️ **只能在 `mode: 1`（小符）下用。** 它每例喂 200 帧但**叶片相位是固定的**，
> 而大符的 tracker 要靠 RANSAC 正弦拟合才能进 TRACKING，相位不动就永远收敛不了
> → 一例都过不了（实测 `mode: 2` 下 0/36，`mode: 1` 下 36/36）。
> 程序在 `mode: 2` 时会打印告警。
>
> 2026-10-12 复测：删掉 `detect_only`（它原本会把这条路径的 PnP 关掉）之后，
> `--selftest-aim --mode 1` 恢复 **36/36 个用例误差 ≤ 2°**。

### 3.5 `record.sh`：把一次运行完整录到 `logs/`

```bash
./record.sh                 # 默认 --no-fire
./record.sh --fire big1     # 带开火；第 2 个参数是标签，写进文件名
```

产物：`logs/<时间戳>_<标签>.log`。事后：

```bash
grep '\[dual\]' logs/*.log
grep 'st=1'     logs/*.log
```

> ⚠️ `hw26_bridge_log.csv` 是**每次运行都截断重写**的。要保留逐帧数据就拿走或改 `log_path`。
> （`--selftest-aim` 已不再写这个文件 —— 它以前会用自己的合成数据把上一次实测的逐帧日志抹掉。）

### 3.6 离线预测器对比（只用日志，不需要模拟器）

```bash
python3 tools/predictor_compare.py logs/*.log
```

从 `[bridge]` 行里取有效观测的 roll 序列，按断帧切成"点亮窗口"，在每个窗口内用最后 N 个点
拟合后向前预测，比较 `const / lin / quad / ar(p) / sine / csine` 的误差，并检查 roll 速率是否
落在官方包络内（`≤ 2.090 rad/s`）。

### 3.7 命令行覆盖（现场试参数最快）

```bash
./build/hw26_sim_bridge --config config/bridge.yaml \
    --conf 0.3 --color 1 --mode 2 --frame face --no-fire --headless --frames 500
```

| 参数 | 作用 |
|---|---|
| `--config <path>` | 配置路径（默认 `config/bridge.yaml`） |
| `--capture <png>` | 抓一帧存盘（配合 `--selftest`） |
| `--selftest <png>` | 对一张图跑检测，退出码 0=有检出 / 4=无检出 |
| `--selftest-aim` | 合成关键点验算瞄准（**仅小符**） |
| `--selftest-center` | 合成"一片在半径 0.7 圆上"的位置序列，验证圆心拟合（§1.10） |
| `--replay <mp4>` | 回放视频；可配 `--closed-loop` |
| `--frames N` | 最多处理 N 帧 |
| `--no-fire` | 关掉开火（只跟瞄） |
| `--headless` | 不开窗口 |
| `--no-filter` | 同时关掉颜色 + 激活两道过滤 |
| `--conf` / `--color` / `--mode` / `--frame` | 覆盖对应配置项 |

---

## 4. 数据流

```text
模拟器 ──rt/image_raw (BIMG 二进制)──┐
       ──rt/gimbal_pose (JSON)──────┤
                                    ▼
   Zenoh 回调 → 【最新帧邮箱】(覆盖式，绝不排队) → 工作线程
                                                       │
     ┌─────────────────────────────────────────────────┘
     ▼
  RGB→BGR → letterbox_image() (HW26) → blobFromImage
  → OpenVINO 推理 (device: GPU，失败自动回退 CPU)
  → decode_detections() (HW26) → 只按置信度过滤
  → Selector(HW26) + 桥的 dual_target 覆盖（保留两片）
  → 桥自实现 PnP (IPPE 双解 + 不依赖身份/一致变换的连续性消歧)  ← 不用 HW26 的 Solver
  → 距离闸 → 观测系变换 → 相位滑动平均(phase_ma_n)
  → local_tracker(桥，local_model)  ∥  BuffTracker::push (HW26 四态状态机)
  → BuffTracker::update → BuffPredictor::predict_position (HW26，到 TRACKING 后优先用它)
  → 弹道迭代(≤5 轮) → yaw/pitch → slew 限速 + 指令低通 + aim 瞬移拒帧
  → 到位残差门 → fire
     │
     ▼ cmd/gimbal {yaw,pitch,roll:0,fire,control}
  模拟器
```

**为什么用"最新帧邮箱"**：单帧推理实测 **18–50 ms**（GPU），模拟器按 60 FPS 出图 → 桥比源慢。
若逐帧排队，`image_age` 单调增长，最终被过期门全程拒帧、云台冻结（sp25 的日志里实测到过
一个会话 p50 时延 **3.1 s** 的"假死"）。覆盖式邮箱则始终处理最新帧，天然不积压。

---

## 5. 参数（`config/bridge.yaml`，扁平 `key: value`，无 YAML 方言坑）

★ = 当前值与前文结论直接相关，改动前先看对应章节。

### 检测

| 项 | 当前 | 说明 |
|---|---|---|
| `model` | `../model/buff_ir.xml` | **只写这一个**。输出形状必须是 `[1,5040,26]`，否则启动即 FATAL |
| `device` | `GPU` | OpenVINO 设备 `CPU/GPU/AUTO/NPU`；**编译失败自动回退 CPU** |
| `confidence_threshold` | `0.5` | 只按置信度过滤（`score > 阈值` 就出） |
| `nms_threshold` | `0.4` | |
| `blade_half_size` | `0.127` | 四角点口径（现无后端使用） |
| `blade_half_size_hw26` | **`0.093`** | ★ HW26 实际生效值（启动时它覆盖 `blade_half_size`，那行是死行）；不要改成 0.114/0.127 |
| `num_reverse` | `0` | `0`=编号 1=A（相位靠前=先打的那片）；`1`=两组编号对调 |
| `filter_color` / `filter_activation` | `1` / `1` | 诊断"检出但 sel=0"时逐个关掉；两个都关 = 退化为"取光轴夹角最小" |
| `mode` | **`2`** | `1`=小符，`2`=大符。**模型分不出大小符，必须显式配**，且要与"车在哪一侧"一致 |
| `my_color` | **`1`** | `0=BLUE, 1=RED`。大符（`FACE_2`，车在 (5,1,5)）实测被分成 `color=1`(RED)；配错会被 `filter_color` 全拒（`n_sel=0`） |

### 双靶标 / 局部 tracker

| 项 | 当前 | 说明 |
|---|---|---|
| `dual_target` | `1` | 保留两片候选（§1.8） |
| `chord_phase` | `0` | 弦相位（实验）：用两片靶心连线方向当相位，理论上与圆心无关。实测反而更差（超物理步 32.6% vs 圆心路径 24.2%），**默认关** |
| `phase_fuse_dual` | `1` | 两片 PnP roll 融合（实验）：不用圆心，只用两片 roll 的对齐平均。默认开 |
| `phase_ma_n` | `6` | 相位滑动平均长度，`1`=关；滞后已由 `base_lead` 补回 |
| `local_tracker` | `1` | 是否启用局部相位 tracker |
| `local_model` | **`7`** | ★ 0=csine / 1=AR / 2=EKF / 3=free / 4=AR+包络 / 5=裸AR（⚠️ 坏的，§9.2）/ 6=查表+滑行 / **7=物理约束+稳健 LM（当前，§1.7）** |
| `local_rms_gate_deg` | **`15`** | rms 门；对需要门的档（0/1/3/**7**）有效。mode 7 下好窗 0.5~8°、自转坏窗 139~286°，15° 能干净分开 |
| `local_bad_rms_deg` | `30` | 坏窗自愈：rms 超过它就判"模型发疯" → 失效创新门 + 丢老样本快速重自举 |
| `local_coast_s` | **`0.2`** | ★ 灭灯多久内允许滑行。**与实测灭灯间隔（1.6–3.2 s）严重不匹配，见 §6.2**；只对 mode 6 有效 |

### 转盘圆心（§1.10）

| 项 | 当前 | 说明 |
|---|---|---|
| `center_source` | **`1`** | ★ `0`=两片联立（第一帧就有，但另一片会污染这一片）/ **`1`=只用一片的历史做固定半径圆拟合（当前）** |
| `center_hist_min` | `8` | 圆拟合所需最少样本数；不够就回退到 0 |
| `center_hist_span_m` | `0.05` | 样本在面内的最小跨度；弧太短直接拒绝（法方程病态） |
| `center_hist_gate_m` | `0.15` | 外点门：样本到圆心的距离与 0.7 之差超过它，就算外点 |
| `pos_gate_m` | `0.12` | 位置内点门：**喂相位前**用圆再判一次 `‖t−C‖ − 0.7`，超门当帧判为 IPPE 位置野值（不喂相位、不做橙点锚定） |
| `center_hist_keep_s` | `8.0` | 历史只保留最近这么久（圆心静止，只为限内存） |

### 相机 / 外参

| 项 | 当前 | 说明 |
|---|---|---|
| `camera_matrix` | `fx=fy=1739.1, cx=720, cy=540` | 对齐实车 CS016 + 6mm（fov_y=34.5°）；模拟器 `rt/camera_info` 实测 1739.0842 |
| `distort_coeffs` | 全 0 | |
| `use_muzzle_offset` | `1` | 是否补偿枪口平移 |
| `muzzle_offset` | `[0.07383008, -0.00108151, -0.02848256]` | 枪口相对相机光心（云台系，米） |

### 弹道 / 瞄准

| 项 | 当前 | 说明 |
|---|---|---|
| `bullet_speed` / `gravity` | `22.0` / `9.81` | |
| `pitch_offset_deg` | `-20.0` | 抵消枪管相对云台固定上仰 20° |
| `yaw_offset_deg` | `0.0` | |
| `predict_time` | `0.10` | 额外提前量 |
| `gimbal_lead_time` | **`0.0`** | 补偿云台稳态滞后；看日志 `err` 长期恒正就加大（当前 0，滞后由 `predict_time` 覆盖） |
| `alg_latency` | `0.0` | `>0` 时用它替代实测单帧耗时 |
| `min_distance` / `max_distance` | `2.0` / `15.0` | 观测质量闸 |

### 观测参考系 / 护栏

| 项 | 当前 | 说明 |
|---|---|---|
| `frame` | **`face`** | ★ 必须 `face`（§1.5.5）；`gimbal` 会自激，`chassis` 有 16~23 cm 偏差 |
| `face_ema` / `face_ema_locked` / `face_lock_frames` | `0.10` / `0.003` / `120` | 符面法向平滑 |
| `max_slew_deg_s` | **`600.0`** | 限速（对齐 ④桥；原 40 对模拟器过慢，`3b864d0` 修正） |
| `cmd_smooth_alpha` | `0.5` | 指令低通 |
| `max_aim_step_px` | `400.0` | aim 瞬移拒帧 |
| `stale_ms` | `200.0` | 与 `max_image_age_ms` 取较小者 |

### 火控

| 项 | 当前 | 说明 |
|---|---|---|
| `fire_enabled` | `1` | 总开关 |
| `fire_err_rad` | **`0.08`** | ★ ≈4.6° 的**云台到位门**（`27886e0` 恢复了对准检查；3.20 rad ≈183° 等于不检查，见 §6.4） |
| `fire_stable_frames` | **`2`** | 连续 2 帧到位才开火 |
| `fire_cooldown` | **`0.10`** | 桥侧"敲门"节流（10 发/s）。模拟器自身冷却 1/1.2 s，**真正的射速限制在模拟器** |
| `fire_require_lock` | `1` | ★ 开火纪律：只在**预测可信**时才允许 `fire=1`（治"已经打中了还继续打"）。命中后换组 → 模型门连续拒绝 → `ready=0` → 自动停火等重锁 |
| `fire_rms_gate_deg` | `15` | 局部模型拟合 rms 上限（度），超过就不算"预测可信" |
| `fire_min_n` | `15` | 局部模型最少样本数 |
| `aim_local_only` | `1` | 瞄准预测来源：本地 mode 7 模型 ready 时**只用它**（HW26 predictor 实测会转圈） |

### 其它

| 项 | 当前 | 说明 |
|---|---|---|
| `visualize` | **`0`** | 当前**关**（`d4e800f` 实验分支，为量化帧率上限）；开窗口约 +9 ms/帧 |
| `visualize_every` | `1` | **1 = 每帧都显示，不要调大** |
| `console_every` | `10` | 屏幕打印节流；CSV 不受影响 |
| `log_every` | `1` | 逐帧写 CSV（为与遥测 1:1 对齐） |
| `log_path` | `hw26_bridge_log.csv` | **每次运行会截断重写** |
| `image_topic` / `pose_topic` / `cmd_topic` | `rt/image_raw` / `rt/gimbal_pose` / `cmd/gimbal` | |
| `max_image_age_ms` | `300.0` | 与 `stale_ms` 取较小者 → 实际 200 ms |

---

## 6. 当前实测状态与已知风险

### 6.1 旧模拟器基线（2026-10-12 17:38，899 帧 / ≈36 s / 中位 25.2 FPS，`local_model: 6`）

> ⚠️ **这一节不是当前配置**：它是旧模拟器（Zenoh 版）+ `local_model: 6` + 旧火控参数（`fire_err_rad: 3.20`）下测的。当前是 `local_model: 7`、`fire_err_rad: 0.08`、`max_slew_deg_s: 600`、`visualize: 0`（见 §5），指标会不同。
> 保留它的价值在于：**"检测 + PnP + 尺度"这条线的健康度结论与参数无关**（见本节末尾的结论行）。

数据源：`logs/20261012_173842_run.log` + 同次运行的 `hw26_bridge_log.csv`。

| 指标 | 值 | 读法 |
|---|---|---|
| `n_det>0` | 456/899 = **50.7%** | 符在闪，其余时间是灭灯 |
| `solved=1` | 456/899，**与 `n_det>0` 完全重合** | 有检出就 100% 解得出 PnP |
| `buff_dis` | 6.16 / **6.53** / 7.10 m | ✅ 与真值 ≈6.4 m 吻合 → 当时 0.088 的尺度对了（当前 0.093，见 §1.5.2） |
| `status` | CONVERGING 414 · **TRACKING 42** · TEMP_LOST 88 · LOST 355 | 只有 **4.7%** 的帧在 TRACKING |
| `control=1` | 43 帧 = 4.8%（42 帧是 TRACKING，仅 1 帧来自局部 tracker） | control 连续段只有 **13 / 17 / 10 / 2 / 1** 帧 |
| `fire=1` | 20 帧 = 2.2% | 只在 control 段内 |
| 单帧耗时 | 中位 24.4 ms，p90 56.5 ms，max 116.6 ms | 其中推理 18–39 ms |
| `reproj_px` | **全程 −1** | 该字段当前没写（§9） |

**结论：检测 + PnP + 尺度这条线是健康的；问题全部集中在"点亮/灭灯的节拍"上。**

### 6.2 ★ 节拍与 `local_coast_s` / `hit_s` 严重不匹配（本节的实测是 **mode 6** 下做的）

> ⚠️ 这 8 个窗口 / 7 个间隔的数据来自 `local_model: 6` 的那次运行（2026-10-12 17:38）。
> 中途切过 mode 5（裸 AR，**已确认接线是坏的**，§9.2），**当前是 `local_model: 7`**（物理约束 + 稳健 LM）：mode 7 不需要 1.0 s 填充、也有 rms 门，
> 所以下面第 1、3 条的"每个窗口从零重来"问题在 mode 7 下不复存在。
> 第 2 条的 `hit_s` 只对 mode 6 生效，现在也不相关。

同一次运行按 `det>0` 切：

| | 时长 |
|---|---|
| 8 个**点亮**窗口 | 1.58 / 3.17 / 3.09 / 2.34 / 2.34 / 2.85 / 1.70 / **0.99** s |
| 7 个**灭灯**间隔 | 3.13 / 3.21 / 2.50 / 2.34 / 3.01 / 1.62 / 1.74 s |

1. **`local_coast_s = 0.2` 覆盖不了任何空窗**：7 个灭灯间隔**全部都 >0.2 s**，最短的 1.62 s 是
   coast 窗口的 **8 倍**，最长 16 倍。滑行在灭灯后约 0.2 s 就把参数清空，于是**每个点亮窗口
   都从零重来**。mode 6 的设计意图是"覆盖空窗" —— 在这个节拍下它覆盖不了。
   提交 `85b8bdf` 刚把 `local_coast_s` 从 0.4 改成 0.2，方向正好相反。
2. **`hit_s = 2.6` 会误报命中**：8 个点亮窗口里有 3 个超过 2.6 s（3.17 / 3.09 / 2.85 s），
   会把正常点亮误判成"刚打中"并冻结参数。诊断里确实出现了 `hit=1`。
3. `mode 6` 本身能锁上：`[dual]` 里每个窗口 `n` 都能涨到 48、`span` 120–183°、`rdy=1`；
   但等到 ready，窗口已经过去一半。

> **建议方向**：若继续用 mode 6，要把 `local_coast_s` 提到 **≥3.5 s**（跨过实测灭灯间隔）、
> `hit_s` 提到 **>3.5 s**（大于最长点亮窗口）。若用 mode 5，则这两条都不需要，但必须靠
> 别的办法抑制它的抖（它没有 rms/跨度门）。

### 6.3 双靶标几何不自洽

见 §1.8 的表。核心数字：`rcd` 0.01–0.60 m（应只有几 cm）、`dres` ≤29.9°、`dlt` 在 72↔144 跳、
`pick` 28/45 是 `(0,0)` 且其余与翻转同现。两个运行都有这个现象（17:27 那版 `dlt` 分布
144:105 / 72:97）。

> **建议方向**：要么先用 `dual_target: 0` 只信 A 片做基准，要么按 `rcd` 设门
> （如 `> 0.2 m` 就本帧不用第二片），阻止错配对把坏相位喂进拟合。

### 6.4 开火门被放水

**历史问题（已修）**：曾经把 `fire_err_rad` 放水到 `3.20`（≈183°）= 取消对准检查，
那时的 `fire=1` 只等价于"有瞄准解 + 冷却到了"，**既不表示对准，也不表示命中**；
实测 43 个 control 帧的到位残差：中位 0.052 rad ≈ 3.0°，最大 0.155 rad ≈ 8.9°
（若按原设计 `0.02 rad`，只有 7/43 帧够格）。

**现状**：`27886e0` 已恢复纪律 —— `fire_err_rad: 0.08`（≈4.6°）+ `fire_stable_frames: 2`
+ `fire_require_lock: 1`（要求 mode 7 ready、rms ≤ `fire_rms_gate_deg`、样本 ≥ `fire_min_n`）。

> **注意**：`0.08 rad` 比上面测到的"中位 3.0° / 最大 8.9°"略宽，所以仍会放行一部分
> 残差偏大的帧。要更严就收到 `0.05`（≈2.9°）。

### 6.5 模型在模拟器上的泛化性（历史风险，当前未复现）

HW26 的模型训练在**真实比赛视频**上；模拟器虽然用官方 `POWER.glb`，但材质/光照与实拍不同。
当前（fov 34.5°）实测 `max_score ≈ 0.92`，裕度足够。若将来又出问题，按下面的诊断树走。

**诊断行（已内置）：**

```text
[det] n=2 sel=2 max_score=0.914 [c=2(RED,INACTIVE) conf=0.91 @739,211] [c=2(RED,INACTIVE) conf=0.90 @536,854]
[det] n=0  max_score=0.312  (阈值 0.50)
```

| 字段 | 含义 |
|---|---|
| `n` | 本帧网络检出数（已过 `confidence_threshold`） |
| `sel` | 经 Selector 后剩下的数量（=0 说明被过滤，不是没检出） |
| `max_score` | 全 5040 候选里的**最高类别分**（不受阈值影响）→ 用来看"离检出差多远" |
| `c=0/1/2/3` | `0=蓝未激活, 1=蓝已激活, 2=红未激活, 3=红已激活` |
| `conf` | 该检出置信度 · `@x,y` 检出位置（像素） |

| 现象 | 结论 | 怎么办 |
|---|---|---|
| `max_score` 长期 < 0.15 | 模型**完全看不出**符 | 换检测源（见下） |
| `max_score` 0.2–0.5 但 `n=0` | 只是**阈值太高** | `--conf 0.25` 试 |
| `n≥1` 但 `sel=0` | **类别不匹配**（颜色或激活） | 看 `c=`：若是红而 `my_color` 配了蓝 → `--color 1`；被激活状态卡住 → `filter_activation: 0` |
| `n≥1 sel≥1` | 正常 | 继续看 `st` |

> **实测参考**（HW26 自带 `buff.mp4`）：`max_score≈0.91`、`c=0(BLUE,INACTIVE)`、`conf≈0.91`。

**确实检不出时的退路**：① 的传统视觉（颜色差分 + 轮廓 + 多边形拟合）产出 HW26 需要的
4 个角点 —— HW26 的 `BuffDetection` **只要 4 个角点**，替换成本最低。

### 6.6 HW26 的 `y-z 平面`前提（16.5–23.4 cm 系统偏差）

`Buff::roll = atan2(Δz, Δy)` 与 `predict_position` 都只在观测系的 y-z 平面里重建半径 0.7 m 的圆，
隐含要求"观测系 x 轴落在符面平面内"。实测在 HW26 自己的视频上有 **16.5–23.4 cm** 的
系统性预测偏差（≥靶标尺寸 23 cm）。详见《HW26接入模拟器可行性报告.md》第 4 节。
`frame: face` 就是为了让这个前提精确成立。

### 6.7 其它

| 项 | 说明 |
|---|---|
| 无车体速度补偿 | 弹丸初速含车体平动速度，本桥不订阅里程计。**保持车体静止时无影响** |
| 弹道是无阻力抛物线 | 模拟器线性阻尼 0.0089，7 m 内影响 <1 cm |
| 大符会慢 | RANSAC `min_inliers=35` 而每帧只出 1 个速度样本 → 至少要几十帧才进 TRACKING；这就是需要 `local_tracker` 的原因 |
| 窗口拖慢吞吐 | `visualize: 1` 时 imshow+waitKey 约 +9 ms/帧（当前 `visualize: 0`） |
| `[stage]` 的 `其余` | 是"合计 − lat"的**有符号差**，lat 大于各段之和时为负，口径奇怪但不影响结论 |

---

## 7. 排错

| 现象 | 处理 |
|---|---|
| 启动即 `FATAL: model output shape is not HW26` | `model:` 指向了 sp25 的 `yolo11_buff_int8.xml`。改回 `../model/buff_ir.xml` |
| **相机被猛拽/抖动** | 确认 `frame: face`（不是 `gimbal`）；再把 `max_slew_deg_s` 调小 |
| 没有窗口 | `visualize: 1`；且必须在**有桌面的环境**运行（SSH 无 X 时会打印"无法创建窗口"并自动降级） |
| `还没收到图像，等待中` | 模拟器没跑。先 `./build/zenoh_probe` 测 Zenoh 连通性 |
| 一直 `det=0` | 看 `max_score`：<0.15 说明模型对该画面无反应，降到 `--conf 0.2` 试 |
| `det` 有值但 `sel=0` | 类别不匹配：看 `c=` 调 `my_color`，或临时 `filter_color: 0` / `filter_activation: 0` |
| `det=2` 但 `st` 上不去 | 大符靠 RANSAC，需要窗口；看 `[dual]` 的 `local n/rdy/span` 和 `dres/rcd` |
| `ctrl=0` 恒为 false | 只有 TRACKING（或局部 tracker ready）才产出指令；看 `local rdy=` 与 `st=` |
| 紫点（原绿点）一个个闪、不连续 | 局部 tracker 没 ready（看 `rdy`、`drop`）；根因通常是 §6.2 的节拍不匹配 |
| 有 `err` 但从不 `fire` | `fire_err_rad` 太小：看 CSV 的 `fire_err_rad` 列，长期偏大就调 `gimbal_lead_time` 或放宽门限 |
| `fire=1` 但打不中 | 当前门限 `fire_err_rad: 0.08`（≈4.6°）。先看 CSV 的 `fire_err_rad` 列：长期偏大就调 `gimbal_lead_time`/`predict_time`，或收紧门限（§6.4） |
| 换了 `log_path` 还是被覆盖 | 每次运行都会 `truncate`，要保留就先复制走 |

---

## 8. 文件清单

```text
bridge/
├── README.md                  # 本文件
├── build.sh                   # 一键编译
├── run.sh                     # 一键启动（接模拟器）
├── record.sh                  # 跑一次并录到 logs/<时间戳>_<标签>.log
├── CMakeLists.txt
├── config/bridge.yaml         # 全部参数
├── model/
│   ├── buff_ir.xml/.bin            # HW26，9 点  ← 唯一在用的后端
│   └── yolo11_buff_int8.xml/.bin   # sp25，6 点  ← 代码路径已删除，仅留档（换过去会崩）
├── src/
│   ├── main.cpp               # Zenoh 收发 + 最新帧邮箱 + replay/capture/selftest + 日志
│   ├── hw26_pipeline.hpp/.cpp # OpenVINO 推理 + 双靶标 + PnP + 坐标变换 + 瞄准 + 弹道 + 开火
│   └── local_tracker.hpp/.cpp # 桥自己的局部相位 tracker（local_model 0–6）
├── tools/                     # 24 个 Python + 1 个 shell + 2 个 C++ 探针，索引见 §8.1
├── logs/                      # record.sh 的产物（gitignore）
└── 运行产物（全部 gitignore）
    ├── hw26_bridge_log.csv            # 逐帧日志（每次运行截断重写）
    ├── hw26_rune_kpts_<时间戳>.jsonl   # 真值 8 点 + 模型原始 18 点（每次运行独立文件名）
    └── <同名>.rtstat.csv               # rt/stat 命中率（真命中 hits / 发弹数）
```

### 8.1 `tools/` 索引（27 个）

| 组 | 脚本 | 用途 |
|---|---|---|
| 模型 | `convert_model.sh` | `buff.onnx` → OpenVINO IR（本桥在用的 `model/buff_ir.xml`） |
| 连通 / 自检 | `zenoh_probe.cpp`、`phys_selftest.cpp` | 最小 Zenoh 探针（订阅 `rt/gimbal_pose`）；物理模型合成自检（真值 + 噪声 + 持续换支/翻转） |
| 逐帧统计 | `csv_eval.py`、`dual_summary.py`、`anomaly_scan.py`、`dlt_seq.py`、`ch_vs_amp.py` | 读 `hw26_bridge_log.csv` / `[dual]` 行汇总；`dlt_seq` 看 `dlt` 跳变、`ch_vs_amp` 对比弦相位与振幅 |
| 真值对比 | `aim_vs_truth.py`、`fit_vs_truth.py`、`fit_vs_truth_time.py`、`fit_deep.py`、`fit_eval.py`、`pred_eval.py`、`predictor_compare.py`、`true_params.py`、`true_delta.py`、`true_delta_runs.py`、`kpts_compare.py`、`rune_kpts_table.py` | 用 `rt/rune_kpts` 真值核对拟合参数 / 预测 / 两片夹角；`fit_vs_truth_time` 按 `true_a` 分段（每段 = 一次激活）；`rune_kpts_table.py` 直接从 `POWER.glb` 导出 FACE_1/FACE_2 真值 8 点 |
| 坏窗 / 卡死诊断 | `badwin_probe.py`、`badwin_seq.py`、`badwin_after.py`、`heal_seq.py`、`stuck_scan.py` | 找连续坏窗、自愈（`heal`）触发点、卡死段 |
| 节拍 / 命中 | `fire_interval.py`、`hit_multi.py` | `launch` 阶跃 → 实际开火间隔；多次运行的"发弹 − 命中"汇总 |

> 每个脚本的详细用法看它文件头注释或 `--help`；本表只做"用哪个"的索引。

工作区根目录还有一套**独立的**遥测工具（不属本仓库）：`tools/telemetry.cpp`、
`tools/analyze.py`、`tools/run_session.sh` —— 录模拟器侧真值并出抖动/命中报告，
产物在 `analysis/<时间戳>/`。

---

## 9. 代码/注释清理记录

### 9.1 已清理（2026-10-12）

| 项 | 处理 |
|---|---|
| `refine_r_center()` 死函数 + `Config::refine_r_center` + 配置项 | 全部删除（无任何调用点；R 中心本就直接取 `keypoints[8]`） |
| `Config::max_cmd_step_deg` + 配置项 | 删除（解析了但从未被使用，护栏已由 `max_slew_deg_s` 覆盖） |
| `Config::detect_only` + 配置项 + `hw26_pipeline.cpp` 里的守卫 | 删除。它**只**挡 `--selftest-aim` 的注入路径，正常相机路径完全无效；而配置里写着 `1`，等于把自检的 PnP 关掉 —— 删掉后 `--selftest-aim --mode 1` 恢复 **36/36 通过** |
| `Output` 里只写不读的字段：`cmd_limited`、`aim_jumped`、`blue_valid`、`blue_range`、`fly_s`、`local_a`、`trk_valid`/`trk_rc_ch`/`trk_u_ch`/`trk_v_ch`/`trk_blade_ch` | 删除（含 cpp 里对应的赋值与 7b 诊断块） |
| 构造函数里模型**编译+打印重复两遍** | 删除重复块，`[hw26] detector=...` 现在只打印一次 |
| `DetectorKind::kHw26` 注释写成"不可用"（与现状相反） | 改正 |
| `blade_half_size` 头注释仍写"两个模型共用" | 改正为"两批物理点、两个值" |
| `viz_every` / `local_drop` / `cmd_jump_deg` / `reproj_px` 等处**多段注释挤在一行** | 拆开并归位 |
| `config/bridge.yaml`：sp25 备用后端说明、`local_model` 只列 0–3、`fire_cooldown` 说"对齐 3 帧"、`gimbal_lead_time` 残留注释、`local_rms_gate_deg` 注释挂错行 | 全部改正 |

### 9.2 仍存在

1. **`local_model: 5` 是坏的（接线漏了）**：`push()` 的分支是
   `mode 0||6` / `mode 1||4` / `mode 2` / `else`（[local_tracker.cpp:421-435](HWauto_buff2026/bridge/src/local_tracker.cpp#L421-L435)），
   **mode 5 掉进了最后的 `else`，调的是 `fitFree()`**；而 `ar_c_` 只在 `fitAR()` 里被赋值
   （[L185-193](HWauto_buff2026/bridge/src/local_tracker.cpp#L185-L193)），所以 mode 5 的 `ar_c_`
   **永远为空** → `predict()` 每次都走"最后两点等速外推"那条兜底
   （[L498-505](HWauto_buff2026/bridge/src/local_tracker.cpp#L498-L505)），而且每帧白跑一遍
   65 点 `fitFree` 网格。它**从来没有真正跑过 AR**，尽管 `cf87d1b`/`7fa5f3e` 的 commit message
   都写着已经接好了。修法是把 `mode_ == 5` 加进 AR 分支（1 行），或把该档改名并去掉白跑的 `fitFree`。
   **当前配置已避开（用 mode 7；mode 5 仍然是坏的，别切过去）。**
2. **CSV 的 `reproj_px` 恒为 −1**：这一列从未被写入，重投影误差目前**没有**被计算。
   （`ex_m` 列是 `face_ex`，即面系下径向矢量的出平面分量 |e.x|，**不是**重投影误差。）
3. `[stage]` 的 `其余` 是"合计 − lat"的有符号差，为负时容易误读成 bug。
4. `Output` 里 `blue_px_x/y` 仍在算并打印，但已不绘制；`fire_aim_*`、`cmd_jump_deg` 等
   诊断量只在部分路径被消费。

---

## 10. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-16 | 清理 `tools/` 里 7 个死路脚本（`aim_accuracy*.py` / `aim_sample.py` / `aim_uv.py` / `fly_sensitivity.py` / `lead_sweep.py` / `ts_overlap.py`），本文件 §8 同步（`84dda5a`） |
| 2026-10-15 | `fix` **span 物理不可能时就地清窗**：原来只置 `phys_valid_=0` 就 `return`，坏样本仍留在窗口里 → `ready()` 永久为假（表现是"rms 很小却卡死"）；现在直接清窗重自举。另：`fire_cooldown` 默认值对齐 `0.1`（hpp / main.cpp 解析默认）、`.gitignore` 补 `*.csv`/`*.jsonl`（`cba0eef` / `008819e` / `7efabdf`） |
| 2026-10-14 | **`rt/stat` 命中率链路**：订阅并写 `<日志名>.rtstat.csv`，用**真命中 `hits`**（`accurate` 是接触次数，不能当命中率），`wall_s` 改定点打印 —— 这是唯一能判断"坏窗值不值得修"的指标。另：坏窗自愈两处（失效创新门 + 丢老样本重自举；自举加样本数下限治空转）、CSV 增 `local_amp`/`local_omega`、实验分支强制关可视化量化帧率上限（`5302be9` / `64bc940` / `0485dd0` / `37af7ad` / `76ad74a` / `2890155` / `d4e800f`） |
| 2026-10-13 | **`local_model: 7`（物理约束 + 稳健 LM）成为当前档**：官方速度模型 `θ₀+sgn·[(2.090−a)t+(a/ω)(cos φ−cos(ωt+φ))]`，参数每帧 LM 精修、Huber 稳健权重、`a∈[0.780,1.045]`/`ω∈[1.884,2.000]` 硬夹、窗口参数热启动、换扇区相位重参考；大符 0.3 s 预测中位 **1.4°**。配套：`ready()` 补 **rms 门**（`local_rms_gate_deg: 15`）+ 物理 span 上限（治"紫点转圈"）、瞄准预测源改为**本地优先/独占**、相位观测统一 `mk_obs`（`bc5e7ca` / `780be22` / `41d1ca6` / `e605f7e` / `fdd2a92`） |
| 2026-10-13 | **开火纪律**：只在"预测可信（mode 7 ready + rms/样本门）+ 云台到位（`fire_err_rad`）"时开火；新增 `fire_require_lock` / `fire_rms_gate_deg` / `fire_min_n`，`fire_err_rad` 3.20 → **0.08**（§6.4）（`27886e0`） |
| 2026-10-13 | **打击点锚定实测橙点（组 A）的预测**，不再从模型圆上另取点；移植 RP 的**相位扇区重参考**（跨换片/换组相位连续）；检出先置信度去重、再按**刚性**选一对（治 `rcd=1.4 m` 的非法配对）；新增**位置内点门**（用圆判 `‖t−C‖−0.7`）+ 修正"位置门切换瞄准约定导致大跳"（`b7012e2` / `9f641d4` / `86c6e04` / `7c4a14e` / `155620c`） |
| 2026-10-13 | **真值日志**：订阅 `rt/rune_kpts` 真值 8 点 + 记录模型原始 18 点，每次运行独立文件名（`hw26_rune_kpts_<时间戳>.jsonl`）；新增 `tools/kpts_compare.py`；`record.sh` 自动保存真值日志（`df477f0` / `f94c977` / `ee1faee` / `df6b654` / `77f695c`） |
| 2026-10-13 | 参数与注释修正：`max_slew_deg_s` 40 → **600** 等"对模拟器实际错误"的值（`3b864d0`）；修正过时注释（标定值 / 已删路径 / 与当前算法冲突的描述）（`8ac5d23`）；CSV 增实测板心 3D 位置（`35d1ab7`） |
| 2026-10-12 | 未来打击点（原最大的绿实心圆）由**绿改紫**，并按用户要求**放大 + 改空心**：`r=8 实心` → `r=14 空心、线宽 3`（BGR 240,32,160 = RGB 160,32,240）。判读依据变成**圆环中心**。字段名 `green_px_*` 与代码/日志里"绿点"的叫法保持不变 |
| 2026-10-12 | 新增 `center_source: 1`（**只用一片的历史做固定半径圆拟合**）并设为当前值，替代"两片联立"的圆心：切断两片互相污染（`rcd` 实测可达 0.60 m），并能剔掉该片自己 IPPE 翻转的野值。新增 `--selftest-center` 自检（6/6 通过，外点剔除后圆心误差 0.9 mm）。顺带改正 README 里 `rcd` 的语义 |
| 2026-10-12 | `local_model: 5 -> 4`（AR + 物理包络）。查代码时发现 **mode 5 接线漏了**：`push()` 没把它接进 `fitAR()`，`ar_c_` 恒空 → 实际退化成"最后两点等速外推"（详见 §9.2）。mode 4 的 `fitAR()` 接线是正确的，且无 rms 门、有物理包络兜底 |
| 2026-10-12 | `local_model: 6 -> 5`（裸 AR）。mode 6 要先攒 1.0 s 才锁定，而实测点亮窗口只有 1.0~3.2 s；mode 5 只要 2 个样本即可用。**代价：无 rms/跨度门**，脏数据也会 ready |
| 2026-10-12 | `--selftest-aim` 不再写逐帧 CSV：它每个用例都会按 `log_path` 截断重写文件，会把上一次实测的日志抹掉。现在自检强制 `log_path` 为空 |
| 2026-10-12 | 删除死代码与失效注释：`refine_r_center`（函数+配置项）、`max_cmd_step_deg`、`detect_only`、一批只写不读的 `Output` 字段、重复的模型编译块；修正 `config/bridge.yaml` 与头文件里错位/过时的注释。删 `detect_only` 后 `--selftest-aim --mode 1` 恢复 36/36 通过 |
| 2026-10-12 | 本文件按当前代码重写：删除 sp25 后端相关全部描述（代码已删）；补 `local_tracker`（`local_model` 0–6）、双靶标现状与诊断、可视化标记、完整参数表、当前实测状态（§6.1–6.4）与不一致清单（§9） |
| 2026-10-11 | 局部相位 tracker 上线；大符 RANSAC `min_inliers` 100→35；相位滑动平均；`face` 参考系；护栏对齐 ④ |
| 2026-10-10 | 适配改版后的模拟器相机（内参 + 外参 + 像素阈值）；CSV 增加 `ex/reproj/gimbal_roll/tilt` |
| 2026-10-09 | 初版：HW26 打符算法接入 Daedalus 模拟器 |
