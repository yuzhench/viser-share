# mujoco_sim：MuJoCo 4.0 里的触觉手（第一版）

在 MuJoCo 4.0 里搭的一只带手套触觉的右手，加一个能拧开瓶盖的饮料瓶，之后用来在仿真里测试用我们数据训练的策略。原文件在 GPU 服务器的 `/root/autodl-tmp/tactile_baseline/setup/`。演示视频见 `demo/index.html`。

## 文件

- `mj_export_mano.py`：导出 MANO 右手（模板网格、关节、蒙皮权重）和手套点位的绑定，写 `results/mujoco_tactile/mano_assets.npz`。
- `taxel_binding.npz`：手套 534 个点位在 MANO 表面上的位置（每个 slot 落在哪个三角形的三个 MANO 顶点编号 + 重心坐标）和 5 个指尖顶点编号。这是我们自己的数据，不含 MANO 的几何。
- `mj_tactile_hand.py`：手的模型 `ManoHand`（MJCF、位姿、读触觉、接触力），带自检。
- `mj_bottle_scene.py`：桌子、瓶子、瓶盖的场景，和抓取规划（numpy 运动学：手指包络、拇指 IK、三指捏）。
- `mj_demo_bottle.py`：三个演示（抓起瓶子、三指拧瓶盖、桌面按压）、读数模型、出视频和统计。
- `demo/`：演示页面和三段视频（左边仿真画面，点的颜色是读数；右边是同一时刻的手套网格图，和网页上真实数据同一个布局）。

## 手和触觉怎么做的

- 手：MANO 右手按蒙皮权重分成 16 节（手掌 + 每根手指 3 节），每个顶点归到权重最大的那一节。每节的碰撞体是它所有顶点的凸包。手的各节之间不互相碰，只和物体、桌子碰。画面里显示的是 MuJoCo skin（MANO 网格 + MANO 蒙皮权重，骨骼就是这 16 节）。
- 触觉点：手套 24×31 网格里的 534 个点位，用网页 hand view 同一套映射贴到 MANO 表面（三角形 + 重心坐标，任何姿态下都跟着皮肤走；映射代码在 `baseline-update` 分支的 `pipeline/baseline/code/hand_view/hand_taxel_map.py`，结果就是 `taxel_binding.npz`）。每节手上的点做成一个 mesh，挂在一个不参与碰撞的 geom 上，由 MuJoCo 4.0 自带的 `tactile` 传感器读出：每个点压进"正在和这节手接触的物体"里的深度（channel 0）。另外两个 channel 是滑动速度，要 mesh 带切向标架才有值；我们的手套只测压力，没用。
- 驱动：每节手是一个自由刚体（重力补偿），用软 weld 拉向一个 mocap 目标；脚本（以后是策略）只给目标位姿：MANO 的 16×3 轴角 + 手腕的 4×4 位姿。不直接拿 mocap 当手，是因为 MuJoCo 把 mocap 的速度当成 0，摩擦带不动物体（拧瓶盖拧不动）。
- 接触软硬：MuJoCo 的软接触力和有效质量成正比，所以每节手按自己的质量设 solref 时间常数，让接触刚度都约为 K = 1.5 N/mm（`ManoHand.mjcf(K=...)`）。每节质量最少按 30 g 算，不然指尖的 weld 太软、捏不住东西（有重力补偿，质量只影响惯性）。
- 抓握控制：`Wrapper`（整手抓）让每根手指沿规划好的包络弯曲，到包络后再继续，直到这根手指受力 4 N；拇指同理。`Pincher`（三指捏）让拇指、食指、中指从张开姿态往捏紧姿态走，到 4 N 停。手松开、再握紧时会自动补力。
- 读数模型（占位，还没标定）：深度先在相邻点之间做 σ = 2.5 mm 的高斯扩散（软层），再换成 0–1 的读数：

  读数 = 1 − exp(−深度 / 1.5 mm)

  没有偏置、噪声和坏点。之后用真实的桌面按压数据拟合这两个参数。

## 场景

- 瓶子：半径 32 mm、高 17 cm、0.5 kg 的自由刚体，立在桌上；拧盖时用一个软 weld 固定（相当于另一只手拿着）。
- 瓶盖：半径 15.5 mm，一个 hinge（转）加一个 slide（升），按螺纹 2.7 mm/圈联动，hinge 上有 0.08 N·m 的干摩擦，只能往拧开的方向转 1.5 圈。
- 两个坑：
  - MJCF 里 hinge 的 `range` 默认单位是度，不是弧度。
  - 转和升的联动用一根固定 tendon（长度保持 0），没用 joint equality。MuJoCo 估算约束对角项时忽略了多项式的斜率，瓶盖的转动惯量又很小，joint equality 会软一万倍左右，等于不起作用。

## 三个演示和现在的结果

数字来自 `results/mujoco_tactile/summary.json`。

1. 抓起瓶子、放下：瓶子抬起 118 mm（指令 120 mm），在手里滑了 1.7 mm；空中总接触力 28 N（拇指 14 N，食指、中指、无名指各约 4.4 N）。有读数的是四个指尖和拇指指尖。
2. 三指拧瓶盖，拧 3 次：手每次转 90°，瓶盖一共转了 303°，按螺纹升起 2.3 mm。只有拇指、食指、中指指尖有读数，每根 4–5 N，掌心是 0。
3. 桌面按压（就是要录的标定动作）：
   - 手掌平按 7 mm：先着桌的是大鱼际（MANO 里它比掌心低约 5 mm），42 个点有读数（掌心区 15 个，拇指和大鱼际区 27 个）。
   - 指尖按 2.5 mm：食指、中指的指腹有读数。

已知问题：
- 脚本整手抓的时候掌心几乎不受力：手指只绕到瓶子侧面，主要靠指尖和拇指夹住；MANO 的手掌是一整块刚体，不会弯过来贴住瓶子。
- 拇指转得多时，拇指根部的皮肤显示会被拉扁。只影响画面，碰撞和触觉点按每节刚体算。
- 手是脚本动的，还没接策略；读数模型没标定。

## 环境和运行

- Python ≥ 3.11：`pip install mujoco==4.0.0 numpy scipy pillow matplotlib imageio imageio-ffmpeg`。
- 离线渲染用 EGL（`MUJOCO_GL=egl`），需要 `libEGL.so.1`。服务器上没有，就把 Ubuntu 的 `libegl1` 包解压到工作目录的 `envs/egl/` 里（`apt-get download libegl1`，再 `dpkg -x libegl1_*.deb envs/egl`），脚本会自动加载；别的位置用环境变量 `EGL_LIB` 指定。
- 路径：工作目录默认是服务器的 `/root/autodl-tmp/tactile_baseline`，用环境变量 `TACTILE_BASELINE` 改；结果写到它下面的 `results/`。
- 步骤：
  1. 生成 `mano_assets.npz`。只需要 MANO 的 `MANO_RIGHT.pkl`（MANO 许可证不允许再分发，所以仓库里没有，去 https://mano.is.tue.mpg.de 注册下载）和一个能 unpickle MANO 的环境（要装 chumpy）；点位绑定直接读这个文件夹里的 `taxel_binding.npz`。在服务器上，如果 `hand_taxel_map.py` 的输出（`results/tactile_hand/`）在，就重新算一遍绑定。

     `MANO_RIGHT_PKL=/path/to/MANO_RIGHT.pkl python mj_export_mano.py`

  2. 自检（点位位置误差约 0.001 mm；一个球压进手掌 5 mm，读到的最大深度 5.00 mm）：

     `python mj_tactile_hand.py`

  3. 演示（视频写到 `results/site_preview/tmp_check/mj_*.mp4`，统计写到 `results/mujoco_tactile/summary.json`）：

     `python mj_demo_bottle.py power twist press`
