# Touch 面板的 hand 视图：taxel heat map 怎么映射到 MANO 手上、怎么渲染

预览页（ours vs VITRA 那三页：Oct 25、Nov 5 19:11、Nov 5 20:32）的 Touch 面板现在默认是 **hand 视图**：两只张开的 MANO 手，从掌心一侧看，每帧把 taxel 值画在手的表面上，跟 3D 时间轴同步。filtered / filled / raw 三种模式都能用，也可以切回原来的 24×31 grid 视图。

整个流程分两步：

1. **离线（Python）**：给 MANO 网格的每个顶点算一个“它对应 grid 上哪个位置”的连续坐标 (col, row)，导出 `hand.json` + `hand.bin`。
2. **网页（WebGL）**：每帧把这一帧的 taxel 值传进 GPU，片元着色器按每个像素插值出来的 grid 坐标去取值、上色。

**这个映射是近似的。** 录下来的数据里没有每个 taxel 的真实 3D 位置：Wuji SDK 的 `tactile_point_cloud` 只在采集界面实时算，没有存下来。所以位置是根据 grid 本身的手形布局估出来的（见第 6 节）。

---

## 0. 代码在哪

路径都相对于服务器上的 `/root/autodl-tmp/tactile_baseline`。

| 文件 | 作用 |
|---|---|
| `setup/hand_mano_base.py` | 读 MANO、摆姿势（自己写的 numpy LBS）、细分网格、部位标签、掌心视角的 z-buffer |
| `setup/hand_taxel_map.py` | grid → 手表面的映射，输出 `results/tactile_hand/hand.json`、`hand.bin`（另有 `hand_debug.npz` 给检查脚本用） |
| `setup/viser_tactile.js` | Touch 面板，拷到每个页面成为 `tactile.js`；hand 视图的 WebGL 渲染在这里 |
| `tmp/nc_blocks.py` | 用 11-07 不接触的 take 核对“哪块 grid 是哪根手指” |
| `tmp/hand_check.py` | 检查图，输出到 `results/site_preview/tmp_check/`：`hand_map_slots_a.png`（每个 slot 放在哪）、`hand_map_slots_b.png`（每个 taxel 在手上占的区域，棋盘格）、`hand_map_frame.png`（同一帧 grid 和 hand 的对比） |
| `setup/make_page_1025.py`、`make_page_1105.py`、`make_page_1105b.py` | 生成页面时把 `hand.json`、`hand.bin` 拷进 `<page>/tactile/` |

重新生成（用 vitra 环境，因为读 MANO 的 pkl 需要 chumpy）：

```bash
envs/vitra/bin/python setup/hand_taxel_map.py   # -> results/tactile_hand/
envs/vitra/bin/python tmp/hand_check.py         # 可选：检查图
```

然后把 `results/tactile_hand/hand.json`、`hand.bin` 拷到各页面的 `tactile/` 里，或者用 `--tactile` 重新跑 `make_page_*`。

---

## 1. 页面上每帧有什么数据

hand 视图和 grid 视图用的是**同一份数据**，只是画法不同：

- `tactile/meta.json`
  - `slots`：两只手套有传感器的 grid 位置的并集，slot id = row × 31 + col。
  - `dead`：这段 take 里从来没有读数的 taxel。
  - `absent`：这只手套没有的 slot。
  - `filled.unfilled`：filled 模式下仍然补不了的坏点。
  - `parts`：每段的起始时间和帧数。
- `tactile/part<k>.bin`：uint8（值 × 255），顺序是 [左 filtered, 左 raw, 右 filtered, 右 raw]，每块 n_frames × n_slots。
- `tactile/part<k>_filled.bin`：[左 filled, 右 filled]。

---

## 2. grid 的布局：哪块是哪根手指

24 行 × 31 列，指尖朝上。下面是左手套有传感器的格子（526 个）。字母表示我们给它分的块：L 小指、R 无名指、M 中指、I 食指、T 拇指、P 手掌、E 手掌小指侧外缘那一列；`.` 表示没有传感器。

```
     0         1         2         3
     0123456789012345678901234567890
   0 ..............MMMM.............
   1 ......RRRR...MMMMMM...III......
   2 .....RRRRRR..MMMMMM..IIIII.....
   3 .....RRRRRR..MMMMMM..IIIII.....
   4 .....RRRRRR..MMMMMM..IIIII.....
   5 .LLL.RRRRRR..MMMMMM..IIIII.....
   6 LLLLLRRRRRR..MMMMMM..IIIII.....
   7 LLLLLRRRRRR..MMMMMM..IIIII.....
   8 LLLLLRRRRRR..MMMMMM..IIIII.....
   9 LLLLLRRRRRR..MMMMMM..IIIII.TTT.
  10 LLLLLPPPPPPPPPPPPPP..PPPPPTTTTT
  11 LLLLLPPPPPPPPPPPPPPPPPPPPPTTTTT
  12 LLLLLPPPPPPPPPPPPPPPPPPPPPTTTTT
  13 LLLL.PPPPPPPPPPPPPPPPPPPP.TTTTT
  14 .....PPPPPPPPPPPPPPPPPPPP.TTTTT
  15 E....PPPPPPPPPPPPPPPPPPPP.TTTTT
  16 E....PPPPPPPPPPPPPPPPPPPPPTTTTT
  17 E....PPPPPPPPPPPPPPPPPPPPP.TTT.
  18 E....PPPPPPPPPPPPPPPPPPPPP.....
  19 E....PPPPPPPPPPPPPPPPPPPPP.....
  20 E....PPPPPPPPPPPPPPPPPPPPP.....
  21 E....PPPPPPPPPPPPPPPPPPP.......
  22 E....PPPPPPPPPPPPPPPPPP........
  23 E....PPPPPPPPPPPPPPPPPP........
```

- 拇指在右、指尖朝上，所以这张图就是**从掌心看的右手**（也等于从手背看的左手）。各指尖的高低也和真手一致：中指最高，无名指和食指次之，小指再低，拇指最低。
- 两只手套用的是**同一张图**，不是左右镜像。右手套多 8 个 slot：(1,25)、(9,26)、(10,19)、(10,20)、(15,1)、(17,26)、(18,27)、(18,28)。
- 第 13–15 行第 25 列本来就没有传感器，正好在虎口的位置。

**核对：哪块是哪根手指（`tmp/nc_blocks.py`）。** 用的是 2026-11-07 23-12-44 那段：55 s，手不碰任何东西，握松拳加单指弯曲。

- 每根手指的弯曲：用手套自带的 `skeleton.npz`（关节位置）算，等于这根手指相邻骨段夹角之和。
- 每块 grid 的反应：块内每个 taxel 的 raw 值减去它的静息水平（5% 分位数），只保留正的部分，整块求和。
- 两条曲线都做 0.25 s 滑动平均，然后算整段的相关系数。

结果：

- 右手套：每块都和自己那根手指相关最高。小指 0.75，无名指 0.86，中指 0.92，食指 0.93，拇指 0.68。
- 左手套：拇指 0.67、小指 0.57、中指 0.87、无名指 0.75 都对得上；食指那块和无名指几乎打平（0.70 对 0.71），按排除法是食指。

所以右手套画在右手 MANO 上；左手套画在**镜像的右手 MANO** 上（x 取负），两只手共用一套映射。

---

## 3. MANO 手怎么来的（离线，`hand_mano_base.py`）

- **模型**：`VITRA/weights/mano/MANO_RIGHT.pkl`。用我们自己写的 numpy LBS（`J_regressor`、`weights`、`posedirs`、`kintree_table`），shape 参数为 0（平均手型）。
- **姿势**：在 MANO 模板手（平摊的手）基础上做三处调整：
  - 食指、中指、无名指、小指的每个关节（MCP、PIP、DIP）都向掌心屈 6°，看起来自然一点；
  - 小指往无名指收 6°；
  - 拇指绕掌心法线往食指收 10°。
- **坐标系（掌心视角）**：先定指尖方向和拇指方向，再用叉乘得到掌心法线：

$$
y = \frac{J_{\text{middle MCP}} - J_{\text{wrist}}}{\lVert J_{\text{middle MCP}} - J_{\text{wrist}} \rVert}
$$

$$
r = J_{\text{index MCP}} - J_{\text{little MCP}}
$$

$$
x = \frac{r - (r \cdot y)\, y}{\lVert r - (r \cdot y)\, y \rVert}
$$

$$
z = x \times y
$$

  - x 指向拇指一侧（屏幕右边），y 指向指尖，z 从掌心指向看的人。
  - 原点取手腕和四个 MCP 的平均。

- **细分**：做两次中点细分，只加密不改形状：778 个顶点、1538 个三角形 → 12337 个顶点、24608 个三角形。法线由原网格的顶点法线插值得到。
- **部位标签**：每个顶点按 MANO 蒙皮权重分到 palm / thumb / index / middle / ring / little，取权重之和最大的那组关节。
- **掌心视角 z-buffer**：把网格正投影到 xy 平面，0.3 mm 一个像素；每个像素记下离看的人最近（z 最大）的那个三角形和重心坐标，也就是“从掌心看过去看到的第一层表面”。后面的 2D → 3D 都靠它。

---

## 4. grid → 手表面（离线，`hand_taxel_map.py`）

目标是给每个网格顶点两样东西：

- 一个连续的 grid 坐标 (col, row)；
- 一个 coverage（0–1），表示这里是不是落在某块传感器上。

网页上每个像素就用插值出来的 (col, row) 去取 taxel 值。

### 4.1 每块的正向映射：grid 坐标 → 掌心视角里的 2D 点

**四根手指（I / M / R / L）**

- **手指轴**：MCP → PIP → DIP → 指尖顶点的折线，取 2D 投影。指尖顶点用 MANO 的顶点编号：食指 317、中指 444、无名指 556、小指 673。
- **行沿手指方向**：这块最下面一行的下边（中指、无名指、食指是第 9.5 行，小指是第 13.5 行）放在 MCP；最上面一行的上边放在离指尖 1.2 mm 处；中间按弧长线性分布。
- **列横跨手指**：
  - 先在部位标签图上量手指宽度：沿轴取 200 个位置，每个位置沿垂线 ±25 mm 扫描，量出这根手指的标签连续占了多宽。拇指侧和小指侧的半宽分开量，做 9 点滑动平均，并限制在手指中段（25%–60%）中位数的 1.15 倍以内。
  - 列从一侧边到另一侧边线性分布，再整体乘 0.85，让边上的 taxel 留在掌面上。
  - 列号越大越靠拇指一侧。

**拇指（T）**

做法和四指一样，只是轴从 CMC 和 MCP 的中点出发，经过 MCP、IP 到指尖顶点 745。第 18.5 行放在起点，第 8.5 行放在指尖。

**手掌（P，第 9.5–23.5 行、第 4.5–25.5 列）**

用薄板样条（thin-plate spline，`scipy.interpolate.RBFInterpolator`）把 (row, col) 映到 2D。一共 15 个控制点：

- **上边（第 9.5 行）对齐指根**：
  - 第 23 列 → 食指 MCP；第 15.5 列 → 中指 MCP；第 7.5 列 → 无名指 MCP；
  - 第 25.5 列 → 食指 MCP 再往拇指侧移出 0.8 个指根半宽；
  - 第 4.5 列 → 从无名指 MCP 往小指 MCP 走 35%。
- **小指一侧**：
  - (13.5, 4.5) → 小指 MCP 往拇指侧 4 mm；
  - (18.5, 4.5)、(23.5, 4.5) → 手掌小指侧的轮廓，往里缩 2 mm。
- **拇指一侧**：
  - (12, 25.5) → 食指掌骨外侧的轮廓；
  - (14.5, 24.5) → 虎口（grid 在这里本来就缺第 25 列）；
  - (18.5, 25.5)、(20.5, 25.5) → 拇指根部那块肉（大鱼际）的外轮廓，往里缩 4 mm；
  - (23.5, 22.5) → 掌根拇指一侧，往里缩 6 mm。
- **中间**：(16.5, 15.5) → 掌心；(23.5, 15.5) → 掌根中点。

这些“轮廓”都是在部位标签图上按高度量出来的。各行的高度在拇指一侧从食指 MCP（第 9.5 行）到掌根（第 23.5 行，网格最低点往上 6 mm）线性分布。

**手掌小指侧外缘那一列（E，第 0–1 列、第 14–23 行）**

不放在掌面上，而是放在手的小指侧侧面：取同一高度上 x 最小的顶点。

### 4.2 从 2D 到 3D

对每块按 0.1 格的步长密集采样 (row, col)，算出 2D 点，再查掌心视角的 z-buffer，用重心坐标得到第一层表面上的 3D 点。一共约 5.8 万个样本点，每个都知道自己的 (col, row) 和所属的块。

### 4.3 三角形归块，角点取 grid 坐标

**每个三角形归到一块。** 取三角形中心，在“它的 MANO 部位允许的块”里找最近的样本点：

- 手指部位的三角形只能归到自己那根手指或手掌；
- 小指部位还可以归到外缘那一列；
- 手掌部位的三角形可以归到任何块。

之后在相邻三角形上做 3 轮多数投票，去掉零星的碎块。

**角点。** 同一个 MANO 顶点如果被不同块的三角形用到，就复制成几个顶点，每块一个。这样一个三角形内部不会在两块之间插值，块和块之间的边界是干净的。复制后一共 12693 个顶点。

**每个顶点的 grid 坐标和 coverage。** 在它那一块的样本点里找最近的一个，取那个点的 (col, row)。再用到这个点的距离 d 算 coverage：

$$
\text{cov} = \operatorname{clip}\!\left(\frac{4.5\ \text{mm} - d}{2.5\ \text{mm}},\ 0,\ 1\right)
$$

- 掌面上的顶点 d ≈ 0，cov = 1；
- 手背、手指侧面、块与块之间的缝离样本点远，cov 在 2–4.5 mm 之间渐变到 0。

### 4.4 导出 `hand.bin`（338 kB）和 `hand.json`

`hand.bin` 是 little-endian，每段从 4 字节对齐的位置开始，各段的偏移写在 `hand.json` 的 `offsets` 里：

| 段 | 类型 | 内容 |
|---|---|---|
| pos | int16 × 3 | 位置（米 × 250000） |
| normal | int8 × 3 | 法线（× 127） |
| grid | uint16 × 3 | (col + 1) × 2000、(row + 1) × 2000、cov × 65535 |
| faces | uint16 × 3 | 三角形 |

`hand.json` 里还有 `bbox`、`sigma = 0.45`、坐标系说明和姿势参数。

---

## 5. 网页上怎么渲染（`tactile.js`，WebGL 1，没用 three.js）

**两个 canvas。** 每只手套一个 canvas，各自一个 WebGL context，用同一份网格。

- 左手套把顶点和法线的 x 取负，镜像成左手。
- 没开背面剔除，所以镜像后三角形绕序反了也没关系。

**顶点着色器。**

- 先做拖动旋转（绕 x、y 轴，双击复位）。
- 然后正交投影：把 bbox 缩放到 canvas（canvas 的宽高比就是 bbox 的宽高比），深度用 −z。

**每帧传给 GPU 的只有三张小纹理。**

- **值纹理**：32×32 单通道。第 row 行第 col 列放这一帧这个 slot 的 uint8 值。
- **掩码纹理**：同样 32×32，标出当前模式下“有读数”的 taxel。
  - filtered 和 raw 去掉坏点；
  - filled 加回已经补上的坏点（`unfilled` 的除外）；
  - 这只手套没有的 slot（`absent`）都去掉。
  - 只有模式或 part 变了才更新。
- **色表纹理**：256×1 RGB，和 grid 视图完全是同一个色表（0 是深灰，0.3 及以上是黄）。

**片元着色器（逐像素）。** 先插值得到这个像素的 grid 坐标 g 和 cov。取 g 附近 3×3 个格子的中心 c_k，做带掩码的高斯加权；σ = 0.45 个格子，m_k 是掩码（0 或 1），s_k 是这个格子的 taxel 值：

$$
w_k = m_k \exp\!\left(-\frac{\lVert g - c_k \rVert^2}{2\sigma^2}\right)
$$

$$
v = \frac{\sum_k w_k\, s_k}{\max\!\left(\sum_k w_k,\ 10^{-4}\right)}
$$

$$
t = \operatorname{clip}\!\left(\frac{\sum_k w_k - 0.15}{0.4},\ 0,\ 1\right)
$$

$$
a = t^2\,(3 - 2t)\cdot \text{cov}
$$

- v 查色表得到这个像素的颜色。
- a 表示“这里有没有有效的 taxel”：掌面上有读数的地方 a ≈ 1；坏点、没有 taxel 的地方、手背上 a → 0。
- σ 取得比较小，所以一整行坏点会露出灰色，而不会被相邻的 taxel“抹平”。真正的补值只在 filled 模式里做。

**光照和混色。** λ 是 Lambert 项，光从左上前方照过来；skin 是灰色 (0.31, 0.33, 0.37)；最后再加一点镜面高光：

$$
\text{color} = (1 - a)\cdot \text{skin}\cdot(0.38 + 0.62\,\lambda) + a\cdot \text{LUT}(v)\cdot(0.72 + 0.28\,\lambda) + \text{specular}
$$

画出来的效果：

- 灰色：这里没有 taxel，或者是坏点；
- 深色：有读数，但值是 0；
- 红到黄：压力，色标 0 到 0.3。

**和时间轴同步。**

- 面板用 `requestAnimationFrame` 读 3D 播放器的时间；暂停时对齐到 0.2 s 的整数倍，和相机视频一致。
- 只有帧号、模式、视图或旋转变了才重画。
- 每只手一次 `drawElements`（约 2.5 万个三角形），每帧只上传 1 KB 的值纹理。

---

## 6. 局限和下一步

**哪些是估的。**

- 每块内部 taxel 的具体位置：行沿手指等分，列按手指宽度等分。
- 手掌样条的控制点、拇指那条从哪里开始、外缘那一列放在侧面，这些都是按 grid 的形状估的。
- 每条手指里哪一侧朝拇指，是按 grid 图本身的方向假设的，没有数据核对过。
- 左手套食指那块的核对是打平的（见第 2 节）。

**其他。**

- 左手用的是镜像的右手 MANO，不是 MANO_LEFT。
- 手的姿势现在是固定的。因为网格顶点上存的是 grid 坐标，和姿势无关，想让手跟着每帧拟合出来的 MANO pose 动，只要每帧对顶点做 LBS（还需要存每个顶点对应的原 MANO 顶点编号和蒙皮权重），着色器不用改。

**怎么做准。** 在采集机上开着 Wuji SDK，手张开放平，把 `tactile_point_cloud` 录几秒。它是手腕坐标系下 526 个 taxel 的 3D 点，SDK 由手部骨架算出。把这些点连同手套骨架一起对齐到 MANO（关节对关节），每个 taxel 在 MANO 表面上就有了真实位置。这样第 4 节里的估计可以换成用真实位置算，第 5 节的渲染基本不用改。
