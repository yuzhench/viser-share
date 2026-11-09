# pipeline/baseline/code：补坏点和 MANO 手视图的代码（参考用）

这里是 baseline 页面（三个 take：`pipeline/baseline/`、`pipeline/baseline_1105/`、`pipeline/baseline_1105b/`）Touch 面板背后的代码副本。原文件在 GPU 服务器的 `/root/autodl-tmp/tactile_baseline/` 下（`setup/` 和 `tmp/`），脚本里写的都是那边的路径。这里没有数据，不能直接跑，用来看具体做法。

文件名后缀对应的 take：`1025` = 2026-10-25 17:42，`1105` = 2026-11-05 19:11（透明杯子），`1105b` = 2026-11-05 20:32（棉花）。同一步骤的三个版本只有路径和 take 编号不同。

## fill/：坏点怎么补

页面上 filtered / filled / raw 三种模式的数据按下面的顺序生成：

1. `tactile_filter_<take>.py`：接触滤波，得到 filtered 值。
   - 静息水平：每个 taxel 每 1 s 取 10% 分位数，再在前后 30 s 内取这些值的 30% 分位数，减掉，负数置 0。
   - 噪声 σ：剩下部分的 1.4826 × MAD。
   - 接触：高于 max(4σ, 0.02) 开始，低于 max(2σ, 0.01) 结束，至少 3 个采样点（25 ms）；还要求 grid 上至少有一个相邻 taxel 也在接触，零星的单点去掉。
2. `export_tactile_web_<take>.py`：导出网页用的 `tactile/meta.json` 和 `tactile/part<k>.bin`（filtered 和 raw，uint8，对齐到 3D 时间轴，每秒 30 帧）。坏点（dead）指整段 take 里读数从来没超过 0.01 的 slot。
3. `tactile_fill_web_<take>.py`：补坏点，写 `tactile/part<k>_filled.bin`。
   - 对每个坏点，在 grid 上找最近的 12 个正常 taxel。距离按只经过有传感器的 slot 的路径长度算（最多 5 步），所以隔着缝的另一根手指不算近。
   - 用一个不带截距的 ridge 回归，从这 12 个相对位置的值预测中心的值。权重在同一只手套、同一个 take 里所有“中心和这 12 个位置都正常”的其他地方一起学（平移不变），用整段 take、每秒 30 帧。能学的地方少于 6 个时，改用 8、6、4 个邻居。
   - 检验：把能学的地方按 grid 左右两半分开（一半太少时按棋盘格分），一半学、另一半算相关系数，两个方向都做。r ≥ 0.3 才补，否则留空（页面上 grid 视图是空心圈，hand 视图是灰色）。
   - 补出来的值取 max(0, 预测)，低于 0.01 的置 0；正常 taxel 保持自己的 filtered 值。所以 filled 模式 = filtered + 补上的坏点。
4. `glove_ref.py`：上面第 2、3 步用到的 `Take` 类（读 take 的元数据和时间）。

补值方法的验证（为什么选这个方法）：

- `tactile_fill_eval.py`（第 1 轮）：用右手套的真实读数检验，在左手套坏点的位置上补右手套，留一个 take 做测试。
- `tactile_fill_transfer.py`（第 2 轮）：在左手套还能读数的那个 take 上，把后来坏掉的行藏起来再补，看能不能补对。
- `tactile_fill_round3.py`（第 3 轮）：藏两列正常的（和真实坏列一样宽），补完和真实读数比；再补真实的坏列，和另一只手套对照着看。
- `nocontact_check.py`：11-07 不接触那段 take 上两只手套的静息水平和噪声。

## hand_view/：MANO 手视图

- `taxel_mano_hand_view.md`：说明文档（grid 怎么映射到 MANO 手上、网页怎么渲染）。
- `hand_mano_base.py`、`hand_taxel_map.py`：生成每个页面的 `tactile/hand.json` 和 `tactile/hand.bin`。
- `hand_check.py`：检查图。
- `nc_blocks.py`：用 11-07 不接触的 take 核对 grid 上哪块是哪根手指。

网页端的代码就是各页面目录下的 `tactile.js`（Touch 面板，hand 视图的 WebGL 渲染也在里面）。
