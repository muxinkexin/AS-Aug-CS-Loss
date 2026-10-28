# Monodepth2 第一步：迁移 RA-Depth 尺度训练策略的 Agent 执行规范

版本：v1.0｜编制日期：2026-10-26｜交付类型：实施说明，非已完成的训练报告

## 0. 当前状态、第一步目标与执行边界

用户已经完成 Monodepth2 环境验证和官方训练复现。执行 Agent 应复用这些成果，从既有仓库、配置、权重和结果开始，不重新安装环境或重新训练官方基线。

**第一步：保持现有 DepthNet、PoseNet 不变，将 RA-Depth 的尺度增强（AS-Aug）与跨尺度一致性（CS-Loss）迁移到 Monodepth2，完成几何验证、开关回归、A0/A1 对照训练与多分辨率评测。**

这一步回答两个问题：

1. 单纯引入尺度训练策略，能否提高用户当前 Monodepth2 的跨分辨率稳定性？
2. 相对于相同增强、相同训练预算的配置，显式一致性约束是否带来额外收益？

本阶段不加入 CDC、LGFI、Bi-CMamba、Uni-CMamba、HAM、HRNet、新教师或额外监督，不修改网络参数量、解码输出尺度与骨干初始化策略。后续模块应在本阶段留下的可追溯配置上独立消融。

本任务是 **RA-Depth 训练思想在 Monodepth2 上的迁移**，不是原论文 Dual HRNet 结构及成绩的逐项复现。本文为便于验证而规定了有效区域处理和损失归一化；这些工程选择必须写入差异记录，不能宣称与作者代码完全等价。

执行顺序为：定位既有成果 → 固定多分辨率评测 → 实现增强和几何 → 实现训练损失 → 通过单测和短跑 → 完成 A0/A1 训练 → 评测与报告。不要跳过几何验收直接启动长训练。

## 1. 先读取既有成果，填写运行清单

### 1.1 定位项目

- 在用户提供或当前授权工作区的 Monodepth2 项目中工作，先读适用的 `AGENTS.md`。
- 用 `git rev-parse --show-toplevel`、`git status --short`、`git rev-parse HEAD` 确认仓库和现有改动。
- 不把本说明所在目录或研究资料缓存目录自动当成用户训练仓库。
- 已有未提交改动必须保留；新增代码、输出和配置使用独立路径。不得自动执行 `git reset --hard`、`git clean` 或覆盖原权重。
- 在当前项目记录中寻找训练命令、`models/opt.json`、日志和评测结果。缺少必需路径时列出具体缺项，不猜造路径或基线成绩。

### 1.2 必须读取并保存的字段

创建 `artifacts/step1/baseline_manifest.json`，至少包含：

| 字段 | 填写内容 |
| --- | --- |
| `repo_root`、`git_commit`、`dirty_patch_path` | 实际仓库、提交及本地改动记录 |
| `python_executable`、`environment_summary` | 当前已通过验证的解释器、PyTorch/CUDA 等版本 |
| `train_command`、`baseline_options` | 用户实际运行命令与完整训练参数 |
| `data_path`、`image_format` | KITTI 实际路径，JPG/PNG 和预处理方式 |
| `train_split`、`eval_split`、`split_hashes` | 实际文件清单及 SHA256，不能只写 Eigen |
| `height`、`width`、`output_scales` | 输入尺寸和解码输出尺度 |
| `training_mode`、`frame_ids`、`pose_model_type` | 单目/双目/混合、帧配置、位姿网络类型 |
| `init_policy`、`seed`、`batch_size` | 权重来源、实际种子、批大小；未知字段写 unknown |
| `optimizer`、`lr_schedule`、`epochs`、`updates` | 既有训练预算与调度 |
| `checkpoint_paths`、`checkpoint_hashes` | 实际 encoder/depth/pose 等权重路径及 SHA256 |
| `gt_source`、`gt_hash`、`crop`、`eval_depth_range` | 真值与评测条件 |
| `median_scaling`、`post_process`、`baseline_metrics` | 实际对齐、后处理和既有七项成绩 |

本规范首版支持单目视频训练、独立 ResNet 位姿分支、当前帧与前后邻帧、默认 Monodepth2 重投影模式。若现有项目使用 stereo/MS、共享 pose encoder、PoseCNN、predictive mask、`v1_multiscale` 或特殊帧配置，不要静默切换配置；先记录差异并扩展对应测试。A0/A1 必须具有相同的明确配置。

**完成条件：** 能明确回答“以哪套源码、数据、初始化、预算和评测协议为基线”。用户已完成训练这一事实无需重新证明；未知的 seed 等只影响后续比较解释。

## 2. 建立实验 ID 和可追溯目录

| 实验 ID | 网络 | 训练视图 | CS 权重 | 用途 |
| --- | --- | --- | --- | --- |
| `B0_existing` | 用户已复现网络 | 原训练方式 | 不适用 | 复用既有结果与权重，多分辨率评测参照 |
| `B0_offcheck` | 同一网络 | 原训练方式 | 0，增强关闭 | 验证新增代码关闭时回到原始路径 |
| `A0_as_only` | 同一网络 | 低/中/高尺度 | 0 | 分离增强收益与一致性收益 |
| `A1_as_cs` | 同一网络 | 与 A0 相同 | 初始实验设为 1.0 | 检验 CS 的增量收益 |

`B0_offcheck` 只做回归，不要求重新训练完整基线。A0/A1 的 CS 权重 0 与 1.0 是本迁移实验的初始设置；损失归一化见第 7 节，不宣称直接复制原文超参数后即可等价。

建议产物路径：

| 路径 | 内容 |
| --- | --- |
| `configs/step1/*.json` | 已解析的实验配置；不依赖额外 YAML 库 |
| `tools/step1/run.py` | 按本说明实现的配置入口 |
| `tools/step1/evaluate_multires.py` | 显式覆盖推理尺寸的评测工具 |
| `tools/step1/visualize_views.py` | 尺度视图、对应区域、有效掩码可视化 |
| `tests/step1/` | 几何、损失、回归和批处理测试 |
| `artifacts/step1/` | 清单、测试报告、指标、图像、差异说明 |
| `runs/step1/<experiment>/<seed>/` | 新日志和新权重 |

以上工具是执行 Agent **需要创建的接口**，本说明不声称这些脚本已经存在或已在用户环境运行。

## 3. 首先固定多分辨率评测，不训练新模型

### 3.1 修正尺寸覆盖入口

核对项目中的 `evaluate_depth.py`。官方参考实现从 `encoder.pth` 字典读取 `height` 和 `width` 创建测试数据集；仅改变普通命令行尺寸参数，可能没有改变实际输入。

新建独立评测入口，要求：

- 支持 `--eval_width` 和 `--eval_height`，显式覆盖输入张量尺寸。
- 每次输出首个 batch 的真实 shape、checkpoint hash、预测 shape 和样本数。
- 不改写 encoder 权重中的训练尺寸，不覆盖原评测脚本和原预测文件。
- 任意尺寸推理需要重建依赖尺寸的几何对象或计时缓存；纯 DepthNet 评测不需要 PoseNet。
- 采用与原评测相同的逆深度转换、插值顺序、有效像素、裁剪、深度范围和后处理设置。
- 特别保持“逆深度 resize 到真值尺寸，再取倒数”的官方参考流程，不擅自改成先求深度再 resize。

### 3.2 固定尺寸与两类指标

参考尺寸为用户实际训练尺寸。默认评测集合为该尺寸，加以下宽×高，重复项去重：

- `416×128`
- `512×160`
- `832×256`
- `1024×320`

以上四个尺寸对应 RA-Depth 分辨率适应实验。若用户基线为 `640×192`，默认共五种输入尺寸；`320×96` 可作为额外压力测试，单独标记，不混入原四尺寸均值。

评测同时保存：

1. **标准成绩：** 每个尺寸分别遵循既有协议，包括该协议要求的逐图 median scaling。指标为 AbsRel、SqRel、RMSE、RMSElog、δ1、δ2、δ3。
2. **尺度漂移诊断：** 同一图像先在参考输入尺寸计算对齐系数 `alpha_i = median(gt_i) / median(pred_ref_i)`；其他尺寸复用同一 `alpha_i`。固定 GT 有效像素和裁剪。此诊断使用了参考尺寸的 GT 对齐，不是无 GT 的度量尺度恢复。

另保存不依赖 GT 的预测一致性：将不同输入尺寸的 **scaled inverse depth** 对齐到相同图像坐标，在共同有效区域计算 `mean(abs(q_r - q_ref) / max(q_ref, eps))`。不要对每个尺寸的 q 单独做均值或 median 归一化后再声称尺度稳定。

### 3.3 评测工具验收

- 同一 checkpoint、参考尺寸、相同样本与相同协议，应复现旧脚本的指标。
- 默认 FP32 验收容差建议：预测 `atol=1e-6, rtol=1e-5`，指标绝对差不超过 `1e-5`；这是工程起始阈值，不是论文结论。若环境数值差异较大，必须定位原因并记录新的合理阈值，不能静默放宽。
- 故意设置一个不同尺寸，日志必须证明输入 shape 确实改变。
- 测试样本顺序必须一致，并使用稳定 image ID；不能用位置错配的预测计算指标。
- 输出 `baseline_multires.csv`、逐样本对齐系数和诊断结果，禁止覆盖旧结果。

## 4. 按代码职责拆分修改，保留原始关闭路径

| 位置 | 需要做的事 | 必须保留的行为 |
| --- | --- | --- |
| `options.py` 或兼容配置适配层 | 增加尺度训练开关、尺度范围、CS 权重、调试输出 | 默认关闭；原命令仍有效 |
| `datasets/mono_dataset.py` 或独立辅助模块 | 为一个 triplet 生成共享几何参数的 L/M/H 视图和元数据 | 原样本读取、split、原始单视图分支 |
| `layers.py` 或新增几何模块 | 坐标映射、内参变换、对应区域采样、掩码归约 | 原投影函数默认接口及行为 |
| `trainer.py` | 同一 DepthNet 处理三视图，中尺度预测 pose，逐视图几何目标，CS | 关闭增强时原 forward/loss/随机数路径 |
| `networks/*` | 本阶段不增加模块、不改变 state_dict 结构 | 原权重严格加载 |
| 独立评测工具 | 显式推理尺寸、标准与固定尺度诊断 | 原 GT/裁剪/对齐协议 |

新增选项的建议语义：

- `scale_training`: `off` 或 `ra_v1`，默认 `off`。
- `scale_low_range`: `[0.7, 0.9]`。
- `scale_high_range`: `[1.1, 2.0]`。
- `cross_scale_weight`: A0 为 0，A1 为 1.0。
- `cross_scale_output`: 首版只用 `disp` 的输出尺度 0。
- `scale_geometry_mode`: 显式记录本文的 `valid_aware_half_pixel_v1`。
- `save_scale_debug`: 保存检查图，不改变训练损失。

参考范围来自 RA-Depth；其余选项名称是本任务接口设计。若项目已有配置框架，可沿用，但必须保存解析后的全部值。

## 5. 实现尺度增强：先让数据和坐标正确

### 5.1 同一个 triplet 共享随机几何

一次样本为当前帧和相邻源帧。低尺度系数、高尺度系数、裁剪起点和水平翻转必须在该 triplet 内共享。颜色增强沿用已复现基线的方式；A0/A1 使用相同设置。首版对 L/M/H 使用同一次抽样的颜色增强参数，网络读增强图，重建目标仍使用未做颜色扰动的对应 RGB。

中尺度视图 M 复用基线输入构造方式；高/低尺度尽量直接从原始读取图像生成，避免先低分辨率化再放大导致细节额外丢失。

### 5.2 三视图都保持相同网络输入尺寸

设基线输入为高 H、宽 W：

| 视图 | 构造方法 | 必须保存的元数据 |
| --- | --- | --- |
| M | 按基线方式得到 H×W 图像 | 基准 K、实际尺寸、flip 约定 |
| H | 原图 resize 到大于 H×W 的实际整数尺寸，再共享裁剪到 H×W | resize 后真实尺寸、裁剪 dx/dy、A_H、K_H |
| L | 原图 resize 到小于 H×W 的实际整数尺寸，再按 RA-Depth 思路复制尾部区域补齐画布 | 真实低尺度 hL/wL、原始有效矩形、A_L、K_L、seam mask |

这里的 H/L 表示场景在固定画布中的尺度，不是把三张不同形状的张量直接塞入同一个 batch。

### 5.3 低尺度拼接区域的明确规则

低尺度有效图像放在画布左上角，超出 `hL×wL` 的区域属于辅助填充。可采用 RA-Depth 的尾部复制方式完成画布，但本迁移首版规定：

- 光度和 CS 的直接几何监督只使用真实低尺度矩形；不把复制块当成一个连续针孔相机平面。
- 源图采样点也必须落在对应真实区域，且深度为正。
- SSIM、局部差分和平滑项的邻域不能跨越拼接接缝；有效掩码需要按对应算子的邻域收缩。
- 辅助填充依然会通过卷积上下文影响预测；有效掩码不意味着网络完全看不到填充。
- 这一处理与原 RA-Depth 训练代码不完全相同，写入 `implementation_differences.md`。A0/A1 使用相同处理，保证 CS 对照可解释。

不要将“复制拼接”替换为黑色 padding 后继续声称是同一增强配置。若日后研究 padding，请另建实验 ID。

### 5.4 固定像素中心约定

新增对应区域采样采用 `align_corners=False` 的半像素约定；原 Monodepth2 重投影路径保持其现有约定，不能全局搜索替换 `align_corners`。

从 M 的像素坐标映射到 resize/crop 后视图，使用实际整数尺寸得到 `sx = W_resize/W`、`sy = H_resize/H`：

`u' = sx*(u + 0.5) - 0.5 - dx`

`v' = sy*(v + 0.5) - 0.5 - dy`

对应的 A 为：第一行 `[sx, 0, (sx-1)/2-dx]`，第二行 `[0, sy, (sy-1)/2-dy]`，第三行 `[0,0,1]`。在同一坐标约定下 `K_view = A @ K_M`。

- 内参来自同一基准坐标系；遵循用户基线现有 flip 约定，不在本阶段顺便重写 flip。
- 保留 double/float 精度直到生成采样网格，避免先对对应位置 round 再损失精度。
- 使用已经减去 crop 的 K 时，投影函数中不再额外加/减相同 crop offset，防止双重补偿。
- 若选择迁移 RA-Depth 原版的 offset 投影约定，则不能同时采用上述 K 补偿；必须另命名模式并补齐一致测试。本规范默认采用上述显式 A 路线。
- 深度值不乘图像缩放系数；Sigmoid `disp` 也不是像素单位的双目视差。

### 5.5 每个样本最少输出的信息

每视图包含：未颜色增强 RGB、网络 RGB、`K`/`inv_K`、A、原始物理有效 mask、实际 resize 尺寸和 crop。元数据必须逐样本存储，不能假定全 batch 使用同一组随机参数。

建议固定数据键 `view in {L,M,H}` 与 `output_scale in {0,1,2,3}` 两个维度，避免都叫 scale 导致覆盖。保留独立变量 `view_id` 和 `output_scale`。

## 6. 实现共享网络与重建，先不加入 CS

1. 三视图使用同一组 DepthNet 参数；先用清晰的逐视图 forward，便于定位问题。
2. PoseNet 仅对中尺度 M 的帧对预测一次位姿，L/M/H 复用相同相机运动；这是同一 triplet 的图像坐标变化，不是新的真实相机运动。
3. 保留 Monodepth2 原输出尺度及全分辨率重建策略。先将每个输出尺度的 `disp` 上采样到该视图的 H×W，再转深度并在视图坐标系重建，CS 只用输出尺度 0。
4. M 的原损失逻辑保留。L/H 采用对应 K、物理有效区域及源坐标有效条件；无效重建候选必须在 min 选择前处理。
5. 每个视图独立执行 minimum reprojection 与 auto-mask；不要复用另一视图的 auto-mask。
6. 平滑的 mean-normalized inverse depth 保留既有定义；L 的均值和相邻差分只使用物理有效区域。此差异记录在迁移说明中。
7. 保持梯度从各视图重建回到深度，并经共享位姿回到 PoseNet。不能为了节省显存意外 detach 位姿。
8. 三次 forward 会改变 BatchNorm 更新次数。A0/A1 必须使用相同 forward 顺序和批处理方式；不要一个逐视图、一个拼接大 batch。

`scale_training=off` 时直接调用原路径，不抽取额外尺度随机数、不做额外 forward、不改变 BatchNorm、loss 或原数据顺序。

完成后先运行 A0 的几何和短跑验证。CS 尚未实现时也不能用未经验证的重建结果启动长训练。

## 7. 实现跨尺度一致性损失

### 7.1 比较对象和坐标

首版统一比较 DepthNet 输出尺度 0 的 Sigmoid `disp`，对应作者代码中的比较对象。它与固定转换区间内的 scaled inverse depth 具有确定对应关系。不要混用一种视图的深度和另一种视图的 disp。

对 LM、MH 两对视图，通过 A 找到同一真实场景点：可将一张视图预测采样到另一张视图的有效坐标中，但两对的方向必须固定并记录。建议都比较到 M 坐标系：

- LM：M 像素经 A_L 映射到 L 的真实矩形，从 L 的预测中采样，与 M 比较。
- MH：M 像素经 A_H 映射到 H 的实际裁剪视图，从 H 的预测中采样，与 M 比较；出界部分排除。

**采样之前**将无效填充位置的坐标和 mask 区分清楚，不能让 border padding 将越界位置伪装成有效对应。

### 7.2 有效掩码

CS mask 至少包含：真实对应区域、采样有效区域和邻域有效性。首版还使用 M 的 auto-mask 可靠区域，且 mask detach。auto-mask 应由候选语义构造，不能写死“索引 0/1 就是不可靠”，因为帧数及候选排列可能变化。

每张图的 CS 只能使用自己的 `[i:i+1]` mask。测试时将 batch 内不同样本分配不同变换，检查结果不串样本。原 RA 参考代码中存在需要复核的批维索引写法，不能直接复制整段函数。

### 7.3 归约与总损失

每对预测采用与参考代码一致形式的 SSIM/L1 混合距离：`0.85*SSIM_loss + 0.15*L1`。SSIM_loss 沿用现有 Monodepth2 `SSIM` 模块返回的距离定义，不能再重复除以 2。其邻域全部有效时才纳入该像素。

先对每个样本、每对视图做 masked mean，再对非空样本/视图对求平均，得到 `L_cs`；整批没有有效对应时返回与预测相连的零标量并记录事件，避免 NaN。此时不允许用零损失证明模型一致。

首版的明确损失定义：

`L_total = (L_MD2_L + L_MD2_M + L_MD2_H)/3 + lambda_cs * L_cs`

每个 `L_MD2_view` 保留既有输出尺度平均和 smoothness 相对权重；L/H 的几何有效区域处理按第 6 节执行。A0/A1 的唯一配置差异应为 `lambda_cs`。

作者代码采用不同的视图求和方式；本文采用平均以便保持基础损失量级，因此 `lambda_cs=1.0` 是本迁移初始设置，不能宣称与原代码完全等权。记录各项数值和梯度后再决定后续验证集调参；本轮不要同时扫多个 loss、增强范围和学习率。

CS 不对每张图或每个视图单独做 median/mean 尺度对齐，否则可能掩盖本来要约束的跨视图尺度漂移。反传到两侧预测；首版不引入 teacher、EMA 或 stop-gradient。

## 8. 必须先通过的测试

使用项目现有测试工具；若没有 pytest，可以用标准库 unittest。以下属于必要几何与研究正确性检查，不是用测试替代训练。

| 编号 | 输入/边界 | 期望结果 |
| --- | --- | --- |
| T01 | 尺度训练关闭，同权重、batch、随机状态 | eval 输出与原路径一致；同随机状态下 train loss/梯度也匹配 |
| T02 | 各视图 A=I、无 crop、无额外颜色变化 | 对齐前后预测一致，CS 为零或插值容差内接近零 |
| T03 | 非中心主点、非整数 resize 和 crop | `project(AK,X)` 等于 `A*project(K,X)`；深度不随图像缩放变化 |
| T04 | 已知坐标斜坡图，crop 在四角及中心 | 半像素坐标对应正确，无固定偏移；有效内部双线性采样与解析值匹配 |
| T05 | 原采样范围的低/高端、整数尺寸取整 | 使用真实 sx/sy；无负切片、空 crop、拼接尺寸错误 |
| T06 | 已知正深度平面，单位位姿 | 真实有效区域重建近零；图像强插值边界另行排除 |
| T07 | 带相机平移、正深度的合成平面 | 非零位姿 warp 正确；仅测试 T=I 不足以验证 K |
| T08 | L 的拼接接缝、H 的出界采样 | 不产生跨接缝监督；无效候选不被 minimum reprojection 选中 |
| T09 | 全部 CS mask 为零或单对为空 | loss 有限、无除零；空样本不稀释其他样本平均；记录空事件 |
| T10 | 两个样本具有不同 crop 和 mask | batched loss 等于单样本非空结果的同定义平均，不使用全 batch mask 广播 |
| T11 | 相同场景预测被人为乘不同尺度 | 未独立归一化的 CS 非零；防止错误尺度归一化让测试假通过 |
| T12 | A0/A1 只有 CS 权重不同 | 模型结构与数据变换相同；A1 有效 CS 对两侧输出的梯度非零 |
| T13 | 实际 B=1、B=2、最后不足配置 batch 的 batch | 不访问 opt.batch_size 固定索引；支持或明确 drop_last，不能静默丢评测样本 |
| T14 | GPU/CPU 张量、FP32 主路径 | 不新增写死 `.cuda()` 或非必要 device 转换；FP32 为首轮验收 |
| T15 | 多分辨率评测尺寸覆盖 | 真实 tensor shape 随输入尺寸改变；原 GT 像素空间和图像 ID 不变 |
| T16 | 网络/配置关闭回归与 state_dict | 无新增模型参数；原 DepthNet 参数量和严格权重加载保持一致 |

解析投影检查建议在 float64 下 `atol=1e-9`；实际插值图像检查按 FP32 采用可解释容差，并单独说明插值/边界误差。不得为过测试而把掩码全部置零。

必须保存一个 triplet 的 L/M/H、真实区域 mask、SSIM 有效 mask、M↔L/H 对应框、重建误差热图及 JSON 元数据。随机抽样多个不同 crop，不只检查中心裁剪。

## 9. 短跑验证：通过后再做完整训练

### 9.1 顺序

1. 运行 B0_offcheck 的单 batch 回归，加载同一既有 checkpoint。
2. 运行 A0 和 A1 各 100 个 optimizer updates，沿用基线实际 batch；显存不足时同步调整 A0/A1 并记录，不单独改变一组。
3. 至少对一个固定 batch 检查 loss、梯度和更新，确认 CS 确实依赖两侧预测。
4. 保存 checkpoint，重新加载，确认推理与保存前一致；验证训练恢复涉及哪些 optimizer、scheduler 和 RNG 状态。
5. 单测、可视化和短跑全部通过后，顺序完成 A0/A1 的既定训练预算。

100 updates 是工程短跑起始长度，用来发现数据、梯度、显存与状态问题，不构成精度结论。短跑可加载既有 checkpoint 加快检查；完整训练的初始化必须按第 10 节重新设定。

### 9.2 每步或固定间隔记录

- 三视图 photo/smooth loss、LM/MH CS、总 loss。
- 各视图 auto-mask 有效比例、几何有效比例、CS 有效比例和空 mask 次数。
- DepthNet/PoseNet 梯度范数，是否出现 NaN/Inf，disp/深度分位数。
- 实际 batch、每步耗时、显存峰值和训练更新数。
- 有效区域 CS 总被关闭、输出常量化或全部深度接近边界时，排查原因；loss 下降本身不算通过。

## 10. A0/A1 正式训练与公平比较

### 10.1 初始化

默认从与用户基线相同的初始化策略开始：例如相同 ImageNet encoder 初始化、相同 pose 初始化策略及受控的随机 decoder。A0/A1 使用相同 seed 和相同初始 state_dict hash。不要把 A1 从 B0 已训练权重微调、A0 从头训练。

若 B0 的 seed 未记录，B0 继续标为 unknown；本轮 A0/A1 可以统一使用 seed=0 并明确这是新实验设置。统一设置 Python、NumPy、PyTorch、CUDA 及 DataLoader worker/generator 的随机种子，记录确定性选项。建议保存一次共同的未训练初始状态供 A0/A1 加载，包含 DepthNet 与 PoseNet，避免因初始化前的随机数消耗不同而出现不同起点。

若明确采用微调路线，两组都从同一 B0 checkpoint 开始，并使用同一新优化器/调度策略，实验名追加 `finetune`。此结果不能冒充与原训练预算完全一致的从头训练。

### 10.2 必须相同的条件

数据/split/帧数、初始化、seed、triplet batch、优化器、学习率、更新数、调度器、数据增强、几何 mask、BatchNorm 更新方式、验证样本、checkpoint 选择规则。lambda_cs 是 A0/A1 的预期区别。

- 三个尺度 view 不等于三个独立 triplet；分别记录 triplets/update 与 views/update。
- 若需要梯度累积，A0/A1 同样设置；梯度累积不能被描述为与更大 BatchNorm batch 完全等价。
- 原 B0 与 A0/A1 的计算预算不同；B0 作为实用参照，A0→A1 才是 CS 的直接消融。
- 可选的“同预算三次相同视图”对照必须另命名，并记录 BN 更新与随机增强差异，不把它假设为数学上完全相同的 B0。
- 不使用测试集选择 lambda、epoch 或 checkpoint；首轮默认评估预定 final epoch，若采用 best-val，必须三组一致且单列。
- 现有 B0 seed 未知时如实记录。首轮匹配 seed 的 A0/A1 可以给初步证据；稳定性结论需要后续重复 seed，不强行把一次提升写成统计显著。

## 11. 执行入口契约与命令模板

执行 Agent 创建工具后，下列命令应在实际 repo 根目录、已经验证的 Python 环境中执行。所有输入路径写入解析后 JSON；JSON 必须填写实际值，不把占位字符串当成有效路径。

### 11.1 配置内容要求

每个 JSON 至少包含：`experiment_id`、`repo_root`、`baseline_manifest`、`resolved_train_options`、`seed`、`scale_training`、`scale_low_range`、`scale_high_range`、`cross_scale_weight`、`scale_geometry_mode`、`output_dir`、`eval_resolutions`。

`resolved_train_options` 来自既有基线；不能把用户未提供的 epochs、batch、学习率等凭常见默认补全。运行前检查路径、范围、尺寸整除、实验目录冲突和 unsupported 模式，失败应返回非零退出码。

### 11.2 命令顺序

```bash
# 工具创建后，先检查配置，不启动训练
python tools/step1/run.py --config configs/step1/A0_as_only.json --stage check
python tools/step1/run.py --config configs/step1/A1_as_cs.json --stage check

# 无 pytest 依赖的测试入口
python -m unittest discover -s tests/step1 -p 'test_*.py' -v

# 复用既有 B0，固定多分辨率评测
python tools/step1/evaluate_multires.py --config configs/step1/B0_existing.json

# 关闭路径回归与尺度训练可视化
python tools/step1/run.py --config configs/step1/B0_offcheck.json --stage regression
python tools/step1/visualize_views.py --config configs/step1/A1_as_cs.json

# 100 updates 短跑，单独命名输出，不占用正式训练目录
python tools/step1/run.py --config configs/step1/A0_as_only.json --stage smoke --max_updates 100
python tools/step1/run.py --config configs/step1/A1_as_cs.json --stage smoke --max_updates 100

# 前述验收通过后，按照配置完成正式训练
python tools/step1/run.py --config configs/step1/A0_as_only.json --stage train
python tools/step1/run.py --config configs/step1/A1_as_cs.json --stage train

# 所有模型使用同一评测工具和协议
python tools/step1/evaluate_multires.py --config configs/step1/A0_as_only.json
python tools/step1/evaluate_multires.py --config configs/step1/A1_as_cs.json
```

入口必须区分 smoke 和 train 输出，防止 smoke checkpoint 被当成正式成绩。`--stage train` 遇到已有结果时明确 resume 或拒绝覆盖，不静默删除。若现有仓库只允许不同入口，Agent 可以调整命令，但必须在交付报告中写出实际可执行命令并运行验证。

## 12. 必须交付的结果与结束条件

### 12.1 工程验收

- [ ] 既有 B0 权重、配置和结果有 hash/路径记录且未覆盖。
- [ ] 关闭增强时输出、loss 和模型参数结构回归通过。
- [ ] T01—T16 完成，失败或未执行项明确标记。
- [ ] 投影、crop、半像素、拼接边界及逐样本 mask 经合成输入检验。
- [ ] A0/A1 100 updates 短跑有限、可反传、可保存加载。
- [ ] 实际多分辨率输入已由张量 shape 证明，参考尺寸评测回归通过。
- [ ] 训练/评测、原文/迁移差异及随机种子记录完整。

工程验收通过代表实现可运行且满足检查，不代表精度提升已经成立。

### 12.2 研究结果表

创建 `artifacts/step1/results.csv`，每行一个 checkpoint、一个输入尺寸、一种尺度对齐：

`experiment_id, seed, commit, checkpoint_hash, init_policy, training_updates, triplet_batch, width, height, actual_tensor_shape, protocol_id, alignment_mode, AbsRel, SqRel, RMSE, RMSElog, delta1, delta2, delta3, prediction_consistency, num_images, gt_hash`

另创建 `training_cost.csv`，记录视图数、训练时间和显存；推理复杂度若网络未变，应核实并报告相同的网络结构。训练成本增加不能写成“完全无开销”。

### 12.3 最终报告必须回答

1. 原 B0 是否确有分辨率变化退化？用实际逐尺寸曲线说明。
2. A0 相对 B0 有何变化？注明额外视图和训练计算差异。
3. A1 相对 A0 是否改善？分别给参考尺寸、原四尺寸均值、最差尺寸与固定参考对齐诊断。
4. 标准 median scaling 的提升，是否伴随真实跨分辨率一致性改善？
5. 是否出现边界变糊、常量预测、尺度漂移、有效监督严重减少等副作用？
6. 有哪些源代码差异、未运行事项或资源限制？

输出 `artifacts/step1/REPORT.md`：包含实际命令、源码差异、测试摘要、结果表、检查图和结论。负结果也完整交付，不通过删除尺寸、调测试集或更换 checkpoint 隐去退化。

若工程检查通过但 A1 没有优于 A0，检查实现与证据后将结论写为“本配置未观察到增量收益”。不自动添加 CDC/Mamba 来掩盖本阶段结果。

若时间/算力不足，只完成到某阶段，最后准确标注 `implemented / unit-tested / smoke-tested / fully-trained / evaluated`，列出剩余命令。禁止将短跑或合成几何测试写成训练复现成功。

## 13. 参考来源、快照与必读位置

以下提交为本说明核查使用的参考快照，不要求将用户已复现仓库切换到它们。执行时保留用户源码，必要时将参考代码放在独立只读目录，并记录新增差异。

| 来源 | 链接 | 必读位置/用途 |
| --- | --- | --- |
| RA-Depth 原论文，ECCV 2022 | https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870557.pdf | §3.2、Algorithm 1、Fig.2、Table 4/5；理解增强、对应区域与消融 |
| RA-Depth 官方仓库 | https://github.com/hmhemu/RA-Depth | 核对作者实现身份 |
| RA-Depth 数据处理快照 | https://github.com/hmhemu/RA-Depth/blob/4f7ebb436c33b61ed31e49ad1f98d66dffc596b9/datasets/mono_dataset.py | `preprocess` / `__getitem__`；resize、拼接、crop、内参 |
| RA-Depth 训练快照 | https://github.com/hmhemu/RA-Depth/blob/4f7ebb436c33b61ed31e49ad1f98d66dffc596b9/trainer.py | `process_batch`、`generate_images_pred`、`compute_losses`；位姿复用、disp 比较、掩码 |
| RA-Depth 几何快照 | https://github.com/hmhemu/RA-Depth/blob/4f7ebb436c33b61ed31e49ad1f98d66dffc596b9/layers.py | `BackprojectDepth` / `Project3D`；理解原版 offset 与本说明 A 路线的区别 |
| Monodepth2 官方仓库 | https://github.com/nianticlabs/monodepth2 | 已有框架和训练/评测约定 |
| Monodepth2 训练快照 | https://github.com/nianticlabs/monodepth2/blob/b017a8164fae339b0ce01f3a9e5048104d5abd0e/trainer.py | 原始重建、minimum reprojection、auto-mask、多输出尺度 |
| Monodepth2 评测快照 | https://github.com/nianticlabs/monodepth2/blob/b017a8164fae339b0ce01f3a9e5048104d5abd0e/evaluate_depth.py | 从 checkpoint 读取尺寸、逆深度 resize、median/crop/GT |

原 RA-Depth 的 ablation baseline 为 HRNet18 encoder 配 Monodepth2 decoder。不要把论文表内 HRNet 基线成绩，填入用户 ResNet18 Monodepth2 的实际基线栏。

静态审查原代码时特别注意：固定 batch 索引、mask 批维广播、写死 `.cuda()`、未覆盖的可选分支、单输出尺度假设，以及某些源版本只对部分视图应用颜色增强。采用本文明确约定后，逐项记录与原代码差异，不为“逐行照抄”保留已确认不正确的实现。

## 14. 可直接交给执行 Agent 的任务说明

> 用户已完成 Monodepth2 环境验证和官方训练复现。请按本文件顺序执行第一步，复用既有成果，保持网络结构不变，实施并验证 RA-Depth 尺度训练策略迁移。先读取项目 AGENTS.md、训练配置和权重，固定 B0 多分辨率评测；再实现有效区域明确的 L/M/H 增强、几何和 CS。关闭路径必须回归原模型，测试通过后运行 A0/A1 短跑及既定预算训练。保留原文件和结果，记录源码快照、实验配置和实际命令，输出测试、图像、指标及 REPORT.md。不要在本阶段加入 CDC、Mamba 或额外教师，不推测精度收益，不将未执行阶段标为完成。必需输入缺失时先从授权项目材料查找，无法取得则精确列出缺项，继续完成不依赖它的工作。

## 15. 本文档自身的验证范围

编制时已对参考官方源码静态核对，并用 NumPy 合成输入检验本说明的半像素 affine、resize/crop 投影协变、低尺度尾部拼接尺寸边界、对应区域采样和空/逐样本掩码归约。也检查了实验配置差异与默认评测尺寸的整除条件。

这些是实施规范的逻辑与几何检查；没有在用户实际项目中实现上述待建工具，没有运行 PyTorch 网络训练，也没有生成新的模型成绩。执行 Agent 必须完成第 8—12 节的实际验收。
