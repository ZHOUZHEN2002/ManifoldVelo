# ManifoldVelo

本仓库提供论文 **Coupling cellular trajectories and RNA kinetics on continuous
expression manifolds with ManifoldVelo** 对应的模型代码。

模型联合拟合细胞时序、连续 U/S 表达轨迹和 RNA 速度，支持环形、多分叉与单主干
轨迹。训练使用配对 U/S；拟合后可用 S-only 新细胞进行时序与速度预测。

许可为 **Academic Research Only：限非商业学术研究、教学和学术评估**。
商业使用需另行取得授权，完整条款见 [LICENSE](LICENSE)。TSvelo 原始 MIT
许可与版权声明单独保留，本许可不撤销第三方独立授予的权利。

## 使用

安装、可运行示例与 Python API 见 [英文 README](README.md)。

- 环形真实数据使用 `family='cycle'`。
- 多峰周期数据使用 `family='multipeak'`。
- 分叉与单主干数据使用 `family='open'`，并安装 `.[graph]` 以构建 DPT/PAGA 图。
- 可提供 root；省略时使用训练 U/S 选择候选起始细胞群及代表细胞。
- 拓扑家族由分析者指定；环形 root 仅固定相位原点，不自动代表 M→G1。

输入为非负、未取对数、基因与细胞顺序对齐的预处理矩阵；U/S 应保留可比较的
量纲。查询仅提供 S、基因名和细胞 ID。输出同时保留动力学速度与曲线导数。

模型实现来自论文冻结版本；导入路径调整、辅助函数提取及来源哈希见
[源码清单](provenance/SOURCE_MANIFEST.json)。仓库包含核心代码、示例生成器、参数
配置、依赖说明和软件测试，不包含真实数据、训练权重、私人记录或论文草稿。
