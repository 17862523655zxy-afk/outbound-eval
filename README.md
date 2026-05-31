# 外呼 Agent 评测平台

自动化评测 AI 外呼 Agent 的综合平台，支持基于 LLM 的用户模拟、多维度评审判分和可视化分析。

## 核心能力

- **多维度评测**：Task Success / Flow Adherence / State Tracking / Recovery / Compliance / Naturalness / Efficiency 七维评分
- **LLM 驱动的用户模拟器**：5 种人物画像（配合型/拒绝型/情绪化/犹豫型/跑题型）× 行为状态机 × 对话记忆
- **Difficulty 分层**：Easy / Medium / Hard 三级难度分布
- **Rule + LLM Hybrid Judge**：规则检查 + LLM 语义判断双重评测
- **可视化 Dashboard**：实时运行监控、评测记录管理、画像/失败/难度/热力图/成功路径 5 维分析
- **Bootstrap CI**：统计置信区间（后端 API 支持）
- **CLI + API**：命令行批量评测 + FastAPI Dashboard

## 快速开始

```bash
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DeepSeek/OpenAI API Key

# CLI 运行评测
python -m outbound_eval run_eval --task feimaotui_contract --scenarios 5

# 启动 Dashboard
python -m outbound_eval serve

# 浏览器访问 http://localhost:8000
```

## 项目结构

```
outbound_eval/
├── agent/           # 被评测的 Agent
├── benchmark/       # 评测核心
├── dataset/         # 评测数据集
├── scenarios/      # 场景生成器
├── events/         # 事件注入器
├── simulator/      # 用户模拟器
├── judge/          # 评测引擎
├── analytics/      # 分析统计
├── experiments/    # 版本对比
├── analyzer/       # 失败分析
├── visualization/  # 可视化
└── dashboard/      # Dashboard
```