# Kardia-R1 模型结构示意图（Mermaid）

> 本文件用 Mermaid 绘制论文《Kardia-R1》的核心架构。在 VS Code 中安装
> **Markdown Preview Mermaid Support** 插件后即可预览，或直接粘贴到
> [mermaid.live](https://mermaid.live) / GitHub 渲染。

---

## 1. 整体架构：数据准备 + 两阶段训练

对应论文 Figure 2，覆盖「Rubric-Guided 数据合成 → 难度划分 → 冷启动 SFT → Rubric-ERL」全链路。

```mermaid
flowchart TB
    subgraph SYNTH["① KardiaBench 数据合成（模型-in-the-loop）"]
        direction TB
        U(["用户画像 u<br/>671 真实画像"]) --> GEN["用户生成器 U"]
        S(["情感情境 s"]) --> GEN
        GEN -->|"开场 x1 = U(u,s)"| RES["共情响应器 A<br/>四段式渐进推理"]
        RES --> EV["Rubric 评估器 R<br/>d ∈ {SOLVED, PASS, FAIL}"]
        EV -->|"内层修正 k ≤ K_max=5"| RES
        EV -->|"情感更新 x_t+1 = U(·)"| GEN
    end

    SYNTH --> STAT["难度统计<br/>p_fail / p_pass / p_solved"]
    STAT --> PART{"难度划分 (Eq.4)<br/>D_easy / D_hard"}

    PART -->|"D_easy（易）"| SFT["Stage 1：冷启动 SFT<br/>四段式输出 · lr=1e-4"]
    PART -->|"D_hard（难）"| RL["Stage 2：Rubric-ERL (GRPO)<br/>N=8 采样 · lr=1e-6"]

    SFT --> ANCHOR["SFT 模型 π_θ0<br/>（作为 KL 锚）"]
    ANCHOR --> RL
    RL --> OUT(["Kardia-R1 模型"])
```

---

## 2. 四段式渐进推理（单轮生成结构）

共情被建模为「理解 → 推理 → 情感识别 → 回复」的换位思考过程。

```mermaid
flowchart LR
    C["上下文 c<br/>画像 + 情境 + 历史"] --> M["策略 π_θ"]
    M --> S1["① understanding<br/>用户意图与情绪线索"]
    S1 --> S2["② reasoning<br/>内在评估 / 换位思考"]
    S2 --> S3["③ emotion<br/>显式情感标签"]
    S3 --> S4["④ response<br/>共情回复"]
    S4 --> O["候选输出 o_j"]
```

---

## 3. Rubric-ERL 奖励组合（Eq. 9–12）

统一奖励 = 格式奖励 + 情感匹配奖励 + 评判器奖励（等权 λ = 1/3）。

```mermaid
flowchart TB
    O["候选 o_j（四段式）"] --> F["格式奖励 r_format (Eq.10)<br/>标签 / 顺序 / 非空"]
    O --> EM["情感匹配奖励 r_emo (Eq.11)<br/>预测 == 真实 ? 1 : 0"]
    O --> RJ["评判器奖励 r_rubric (Eq.12)"]
    RJ --> JUDGE["评判器 LLM_rubric (Qwen3-8B)<br/>5 维：相关性 / 流畅度 / 共情 / 人设一致性 / 安全"]
    F --> W(("统一奖励 r_j (Eq.9)<br/>λ_f = λ_e = λ_r = 1/3"))
    EM --> W
    RJ --> W
    W --> ADV["组内优势 A_j (Eq.6)<br/>z-score 归一化"]
    ADV --> GRPO["GRPO 更新 (Eq.7-8)<br/>裁剪代理 + KL 锚"]
```

---

## 4. 数据合成 Markov 过程（Algorithm 1）

```mermaid
flowchart TB
    START(["输入：画像 u，情境 s"]) --> X1["开场 x1 = U(u,s)"]
    X1 --> T{"t ≤ T_max (10)?"}
    T -->|是| INNER{"内层 k ≤ K_max (5)?"}
    INNER -->|是| A["生成 y_t = A(x≤t, u)"]
    A --> R["评分 d_t = R(x_t, y_t)"]
    R --> DEC{"d_t 达标?"}
    DEC -->|"SOLVED"| OUT(["返回轨迹（终止）"])
    DEC -->|"PASS（跳出内层）"| UPD["情感更新<br/>x_t+1 = U(u,s,x≤t,y_t)"]
    DEC -->|"FAIL 且 k<5"| INNER
    INNER -->|否| UPD
    UPD --> T
    T -->|否| OUT
```

---

## 5. 各模块在代码中的对应关系

| Mermaid 图 | 对应论文公式 | 代码位置 |
|---|---|---|
| 图 2 四段式 | Section 4 | `kardia/tags.py`、`kardia/data.py` |
| 图 3 奖励 | Eq. 9–12 | `kardia/rewards.py`、`kardia/judge.py` |
| 图 4 数据合成 | Algorithm 1 | `scripts/generate_data.py` |
| 难度划分 | Eq. 4 | `kardia/partition.py` |
| GRPO 更新 | Eq. 6–8 | `kardia/grpo.py` |
