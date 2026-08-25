<div align="center">

<img alt="SimpleMem 标志" src="https://github.com/user-attachments/assets/6ea54ad1-e007-442c-99d7-1174b10d1fec" width="450">

<div align="center">

## 面向 LLM 智能体的高效终身记忆 — 文本与多模态

<small>通过语义无损压缩存储、压缩并检索长期记忆。现已支持文本、图像、音频与视频多模态。</small>

</div>

<p><b>兼容任何支持 MCP（文本记忆）或 Python 集成（完整多模态）的 AI 平台</b></p>

<!-- Atlas Cloud — SimpleMem 可运行的 OpenAI 兼容后端之一 -->
<p align="center">
  <a href="https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=SimpleMem">
    <img src="../../fig/atlas-cloud-logo.png" alt="Atlas Cloud" width="180">
  </a>
</p>

<p align="center">
🎁 <b><a href="https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=SimpleMem">Atlas Cloud</a></b> 是一个全模态、OpenAI 兼容的 AI 推理平台——把 SimpleMem 的 <code>OPENAI_BASE_URL</code> 指向它，即可用 DeepSeek、Qwen、GLM、Kimi、MiniMax 等模型，通过单一 API 完成记忆构建、检索与评估，无需对接多家厂商。提供高性价比的 <a href="https://www.atlascloud.ai/console/coding-plan">coding plan</a>。
</p>

<table>
<tr>

<td align="center" width="100">
  <a href="https://www.anthropic.com/claude">
    <img src="https://cdn.simpleicons.org/claude/D97757" width="48" height="48" alt="Claude Desktop" />
  </a><br/>
  <sub>
    <a href="https://www.anthropic.com/claude"><b>Claude Desktop</b></a>
  </sub>
</td>

<td align="center" width="100">
  <a href="https://cursor.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/cursor/FFFFFF">
      <img src="https://cdn.simpleicons.org/cursor/000000" width="48" height="48" alt="Cursor" />
    </picture>
  </a><br/>
  <sub>
    <a href="https://cursor.com"><b>Cursor</b></a>
  </sub>
</td>

<td align="center" width="100">
  <a href="https://lmstudio.ai">
    <img src="https://github.com/lmstudio-ai.png?size=200" width="48" height="48" alt="LM Studio" />
  </a><br/>
  <sub>
    <a href="https://lmstudio.ai"><b>LM Studio</b></a>
  </sub>
</td>

<td align="center" width="100">
  <a href="https://cherry-ai.com">
    <img src="https://github.com/CherryHQ.png?size=200" width="48" height="48" alt="Cherry Studio" />
  </a><br/>
  <sub>
    <a href="https://cherry-ai.com"><b>Cherry Studio</b></a>
  </sub>
</td>

<td align="center" width="100">
  <a href="https://pypi.org/project/simplemem/">
    <img src="https://cdn.simpleicons.org/pypi/3775A9" width="48" height="48" alt="PyPI" />
  </a><br/>
  <sub>
    <a href="https://pypi.org/project/simplemem/"><b>PyPI 包</b></a>
  </sub>
</td>

<td align="center" width="100">
  <sub><b>+ 任意 MCP<br/>客户端</b></sub>
</td>

</tr>
</table>

<div align="center">

<br/>

[🇬🇧 English](../../README.md) •
**🇨🇳 中文** •
[🇯🇵 日本語](./README.ja.md) •
[🇰🇷 한국어](./README.ko.md) •
[🇪🇸 Español](./README.es.md) •
[🇫🇷 Français](./README.fr.md) •
[🇩🇪 Deutsch](./README.de.md) •
[🇧🇷 Português](./README.pt-br.md)<br/>
[🇷🇺 Русский](./README.ru.md) •
[🇸🇦 العربية](./README.ar.md) •
[🇮🇹 Italiano](./README.it.md) •
[🇻🇳 Tiếng Việt](./README.vi.md) •
[🇹🇷 Türkçe](./README.tr.md)

<br/>

[![项目主页](https://img.shields.io/badge/🎬_交互演示-访问我们的网站-FF6B6B?style=for-the-badge&labelColor=FF6B6B&color=4ECDC4&logoColor=white)](https://aiming-lab.github.io/SimpleMem-Page)

<p align="center">
  <a href="https://arxiv.org/abs/2601.02553"><img src="https://img.shields.io/badge/arXiv-2601.02553-b31b1b?style=flat&labelColor=555" alt="arXiv"></a>
  <a href="https://github.com/aiming-lab/SimpleMem"><img src="https://img.shields.io/badge/github-SimpleMem-181717?style=flat&labelColor=555&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/github/license/aiming-lab/SimpleMem?style=flat&label=license&labelColor=555&color=2EA44F" alt="许可证"></a>
  <a href="https://github.com/aiming-lab/SimpleMem/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat&labelColor=555" alt="欢迎 PR"></a>
  <br/>
  <a href="https://pypi.org/project/simplemem/"><img src="https://img.shields.io/pypi/v/simplemem?style=flat&label=pypi&labelColor=555&color=3775A9&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://pypi.org/project/simplemem/"><img src="https://img.shields.io/pypi/pyversions/simplemem?style=flat&label=python&labelColor=555&color=3775A9&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://mcp.simplemem.cloud"><img src="https://img.shields.io/badge/MCP-mcp.simplemem.cloud-14B8A6?style=flat&labelColor=555" alt="MCP 服务器"></a>
  <a href="https://github.com/aiming-lab/SimpleMem"><img src="https://img.shields.io/badge/Claude_Skills-supported-FFB000?style=flat&labelColor=555" alt="Claude Skills"></a>
  <br/>
  <a href="https://discord.gg/KA2zC32M"><img src="https://img.shields.io/badge/Discord-加入聊天-5865F2?style=flat&labelColor=555&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="../../fig/wechat_logo3.JPG"><img src="https://img.shields.io/badge/WeChat-群组-07C160?style=flat&labelColor=555&logo=wechat&logoColor=white" alt="WeChat"></a>
</p>

<br/>

[🚀 快速开始](#-快速开始) • [🌟 概述](#-概述) • [📦 安装](#-安装) • [🔌 MCP 服务器](#-mcp-服务器文本记忆) • [📊 复现](#-复现论文结果) • [📝 引用](#-引用)

</div>

</div>

<br/>

## 🔥 最新动态

- **[05/21/2026]** 📦 **统一 `simplemem` 包 — 一次导入，自动路由！** SimpleMem、Omni-SimpleMem 和 EvolveMem 现已整合至单一包中。`from simplemem import SimpleMem` 会根据你首次调用的方法自动选择文本或多模态后端，而 `simplemem.optimize(...)` 则调用 EvolveMem 的自进化循环。只需一步即可安装：`pip install -e .`。
- **[05/14/2026]** 🧬 **EvolveMem (v3.0) — 基于 AutoResearch 的自进化记忆！** 检索基础设施现在通过 LLM 驱动的闭环诊断实现自我进化。在 LoCoMo 上，EvolveMem 比最强基准高出**相对提升 +25.7%**；在 MemBench 上，高出**相对提升 +18.9%**。系统能发现原始设计中不存在的全新检索维度。[查看 EvolveMem →](../../EvolveMem/)
- **[04/02/2026]** 🧠 **Omni-SimpleMem (v2.0) — 多模态记忆来了！** SimpleMem 现已支持**文本、图像、音频与视频**记忆。在 LoCoMo 上实现**新 SOTA（F1=0.613，+47%）**，在 Mem-Gallery 上实现**（F1=0.810，+51%）**，超越此前最优方法。[查看 Omni-SimpleMem →](../../OmniSimpleMem/)
- **[02/09/2026]** 🚀 **跨会话记忆 — 超越 Claude-Mem 64%！** [查看跨会话文档 →](../../cross/README.md)
- **[01/20/2026]** 📦 **SimpleMem 已上线 PyPI！** 通过 `pip install simplemem` 安装。[查看包使用指南 →](../PACKAGE_USAGE.md)
- **[01/14/2026]** 🎉 **SimpleMem MCP 服务器正式上线！** 云托管地址：[mcp.simplemem.cloud](https://mcp.simplemem.cloud)。[查看 MCP 文档 →](../../MCP/README.md)
- **[01/05/2026]** SimpleMem 论文已发布至 [arXiv](https://arxiv.org/abs/2601.02553)！

---

## 📑 目录

- [🚀 快速开始](#-快速开始)
- [🌟 概述](#-概述)
- [📦 安装](#-安装)
- [🐳 Docker](#-使用-docker-运行)
- [🔌 MCP 服务器](#-mcp-服务器文本记忆)
- [📊 复现论文结果](#-复现论文结果)
- [🗺️ 路线图](#️-路线图)
- [📝 引用](#-引用)

---

## 🚀 快速开始

### 🧠 理解基本工作流程

从宏观角度看，SimpleMem 是一个面向基于 LLM 的智能体的长期记忆系统。工作流程由三个简单步骤组成：

1. **存储信息** — 对话或事实经过处理后转化为结构化的原子记忆单元。
2. **索引记忆** — 利用语义嵌入和结构化元数据对存储的记忆进行组织。
3. **检索相关记忆** — 当收到查询时，SimpleMem 基于语义而非关键词检索最相关的存储信息。

这种设计使 LLM 智能体能够保持上下文、高效回忆过往信息，并避免重复处理冗余历史记录。

### 🎓 基本用法

SimpleMem 以单一 `simplemem` 包的形式提供。默认的 `mode="auto"` 会根据你调用的方法**自动检测**所需后端 — 无需手动配置：

```python
from simplemem import SimpleMem

mem = SimpleMem()  # mode="auto" — backend chosen by first call
```

你首次调用的方法决定使用的后端：

| 首次调用 | 选择的后端 | 原因 |
|:--|:--|:--|
| `add_dialogue()` | **文本**（SimpleMem） | 基于对话的 API → 文本模式 |
| `add_text()` / `add_image()` / `add_audio()` / `add_video()` | **Omni**（Omni-SimpleMem） | 多模态 API → omni 模式 |

<table>
<tr>
<td width="50%">

**📝 Auto → 文本**（纯文本输入）

```python
from simplemem import SimpleMem

mem = SimpleMem()  # auto mode

# add_dialogue() → text backend auto-selected
mem.add_dialogue(
    "Alice",
    "Bob, let's meet at Starbucks tomorrow at 2pm",
    "2025-11-15T14:30:00",
)
mem.add_dialogue(
    "Bob",
    "Sure, I'll bring the market analysis report",
    "2025-11-15T14:31:00",
)
mem.finalize()

answer = mem.ask("When and where will Alice and Bob meet?")
# → "16 November 2025 at 2:00 PM at Starbucks"
```

</td>
<td width="50%">

**🧠 Auto → Omni**（多模态输入）

```python
from simplemem import SimpleMem

mem = SimpleMem()  # auto mode

# add_image() → omni backend auto-selected
mem.add_text(
    "User loves hiking in the Rocky Mountains.",
    tags=["session_id:D1"],
)
mem.add_image("photo.jpg", tags=["session_id:D1"])
mem.add_audio("voice_note.wav", tags=["session_id:D1"])

result = mem.query("What does the user enjoy?", top_k=5)
for item in result.items:
    print(item["summary"])

mem.close()
```

</td>
</tr>
</table>

> **💡 提示**：Auto 模式会选择适合你数据的最轻量后端。如有需要，你仍可显式使用 `mode="text"` 或 `mode="omni"`。

---

### 🧬 进阶：优化检索配置

在自有开发集上离线调整检索超参数，然后部署生成的 `Config` 用于推理。这是对 EvolveMem 自进化循环的轻量封装：

```python
import simplemem
from simplemem import SimpleMem, load_config

# mem is a finalized SimpleMem instance with memories already built
dev_questions = [
    ("When is the meeting?", "2pm tomorrow at Starbucks"),
    ("What should Bob prepare?", "market analysis report"),
]
config = simplemem.optimize(mem, dev_questions, max_rounds=3)
config.save("my_config.json")

# Later, deploy with the optimized config
config = load_config("my_config.json")
mem = SimpleMem(config=config)
```

> EvolveMem 针对你的开发问题执行 LLM 驱动的"评估 → 诊断 → 提议 → 守护"循环，调整全局检索标志（top_k、融合模式、答案验证、反思轮次等）。如需完整的独立版本（含基准适配器和分类覆盖），请参见 [`EvolveMem/`](../../EvolveMem/)。

---

### 🚄 进阶：并行处理

对于大规模对话处理，可启用并行模式：

```python
from simplemem import create

mem = create(
    mode="text",
    clear_db=True,
    enable_parallel_processing=True,  # ⚡ Parallel memory building
    max_parallel_workers=8,
    enable_parallel_retrieval=True,   # 🔍 Parallel query execution
    max_retrieval_workers=4
)
```

> **💡 专业提示**：并行处理可显著降低批量操作的延迟！

---

## 🌟 概述

**SimpleMem** 是面向 LLM 智能体的统一记忆栈，基于一个核心原则：以高信息密度存储*语义无损*记忆，使智能体在召回更多信息的同时消耗更少的 token。该包整合了三项共享此原则、但各自攻克不同问题的研究成果。

### 📝 SimpleMem：效率核心（文本）

大多数记忆系统面临两难困境：要么被动积累原始交互历史（冗余、耗费大量 token），要么运行昂贵的推理循环来过滤噪声（缓慢、成本高）。SimpleMem 通过三阶段流水线压缩交互内容：

| 阶段 | 功能 |
|:--|:--|
| **1. 语义结构化压缩** | 将非结构化交互提炼为紧凑的记忆单元（含消解指代与绝对时间戳的自包含事实），每条记忆通过多个互补视图进行索引，支持灵活检索。 |
| **2. 在线语义合成** | 在会话内合并相关上下文，形成统一的抽象表示，在构建记忆时消除冗余，而非等到查询时才处理。 |
| **3. 意图感知检索规划** | 推断查询背后的搜索意图，决定*检索什么*并组装精准、紧凑的上下文。 |

在 LoCoMo 基准上，该方法相比以往系统实现了平均 F1 提升 26.4%，同时将推理时 token 消耗减少约 30 倍。机制细节（混合索引层、压缩示例、检索规划）：[**SimpleMem 文本记忆 →**](../text-memory.md)。

### 🧠 Omni-SimpleMem：多模态记忆（文本、图像、音频、视频）

Omni-SimpleMem 将"压缩优先"理念扩展至四种模态，基于三个原则构建：**选择性摄取**（按模态进行熵驱动过滤）、**渐进式检索**（FAISS + BM25 混合检索，配合金字塔式 token 预算扩展）和**知识图谱增强**（多跳跨模态推理）。其架构并非人工设计，而是由一个在两个基准上运行约 50 次实验的自主研究流水线*发现*的——该流水线诊断失败模式、提议架构变更，甚至在无人介入内循环的情况下修复数据管道缺陷。值得注意的是，缺陷修复和架构变更各自的贡献均超过所有超参数调优的总和，使系统从朴素基准跃升至 LoCoMo 和 Mem-Gallery 双榜 SOTA。完整文档：[**Omni-SimpleMem →**](../../OmniSimpleMem/)。

### 🧬 EvolveMem：自进化检索

EvolveMem 填补了几乎所有记忆系统共有的盲点：存储内容在演化，但*检索*机制（评分函数、融合策略、答案生成策略）在部署后就被冻结了。EvolveMem 运行闭环 AutoResearch 流程（**评估 → 诊断 → 提议 → 守护 → 循环**），由 LLM 诊断逐题失败原因并提议配置变更，在出现退步时自动回滚，在停滞时给予探索激励。系统能发现原始设计中不存在的新检索维度（查询分解、实体替换、答案验证），在 LoCoMo 上相对最强基准提升 25.7%，且进化后的配置可在基准间正向迁移。完整文档：[**EvolveMem →**](../../EvolveMem/)。

### 三者如何协作

`from simplemem import SimpleMem` 提供文本核心，并自动路由至多模态后端；`simplemem.optimize(...)` 调用 EvolveMem，针对你的数据调优检索。一个包，一套心智模型：无损压缩，按意图检索，让系统持续自我提升。

---

## 📦 安装

### 📝 初次使用说明

- 请确保**在当前激活的环境中使用 Python 3.10+**，而非仅全局安装。
- 在运行任何记忆构建或检索操作之前，必须**配置兼容 OpenAI 的 API 密钥**，否则初始化可能失败。
- 使用非 OpenAI 提供商（如 Qwen 或 Azure OpenAI）时，请在 `config.py` 中核实模型名称和 `OPENAI_BASE_URL`。
- 对于大型对话数据集，启用并行处理可显著缩短记忆构建时间。

### 📋 要求

- 🐍 Python 3.10+
- 🔑 兼容 OpenAI 的 API（OpenAI、Qwen、Azure OpenAI 等）

### 🛠️ 安装步骤

```bash
# 📥 Clone repository
git clone https://github.com/aiming-lab/SimpleMem.git
cd SimpleMem

# 📦 Install dependencies (pinned versions)
pip install -r requirements.txt

# — OR — install as an editable package
pip install -e .                  # default: text + multimodal + evolver
pip install -e ".[server]"        # + MCP / HTTP server (mcp, fastapi, ...)
pip install -e ".[all]"           # everything, including dev tools

# ⚙️ Configure API settings
cp config.py.example config.py
# Edit config.py with your API key and preferences
```

### ⚙️ 配置示例

```python
# config.py
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = None  # or custom endpoint for Qwen/Azure

LLM_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # State-of-the-art retrieval
```

由于 SimpleMem 对接任意 OpenAI 兼容端点，你也可以通过设置 `OPENAI_BASE_URL` 把它指向像 [Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=SimpleMem) 这样的托管平台：

```python
# config.py —— 使用 Atlas Cloud 作为 OpenAI 兼容后端
OPENAI_API_KEY = "your-atlascloud-api-key"
OPENAI_BASE_URL = "https://api.atlascloud.ai/v1"

LLM_MODEL = "deepseek-ai/deepseek-v4-pro"   # 推理模型，注意预留足够的输出 token
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
```

> `deepseek-ai/deepseek-v4-pro` 是推理模型，请预留足够的输出预算；若返回为空，请调大 token 上限（>= 512）。Atlas Cloud 上其他 OpenAI 兼容聊天模型（如 `Qwen/Qwen3-Next-80B-A3B-Instruct`、`zai-org/glm-5`、`moonshotai/kimi-k2.6`）用法相同。

---

## 🐳 使用 Docker 运行

**MCP 服务器**可在 Docker 中运行，提供一致、隔离的环境。数据（LanceDB 和用户数据库）持久化存储在宿主机卷中。

### 前置条件

- [Docker](https://docs.docker.com/get-docker/) 和 [Docker Compose](https://docs.docker.com/compose/install/)

### 快速启动

```bash
# From the repository root
docker compose up -d
```

- **Web UI：** http://localhost:8000/
- **REST API：** http://localhost:8000/api/
- **MCP (SSE)：** http://localhost:8000/mcp/sse?token=&lt;TOKEN&gt;

数据存储在宿主机的 `./data` 目录（自动创建）。

### 自定义配置

1. 复制环境模板并编辑：
   ```bash
   cp .env.example .env
   # Edit .env: set JWT_SECRET_KEY, ENCRYPTION_KEY, LLM_PROVIDER, model URLs, etc.
   ```
2. 使用环境文件启动：
   ```bash
   docker compose --env-file .env up -d
   ```

### 使用宿主机上的 Ollama

当 `LLM_PROVIDER=ollama` 且 Ollama 运行在本机（非 Docker 内）时，在 `.env` 中设置：

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
```

在 Linux 上，`host.docker.internal` 已通过 Compose 文件自动启用。

### 常用命令

```bash
docker compose logs -f simplemem   # Follow logs
docker compose down                 # Stop and remove containers
```

> 📖 如需自托管 MCP 服务器（Docker 或裸机），请参阅 [MCP 文档](../../MCP/README.md)。

---

## 🔌 MCP 服务器（文本记忆）

SimpleMem 通过模型上下文协议（MCP）提供**云托管记忆服务**，可与 Claude Desktop、Cursor 等 MCP 兼容客户端无缝集成。

**🌐 云服务**：[mcp.simplemem.cloud](https://mcp.simplemem.cloud) — 或使用 [Docker](#-使用-docker-运行) 在本地自托管 MCP 服务器。

### 核心功能

| 功能 | 描述 |
|---------|-------------|
| **可流式 HTTP** | MCP 2025-03-26 协议，采用 JSON-RPC 2.0 |
| **多租户隔离** | 每用户数据表，支持 token 鉴权 |
| **混合检索** | 语义搜索 + 关键词匹配 + 元数据过滤 |
| **生产优化** | 通过 OpenRouter 集成加速响应 |

### 快速配置

```json
{
  "mcpServers": {
    "simplemem": {
      "url": "https://mcp.simplemem.cloud/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```

> 📖 详细安装说明及自托管指南，请参阅 [MCP 文档](../../MCP/README.md)

---

## 📊 复现论文结果

复现论文中 LoCoMo / MemBench / Mem-Gallery 的实验数据。每个支柱在其各自目录中都有独立的基准运行器。请先安装基准扩展：`pip install -e ".[benchmark]"`。

### 📝 SimpleMem（文本）— LoCoMo

从仓库根目录运行：

```bash
python test_locomo10.py                       # full LoCoMo benchmark
python test_locomo10.py --num-samples 5       # quick subset
python test_locomo10.py --result-file my_results.json
```

### 🧬 EvolveMem — 自进化 + LoCoMo / MemBench

从 `EvolveMem/` 目录运行（参见 [`EvolveMem/README.md`](../../EvolveMem/README.md)）：

```bash
cd EvolveMem
python run_evolution.py --data data/locomo10.json --max-rounds 7
python run_benchmark.py locomo --sample 0 --initial weak --max-rounds 3
python run_benchmark.py membench --agent FirstAgent --max-rounds 3
```

### 🧠 Omni-SimpleMem — LoCoMo / Mem-Gallery

从 `OmniSimpleMem/` 目录运行（参见 [`OmniSimpleMem/README.md`](../../OmniSimpleMem/README.md)）：

```bash
cd OmniSimpleMem
python benchmarks/locomo/run_locomo.py --data-path /path/to/locomo10.json --model gpt-4o
```

---

## 🗺️ 路线图

各集成渠道当前支持的能力：

| 能力 | Python（`pip install`） | MCP 服务器（Claude Desktop、Cursor 等） |
|:--|:--:|:--:|
| 文本记忆 | ✅ | ✅ |
| 多模态（图像 / 音频 / 视频） | ✅ | ⬜ 计划中 |
| `optimize()` 自进化检索 | ✅ | ⬜ 计划中 |

计划填补差距的工作（MCP 服务器是独立的多租户文本服务；以下是真实功能，而非文档修复）：

- [ ] **MCP 多模态支持。** 新增 `memory_add_image` / `memory_add_audio` / `memory_add_video` 工具。需要文件上传路径（base64 或 URL，因为 MCP 无法传递本地文件路径）、Omni-SimpleMem 存储后端的多租户适配，以及服务端视觉/音频模型访问能力。
- [ ] **MCP EvolveMem 支持。** 将 `optimize()` 暴露为 MCP 工具。相比多模态更易实现（文本输入，JSON 配置输出，无需文件传输），但 MCP 检索器目前仅支持 EvolveMem 进化的约 10 个维度中的 `semantic_top_k` / `keyword_top_k`。需要扩展 MCP 检索器以支持其余参数（结构化 top_k、融合模式/权重、实体替换、查询分解、答案验证），实现在租户存储记忆上运行进化循环的适配器、每租户配置持久化，以及异步执行（循环依赖大量 LLM 调用，同步请求会超时）。
- [ ] **Docker** 在 MCP 服务器支持上述功能后自动继承（向镜像添加多模态依赖和 Omni 存储卷）。

如需立即使用完整多模态和自进化检索功能，请使用 Python API（参见[快速开始](#-快速开始)）。

---

## 📝 引用

如果你在研究中使用了 SimpleMem，请引用：

```bibtex
@article{simplemem2026,
  title={SimpleMem: Efficient Lifelong Memory for LLM Agents},
  author={Liu, Jiaqi and Su, Yaofeng and Xia, Peng and Zhou, Yiyang and Han, Siwei and  Zheng, Zeyu and Xie, Cihang and Ding, Mingyu and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2601.02553},
  year={2026},
  url={https://arxiv.org/abs/2601.02553}
}
```

```bibtex
@article{evolvemem2026,
  title={EvolveMem: Self-Evolving Memory Architecture via AutoResearch for LLM Agents},
  author={Liu, Jiaqi and Ye, Xinyu and Xia, Peng and Zheng, Zeyu and Xie, Cihang and Ding, Mingyu and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2605.13941},
  year={2026},
  url={https://arxiv.org/abs/2605.13941}
}
```

```bibtex
@article{omnisimplemem2026,
  title   = {Omni-SimpleMem: Autoresearch-Guided Discovery of Lifelong Multimodal Agent Memory},
  author  = {Liu, Jiaqi and Ling, Zipeng and Qiu, Shi and Liu, Yanqing and Han, Siwei and Xia, Peng and Tu, Haoqin and Zheng, Zeyu and Xie, Cihang and Fleming, Charles and Ding, Mingyu and Yao, Huaxiu},
  journal = {arXiv preprint arXiv:2604.01007},
  year    = {2026},
}
```

---

## 📄 许可证

本项目采用 **MIT 许可证** — 详情请参阅 [LICENSE](../../LICENSE) 文件。

---

## 🙏 致谢

我们衷心感谢以下项目与团队：

- 🔍 **嵌入模型**：[Qwen3-Embedding](https://github.com/QwenLM/Qwen) — 业界领先的检索性能
- 🗄️ **向量数据库**：[LanceDB](https://lancedb.com/) — 高性能列式存储
- 📊 **基准**：[LoCoMo](https://github.com/snap-research/locomo) — 长上下文记忆评估框架
