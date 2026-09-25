# Evidence First RAG

**证据优先的资料问答实验室：检索、引用、拒答与可复现实验。**

这是一个从零编写的学习与作品集项目。当前为 v0.2 原型，不是已验证的研究创新或生产服务。

## 已实现

- 无第三方依赖即可运行的中文/英文 BM25 检索、原文摘录与来源展示。
- 可选的 Ollama 本地向量检索，以及 BM25 + 向量检索的 RRF 融合。
- 可选的 GPT（OpenAI Responses API）或本地模型生成，引用编号检查与低词项覆盖率拒答。
- 网页演示、命令行评测、单元与 HTTP 测试。
- 原创虚构 Atlas 产品手册及小型人工编写问题集，不含真实业务或科研数据。

**运行状态须区分：**BM25 与网页可以直接使用；本地语义检索和本地生成需要另外安装 Ollama 并准备模型；GPT 生成需要 OpenAI API 密钥和可用额度。当前仓库测试包含模拟模型接口测试；GPT 接入未配置真实密钥，尚未完成真实 API 端到端验证。

## 一分钟启动

需要 Python 3.11 或更新版本，无需安装 Python 依赖。在本仓库目录执行：

```sh
python app.py
```

打开 <http://127.0.0.1:8765>。默认模式返回检索到的原文，不调用大模型，也不会把资料发送到云端。页面上的“原文摘录”不应当被当作生成式问答成绩。

```sh
python -m unittest discover -s tests -v
python evaluate.py
```

逐题结果保存到 `reports/smoke.json`。语料和问题文件的 SHA-256 随报告记录，便于确认实验输入。

## 接入 GPT API（推荐的下一步）

先在 [OpenAI 平台](https://platform.openai.com/api-keys) 准备 API 密钥和可用额度。不要把密钥发到聊天或提交到 GitHub。

Windows 用户可双击 `start_gpt.cmd`：它会在后台启动服务，确认就绪后自动打开 <http://127.0.0.1:8766>；重复运行会复用已启动的服务。服务无需密钥即可启动；在网页顶部“连接 GPT”中输入密钥，点击“保存到本次会话”即可启用生成。也可执行：

```sh
python launch.py
```

电脑重启后需要再次启动，程序未设置开机自启。仅打开网页链接不会启动本地服务。后台启动失败时查看 `.runtime/server.log`（该目录不提交到 GitHub）。如需在终端前台运行：

```sh
python app.py --provider openai --port 8766
```

网页输入框使用密码样式，提交后立即清空，不使用浏览器本地存储。密钥经本机接口保存到当前 Python 进程内，不写入文件，也不会由服务返回给网页。服务重启后需重新输入。已有环境变量 `OPENAI_API_KEY` 时自动使用；若希望在终端隐藏输入，可额外添加 `--prompt-api-key`。本程序不自动读取 `.env` 文件。保存密钥不会调用付费接口，密钥有效性在首次生成时验证。

默认模型为 `gpt-4.1-mini`。可通过 `--generation-model` 或环境变量 `OPENAI_MODEL` 修改；优先级为命令行、环境变量、默认值。实际可用模型取决于你的 API 项目权限。输出上限默认 1024 tokens，可用 `--max-output-tokens` 设置 64–4096。

网页默认不勾选生成。勾选“使用 GPT 生成回答”再查询时，才会向 OpenAI 发送问题及 Top-k 资料块的编号、标题和正文；不发送整库、来源路径或密钥到前端。BM25 检索始终在本机运行。证据门槛未通过时直接拒答，不调用 GPT。当前 GPT 模式不支持向量/混合检索，后续再独立接入嵌入模型。

请求使用 Responses API，设置 `store=false`，不使用文件上传、工具或自动重试。`store=false` 不等同于所有服务端日志零保留，数据处理仍以 [OpenAI 数据说明](https://developers.openai.com/api/docs/guides/your-data) 为准。超时可能已经产生用量，重试需用户手动触发。

模型返回后仍检查引用编号。界面显示实际输入/输出 token 用量，不把它换算为未经核实的金额。生成截断、模型拒绝、密钥失效、额度或速率限制均会给出明确错误，不把原始服务异常或密钥显示在网页上。

官方参考：[API 快速开始](https://developers.openai.com/api/docs/quickstart)、[GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini)。实现使用 Python 标准库直接调用官方 HTTPS 接口，因此无需新增依赖。

## 接入本地模型

先自行在 Ollama 中安装适合机器配置的嵌入与生成模型，然后使用实际模型名称：

```sh
python app.py --embedding-model YOUR_EMBEDDING_MODEL --generation-model YOUR_CHAT_MODEL
python evaluate.py --embedding-model YOUR_EMBEDDING_MODEL --modes bm25 dense hybrid --output reports/comparison.json
```

将示例中的名称替换成已安装模型名。程序只连接 `127.0.0.1:11434`，不会自动下载模型或产生付费 API 请求。配置后界面才开放相应模式；服务不可用时明确报错，不静默退回其他方法。

接口依据 Ollama 官方 [embed 文档](https://docs.ollama.com/api/embed) 与 [generate 文档](https://docs.ollama.com/api/generate)。生成接口返回的实际 token 计数会展示在 API 结果的 `usage` 中；当前未估算货币成本。

## 数据格式

`data/corpus.json` 是已经分块的资料数组，每块包含 `id`、`title`、`source`、`text`。`id` 必须唯一且仅由英文字母、数字、下划线或连字符组成。

```json
[{"id":"guide-01","title":"标题","source":"公开文档 URL 或说明","text":"资料正文"}]
```

```sh
python app.py --corpus PATH_TO_YOUR_PUBLIC_CORPUS.json
```

目前不支持 PDF 解析、自动分块、文件上传或持久化向量索引。默认语料为本项目原创的虚构演示文本；真实业务资料不应提交到公开仓库。`.gitignore` 排除了 `data/private/`、模型和环境配置。

## 架构与关键选择

```text
问题 → BM25 / 本地向量 / RRF 混合 → Top-k 证据
                                       ↓
                              词项覆盖率拒答门槛
                                       ↓
                          原文摘录 / GPT 或本地生成
                                       ↓
                           引用 ID 检查 + 来源展示
```

- BM25 中文分词使用连续双字片段，方便理解和复现，但对同义表达有限。
- RRF 合并排名，避免把 BM25 分数与余弦相似度直接相加；这是已有方法，不是本项目提出的新算法。
- 拒答门槛使用查询词项与单块证据的最大重合比例，默认 0.15。它是未经校准的启发式，不是置信度；语义改写可能被误拒答，关键词相似的无答案问题也可能漏拒答。
- 生成引用校验只检查引用 ID 是否存在，不能证明陈述被证据蕴含；页面明确显示“语义支持待核验”。提示词中的资料隔离也不能保证抵抗所有提示注入。
- 向量索引在进程内首次使用时生成、随后复用；重启后重建。当前适合小型实验，不适合大规模资料库。
- 自带 HTTP 服务只绑定本机，不提供生产环境认证、限流或隔离。

## 评测边界

目前共有 8 个资料块、14 个问题（10 个有答案，4 个无答案）。这是与实现一起编写的合成冒烟集，未进行独立测试集划分，不能代表公开基准或实际业务准确率。

报告包括有答案问题的 Recall@k、MRR@k，无答案拒答率、有答案接受率及耗时。延迟包括检索和摘录；向量模式首个问题包含建索引耗时。当前评测不调用生成模型，也不评估生成正确率、引用蕴含度或费用。

初次 BM25 冒烟测试在这组数据上 Recall@3 和 MRR@3 均为 1.0，但无答案拒答率仅为 0.75：涉及“审计日志导出 PDF”的问题出现漏拒答。这是接下来要解决的已知问题，不应把小样本检索成绩写成“问答准确率 100%”。实际结果见 `reports/smoke.json`。

## 接下来的改进实验

详见 [项目路线与实验设计](docs/ROADMAP.md)。重点研究“证据不足时是否应该回答”，通过基线、改进和消融实验验证价值。当前没有声明改进幅度。

## 学习来源与许可证

学习路线参考 Nir Diamant 的 [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)。该参考项目使用[自定义非商业许可证](https://github.com/NirDiamant/RAG_Techniques/blob/main/LICENSE)。本仓库未复制其代码、Notebook 或数据，采用独立编写的代码与演示资料；不代表与原作者有合作或背书关系。

本仓库代码与原创演示数据按 [MIT](LICENSE) 许可。以后引入第三方资料或模型时，需要逐项保留其来源和适用许可。
