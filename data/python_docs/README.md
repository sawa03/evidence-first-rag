# Python 文档中文评测集 v0.1

这是基于公开技术事实独立编写的中文学习笔记和问题集，不包含真实业务数据。它是 AI 辅助编写、尚未经过独立人工审核的自建挑战集，不是 Python 官方数据集或成熟公开基准。

## 来源与使用范围

核对日期：2026-09-25。参考页面显示 Python 3.14.7 文档；URL 为会随版本更新的 `/3/`。实验使用仓库内冻结的中文笔记，不在运行时下载网页；报告记录语料和完整题库 SHA-256。

- [Data Structures](https://docs.python.org/3/tutorial/datastructures.html)：列表增删、弹出、栈、队列。
- [More Control Flow Tools](https://docs.python.org/3/tutorial/controlflow.html)：range、循环控制、循环 else、pass。
- [Input and Output](https://docs.python.org/3/tutorial/inputoutput.html)：文件模式、二进制、关闭、JSON。
- [Errors and Exceptions](https://docs.python.org/3/tutorial/errors.html)：处理、else、raise、finally。
- [Python 文档版权与许可](https://docs.python.org/3/license.html)：原文版权属于 Python Software Foundation 等原权利人，原文适用该页许可；本项目不对原文重新授权。

每个 chunk 的 source 指向对应章节。仓库仅包含简短的原创事实笔记、问题和答案，不收录原文全文、逐句翻译或官方示例代码；这些原创文件采用仓库 MIT 许可。笔记经过简化，不能替代完整文档。

## 构成与划分

16 个资料块、100 道题：32 道直接问答、32 道表达改写、4 道跨块问答、32 道近主题无答案题。不是 100 个独立知识点，同一主题包含相关问题。

开发集和测试集各 50 道：34 道有答案、16 道无答案。开发集使用数据结构与控制流主题；测试集使用文件与异常主题。相关证据和主题不能跨集合；跨块问题仅组合本集合的证据。所有 16 块都在检索库中，以保留跨主题干扰项。

无答案只指冻结的 16 块资料不足以支持回答，不能用模型的外部知识作答；例如资料提到文件模式，但没有给出特定机器上的文件上限。`relevant` 保存回答需要的证据集合，`reference_answer` 供人工复核使用，不自动视为字符串匹配标准。

这是同一开发流程编写的两个集合，存在措辞相关性和标签偏差；不是盲测，也没有人工签字验收。测试集本次只运行固定默认参数，之后不得据此调参并继续宣称独立测试成绩。应先在开发集改进，再新增未见题复验。

## 复现

在仓库根目录运行：

```sh
python evaluate.py --corpus data/python_docs/corpus.json --questions data/python_docs/questions.json --dataset-name python-docs-zh-v0.1 --split dev --output reports/python-dev.json
python evaluate.py --corpus data/python_docs/corpus.json --questions data/python_docs/questions.json --dataset-name python-docs-zh-v0.1 --split test --output reports/python-test.json
```

默认 BM25、k=3、词项覆盖门槛 0.15；不调用 GPT，不产生 API 用量。可用 `--min-overlap` 在开发集实验。报告包含题型分组、逐题检索、拒答、摘录、参考答案以及输入哈希。

Recall/MRR 衡量检索，全部证据召回率衡量多证据是否齐全；无答案误接受率反映拒答门槛的不足。接受只是放行原文摘录，不是生成回答正确。没有相应样本或没有拒答时，对应分母为空的指标为 null。

下一步需要人工复核全部参考答案和无答案标注，再进行固定 GPT 模型的小规模生成实验、回答和引用支持程度审核。不能把本次检索成绩写为 GPT 正确率。
