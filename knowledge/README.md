# S20 minimal agent notes

S20 的核心不是“把提示词写得很长”，而是一个可观察的控制循环：

1. 把用户消息放入会话历史。
2. 将系统提示、摘要和最近消息送给模型。
3. 如果模型返回工具调用，执行工具并追加 `tool` 结果。
4. 回到模型，直到得到最终文本或达到步数上限。

这个练习项目只保留 calculator、search、read_docs 三个工具。真实模型通过
OpenAI-compatible Chat Completions 接口接入；没有 API key 时用 DemoClient 做确定性测试。
