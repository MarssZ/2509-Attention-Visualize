  ---
  description: 基于需求创建设计文档
  ---

  # 目标

  基于已批准的需求文档，创建功能设计文档。

  ## 核心要求

  1. **前置条件**：确保需求文档已存在且已批准，它位于 `specs/{feature_name}/requirements.md`
  2. **研究导向**：识别技术难点，在对话中进行必要研究
  3. **设计内容**：
     - Overview
     - Architecture
     - Components and Interfaces
     - Data Models(if need)
     - Error Handling
     - Testing Strategy
  4. **设计原则**：
     - 保持简单，避免过度设计
     - 解释关键决策的理由
     - 适时使用图表说明（Mermaid）
  5. **用户参与**：
     - 完成初稿后征求反馈
     - 根据反馈迭代直到批准
     - 批准后创建设计文档。

  ## 输出

  在 `specs/{feature_name}/design.md` 创建设计文档。


 ## 任务
 现在开始基于以下需求文档，分析并与用户探讨设计文档的内容：