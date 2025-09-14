# 任务清单

## 按骨干完整性排序 (最小验证→骨干→完善→优化)

### 最小验证任务 (优先级1) - 技术风险消除

- [x] 1. 实现最小验证DEMO
  - 来源：verify.md第39-62行的最小验证DEMO
  - 文件：`test_basic_verification.py`
  - 验证：Console显示 `"权重形状: torch.Size([1, 8])"`，证明核心算法可行

### 骨干任务 (优先级2) - 端到端最简流程

- [x] 2-5. 实现完整可视化流程 (简化为单函数方案)
  - **设计决策**：采用单文件简洁方案，避免过度设计的多组件架构
  - 文件：`test_basic_verification.py` 的 `visualize_attention()` 函数
  - 功能集成：
    - ✅ 注意力提取：`outputs.attentions[-1]`
    - ✅ 权重聚合：`.mean(dim=1).sum(dim=-2)` 
    - ✅ HTML渲染：红色深浅表示权重强度
    - ✅ 主接口：一个函数完成全流程
    - ✅ 中文token显示修复：使用 `tokenizer.decode([token_id])` 正确显示中文字符
  - 验证：✅ 生成 `attention_viz.html`，浏览器显示红色高亮文本

- [x] 6. 实现Web交互式界面 (新增核心功能)
  - **用户需求**：在浏览器中动态输入文本，实时显示注意力可视化
  - **设计决策**：Flask轻量级Web服务，包装现有可视化逻辑，保持零破坏性
  - 文件：`web_app.py`
  - 功能特性：
    - ✅ 交互式Web界面：输入框 + 实时可视化
    - ✅ AJAX异步处理：无需刷新页面
    - ✅ 美观现代化UI：加载状态、错误处理、快捷键支持
    - ✅ 复用核心逻辑：完全基于 `test_basic_verification.py` 的模型推理
  - 验证：✅ 访问 http://localhost:5000，输入文本，实时显示注意力可视化

- [x] 6.1. 修正注意力计算逻辑 ⭐**关键修复** (2025-09-14)
  - **问题诊断**：原始逻辑使用 `averaged.sum(dim=-2)` 导致计算错误
    - 该操作将所有位置对某个token的注意力相加，破坏了softmax的概率性质
    - 权重总和不再为1，失去了统计意义
  - **修正方案**：
    ```python
    # 修正前（错误）：
    weights = averaged.sum(dim=-2)  # 毫无意义的聚合

    # 修正后（正确）：
    last_token_attention = averaged[0, -1, :]  # 最后一个token的注意力分布
    ```
  - **技术要点**：
    - ✅ 使用最后一个token对所有位置的注意力分布
    - ✅ 保持softmax的概率性质（权重和为1）
    - ✅ 反映模型在处理整个序列时的注意力分配
    - ✅ 符合Transformer因果注意力的实际机制
  - **修改文件**：
    - `web_app.py:34-40` - 更新注意力权重提取逻辑
    - `web_app.py:49-57` - 改进归一化错误处理
  - 验证：✅ 注意力权重现在具有明确的语义含义，可视化结果更准确

### 完善任务 (优先级3) - 功能增强

- [ ] 7. 添加模型兼容性检查
  - 来源：requirements.md需求4 + design.md错误处理
  - 文件：`web_app.py` 或 `test_basic_verification.py`
  - 验证：不支持的模型显示明确错误提示

- [ ] 8. 实现长文本处理优化
  - 来源：requirements.md需求2.3 + design.md布局调整
  - 文件：`web_app.py` 的前端界面
  - 验证：500个token文本正常显示和滚动

### 优化任务 (优先级4) - 边缘情况和性能

- [ ] 9. 完善错误处理机制
  - 来源：requirements.md需求3.4 + design.md错误处理设计
  - 文件：`web_app.py` 的API错误处理
  - 验证：异常模型显示有用的错误信息和解决建议

- [ ] 10. 添加性能监控和优化
  - 来源：requirements.md需求3.3 + design.md性能要求
  - 文件：`web_app.py` 
  - 验证：Web界面显示处理耗时且在10秒内完成

- [ ] 11. 实现批量可视化功能
  - 来源：requirements.md需求5.2 (P2未来功能)
  - 文件：`web_app.py` 扩展多输入功能
  - 验证：支持多个提示词的对比可视化

- [ ] 12. 添加最新模型支持 (未来任务)
  - 来源：requirements.md需求4.4 (P2未来功能)
  - 文件：`web_app.py` 的模型加载逻辑
  - 验证：检测transformers版本，支持Qwen3等新模型或提供升级提示

## 项目结构 (实际采用的简洁架构)

```
├── test_basic_verification.py  # 核心文件：包含完整可视化流程
├── web_app.py                  # Web交互式界面：Flask服务器
├── attention_viz.html          # 生成的可视化HTML文件
├── specs/                      # 规范文档目录
├── pyproject.toml             # 项目依赖配置（uv管理）
└── README.md                   # 项目说明
```

**设计哲学**：遵循 "Less is More" 原则，最小文件数量，最大功能价值

## 关键验证命令

```bash
# 方式1：命令行可视化 (静态HTML文件)
python test_basic_verification.py

# 预期输出：
# - Console: "权重形状: torch.Size([1, 8])"
# - Console: "生成HTML文件: attention_viz.html" 
# - 文件: attention_viz.html (可在浏览器打开查看可视化效果)

# 方式2：Web交互式界面 (推荐)
python web_app.py
# 然后访问: http://localhost:5000
# 在浏览器中动态输入文本，实时查看注意力可视化

# 自定义文本可视化（命令行版本）
python -c "
from test_basic_verification import visualize_attention
visualize_attention('我爱中国')
"
```