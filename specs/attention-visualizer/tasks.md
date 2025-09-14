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
  - 验证：✅ 生成 `attention_viz.html`，浏览器显示红色高亮文本

### 完善任务 (优先级3) - 功能增强

- [ ] 6. 添加模型兼容性检查
  - 来源：requirements.md需求4 + design.md错误处理
  - 文件：`src/attention_extractor.py`
  - 验证：不支持的模型显示明确错误提示

- [ ] 7. 实现长文本处理优化
  - 来源：requirements.md需求2.3 + design.md布局调整
  - 文件：`src/html_renderer.py`
  - 验证：500个token文本正常显示和滚动

### 优化任务 (优先级4) - 边缘情况和性能

- [ ] 8. 完善错误处理机制
  - 来源：requirements.md需求3.4 + design.md错误处理设计
  - 文件：`src/attention_extractor.py`, `src/visualize_attention.py`
  - 验证：异常模型显示有用的错误信息和解决建议

- [ ] 9. 添加性能监控和优化
  - 来源：requirements.md需求3.3 + design.md性能要求
  - 文件：`src/visualize_attention.py`
  - 验证：Console显示处理耗时且在10秒内完成

- [ ] 10. 实现批量可视化功能
  - 来源：requirements.md需求5.2 (P2未来功能)
  - 文件：`src/visualize_attention.py`
  - 验证：支持多个提示词的对比可视化

- [ ] 11. 添加最新模型支持 (未来任务)
  - 来源：requirements.md需求4.4 (P2未来功能)
  - 文件：`src/attention_extractor.py`
  - 验证：检测transformers版本，支持Qwen3等新模型或提供升级提示

## 项目结构 (实际采用的简洁架构)

```
├── test_basic_verification.py  # 核心文件：包含完整可视化流程
├── attention_viz.html          # 生成的可视化HTML文件
├── specs/                      # 规范文档目录
└── README.md                   # 项目说明
```

**设计哲学**：遵循 "Less is More" 原则，单文件包含完整功能链路

## 关键验证命令

```bash
# 完整流程测试 (一键运行)
python test_basic_verification.py

# 预期输出：
# - Console: "权重形状: torch.Size([1, 8])"
# - Console: "生成HTML文件: attention_viz.html" 
# - 文件: attention_viz.html (可在浏览器打开查看可视化效果)

# 自定义文本可视化
python -c "
from test_basic_verification import visualize_attention
visualize_attention('我爱中国')
"
```