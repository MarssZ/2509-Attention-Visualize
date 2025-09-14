# 任务清单

## 按骨干完整性排序 (最小验证→骨干→完善→优化)

### 最小验证任务 (优先级1) - 技术风险消除

- [ ] 1. 实现最小验证DEMO
  - 来源：verify.md第39-62行的最小验证DEMO
  - 文件：`test_basic_verification.py`
  - 验证：Console显示 `"权重形状: torch.Size([1, 4])"`，证明核心算法可行

### 骨干任务 (优先级2) - 端到端最简流程

- [ ] 2. 实现核心注意力提取组件
  - 来源：requirements.md需求1 + design.md的AttentionExtractor
  - 文件：`src/attention_extractor.py`
  - 验证：Console显示具体注意力张量形状

- [ ] 3. 实现权重聚合组件
  - 来源：requirements.md需求2.1 + design.md的WeightAggregator  
  - 文件：`src/weight_aggregator.py`
  - 验证：Console显示聚合后的权重分布

- [ ] 4. 实现HTML渲染组件
  - 来源：requirements.md需求2.2-2.4 + design.md的HTMLRenderer
  - 文件：`src/html_renderer.py`
  - 验证：生成HTML文件，浏览器显示红色高亮文本

- [ ] 5. 集成主接口函数
  - 来源：requirements.md需求3 + design.md主接口设计
  - 文件：`src/visualize_attention.py`
  - 验证：调用函数 → 完整流程跑通，生成可视化HTML

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

## 项目结构

```
src/
├── visualize_attention.py      # 主接口函数
├── attention_extractor.py      # 注意力提取组件
├── weight_aggregator.py        # 权重聚合组件
├── html_renderer.py           # HTML渲染组件
└── __init__.py

tests/
├── test_model_compatibility.py
├── test_attention_extraction.py
└── test_html_rendering.py
```

## 关键验证命令

```bash
# 基础功能测试
python -c "from src import visualize_attention; print('✅ 导入成功')"

# 完整流程测试
python test_basic_flow.py  # 应该生成 attention_viz.html

# 性能测试
python test_performance.py  # Console显示耗时信息
```