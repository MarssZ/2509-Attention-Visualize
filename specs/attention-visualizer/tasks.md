# 任务清单

## 按验证容易程度排序 (技术验证优先，避免错误传染)

### Console验证 - 技术基础验证 (最高优先级)

- [ ] 1. 实现verify.md中的最小验证DEMO
  - 需求：核心技术可行性验证
  - 文件：`test_basic_verification.py`
  - 验证：Console显示 `"权重形状: torch.Size([1, 10])"`，证明核心链路可行

- [ ] 2. 实现注意力权重提取
  - 需求：REQ-1 (注意力权重提取)
  - 文件：`src/attention_extractor.py`
  - 验证：Console显示 `"AttentionExtractor: extracted shape (1, 12, 10, 10)"`

- [ ] 3. 创建权重聚合功能
  - 需求：REQ-1.3 + REQ-2.1 (权重张量处理)
  - 文件：`src/weight_aggregator.py`
  - 验证：Console显示 `"WeightAggregator: aggregated to shape (10,) with max=0.85"`

### 5秒验证 - 用户可见效果

- [ ] 4. 创建HTML基础模板
  - 需求：REQ-2 (HTML文本高亮可视化)
  - 文件：`src/html_renderer.py`
  - 验证：打开HTML文件 → 看到基础页面结构和样式

- [ ] 5. 实现简单文本着色功能
  - 需求：REQ-2.2 (token级别标注)
  - 文件：`src/html_renderer.py`
  - 验证：输入"Hello World" → HTML显示红色高亮文本

- [ ] 6. 实现主接口函数框架
  - 需求：REQ-3 (简化接口)
  - 文件：`src/visualize_attention.py`
  - 验证：调用函数 → 生成HTML文件并自动打开浏览器

- [ ] 7. 添加Token对齐处理
  - 需求：REQ-2.2 (token标签显示)
  - 文件：`src/html_renderer.py`
  - 验证：输入带特殊字符的文本 → HTML正确显示所有token

### 集成验证 - 完整流程测试

- [ ] 8. 添加模型兼容性检查
  - 需求：REQ-4 (多模型兼容性)
  - 文件：`src/attention_extractor.py`
  - 验证：Console显示 `"Model compatibility: ✅ supports output_attentions"` 或警告信息

- [ ] 9. 集成完整数据流
  - 需求：REQ-1到REQ-3的集成
  - 文件：`src/visualize_attention.py`
  - 验证：Console显示 `"Pipeline complete: model → attention → HTML in 3.2s"`

### 状态验证 - 需要特殊条件

- [ ] 10. 测试小型模型兼容性
  - 需求：REQ-4.1 (Qwen系列支持)
  - 文件：`tests/test_model_compatibility.py`
  - 验证：使用bert-base-uncased → 生成正确的注意力可视化

- [ ] 11. 实现长文本处理
  - 需求：REQ-2.3 (长序列自动调整)
  - 文件：`src/html_renderer.py`
  - 验证：输入500个token的文本 → HTML支持滚动查看且布局正常

- [ ] 12. 添加色彩映射优化
  - 需求：REQ-2.4 (颜色深度对应权重强度)
  - 文件：`src/html_renderer.py`
  - 验证：相同文本多次运行 → 最高权重token始终是最深红色

### 边缘情况处理 - 异常和优化

- [ ] 13. 实现错误处理机制
  - 需求：REQ-3.4 (清晰错误信息)
  - 文件：`src/attention_extractor.py`, `src/visualize_attention.py`
  - 验证：传入不支持注意力的模型 → 显示明确错误提示和解决建议

- [ ] 14. 添加性能监控
  - 需求：REQ-3.3 (10秒内完成可视化)
  - 文件：`src/visualize_attention.py`
  - 验证：运行大模型推理 → Console显示耗时且在10秒内完成

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