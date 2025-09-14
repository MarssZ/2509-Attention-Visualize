# 🔍 Attention Visualizer - 注意力可视化工具

> **交互式Transformer注意力权重可视化工具**
> 通过Web界面实时分析模型的注意力分布，理解AI如何"关注"输入文本的不同部分

## ✨ 功能特性

- 🎯 **实时可视化** - 输入文本，立即看到注意力权重分布
- 🌐 **Web交互界面** - 现代化的浏览器界面，支持快捷键和实时分析
- 📊 **科学准确** - 基于Transformer最后一层的真实注意力权重
- 🔧 **技术细节** - 显示tokenizer信息、权重统计和分词结果
- 🎨 **直观呈现** - 红色深浅表示注意力强度，支持悬停查看具体数值

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 支持CUDA的GPU（推荐，CPU也可运行）

### 安装依赖
```bash
# 使用uv包管理器（推荐）
uv add torch transformers flask modelscope

# 或使用pip
pip install torch transformers flask modelscope
```

### 启动应用
```bash
python app.py
```

然后访问：http://localhost:5000

## 🎮 使用方法

1. **启动服务** - 运行 `python app.py`
2. **输入文本** - 在网页中输入要分析的中文文本
3. **查看结果** - 观察每个token的注意力权重和可视化效果
4. **技术分析** - 查看tokenizer信息和详细的权重统计

### 示例输出
```
输入："我爱中国"

可视化结果：
[0] "我"   (权重: 0.245)  ██████
[1] "爱"   (权重: 0.189)  █████
[2] "中国" (权重: 0.566)  ███████████

权重总和: 1.000 ✓
```

## 🔬 技术原理

### 注意力权重提取
本工具采用**因果注意力**的正确计算方法：

```python
# 获取最后一层的注意力矩阵
attention = outputs.attentions[-1]  # [batch, heads, seq_len, seq_len]

# 平均所有注意力头
averaged = attention.mean(dim=1)    # [batch, seq_len, seq_len]

# 提取最后一个token对所有位置的注意力分布
last_token_attention = averaged[0, -1, :]  # [seq_len]
```

## 📁 项目结构

```
├── app.py                        # Web交互式界面（主要功能）
├── specs/attention-visualizer/
│   └── tasks.md                  # 项目任务和开发记录
└── README.md                     # 项目文档
```

## 🎯 支持的模型

目前支持基于Transformer架构的模型：

- ✅ **Qwen系列** - Qwen2-0.5B-Instruct（默认）
- ✅ **其他Transformer模型** - 只要支持 `output_attentions=True`

### 模型切换
```python
# 在 app.py 中修改模型名称
model_name = "你的模型名称"
```

## 🔧 高级功能

### 技术信息显示
- **模型信息** - 显示使用的模型和tokenizer类型
- **词汇表大小** - 模型的词汇表统计
- **分词详情** - 完整的token分解过程
- **权重统计** - 原始权重分布和归一化信息

### 可视化选项
- **颜色映射** - 红色深浅表示注意力强度
- **悬停提示** - 鼠标悬停查看精确权重值
- **响应式设计** - 适配不同屏幕大小

## ⚠️ 重要提醒

1. **分词理解** - 不同模型的tokenizer会产生不同的分词结果
2. **权重解释** - 注意力权重反映的是token级别的关系，不是词级别
3. **模型限制** - 闭源API模型（如GPT-4）无法获取内部注意力权重
4. **计算资源** - 大型模型需要相应的GPU资源

## 🛠️ 开发历史

### 已完成的关键修复
- ✅ **注意力计算逻辑修正** - 从错误的 `sum(dim=-2)` 改为正确的最后token注意力
- ✅ **权重归一化优化** - 移除不当的min-max归一化，保持概率分布性质
- ✅ **Web交互界面** - 实现现代化的实时可视化Web应用
- ✅ **中文支持优化** - 正确处理中文token的显示和编码

### 技术债务和改进空间
- [ ] 支持更多模型架构

## 🔮 未来功能

### 闭源模型支持
正在探索通过**输入重要性分析**支持闭源API：
```python
def analyze_input_importance_via_api(text, api_client):
    """通过遮蔽token分析重要性（伪注意力）"""
    # 1. 获取完整输出
    # 2. 逐个遮蔽token，观察输出变化
    # 3. 计算重要性分数
```

## 📄 许可证

MIT License - 自由使用和修改

## 🤝 贡献

欢迎提交Issue和Pull Request！

### 开发指南
1. Fork本项目
2. 创建feature分支
3. 提交变更
4. 发起Pull Request

## 📞 联系

- 项目维护者：Marss
- 联系邮箱：Seraphim999@163.com
- 技术问题：请提交GitHub Issue
- 功能建议：请提交GitHub Issue

---

**⭐ 如果这个工具对你有帮助，请给个Star支持一下！**