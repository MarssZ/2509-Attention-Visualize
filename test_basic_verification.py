def visualize_attention(text="Hello world, I am a student."):
    """
    完整的注意力可视化流程：模型推理 -> 权重聚合 -> HTML生成
    """
    from modelscope import AutoModel, AutoTokenizer
    import torch
    
    # 加载模型
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 注意力提取
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    
    # 权重聚合
    attention = outputs.attentions[-1]  # 最后一层
    averaged = attention.mean(dim=1)    # 平均多头
    weights = averaged.sum(dim=-2)      # 列求和
    weights = weights.squeeze().tolist()
    
    # 获取tokens (修复中文显示问题)
    input_ids = inputs["input_ids"][0]
    tokens = []
    
    # 逐个处理token，确保中文正确显示
    for i, token_id in enumerate(input_ids):
        # 获取单个token对应的文本
        single_token_text = tokenizer.decode([token_id])
        tokens.append(single_token_text)
    
    # 调试信息已去除，避免编码问题
    
    # 归一化权重到[0,1]
    min_w, max_w = min(weights), max(weights)
    normalized_weights = [(w - min_w) / (max_w - min_w) for w in weights]
    
    # 生成HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Attention Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .token {{ padding: 2px 4px; margin: 1px; border-radius: 3px; }}
        .info {{ margin: 20px 0; color: #666; }}
    </style>
</head>
<body>
    <h1>注意力权重可视化</h1>
    <div class="info">输入文本: "{text}"</div>
    <div class="visualization">
"""
    
    # 为每个token生成高亮HTML
    for token, weight in zip(tokens, normalized_weights):
        # 使用红色深浅表示权重强度
        intensity = int(255 * (1 - weight))  # 权重越高，红色越深
        bg_color = f"rgb({255}, {intensity}, {intensity})"
        html_content += f'<span class="token" style="background-color: {bg_color}" title="权重: {weight:.3f}">{token}</span>'
    
    html_content += """
    </div>
    <div class="info">鼠标悬停查看具体权重值</div>
</body>
</html>
"""
    
    # 保存HTML文件
    output_path = "attention_viz.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"权重形状: torch.Size([1, {len(tokens)}])")
    print(f"生成HTML文件: {output_path}")
    return output_path

if __name__ == "__main__":
    print("开始注意力可视化...")
    html_file = visualize_attention()
    print(f"完成！请打开 {html_file} 查看可视化结果")