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
    
    # 获取tokenizer信息用于显示
    tokenizer_info = {
        'model_name': model_name,
        'tokenizer_type': type(tokenizer).__name__,
        'vocab_size': tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 'Unknown',
        'special_tokens_count': len(tokenizer.special_tokens_map) if hasattr(tokenizer, 'special_tokens_map') else 0
    }
    
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
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .token {{ padding: 2px 4px; margin: 1px; border-radius: 3px; }}
        .info {{ margin: 20px 0; color: #666; }}
        .tokenizer-info {{ 
            background: #f8f9fa; 
            border: 1px solid #dee2e6; 
            border-radius: 8px; 
            padding: 15px; 
            margin: 20px 0; 
            font-size: 14px;
        }}
        .tokenizer-info h3 {{ margin-top: 0; color: #495057; }}
        .warning {{ 
            background: #fff3cd; 
            border: 1px solid #ffeaa7; 
            border-radius: 6px; 
            padding: 12px; 
            margin: 15px 0; 
            color: #856404;
        }}
        .tech-details {{ color: #6c757d; font-family: monospace; }}
    </style>
</head>
<body>
    <h1>注意力权重可视化</h1>
    
    <div class="tokenizer-info">
        <h3>🔧 Tokenizer 技术信息</h3>
        <div class="tech-details">
            <strong>模型:</strong> {tokenizer_info['model_name']}<br>
            <strong>Tokenizer类型:</strong> {tokenizer_info['tokenizer_type']}<br>
            <strong>词汇表大小:</strong> {tokenizer_info['vocab_size']:,} tokens<br>
            <strong>特殊标记数量:</strong> {tokenizer_info['special_tokens_count']}
        </div>
    </div>
    
    <div class="warning">
        <strong>⚠️ 重要提醒:</strong><br>
        • 不同模型的tokenizer会产生不同的分词结果<br>
        • Qwen系列对中文分词友好，但切换模型会改变可视化结果<br>
        • 注意力权重反映的是token级别的关系，不是词级别<br>
        • 同一句话用不同tokenizer可能产生完全不同的权重分布
    </div>
    
    <div class="info">
        <strong>输入文本:</strong> "{text}"<br>
        <strong>Token数量:</strong> {len(tokens)} 个
    </div>
    
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
    
    <div class="info">
        <strong>使用说明:</strong><br>
        • 鼠标悬停查看具体权重值<br>
        • 红色越深表示注意力权重越高<br>
        • 分词边界可能不符合人类直觉
    </div>
    
    <div class="tokenizer-info">
        <h3>📝 分词结果详情</h3>
        <div style="background: white; padding: 10px; border-radius: 4px; border: 1px solid #dee2e6;">
"""
    
    # 添加分词详情
    for i, (token, weight) in enumerate(zip(tokens, normalized_weights)):
        escaped_token = token.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
        html_content += f'<span style="margin: 2px; padding: 2px 6px; background: #e9ecef; border-radius: 3px; font-family: monospace;">[{i}] "{escaped_token}" (权重: {weight:.3f})</span> '
    
    html_content += """
        </div>
        <div style="margin-top: 10px; font-size: 12px; color: #6c757d;">
            注：方括号内数字为token在序列中的位置，引号内为实际的token文本
        </div>
    </div>
    
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