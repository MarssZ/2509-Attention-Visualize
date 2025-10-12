"""
交互式注意力可视化Web应用
基于现有的 visualize_attention 逻辑
"""

from flask import Flask, request, jsonify

app = Flask(__name__)

def get_attention_visualization_data(text):
    """
    提取现有逻辑的核心部分，返回可视化数据和tokenizer信息
    """
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    import torch

    # 加载模型（改用CausalLM以支持预测）
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        attn_implementation="eager"  # 强制使用eager注意力以支持output_attentions
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 获取tokenizer技术信息（新增）
    tokenizer_info = {
        'model_name': model_name,
        'tokenizer_type': type(tokenizer).__name__,
        'vocab_size': tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 'Unknown',
        'special_tokens_count': len(tokenizer.special_tokens_map) if hasattr(tokenizer, 'special_tokens_map') else 0
    }
    
    # 注意力提取（与原代码相同）
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    
    # 正确的注意力权重提取：最后一个token对所有位置的注意力分布
    attention = outputs.attentions[-1]  # [batch, heads, seq_len, seq_len]
    averaged = attention.mean(dim=1)    # [batch, seq_len, seq_len] - 平均所有注意力头

    # 取最后一个token对所有位置的注意力（这才是正确的注意力可视化）
    last_token_attention = averaged[0, -1, :]  # [seq_len] - 最后位置的注意力分布
    weights = last_token_attention.detach().numpy().tolist()
    
    # 获取可读tokens和token_ids
    input_ids = inputs["input_ids"][0]
    tokens = []
    token_ids = input_ids.tolist()  # 获取token_ids列表
    for token_id in input_ids:
        single_token_text = tokenizer.decode([token_id])
        tokens.append(single_token_text)

    # 注意力权重已经是softmax的结果（和为1），直接使用原始权重进行可视化
    # 保持概率分布的真实性质，不做min-max归一化

    # 新增：获取下一token预测（Top-K候选）
    logits = outputs.logits[0, -1, :]  # 最后位置的预测分数 [vocab_size]
    probs = torch.softmax(logits, dim=-1)  # 转换为概率分布

    # 获取Top-10候选token
    top_k = 10
    top_probs, top_token_ids = torch.topk(probs, top_k)

    # 构建候选列表
    candidates = []
    for i in range(top_k):
        token_id = top_token_ids[i].item()
        prob = top_probs[i].item()
        token_text = tokenizer.decode([token_id])
        candidates.append({
            'token': token_text,
            'probability': prob,
            'token_id': token_id,
            'rank': i + 1
        })

    # 预测信息（保持向后兼容，主预测是第一个）
    prediction_info = {
        'token': candidates[0]['token'],
        'probability': candidates[0]['probability'],
        'token_id': candidates[0]['token_id'],
        'top_candidates': candidates  # 新增Top-K候选列表
    }

    return tokens, weights, tokenizer_info, token_ids, prediction_info

@app.route('/')
def index():
    """主页面 - 交互式界面"""
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>交互式注意力可视化</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .input-section { margin-bottom: 30px; }
        .input-section label { font-weight: bold; display: block; margin-bottom: 10px; }
        .input-section textarea { width: 100%; height: 80px; padding: 10px; border: 2px solid #ddd; border-radius: 5px; font-size: 16px; resize: vertical; }
        .input-section button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; margin-top: 10px; }
        .input-section button:hover { background: #0056b3; }
        .input-section button:disabled { background: #ccc; cursor: not-allowed; }
        .result-section { margin-top: 30px; }
        .token { padding: 3px 6px; margin: 2px; border-radius: 4px; display: inline-block; }
        .info { margin: 20px 0; color: #666; }
        .loading { color: #007bff; font-style: italic; }
        .error { color: #dc3545; background: #f8d7da; padding: 10px; border-radius: 5px; }
        .info-row {
            display: flex;
            gap: 15px;
            margin: 15px 0;
        }
        .info-box {
            flex: 1;
            background: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 12px;
            font-size: 13px;
        }
        .info-box h3 { 
            margin: 0 0 8px 0; 
            color: #666; 
            font-size: 14px;
        }
        .tech-details { color: #555; font-family: monospace; }

        /* Tooltip样式 */
        .prediction-token {
            position: relative;
            cursor: pointer;
        }
        .tooltip {
            visibility: hidden;
            width: 300px;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 12px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .tooltip::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }
        .prediction-token:hover .tooltip {
            visibility: visible;
            opacity: 1;
        }
        .candidate-item {
            margin: 3px 0;
            padding: 2px 0;
            border-bottom: 1px solid #555;
        }
        .candidate-item:last-child {
            border-bottom: none;
        }
        .candidate-rank {
            color: #87CEEB;
            font-weight: bold;
            width: 20px;
            display: inline-block;
        }
        .candidate-token {
            color: #fff;
            font-weight: bold;
            margin: 0 8px;
        }
        .candidate-prob {
            color: #ccc;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Prompts-注意力可视化 by:Marss</h1>
        
        <div class="input-section">
            <label for="textInput">输入您的文本:</label>
            <textarea id="textInput" placeholder="请输入要分析的文本，例如：我爱中国">我爱中国</textarea>
            <button onclick="visualizeAttention()" id="visualizeBtn">🔍 分析注意力</button>
        </div>
        
        <div class="result-section" id="resultSection" style="display: none;">
            <h3>注意力权重可视化结果</h3>
            

            
            <div class="info" id="inputInfo"></div>
            <div id="visualization"></div>
            
            <div class="info">
                <strong>使用说明:</strong><br>
                • <span style="color: #d32f2f;">红色权重</span> 显示模型对历史信息的关注度<br>
                • <span style="color: #1976d2;">蓝色预测</span> 是基于红色权重计算的下一个token<br>
                • 红色越深 → 对预测蓝色token的贡献越大<br>
                • 鼠标悬停红色token查看具体权重值<br>
                • <strong>鼠标悬停蓝色预测token查看Top-10候选及概率</strong>
            </div>
            
            <div class="info-row" id="infoRow" style="display: none;">
                <div class="info-box" id="tokenizerInfo">
                    <h3>🔧 Tokenizer 技术信息</h3>
                    <div class="tech-details" id="techDetails">
                        <!-- 技术信息将通过JavaScript动态填充 -->
                    </div>
                </div>
                
                <div class="info-box" id="warningInfo">
                    <h3>⚠️ 重要提醒</h3>
                    <div style="font-size: 12px; line-height: 1.4;">
                        • 不同模型的tokenizer会产生不同的分词结果<br>
                        • Qwen系列对中文分词友好，但切换模型会改变可视化结果<br>
                        • 注意力权重反映的是token级别的关系，不是词级别<br>
                        • 同一句话用不同tokenizer可能产生完全不同的权重分布
                    </div>
                </div>
            </div>

            <div class="info-box" id="tokenDetails" style="display: none; margin: 15px 0;">
                <h3>📝 分词结果详情</h3>
                <div style="background: white; padding: 8px; border-radius: 4px; border: 1px solid #ddd;" id="tokenDetailsContent">
                    <!-- 分词详情将通过JavaScript动态填充 -->
                </div>
                <div style="margin-top: 8px; font-size: 11px; color: #666;">
                    注：[序号] "token文本" (ID: token_id) | 进度条长度反映权重大小 | 权重总和应约等于1.0
                </div>
            </div>
        </div>
    </div>

    <script>
        async function visualizeAttention() {
            const textInput = document.getElementById('textInput');
            const visualizeBtn = document.getElementById('visualizeBtn');
            const resultSection = document.getElementById('resultSection');
            const inputInfo = document.getElementById('inputInfo');
            const visualization = document.getElementById('visualization');
            
            const text = textInput.value.trim();
            if (!text) {
                alert('请输入文本');
                return;
            }
            
            // 显示加载状态
            visualizeBtn.disabled = true;
            visualizeBtn.innerHTML = '🔄 分析中...';
            resultSection.style.display = 'block';
            
            // 重置所有信息面板
            document.getElementById('infoRow').style.display = 'none';
            document.getElementById('tokenDetails').style.display = 'none';
            
            inputInfo.innerHTML = `输入文本: "${text}"`;
            visualization.innerHTML = '<div class="loading">正在加载模型并分析注意力权重，请稍候...</div>';
            
            try {
                const response = await fetch('/visualize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // 显示可视化结果 + 预测token
                    let visualizationHtml = data.html;

                    // 添加预测token显示（蓝色背景 + tooltip）
                    if (data.prediction) {
                        const predToken = data.prediction.token;
                        const predProb = (data.prediction.probability * 100).toFixed(1);

                        // 构建Top-K候选tooltip内容
                        let tooltipContent = '<div><strong>Top-10 候选预测:</strong></div>';
                        if (data.prediction.top_candidates) {
                            data.prediction.top_candidates.forEach(candidate => {
                                const prob = (candidate.probability * 100).toFixed(1);
                                const escapedToken = candidate.token.replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/&/g, '&amp;');
                                tooltipContent += `
                                    <div class="candidate-item">
                                        <span class="candidate-rank">${candidate.rank}.</span>
                                        <span class="candidate-token">"${escapedToken}"</span>
                                        <span class="candidate-prob">${prob}%</span>
                                    </div>
                                `;
                            });
                        }

                        visualizationHtml += `
                            <span class="token prediction-token" style="background-color: #87CEEB; border: 2px solid #4682B4; margin-left: 5px;">
                                ${predToken}(${predProb}%)
                                <div class="tooltip">${tooltipContent}</div>
                            </span>
                        `;
                    }

                    visualization.innerHTML = visualizationHtml;
                    
                    // 显示tokenizer技术信息和警告信息
                    if (data.tokenizer_info) {
                        const techDetails = document.getElementById('techDetails');
                        techDetails.innerHTML = `
                            <strong>模型:</strong> ${data.tokenizer_info.model_name}<br>
                            <strong>Tokenizer类型:</strong> ${data.tokenizer_info.tokenizer_type}<br>
                            <strong>词汇表大小:</strong> ${data.tokenizer_info.vocab_size.toLocaleString()}<br>
                            <strong>特殊标记:</strong> ${data.tokenizer_info.special_tokens_count}
                        `;
                        // 显示整个信息行
                        document.getElementById('infoRow').style.display = 'flex';
                    }
                    
                    // 更新输入信息
                    inputInfo.innerHTML = `
                        <strong>输入文本:</strong> "${text}"<br>
                        <strong>Token数量:</strong> ${data.token_count} 个
                    `;
                    
                    // 显示分词结果详情
                    const tokenDetails = document.getElementById('tokenDetails');
                    const tokenDetailsContent = document.getElementById('tokenDetailsContent');
                    if (data.token_details_html) {
                        tokenDetailsContent.innerHTML = data.token_details_html;
                        tokenDetails.style.display = 'block';
                    }
                    
                } else {
                    visualization.innerHTML = `<div class="error">错误: ${data.error}</div>`;
                }
            } catch (error) {
                visualization.innerHTML = `<div class="error">请求失败: ${error.message}</div>`;
            }
            
            // 恢复按钮状态
            visualizeBtn.disabled = false;
            visualizeBtn.innerHTML = '🔍 分析注意力';
        }
        
        // 支持Enter键提交
        document.getElementById('textInput').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                visualizeAttention();
            }
        });
    </script>
</body>
</html>
    """
    return html_template

@app.route('/visualize', methods=['POST'])
def visualize():
    """处理可视化请求的API"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'success': False, 'error': '文本不能为空'})
        
        # 调用现有的可视化逻辑（新增预测信息）
        tokens, normalized_weights, tokenizer_info, token_ids, prediction_info = get_attention_visualization_data(text)
        
        # 生成HTML可视化片段
        html_parts = []
        for token, weight in zip(tokens, normalized_weights):
            intensity = int(255 * (1 - weight))
            bg_color = f"rgb({255}, {intensity}, {intensity})"
            html_parts.append(
                f'<span class="token" style="background-color: {bg_color}" title="权重: {weight:.3f}">{token}</span>'
            )
        
        visualization_html = ''.join(html_parts)
        
        # 生成分词详情HTML - 每个token一行，带进度条和token_id
        token_details_html = []
        weights_sum = sum(normalized_weights)
        max_weight = max(normalized_weights) if normalized_weights else 1

        for i, (token, weight, token_id) in enumerate(zip(tokens, normalized_weights, token_ids)):
            escaped_token = token.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
            # 计算进度条长度 (最大15个字符)
            bar_length = int((weight / max_weight) * 15) if max_weight > 0 else 0
            progress_bar = '█' * bar_length + '░' * (15 - bar_length)

            token_details_html.append(f'''
                <div style="font-family: monospace; margin: 3px 0; padding: 5px; background: #f8f9fa; border-radius: 4px; border-left: 4px solid #007bff;">
                    <span style="color: #666;">[{i}]</span>
                    <span style="font-weight: bold; color: #333;">"{escaped_token}"</span>
                    <span style="color: #666;">(ID: {token_id})</span><br>
                    <span style="color: #666; font-size: 12px;">权重: {weight:.3f}</span>
                    <span style="color: #007bff; margin-left: 10px;">{progress_bar}</span>
                </div>
            ''')

        # 添加权重总和验证
        validation_icon = "✓" if abs(weights_sum - 1.0) < 0.001 else "⚠"
        validation_color = "#28a745" if abs(weights_sum - 1.0) < 0.001 else "#dc3545"

        token_details_html.append(f'''
            <div style="margin-top: 15px; padding: 10px; background: #e9ecef; border-radius: 4px; font-family: monospace;">
                <strong>权重总和: {weights_sum:.3f}</strong>
                <span style="color: {validation_color}; font-size: 18px; margin-left: 8px;">{validation_icon}</span>
            </div>
        ''')
        
        return jsonify({
            'success': True,
            'html': visualization_html,
            'token_count': len(tokens),
            'tokenizer_info': tokenizer_info,
            'token_details_html': ''.join(token_details_html),
            'prediction': prediction_info  # 新增预测信息
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("启动交互式注意力可视化服务...")
    print("访问地址: http://localhost:5000")
    app.run(debug=True, host='localhost', port=5000)