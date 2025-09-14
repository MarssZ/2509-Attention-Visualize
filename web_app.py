"""
交互式注意力可视化Web应用
基于现有的 visualize_attention 逻辑
"""

from flask import Flask, request, jsonify, render_template_string
import json

# 导入现有的可视化逻辑
from test_basic_verification import visualize_attention as _visualize_attention

app = Flask(__name__)

def get_attention_visualization_data(text):
    """
    提取现有逻辑的核心部分，返回可视化数据和tokenizer信息
    """
    from modelscope import AutoModel, AutoTokenizer
    import torch
    
    # 加载模型（与原代码相同）
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
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
    
    # 权重聚合（与原代码相同）
    attention = outputs.attentions[-1]
    averaged = attention.mean(dim=1)
    weights = averaged.sum(dim=-2)
    weights = weights.squeeze().tolist()
    
    # 获取可读tokens（与原代码相同）
    input_ids = inputs["input_ids"][0]
    tokens = []
    for token_id in input_ids:
        single_token_text = tokenizer.decode([token_id])
        tokens.append(single_token_text)
    
    # 归一化权重（与原代码相同）
    min_w, max_w = min(weights), max(weights)
    normalized_weights = [(w - min_w) / (max_w - min_w) for w in weights]
    
    return tokens, normalized_weights, tokenizer_info

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
                • 鼠标悬停查看具体权重值<br>
                • 红色越深表示注意力权重越高<br>
                • 分词边界可能不符合人类直觉
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
                    注：方括号内数字为token在序列中的位置，引号内为实际的token文本
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
                    // 显示可视化结果
                    visualization.innerHTML = data.html;
                    
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
        
        # 调用现有的可视化逻辑
        tokens, normalized_weights, tokenizer_info = get_attention_visualization_data(text)
        
        # 生成HTML可视化片段
        html_parts = []
        for token, weight in zip(tokens, normalized_weights):
            intensity = int(255 * (1 - weight))
            bg_color = f"rgb({255}, {intensity}, {intensity})"
            html_parts.append(
                f'<span class="token" style="background-color: {bg_color}" title="权重: {weight:.3f}">{token}</span>'
            )
        
        visualization_html = ''.join(html_parts)
        
        # 生成分词详情HTML
        token_details_html = []
        for i, (token, weight) in enumerate(zip(tokens, normalized_weights)):
            escaped_token = token.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
            token_details_html.append(
                f'<span style="margin: 2px; padding: 2px 6px; background: #e9ecef; border-radius: 3px; font-family: monospace;">[{i}] "{escaped_token}" (权重: {weight:.3f})</span> '
            )
        
        return jsonify({
            'success': True,
            'html': visualization_html,
            'token_count': len(tokens),
            'tokenizer_info': tokenizer_info,
            'token_details_html': ''.join(token_details_html)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("启动交互式注意力可视化服务...")
    print("访问地址: http://localhost:5000")
    app.run(debug=True, host='localhost', port=5000)