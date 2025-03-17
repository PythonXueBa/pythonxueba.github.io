# -*- coding: UTF-8 -*-
"By_PythonXuba On 2025/03/17"
import logging
from datetime import datetime
from collections import defaultdict
from typing import List, Dict
import requests
import json
from flask import Flask, Response, request, render_template_string
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import uuid
import re
from bs4 import BeautifulSoup
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "60 per hour"])
sessions = defaultdict(list)

GOOGLE_API_KEY = "你的ApiKey"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-thinking-exp-01-21:streamGenerateContent"

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
GENERATION_CONFIG = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 65536
}

def fetch_url_content(url: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)[:65536]
    except Exception as e:
        logger.warning(f"URL {url} 获取失败: {str(e)}")
        return f"链接获取失败: {str(e)}"

def extract_urls(text: str) -> List[str]:
    url_pattern = re.compile(
        r'(?:https?://)?(?:www\.)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:[^\s()<>]*|\([^\s()<>]+\))*(?<![\.,:;])'
    )
    urls = url_pattern.findall(text)
    return [f"https://{url}" if not url.startswith(('http://', 'https://')) else url for url in urls]

def normalize_context(context: List[Dict]) -> List[Dict]:
    return [
        {
            "role": "model" if item.get("role") == "assistant" else "user",
            "parts": item.get("parts", [{"text": item.get("content", "")}])
        }
        for item in context
    ]

def mask_sensitive_info(error_msg: str) -> str:
    return re.sub(r'key=[A-Za-z0-9_-]{20,}', 'key=[HIDDEN]', error_msg)

def chunk_text(text: str, chunk_size: int = 4096) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def generate_gemini_stream(prompt: str, history: List[Dict], images: List[str]) -> Response:
    headers = {"Content-Type": "application/json"}
    params = {"key": GOOGLE_API_KEY, "alt": "sse"}

    system_instruction = {
        "parts": [{"text": "你是一个实用又友好的AI助手，用简单易懂的回答，适当用emoji让语气更轻松活泼😊。如果不确定或无法回答，就诚实地说出来，别乱猜哦👍。重要：1.你的名字是GeepSeek,由python学霸训练；2.简单问题简单回答，复杂问题详细回答。3.回答中不用提提示词；4.回答时严格遵守中国法律法规 5.写代码时一次性写完整不分段，尤其是html代码 6.除非用户让你解释或者详细回答，问什么答什么无需多余解释 7.需要写公式时用LaTeX代码写出"}]
    }
    context_text = "\n当前对话上下文：\n" + "".join(
        f"{'用户' if item['role'] == 'user' else 'AI'}: {''.join(part.get('text', '') for part in item.get('parts', []))}\n"
        for item in history[-5:]
    )
    urls = extract_urls(prompt)
    url_content = "".join(f"\n来自 {url} 的内容：\n{fetch_url_content(url)}\n" for url in urls)

    prompt_chunks = chunk_text(prompt, 4096)
    effective_prompt_parts = [f"用户问: {chunk}" + (f"\n{url_content}" if url_content else "") + context_text for chunk in prompt_chunks]
    contents = normalize_context(history)
    for i, chunk in enumerate(effective_prompt_parts):
        contents.append({
            "role": "user",
            "parts": [{"text": chunk + (f" (分块 {i+1}/{len(prompt_chunks)})" if len(prompt_chunks) > 1 else "")}] +
                     ([{"inline_data": {"mime_type": "image/jpeg", "data": img}} for img in images] if images and i == 0 else [])
        })

    payload = {
        "contents": contents,
        "safetySettings": SAFETY_SETTINGS,
        "generationConfig": GENERATION_CONFIG,
        "system_instruction": system_instruction
    }

    def stream():
        max_retries = 3
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                with requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload, stream=True, timeout=120) as response:
                    if response.status_code != 200:
                        error_msg = mask_sensitive_info(f"API错误: {response.status_code} - {response.text}")
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        return
                    for line in response.iter_lines():
                        if line and line.startswith(b"data: "):
                            sse_data = line[6:].decode("utf-8", errors='ignore')
                            try:
                                json_data = json.loads(sse_data)
                                text_content = "".join(part.get("text", "") for candidate in json_data.get("candidates", []) for part in candidate.get("content", {}).get("parts", []))
                                if text_content:
                                    yield f"data: {json.dumps({'text': text_content[:65536] + ('... [truncated]' if len(text_content) > 65536 else '')})}\n\n"
                            except json.JSONDecodeError:
                                logger.error(f"响应解析失败: {sse_data}")
                                yield f"data: {json.dumps({'error': '响应解析失败，请稍后再试'})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    return
            except requests.exceptions.RequestException as e:
                error_msg = mask_sensitive_info(str(e))
                if attempt < max_retries - 1:
                    yield f"data: {json.dumps({'text': f'连接失败，正在重试 ({attempt + 1}/{max_retries})...请稍等'})}\n\n"
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    yield f"data: {json.dumps({'error': f'服务器响应失败: {error_msg}，请检查网络或稍后再试'})}\n\n"
                    return

    return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        context = data.get('context', [])
        images = data.get('images', [])

        sessions[session_id].append({'role': 'user', 'parts': [{"text": message}], 'timestamp': datetime.now().isoformat(), 'images': images})
        return generate_gemini_stream(message, context, images)
    except Exception as e:
        logger.error(f"请求处理失败: {str(e)}", exc_info=True)
        return Response(
            f"data: {json.dumps({'error': f'服务器内部错误: {str(e)}，请稍后再试'})}",
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )

@app.route('/')
def index():
    try:
        return render_template_string(r'''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>GeepSeek-智能AI</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/marked@4.3.0/lib/marked.umd.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
    <style>
        :root {
            --primary: #000000;
            --secondary: #666666;
            --bg: #ffffff;
            --text: #000000;
            --border: #cccccc;
            --shadow: rgba(0, 0, 0, 0.1);
            --hover: #f0f0f0;
            --code-bg: #1e1e1e;
            --code-text: #d4d4d4;
            --error: #ff4444;
            --button-bg: #e0e0e0;
            --button-hover: #cccccc;
            --ai-bg: #e0e0e0;
            --user-bg: #f1f1f1;
            --markdown-bg: #f9f9f9;
            --blockquote-border: #3498db;
            --table-border: #ddd;
        }
        [data-theme="dark"] {
            --primary: dimgray;
            --secondary: #999999;
            --bg: #1a1a1a;
            --text: #e0e0e0;
            --border: #404040;
            --shadow: rgba(0, 0, 0, 0.3);
            --hover: #2a2a2a;
            --code-bg: #2d2d2d;
            --code-text: #e0e0e0;
            --error: #ff6666;
            --button-bg: #404040;
            --button-hover: #505050;
            --ai-bg: #2a2a2a;
            --user-bg: #252525;
            --markdown-bg: #222222;
            --blockquote-border: #2980b9;
            --table-border: #505050;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            font-size: 18px;
            min-height: 100vh;
            overflow-x: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            transition: background 0.3s, color 0.3s;
        }
        .container {
            display: flex;
            height: 100vh;
            width: 100%;
            max-width: 1400px;
            border-radius: 12px;
            overflow: hidden;
            background: var(--bg);
            box-shadow: 0 4px 12px var(--shadow);
            transition: background 0.3s;
        }
        .sidebar {
            width: 25%;
            max-width: 350px;
            background: var(--user-bg);
            border-right: 1px solid var(--border);
            height: 100vh;
            overflow-y: auto;
            position: fixed;
            left: -350px;
            transition: left 0.3s ease;
            z-index: 1000;
            display: flex;
            flex-direction: column;
        }
        .sidebar.active { left: 0; }
        .theme-toggle {
            margin: 20px 8%;
            padding: 12px;
            background: var(--button-bg);
            color: var(--text);
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.3s;
        }
        .theme-toggle:hover { background: var(--button-hover); }
        .menu-toggle {
            position: fixed;
            top: 10px;
            left: 10px;
            background: none;
            border: none;
            cursor: pointer;
            z-index: 1001;
            padding: 8px;
        }
        .menu-toggle.hidden { display: none; }
        .new-chat-btn {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 8px;
            background: var(--primary);
            color: #fff;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: background 0.3s;
            box-shadow: 0 3px 8px var(--shadow);
            width: 40px;
            height: 40px;
            z-index: 1001;
        }
        .new-chat-btn:hover { background: #333333; }
        .history-search {
            margin: 8% 8% 5%;
            padding: 12px;
            border: 1px solid var(--border);
            border-radius: 8px;
            width: 85%;
            font-size: 16px;
            background: var(--bg);
            box-shadow: inset 0 2px 4px var(--shadow);
            color: var(--text);
        }
        .chat-history {
            flex: 1;
            overflow-y: auto;
            padding: 0 5%;
            scrollbar-width: thin;
            scrollbar-color: var(--secondary) var(--bg);
        }
        .chat-history::-webkit-scrollbar { width: 8px; }
        .chat-history::-webkit-scrollbar-thumb { background: var(--secondary); border-radius: 4px; }
        .history-item {
            padding: 12px;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            transition: background 0.2s;
            border-radius: 8px;
            margin: 5px 0;
            font-size: 16px;
            color: var(--text);
        }
        .history-item:hover { background: var(--hover); }
        .history-item.active { background: var(--ai-bg); }
        .history-item .title { font-weight: 600; font-size: 18px; color: var(--text); }
        .history-item .preview { font-size: 14px; color: var(--secondary); margin-top: 5px; }
        .history-item .time { font-size: 12px; color: var(--secondary); margin-top: 5px; display: block; }
        .clear-history-container {
            position: sticky;
            bottom: 0;
            padding: 10px 8%;
            background: var(--user-bg);
            border-top: 1px solid var(--border);
        }
        .clear-history {
            padding: 12px 20px;
            background: var(--secondary);
            color: #fff;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.3s;
            box-shadow: 0 3px 8px var(--shadow);
            width: 100%;
            font-size: 18px;
            display: flex;
            justify-content: center;
        }
        .clear-history:hover { background: #4d4d4d; }
        .chat-container {
            flex: 3;
            display: flex;
            flex-direction: column;
            padding: 20px;
            background: var(--bg);
            border-radius: 0 12px 12px 0;
            width: 100%;
            height: 100vh;
            position: relative;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 0 10px;
            scrollbar-width: thin;
            scrollbar-color: var(--secondary) var(--bg);
            display: flex;
            flex-direction: column;
            margin-top: 40px;
        }
        .chat-messages::-webkit-scrollbar { width: 8px; }
        .chat-messages::-webkit-scrollbar-thumb { background: var(--secondary); border-radius: 4px; }
        .message {
            display: flex;
            margin: 12px 0;
            max-width: 80%;
            border-radius: 12px;
            animation: fadeIn 0.3s ease;
            box-shadow: 0 2px 6px var(--shadow);
            font-size: 18px;
            overflow-wrap: break-word;
            word-break: break-word;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
        .message.user {
            margin-left: auto;
            background: var(--user-bg);
            color: var(--text);
            border-bottom-right-radius: 4px;
        }
        .message.ai {
            margin-right: auto;
            background: var(--ai-bg);
            color: var(--text);
            border-bottom-left-radius: 4px;
        }
        .message-content {
            line-height: 1.6;
            overflow-wrap: break-word;
            word-break: break-word;
            width: 100%;
            background: var(--markdown-bg);
            padding: 10px;
            padding-left: 20px;
            border-radius: 8px;
            position: relative;
        }
        .message-content p { margin: 10px 0; }
        .message-content a { color: #2980b9; text-decoration: none; transition: color 0.3s; }
        .message-content a:hover { color: #3498db; text-decoration: underline; }
        .message-content ul, .message-content ol { padding-left: 10px; margin: 10px 0; }
        .message-content li { margin: 5px 0; }
        .message-content blockquote {
            margin: 10px 0;
            padding: 10px 15px;
            background: var(--user-bg);
            border-left: 4px solid var(--blockquote-border);
            color: var(--text);
            font-style: italic;
            border-radius: 4px;
        }
        .message-content h1, .message-content h2, .message-content h3,
        .message-content h4, .message-content h5, .message-content h6 {
            margin: 15px 0 10px;
            color: var(--primary);
            font-weight: 600;
        }
        .message-content h1 { font-size: 1.5em; }
        .message-content h2 { font-size: 1.3em; }
        .message-content h3 { font-size: 1.2em; }
        .message-content h4 { font-size: 1.1em; }
        .message-content h5, .message-content h6 { font-size: 1em; }
        .message-content table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            background: var(--bg);
            box-shadow: 0 1px 3px var(--shadow);
        }
        .message-content th, .message-content td {
            padding: 8px 12px;
            border: 1px solid var(--table-border);
            text-align: left;
        }
        .message-content th {
            background: var(--ai-bg);
            font-weight: 600;
            color: var(--primary);
        }
        .message-content pre {
            background: var(--code-bg);
            color: var(--code-text);
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre-wrap;
            width: auto;
            max-width: 99%;
            margin: 10px 0;
            font-size: 16px;
            line-height: 1.5;
            position: relative;
            box-shadow: 0 2px 5px var(--shadow);
        }
        .message-content code {
            background: var(--code-bg);
            color: var(--code-text);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 16px;
        }
        .message-content pre code {
            background: none;
            padding: 0;
            display: block;
            width: 100%;
        }
        .message-content .katex {
        display: inline;
            vertical-align: middle;
            max-width: 100%;
            overflow-x: auto;
            white-space: nowrap;
            padding: 5px;
            font-size: 1.2em;
            color: var(--text);
        }
        .message-content .katex-display {
            display: block;
            max-width: 100%;
            overflow-x: auto;
            padding: 10px;
            margin: 10px 0;
            background: var(--user-bg);
            border-radius: 4px;
            box-sizing: content-box;
            color: var(--text);
        }
        .code-actions {
            position: absolute;
            bottom: 8px;
            right: 8px;
            display: flex;
            flex-wrap: nowrap;
            gap: 6px;
            z-index: 10;
            transition: flex-direction 0.1s;
        }
        .code-actions button {
            background: var(--button-bg);
            color: var(--text);
            border: none;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.3s;
            white-space: nowrap;
        }
        .code-actions button:hover { background: var(--button-hover); }
        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .loading-dots {
            display: flex;
            gap: 8px;
        }
        .loading-dots span {
            width: 12px;
            height: 12px;
            background: var(--primary);
            border-radius: 50%;
            animation: bounce 1.2s infinite ease-in-out both;
        }
        .loading-dots span:nth-child(1) { animation-delay: -0.3s; }
        .loading-dots span:nth-child(2) { animation-delay: -0.15s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 15px;
            border-top: 1px solid var(--border);
            background: var(--bg);
            z-index: 100;
            width: 100%;
            max-width: 1400px;
            margin: 0 auto;
        }
        .input-wrapper {
            display: flex;
            gap: 12px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px;
            align-items: center;
            box-shadow: 0 2px 5px var(--shadow);
        }
        #message-input {
            flex: 1;
            border: none;
            outline: none;
            padding: 5px;
            resize: none;
            max-height: 20vh;
            font-size: 18px;
            background: transparent;
            color: var(--text);
            overflow-y: auto;
            -webkit-user-select: text;
            -moz-user-select: text;
            -ms-user-select: text;
            user-select: text;
            pointer-events: auto;
        }
        #send-button, #stop-button, .upload-button {
            background: none;
            border: none;
            cursor: pointer;
            padding: 8px;
            color: var(--primary);
            transition: color 0.3s;
        }
        #send-button:hover, #stop-button:hover, .upload-button:hover { color: #333333; }
        #stop-button { display: none; }
        .image-preview-container {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 12px;
        }
        .image-preview-item {
            max-width: 100px;
            max-height: 100px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px var(--shadow);
            position: relative;
            transition: transform 0.2s;
        }
        .image-preview-item:hover { transform: scale(1.05); }
        .image-preview-item img { width: 100%; height: 100%; object-fit: cover; }
        .remove-image {
            position: absolute;
            top: 5px;
            right: 5px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            border: none;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            cursor: pointer;
            font-size: 12px;
            line-height: 20px;
            text-align: center;
        }
        .error-message {
            color: var(--error);
            font-size: 16px;
            margin: 10px 0;
            text-align: center;
        }
        @media (max-width: 600px) {
            .container { flex-direction: column; height: auto; }
            .sidebar { width: 80%; max-width: none; height: 100vh; left: -100%; }
            .sidebar.active { left: 0; }
            .chat-container { flex: none; margin: 0; padding: 15px; }
            .chat-messages { font-size: 14px; }
            .input-container { padding: 10px; }
            .input-wrapper { padding: 10px; }
            .message { max-width: 90%; font-size: 16px; }
            .menu-toggle { top: 5px; left: 5px; }
            .new-chat-btn { top: 5px; right: 5px; width: 36px; height: 36px; }
            .message-content pre { font-size: 12px; padding: 12px; max-height: 100%; margin-bottom: 20px; }
            .code-actions { flex-direction: row; gap: 4px; bottom: 2px; right: 2px; }
            .code-actions button { padding: 4px 8px; font-size: 12px; min-width: 50px; }
            #message-input { font-size: 14px; }
            .clear-history { font-size: 16px; padding: 10px 15px; }
            .message-content h1 { font-size: 1.3em; }
            .message-content h2 { font-size: 1.2em; }
            .message-content h3 { font-size: 1.1em; }
            .message-content .katex { font-size: 0.85em; padding: 3px; }
            .message-content .katex-display { padding: 8px; margin: 8px 0; }
        }
    </style>
</head>
<body>
    <button class="menu-toggle">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M3 12h18M3 6h18M3 18h18" stroke="var(--primary)" stroke-width="2" stroke-linecap="round"/>
        </svg>
    </button>
    <button class="new-chat-btn">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M8 3.33337V12.6667M3.33333 8H12.6667" stroke="white" stroke-width="2" stroke-linecap="round"/>
        </svg>
    </button>
    <div class="container">
        <div class="sidebar">
            <button class="theme-toggle" onclick="toggleTheme()">切换夜间模式</button>
            <input type="text" class="history-search" placeholder="搜索历史记录...">
            <div class="chat-history"></div>
            <div class="clear-history-container">
                <button class="clear-history">清空历史</button>
            </div>
        </div>
        <div class="chat-container">
            <div class="chat-messages"></div>
            <div class="input-container">
                <div class="image-preview-container" id="image-preview"></div>
                <div class="input-wrapper">
                    <button class="upload-button" id="upload-button">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                            <path d="M14 10V12H2V10H0V14H16V10H14ZM8 2L12 6H9V10H7V6H4L8 2Z" fill="var(--primary)"/>
                        </svg>
                    </button>
                    <textarea id="message-input" placeholder="输入消息或Ctrl+V粘贴图片..." rows="1"></textarea>
                    <button id="send-button">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                            <path d="M14.6667 1.33337L7.33333 8.66671M14.6667 1.33337L10 14.6667L7.33333 8.66671L1.33333 6.00004L14.6667 1.33337Z" stroke="var(--primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </button>
                    <button id="stop-button">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                            <path d="M4 4H12V12H4V4Z" stroke="var(--primary)" stroke-width="2" stroke-linejoin="round"/>
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    </div>
    <div id="html-preview-modal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.6); z-index: 1001; overflow: auto;">
        <div style="background: var(--bg); margin: 10% auto; padding: 20px; border-radius: 12px; width: 90%; max-width: 800px; position: relative; box-shadow: 0 5px 20px var(--shadow);">
            <button id="close-preview" style="position: absolute; right: 5px; top: 10px; background: none; border: none; cursor: pointer; font-size: 1.5rem; color: var(--text);">×</button>
            <iframe id="html-preview-content" style="width: 100%; height: 500px; border: none;"></iframe>
        </div>
    </div>
    <script>
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            html.setAttribute('data-theme', currentTheme === 'dark' ? 'light' : 'dark');
            localStorage.setItem('theme', html.getAttribute('data-theme'));
            sidebar.classList.toggle('active');
            menuToggle.classList.toggle('hidden');
        }

        function hasMath(content) {
            return /\$[\s\S]+?\$|\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\(.*?\\\)|\\begin\{[a-z]*\}[^]*?\\end\{[a-z]*\}/.test(content);
        }

        function preprocessMath(content) {
            return content
                .replace(/\\\{/g, '{')
                .replace(/\\\}/g, '}')
                .replace(/([^\\])\$/g, '$1 \$')
                .replace(/([^\\])\$\$/g, '$1 \$\$');
        }

        function renderMath(contentDiv) {
            contentDiv.querySelectorAll('.katex').forEach(el => {
                el.outerHTML = el.innerHTML;
            });

            renderMathInElement(contentDiv, {
                delimiters: [
                    { left: "$$", right: "$$", display: true },
                    { left: "\\[", right: "\\]", display: true },
                    { left: "$", right: "$", display: false },
                    { left: "\\(", right: "\\)", display: false },
                    { left: "\\begin{align}", right: "\\end{align}", display: true },
                    { left: "\\begin{equation}", right: "\\end{equation}", display: true }
                ],
                throwOnError: false,
                strict: false,
                trust: true,
                errorColor: '#ff0000',
                maxSize: 50,
                maxExpand: 1000
            });
        }

        function escapeHtml(unsafe) {
            return unsafe
        }

        document.addEventListener('DOMContentLoaded', () => {
            marked.setOptions({
                gfm: true,
                tables: true,
                breaks: true,
                pedantic: false,
                smartLists: true,
                smartypants: true,
                highlight: function(code, lang) {
                    const escapedCode = escapeHtml(code);
                    if (lang && hljs.getLanguage(lang)) {
                        return hljs.highlight(escapedCode, { language: lang }).value;
                    }
                    return hljs.highlightAuto(escapedCode).value;
                }
            });

            const savedTheme = localStorage.getItem('theme');
            if (savedTheme) {
                document.documentElement.setAttribute('data-theme', savedTheme);
            }

            initializeChat();
        });

        const Config = {
            MAX_CONTEXT_LENGTH: 10,
            MAX_MESSAGE_LENGTH: 65536,
            MAX_IMAGE_UPLOADS: 5,
            CHUNK_SIZE: 4096
        };
        let conversations = JSON.parse(localStorage.getItem('conversations')) || [];
        let currentConversationId = localStorage.getItem('currentConversationId') || Date.now().toString();
        let currentContext = [];
        let eventSource = null;
        let uploadedImages = [];
        let isGenerating = false;

        const chatMessages = document.querySelector('.chat-messages');
        const messageInput = document.querySelector('#message-input');
        const sendButton = document.querySelector('#send-button');
        const stopButton = document.querySelector('#stop-button');
        const uploadButton = document.querySelector('#upload-button');
        const imagePreview = document.querySelector('#image-preview');
        const sidebar = document.querySelector('.sidebar');
        const menuToggle = document.querySelector('.menu-toggle');
        const newChatBtn = document.querySelector('.new-chat-btn');
        const clearHistoryBtn = document.querySelector('.clear-history');
        const historySearch = document.querySelector('.history-search');
        const htmlPreviewModal = document.querySelector('#html-preview-modal');
        const htmlPreviewContent = document.querySelector('#html-preview-content');
        const closePreviewBtn = document.querySelector('#close-preview');

        function initializeChat() {
            conversations.forEach(conv => {
                conv.messages.forEach(msg => {
                    if (msg.content && !msg.parts) {
                        msg.parts = [{"text": msg.content}];
                        delete msg.content;
                    }
                    if (msg.role === "assistant") msg.role = "model";
                    if (msg.role === "system") msg.role = "user";
                });
            });
            loadConversation(currentConversationId);
            updateChatHistory();
            setupEventListeners();
            adjustInputHeight();
            adjustContainerHeight();
            setTimeout(() => messageInput.focus(), 100);
        }

        function setupEventListeners() {
            menuToggle.addEventListener('click', () => {
                sidebar.classList.toggle('active');
                menuToggle.classList.toggle('hidden');
            });
            document.addEventListener('click', e => {
                if (!sidebar.contains(e.target) && !menuToggle.contains(e.target) && sidebar.classList.contains('active')) {
                    sidebar.classList.remove('active');
                    menuToggle.classList.remove('hidden');
                }
            });
            sendButton.addEventListener('click', () => {
                debounce(sendMessage, 100)();
                setTimeout(ensureInputFocusable, 100);
            });
            stopButton.addEventListener('click', stopGenerating);
            messageInput.addEventListener('keydown', e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    debounce(sendMessage, 100)();
                }
            });
            messageInput.addEventListener('click', () => messageInput.focus());
            messageInput.addEventListener('input', adjustInputHeight);
            messageInput.addEventListener('paste', handlePaste);
            uploadButton.addEventListener('click', () => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = 'image/*';
                input.multiple = true;
                input.onchange = handleImageUpload;
                input.click();
            });
            newChatBtn.addEventListener('click', startNewChat);
            clearHistoryBtn.addEventListener('click', clearHistory);
            historySearch.addEventListener('input', debounce(e => updateChatHistory(e.target.value.toLowerCase()), 300));
            window.addEventListener('resize', adjustContainerHeight);
            closePreviewBtn.addEventListener('click', () => {
                htmlPreviewModal.style.display = 'none';
            });
        }

        function debounce(func, wait) {
            let timeout;
            return function (...args) {
                clearTimeout(timeout);
                timeout = setTimeout(() => func.apply(this, args), wait);
            };
        }

        function adjustInputHeight() {
            messageInput.style.height = 'auto';
            messageInput.style.height = `${Math.min(messageInput.scrollHeight, window.innerHeight * 0.2)}px`;
        }

        function adjustContainerHeight() {
            const chatContainer = document.querySelector('.chat-container');
            const inputContainer = document.querySelector('.input-container');
            chatContainer.style.paddingBottom = `${inputContainer.offsetHeight + 10}px`;
        }

        function ensureInputFocusable() {
            messageInput.style.pointerEvents = 'auto';
            messageInput.style.userSelect = 'text';
            messageInput.focus();
        }

        function handleImageUpload(e) {
            const remainingSlots = Config.MAX_IMAGE_UPLOADS - uploadedImages.length;
            if (remainingSlots <= 0) {
                addMessage(`最多只能上传 ${Config.MAX_IMAGE_UPLOADS} 张图片哦！`, 'ai');
                return;
            }
            const filesToProcess = Array.from(e.target.files).slice(0, remainingSlots);
            filesToProcess.forEach(file => {
                const reader = new FileReader();
                reader.onload = event => {
                    addImageToPreview(event.target.result.split(',')[1]);
                };
                reader.readAsDataURL(file);
            });
        }

        function handlePaste(e) {
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            const remainingSlots = Config.MAX_IMAGE_UPLOADS - uploadedImages.length;
            if (remainingSlots <= 0) return;
            let imageCount = 0;
            for (const item of items) {
                if (item.type.indexOf('image') !== -1 && imageCount < remainingSlots) {
                    const blob = item.getAsFile();
                    const reader = new FileReader();
                    reader.onload = event => addImageToPreview(event.target.result.split(',')[1]);
                    reader.readAsDataURL(blob);
                    imageCount++;
                }
            }
        }

        function addImageToPreview(base64Data) {
            if (uploadedImages.length >= Config.MAX_IMAGE_UPLOADS) return;
            const imgDiv = document.createElement('div');
            imgDiv.className = 'image-preview-item';
            imgDiv.innerHTML = `<img src="data:image/jpeg;base64,${base64Data}" alt="Uploaded Image">
                                <button class="remove-image" onclick="this.parentElement.remove(); uploadedImages = uploadedImages.filter(img => img !== '${base64Data}');">×</button>`;
            imagePreview.appendChild(imgDiv);
            uploadedImages.push(base64Data);
        }

        function updateChatHistory(searchTerm = '') {
            const chatHistory = document.querySelector('.chat-history');
            chatHistory.innerHTML = '';
            const filteredConversations = conversations
                .sort((a, b) => new Date(b.messages?.slice(-1)[0]?.timestamp || 0) - new Date(a.messages?.slice(-1)[0]?.timestamp || 0))
                .filter(c => !searchTerm || c.messages.some(m => m.parts?.[0]?.text.toLowerCase().includes(searchTerm)));
            const initialLoad = filteredConversations.slice(0, 20);
            initialLoad.forEach(c => renderHistoryItem(c, chatHistory));
            let loadedCount = initialLoad.length;
            chatHistory.addEventListener('scroll', () => {
                if (chatHistory.scrollTop + chatHistory.clientHeight >= chatHistory.scrollHeight - 50 && loadedCount < filteredConversations.length) {
                    const nextBatch = filteredConversations.slice(loadedCount, loadedCount + 10);
                    nextBatch.forEach(c => renderHistoryItem(c, chatHistory));
                    loadedCount += 10;
                }
            });
        }

        function renderHistoryItem(c, chatHistory) {
            if (!c.messages?.length) return;
            const firstUserMessage = c.messages.find(m => m.role === 'user');
            const titleText = firstUserMessage?.parts?.[0]?.text.slice(0, 30) + (firstUserMessage?.parts?.[0]?.text.length > 30 ? '...' : '') || '新对话';
            const div = document.createElement('div');
            div.className = `history-item ${c.id === currentConversationId ? 'active' : ''}`;
            div.innerHTML = `
                <div class="title">${titleText}</div>
                <div class="preview">${c.messages.slice(-2).map(m => `<div>${m.parts?.[0]?.text.slice(0, 50)}${m.parts?.[0]?.text.length > 50 ? '...' : ''}</div>`).join('')}</div>
                <div class="time">${new Date(c.messages.slice(-1)[0].timestamp).toLocaleString()}</div>
            `;
            div.onclick = () => loadConversation(c.id);
            chatHistory.appendChild(div);
        }

        function loadConversation(id) {
            currentConversationId = id;
            localStorage.setItem('currentConversationId', id);
            const conversation = conversations.find(c => c.id === id);
            if (!conversation) return;
            chatMessages.innerHTML = '';
            conversation.messages.forEach(msg => {
                const images = msg.parts.filter(part => part.inline_data).map(part => part.inline_data.data);
                addMessage(msg.parts.find(part => part.text)?.text || '', msg.role === 'user' ? 'user' : 'ai', images);
            });
            currentContext = conversation.messages.map(msg => ({
                role: msg.role,
                parts: msg.parts.map(part => part.inline_data ? {inline_data: part.inline_data} : {text: part.text})
            }));
            updateChatHistory();
            sidebar.classList.remove('active');
            menuToggle.classList.remove('hidden');
        }

        async function addMessage(text, type = 'user', images = null) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            messageDiv.appendChild(contentDiv);
            chatMessages.appendChild(messageDiv);

            let content = text || '（无内容）';
            if (type === 'user') {
                contentDiv.textContent = content;
            } else {
                content = preprocessMath(content);
                const markedContent = await marked.parse(content);
                contentDiv.innerHTML = markedContent || '（AI 未返回内容）';
                addCodeActions(contentDiv);
                hljs.highlightAll();
                if (hasMath(content)) {
                    renderMath(contentDiv);
                }
            }

            if (images?.length) {
                const imgContainer = document.createElement('div');
                imgContainer.className = 'image-preview-container';
                images.forEach(imgData => {
                    const img = document.createElement('div');
                    img.className = 'image-preview-item';
                    img.innerHTML = `<img src="data:image/jpeg;base64,${imgData}" alt="Image">`;
                    imgContainer.appendChild(img);
                });
                messageDiv.appendChild(imgContainer);
            }
            chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
        }

        function addCodeActions(contentDiv) {
            contentDiv.querySelectorAll('pre code').forEach(block => {
                if (block.parentNode.querySelector('.code-actions')) return;
                const pre = block.parentNode;
                const actions = document.createElement('div');
                actions.className = 'code-actions';

                const isHtml = block.textContent.trim().toLowerCase().startsWith('<!doctype html') ||
                             block.textContent.trim().startsWith('<html');
                const buttonCount = isHtml ? 2 : 1;
                const minWidthNeeded = buttonCount * 80;

                actions.innerHTML = `
                    <button onclick="copyCode(this)">复制</button>
                    ${isHtml ? '<button onclick="previewHTML(this)">预览</button>' : ''}
                `;

                pre.style.position = 'relative';
                pre.appendChild(actions);

                function adjustButtonLayout() {
                    const preWidth = pre.offsetWidth;
                    if (preWidth < minWidthNeeded + 20) {
                        actions.style.flexDirection = 'column';
                        actions.style.alignItems = 'flex-end';
                    } else {
                        actions.style.flexDirection = 'row';
                        actions.style.alignItems = 'center';
                    }
                }

                adjustButtonLayout();
                window.addEventListener('resize', adjustButtonLayout);
            });
        }

        function copyCode(button) {
            const codeBlock = button.closest('pre').querySelector('code');
            const code = codeBlock.textContent;

            const tempTextarea = document.createElement('textarea');
            tempTextarea.value = code;
            tempTextarea.style.position = 'fixed';
            tempTextarea.style.opacity = '0';
            document.body.appendChild(tempTextarea);
            tempTextarea.select();

            try {
                if (navigator.clipboard && window.isSecureContext) {
                    navigator.clipboard.writeText(code).then(() => {
                        showCopyFeedback(button, '已复制');
                    }).catch(() => {
                        document.execCommand('copy');
                        showCopyFeedback(button, '已复制');
                    });
                } else {
                    document.execCommand('copy');
                    showCopyFeedback(button, '已复制');
                }
            } catch (err) {
                console.error('复制失败:', err);
                showCopyFeedback(button, '复制失败');
            } finally {
                document.body.removeChild(tempTextarea);
            }
        }

        function showCopyFeedback(button, text) {
            const originalText = button.textContent;
            button.textContent = text;
            button.disabled = true;
            setTimeout(() => {
                button.textContent = originalText;
                button.disabled = false;
            }, 2000);
        }

        function previewHTML(button) {
            const code = button.closest('pre').querySelector('code').textContent;
            htmlPreviewContent.srcdoc = code;
            htmlPreviewModal.style.display = 'block';
        }

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message && !uploadedImages.length || isGenerating) return;

            startGeneratingState();
            if (eventSource) eventSource.close();

            const imagesToSend = [...uploadedImages];
            messageInput.value = '';
            imagePreview.innerHTML = '';
            uploadedImages = [];
            adjustInputHeight();

            const userParts = [{"text": message || "请描述这些图片"}].concat(
                imagesToSend.map(img => ({"inline_data": {"mime_type": "image/jpeg", "data": img}}))
            );
            currentContext.push({ role: 'user', parts: userParts });
            await addMessage(message || '[图片消息]', 'user', imagesToSend);
            const loadingDiv = addLoadingMessage();

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message || "请描述这些图片",
                        session_id: currentConversationId,
                        context: currentContext,
                        images: imagesToSend
                    })
                });

                if (!response.ok) throw new Error(`服务器错误: ${response.status} - ${await response.text()}`);

                let accumulatedText = '';
                const aiMessageDiv = document.createElement('div');
                aiMessageDiv.className = 'message ai';
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                aiMessageDiv.appendChild(contentDiv);
                chatMessages.replaceChild(aiMessageDiv, loadingDiv);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                eventSource = { close: () => { reader.cancel(); endGeneratingState(); } };

                async function processStream() {
                    const { done, value } = await reader.read();
                    if (done) {
                        if (!accumulatedText) accumulatedText = '（AI 未返回内容）';
                        currentContext.push({ role: 'model', parts: [{"text": accumulatedText}] });
                        currentContext = currentContext.slice(-Config.MAX_CONTEXT_LENGTH);
                        saveToHistory(message || "请描述这些图片", accumulatedText, imagesToSend);
                        hljs.highlightAll();
                        if (hasMath(accumulatedText)) renderMath(contentDiv);
                        endGeneratingState();
                        eventSource = null;
                        return;
                    }
                    const chunk = decoder.decode(value, { stream: true });
                    chunk.split('\n\n').forEach(async line => {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.text) {
                                    accumulatedText += data.text;
                                    contentDiv.innerHTML = await marked.parse(preprocessMath(accumulatedText));
                                    addCodeActions(contentDiv);
                                    hljs.highlightAll();
                                    if (hasMath(accumulatedText)) renderMath(contentDiv);
                                    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
                                } else if (data.error) {
                                    contentDiv.innerHTML = `<span class="error-message">抱歉，出了点问题：${escapeHtml(data.error)}</span>`;
                                    endGeneratingState();
                                    eventSource.close();
                                    eventSource = null;
                                } else if (data.done) {
                                    if (!accumulatedText) accumulatedText = '（AI 未返回内容）';
                                    currentContext.push({ role: 'model', parts: [{"text": accumulatedText}] });
                                    saveToHistory(message || "请描述这些图片", accumulatedText, imagesToSend);
                                    if (hasMath(accumulatedText)) renderMath(contentDiv);
                                    endGeneratingState();
                                    eventSource = null;
                                }
                            } catch (e) {
                                console.error("Stream parse error:", e, "Raw data:", line);
                                contentDiv.innerHTML += `<span class="error-message">响应解析错误，请稍后再试</span>`;
                            }
                        }
                    });
                    if (isGenerating) processStream();
                    else {
                        reader.cancel();
                        endGeneratingState();
                        eventSource = null;
                    }
                }
                processStream();
            } catch (error) {
                console.error("Fetch error:", error);
                chatMessages.removeChild(loadingDiv);
                addMessage(`哎呀，出错了：${error.message || '未知错误'}，请稍后再试哦😅`, 'ai');
                saveToHistory(message || "请描述这些图片", `[请求失败: ${error.message}]`, imagesToSend);
                endGeneratingState();
                eventSource = null;
            }
        }

        function stopGenerating() {
            if (eventSource && isGenerating) {
                isGenerating = false;
                stopButton.style.display = 'none';
                sendButton.style.display = 'inline-block';
                eventSource.close();
            }
        }

        function startGeneratingState() {
            isGenerating = true;
            sendButton.style.display = 'none';
            stopButton.style.display = 'inline-block';
        }

        function endGeneratingState() {
            isGenerating = false;
            sendButton.style.display = 'inline-block';
            stopButton.style.display = 'none';
        }

        function addLoadingMessage() {
            const div = document.createElement('div');
            div.className = 'message ai';
            div.innerHTML = `
                <div class="message-content">
                    <div class="loading-container">
                        <div class="loading-dots"><span></span><span></span><span></span></div>
                    </div>
                </div>`;
            chatMessages.appendChild(div);
            chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
            return div;
        }

        function saveToHistory(message, response, images) {
            let conversation = conversations.find(c => c.id === currentConversationId);
            if (!conversation) {
                conversation = { id: currentConversationId, messages: [] };
                conversations.push(conversation);
            }
            const userParts = [{"text": message}].concat(
                images.map(img => ({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img
                    }
                }))
            );
            conversation.messages.push(
                { role: 'user', parts: userParts, timestamp: new Date().toISOString() },
                { role: 'model', parts: [{"text": response}], timestamp: new Date().toISOString() }
            );
            localStorage.setItem('conversations', JSON.stringify(conversations));
            updateChatHistory();
        }

        function startNewChat() {
            if (isGenerating) {
                stopGenerating();
            }
            currentConversationId = Date.now().toString();
            localStorage.setItem('currentConversationId', currentConversationId);
            chatMessages.innerHTML = '';
            imagePreview.innerHTML = '';
            uploadedImages = [];
            currentContext = [];
            conversations.push({ id: currentConversationId, messages: [] });
            localStorage.setItem('conversations', JSON.stringify(conversations));
            updateChatHistory();
            sidebar.classList.remove('active');
            menuToggle.classList.remove('hidden');
        }

        function clearHistory() {
            if (confirm('确定要清空历史记录吗？')) {
                if (isGenerating) stopGenerating();
                conversations = [];
                localStorage.setItem('conversations', JSON.stringify(conversations));
                startNewChat();
            }
        }
    </script>
</body>
</html>
''')
    except Exception as e:
        logger.error(f"渲染首页错误: {str(e)}")
        return "模板加载失败", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)