# -*- coding: utf-8 -*-
"""NSM Middleware Server — Ollama 兼容的中间层服务。

架构：
  用户(自然语言) ──► 本服务
      ├─ 含 DFA 任务？ → 1) 调用 Ollama 解析 NL → JSON spec（few-shot + 排列纠错）
      │                  2) Neural DFA Executor 执行（232K，k≤32，任意长度）
      │                  3) 激活通路写盘（神经记号文件，可审计/重放）
      │                  4) 返回 状态轨迹 + 判定 + 纸带引用
      └─ 普通请求 → 原样代理给 Ollama

API（兼容 Ollama）：
  POST /api/generate   {model, prompt, options}
  POST /api/chat       {model, messages, options}
  GET  /api/tags       列出可用模型（含虚拟模型 "dfa-executor"）
  GET  /tape/<id>      取回某次执行的纸带文件

用法：
  python nsm_server.py            # 启动，默认端口 11435
  # 然后像用 Ollama 一样调用，model="dfa-executor"
"""
import json
import os
import re
import time
import uuid
from typing import Optional

import torch
from flask import Flask, jsonify, request, send_file

from nl_middleware import (
    ollama_complete, parse_spec, permutation_repair,
)
from nsm.dfa_executor import ExecutorConfig, NeuralDFAExecutor, build_executor_batch

app = Flask(__name__)

# ---------------------------------------------------------------- 全局资源
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EXECUTOR = None
TAPE_DIR = os.path.join(os.path.dirname(__file__), "..", "tapes")
os.makedirs(TAPE_DIR, exist_ok=True)

PARSER_MODEL = "dfa-parser"   # 解析用模型（Ollama 注册的 few-shot 版）
OLLAMA_BASE = "http://localhost:11434"


def load_executor():
    global EXECUTOR
    EXECUTOR = NeuralDFAExecutor(ExecutorConfig(d_model=256, max_k=64, max_trans=128))
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "essential",
                             "dfa_executor_k32_final.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "essential",
                                 "dfa_executor_final_final.pt")
    EXECUTOR.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    EXECUTOR.to(DEVICE)
    EXECUTOR.eval()
    print(f"[nsm] executor loaded from {ckpt_path} ({EXECUTOR.count_params()/1e3:.0f}k params)")


# ---------------------------------------------------------------- 执行 + 纸带
def execute_with_tape(spec, word, tape_id: str):
    """Executor 执行并写激活通路纸带。返回 (states, verdict, tape_path)。"""
    dfa = {"k": spec["k"], "m": 2, "start": spec["start"],
           "accepting": sorted(spec["accepting"]), "delta": spec["delta"]}
    batch = [(dfa, tuple(word))]
    inp = build_executor_batch(batch, device=DEVICE)
    nT = inp["trans_tokens"].shape[1]
    q_idx = ((inp["trans_tokens"] - EXECUTOR.TRANSITION_OFFSET) // 2).clamp(0, 64 - 1)
    sym_idx = ((inp["trans_tokens"] - EXECUTOR.TRANSITION_OFFSET) % 2).clamp(0, 1)
    keys = EXECUTOR.k_proj(EXECUTOR.state_emb(q_idx) + EXECUTOR.sym_emb(sym_idx))
    vals = EXECUTOR.v_proj(EXECUTOR.state_emb(inp["nxt_tokens"].clamp(0, 64 - 1)))
    r = EXECUTOR.state_emb(torch.zeros(1, dtype=torch.long, device=DEVICE))
    sym_e = EXECUTOR.sym_emb(inp["sym_ids"])
    states = []
    tape_lines = []
    with torch.no_grad():
        for t in range(len(word)):
            q = EXECUTOR.q_proj(r + sym_e[:, t])
            scores = torch.bmm(q.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
            scores = scores.masked_fill(~inp["trans_mask"], float('-inf'))
            attn_idx = scores.argmax(dim=-1).item()
            attn = torch.softmax(scores, dim=-1)
            retr = torch.bmm(attn.unsqueeze(1), vals).squeeze(1)
            r = EXECUTOR.norm(retr)
            sid = EXECUTOR.state_head(r).argmax(-1).item()
            states.append(sid)
            tape_lines.append(f"{attn_idx} {sid}")
    verdict = 1 if states[-1] in set(spec["accepting"]) else 0
    tape_path = os.path.join(TAPE_DIR, f"{tape_id}.tape")
    import hashlib
    spec_summary = {"k": spec["k"], "start": spec["start"],
                    "accepting": sorted(spec["accepting"]),
                    "delta": [[q, s, spec["delta"][(q, s)]]
                              for q in range(spec["k"]) for s in range(2)]}
    spec_hash = hashlib.md5(json.dumps(spec_summary).encode()).hexdigest()[:8]
    with open(tape_path, "w") as f:
        f.write(f"# k={spec['k']} start={spec['start']} accepting={sorted(spec['accepting'])} "
                f"spec_hash={spec_hash}\n")
        f.write("\n".join(tape_lines))
    return states, verdict, tape_path


# ---------------------------------------------------------------- DFA 任务检测
DFA_HINT = re.compile(r'(state|transition|automaton|finite|accepting|DFA|状态|转移|自动机)', re.IGNORECASE)


def looks_like_dfa_task(text: str) -> bool:
    return bool(DFA_HINT.search(text))


def parse_nl_to_spec(nl_text: str) -> Optional[dict]:
    """NL → spec：用 Ollama 注册的 dfa-parser（预置 few-shot 系统提示）+ 排列纠错。"""
    out = ollama_complete("Translate this into JSON:\n" + nl_text, model=PARSER_MODEL)
    spec = parse_spec(out)
    if spec is not None:
        spec = permutation_repair(spec)
    return spec


def extract_word(text: str):
    """从用户文本提取二进制串：优先 "bits:" / "string:" / "0 1 0 ..." 模式。"""
    m = re.search(r'(?:bits|string|序列|串)\s*[:：]?\s*([01\s]+)', text, re.IGNORECASE)
    if m:
        bits = [int(b) for b in re.findall(r'[01]', m.group(1))]
        if bits:
            return tuple(bits)
    # 兜底：找最长的 0/1 串
    seqs = re.findall(r'[01]{4,}', text)
    if seqs:
        return tuple(int(c) for c in max(seqs, key=len))
    return None


def run_dfa_pipeline(text: str) -> dict:
    """NL DFA 任务完整管线：解析 → 执行 → 纸带。返回 dict。"""
    spec = parse_nl_to_spec(text)
    if spec is None:
        return {"error": "无法从描述中解析出合法 DFA 规格。"}
    word = extract_word(text)
    if word is None:
        return {"error": "未找到二进制输入串（请包含如 'bits: 0 1 1 0 ...'）。"}
    tape_id = uuid.uuid4().hex[:12]
    states, verdict, tape_path = execute_with_tape(spec, word, tape_id)
    return {
        "spec": {"k": spec["k"], "start": spec["start"], "accepting": spec["accepting"]},
        "word": "".join(str(b) for b in word),
        "states": states,
        "verdict": "ACCEPT" if verdict else "REJECT",
        "tape_id": tape_id,
        "tape_url": f"/tape/{tape_id}",
    }


def format_dfa_result(res: dict) -> str:
    if "error" in res:
        return res["error"]
    lines = [
        f"[NSM Middleware] 已解析 DFA（k={res['spec']['k']}, start={res['spec']['start']}, "
        f"accepting={res['spec']['accepting']}）",
        f"输入串: {res['word']}",
        f"状态轨迹: {res['states']}",
        f"最终判定: {res['verdict']}",
        f"纸带(激活通路): {res['tape_url']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- Ollama 兼容 API
@app.route("/api/tags", methods=["GET"])
def tags():
    return jsonify({"models": [
        {"name": "dfa-executor:latest", "modified_at": "2026-08-08", "size": 0},
        {"name": "qwen2.5:7b", "modified_at": "", "size": 4700000000},
    ]})


def _handle(model, prompt, messages, options):
    text = prompt or (messages[-1].get("content", "") if messages else "")
    if model.startswith("dfa-executor") or looks_like_dfa_task(text):
        res = run_dfa_pipeline(text)
        return format_dfa_result(res)
    # 代理到 Ollama
    out = ollama_complete(text, model=model or "qwen2.5:7b",
                          temperature=options.get("temperature", 0.0) if options else 0.0)
    return out


@app.route("/api/generate", methods=["POST"])
def generate():
    body = request.get_json(force=True)
    model = body.get("model", "qwen2.5:7b")
    prompt = body.get("prompt", "")
    options = body.get("options", {})
    t0 = time.time()
    out = _handle(model, prompt, None, options)
    return jsonify({
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": out,
        "done": True,
        "total_duration": int((time.time() - t0) * 1e9),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    model = body.get("model", "qwen2.5:7b")
    messages = body.get("messages", [])
    options = body.get("options", {})
    t0 = time.time()
    out = _handle(model, None, messages, options)
    return jsonify({
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": out},
        "done": True,
        "total_duration": int((time.time() - t0) * 1e9),
    })


@app.route("/tape/<tape_id>", methods=["GET"])
def get_tape(tape_id):
    path = os.path.join(TAPE_DIR, f"{tape_id}.tape")
    if not os.path.exists(path):
        return jsonify({"error": "tape not found"}), 404
    return send_file(path, mimetype="text/plain")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "executor": EXECUTOR is not None, "device": DEVICE})


# ---------------------------------------------------------------- 网页界面
WEB_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>NSM Middleware — DFA Executor</title>
<style>
  body { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; background:#0d1117; color:#e6edf3;
         margin:0; padding:24px; max-width:900px; margin:0 auto; }
  h1 { font-size:20px; color:#58a6ff; }
  .hint { color:#8b949e; font-size:13px; margin-bottom:12px; }
  textarea { width:100%; height:180px; background:#161b22; color:#e6edf3; border:1px solid #30363d;
             border-radius:6px; padding:10px; font-family:Consolas,monospace; font-size:14px; box-sizing:border-box; }
  select { background:#161b22; color:#e6edf3; border:1px solid #30363d; border-radius:6px; padding:6px; margin:10px 0; }
  button { background:#238636; color:#fff; border:none; border-radius:6px; padding:8px 20px; font-size:14px;
           cursor:pointer; }
  button:hover { background:#2ea043; }
  #out { margin-top:16px; background:#161b22; border:1px solid #30363d; border-radius:6px; padding:14px;
         white-space:pre-wrap; font-family:Consolas,monospace; font-size:14px; min-height:120px; }
  #out a { color:#58a6ff; }
  .ex { background:none; border:1px solid #30363d; color:#8b949e; font-size:12px; padding:4px 10px;
        border-radius:12px; margin-right:8px; cursor:pointer; }
  .ex:hover { color:#e6edf3; border-color:#58a6ff; }
  #status { color:#8b949e; font-size:12px; margin-left:12px; }
</style>
</head>
<body>
  <h1>NSM Middleware — 神经状态机执行器</h1>
  <div class="hint">输入自然语言 DFA 描述 + 二进制串，由 <b>dfa-parser</b> 解析、<b>Neural DFA Executor</b>（232K）执行，激活通路纸带可下载审计。</div>
  <div style="margin-bottom:8px;">
    <button class="ex" onclick="fill(1)">示例1 表格</button>
    <button class="ex" onclick="fill(2)">示例2 规则式</button>
    <button class="ex" onclick="fill(3)">示例3 打乱句</button>
  </div>
  <textarea id="inp" placeholder="例如：\nThere are 4 states: A, B, C, D. The start state is A. The accepting states are B, D.\nA -0-> B / A -1-> D / ...\nBinary string: 0 1 0 0 1 1 0 1"></textarea>
  <div>
    <select id="model">
      <option value="dfa-executor">dfa-executor（中间层）</option>
      <option value="qwen2.5:7b">qwen2.5:7b（纯 LLM 对照）</option>
    </select>
    <button onclick="send()">执行</button>
    <span id="status"></span>
  </div>
  <div id="out">输出会显示在这里。</div>
<script>
const EXAMPLES = {
  1: `There are 4 states: A, B, C, D.
The start state is A.
The accepting states are B, D.
The transition rules are:
  A -0-> B
  A -1-> D
  B -0-> A
  B -1-> B
  C -0-> C
  C -1-> C
  D -0-> D
  D -1-> A

Binary string: 0 1 0 0 1 1 0 1`,
  2: `There are 4 states: A, B, C, D.
Start: A. Accepting: B, D.
In state A, on symbol 0 go to B, on symbol 1 go to D.
In state B, on symbol 0 go to A, on symbol 1 go to B.
In state C, on symbol 0 go to C, on symbol 1 go to C.
In state D, on symbol 0 go to D, on symbol 1 go to A.

Binary string: 0 1 0 0 1 1 0 1`,
  3: `We have a machine with 4 states named A, B, C, D.
It starts in A and accepts in B, D.
if we are in A and see symbol 0, it enters B.
when we are in D and see symbol 1, it jumps to A.
upon we are in B and see symbol 1, it becomes B.
whenever we are in A and see symbol 1, it goes to D.
if we are in B and see symbol 0, it transitions to A.
when we are in C and see symbol 0, it enters C.
upon we are in C and see symbol 1, it moves to C.
if we are in D and see symbol 0, it becomes D.

Binary string: 0 1 0 0 1 1 0 1`
};
function fill(n){ document.getElementById('inp').value = EXAMPLES[n]; }
async function send(){
  const inp = document.getElementById('inp').value;
  const model = document.getElementById('model').value;
  const out = document.getElementById('out');
  const status = document.getElementById('status');
  out.textContent = '处理中...';
  status.textContent = '';
  try{
    const resp = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model, messages:[{role:'user', content: inp}]})
    });
    const data = await resp.json();
    out.textContent = data.message.content;
    status.textContent = '耗时 ' + (data.total_duration/1e9).toFixed(2) + 's';
  }catch(e){
    out.textContent = '请求失败: ' + e;
  }
}
</script>
</body>
</html>"""


@app.route("/", methods=["GET"])
def index():
    return WEB_PAGE


if __name__ == "__main__":
    load_executor()
    port = int(os.environ.get("NSM_PORT", 11435))
    print(f"[nsm] middleware server on http://localhost:{port} (Ollama-compatible)")
    print(f"[nsm] 用 model='dfa-executor' 触发 DFA 中间层；其他 model 代理到 Ollama")
    print(f"[nsm] 浏览器打开 http://localhost:{port} 直接使用")
    app.run(host="127.0.0.1", port=port, debug=False)

