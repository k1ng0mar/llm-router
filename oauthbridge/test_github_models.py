import json, urllib.request, urllib.error, yaml, concurrent.futures as cf
from pathlib import Path

cfg = yaml.safe_load(Path('/home/ubuntu/llm-router/router.yaml').read_text())
key = cfg['router_key']
R = 'http://127.0.0.1:8015'

def api(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(R+path, data=data, method=method,
        headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode() or '{}')

# ---- restore old chat pool + trim code pool ----
old_chat = ['Bai:deepseek-v4-flash','xkiro:qwen/qwen3.8-max','hypercharm:deepseek-v4-flash-0731',
            'opencode:mimo-v2.5-free','groq:openai/gpt-oss-20b','General Compute:gemma-4-31B-it']
pools = api('GET','/api/config').get('pools') or {}
cur_code = list(pools.get('code') or [])
new_code = [e for e in cur_code if not (e.startswith('grok-oauth') or e.startswith('codex-oauth'))]
print('chat ->', api('POST','/api/config/pools',{'pool':'chat','entries':old_chat}).get('entries'))
print('code ->', api('POST','/api/config/pools',{'pool':'code','entries':new_code}).get('entries'))

# ---- fetch fresh github/copilot catalog ----
req = urllib.request.Request('http://127.0.0.1:8648/v1/models', headers={'Authorization':'Bearer proxy'})
cat = json.loads(urllib.request.urlopen(req, timeout=20).read().decode()).get('data') or []
ids = sorted({m['id'] for m in cat})
outdated = lambda mid: mid=='gpt-4' or mid.startswith('gpt-4-') or mid.startswith('gpt-4o')
def ping(mid):
    body = {'model':mid,'messages':[{'role':'user','content':'Reply with exactly: PONG'}],
            'max_tokens':8,'stream':False}
    data=json.dumps(body).encode()
    r=urllib.request.Request('http://127.0.0.1:8648/v1/chat/completions', data=data, method='POST',
        headers={'Content-Type':'application/json','Authorization':'Bearer proxy'})
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            d=json.loads(resp.read().decode() or '{}')
            if 'choices' in d:
                return (mid, 'OK', (d['choices'][0].get('message') or {}).get('content','').strip()[:20])
            return (mid, resp.status, str(d.get('error') or d)[:80])
    except urllib.error.HTTPError as e:
        try: return (mid, e.code, json.loads(e.read().decode() or '{}').get('error',{}).get('message',str(e))[:80])
        except Exception: return (mid, e.code, 'err')
    except Exception as e:
        return (mid, 'TIMEOUT/ERR', str(e)[:60])

eligible = [m for m in ids if not outdated(m)]
print(f'\n=== testing {len(eligible)} non-gpt4 models ===')
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(ping, eligible))
for mid, st, msg in sorted(results, key=lambda x:(x[1]!='OK', x[0])):
    print(f'{st:12} {mid:32} {msg}')
