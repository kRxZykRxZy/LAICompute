#!/usr/bin/env python3
"""End-to-end test of every LAICompute API endpoint on the Raspberry Pi."""
import json
import socket
import sys
import time
import urllib.request
import urllib.error

HOST = "raspberrypi.local"
PORT = 8080
results = []

def resolve_ips(host):
    try:
        infos = socket.getaddrinfo(host, PORT, socket.AF_UNSPEC, socket.SOCK_STREAM)
        return sorted(set(i[4][0] for i in infos))
    except Exception as e:
        print(f"  (dns resolution failed: {e})")
        return []

def base_url_for(ip):
    return f"http://[{ip}]:{PORT}" if ":" in ip else f"http://{ip}:{PORT}"

def pick_base():
    ips = resolve_ips(HOST)
    print(f"  {HOST} resolves to: {ips}")
    if not ips:
        return f"http://{HOST}:{PORT}"
    for ip in ips:
        if not ip.startswith("fe80:"):
            test = base_url_for(ip)
            try:
                with urllib.request.urlopen(test + "/api/status", timeout=8) as r:
                    r.read()
                    return test
            except Exception:
                continue
    return f"http://{HOST}:{PORT}"

BASE = pick_base()
print(f"  using base: {BASE}\n")

def req(method, path, body=None, expect=200):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            raw = resp.read().decode(errors="replace")
            code = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        code = e.code
    ok = code == expect
    results.append((method, path, code, ok))
    status = "OK" if ok else f"FAIL (expected {expect})"
    print(f"[{status}] {method} {path} -> {code}")
    if raw.strip():
        try:
            parsed = json.loads(raw)
            print(f"        {parsed if isinstance(parsed, str) else json.dumps(parsed)}")
        except Exception:
            snippet = raw[:200].replace("\n", " ")
            print(f"        {snippet}")
    return code, raw

def main():
    print(f"=== LAICompute API test against {BASE} ===")

    # 1. status
    req("GET", "/api/status")

    # 2. models
    req("GET", "/api/models")

    # 3. backends
    req("GET", "/api/backends")

    # 4. gpu/stats
    req("GET", "/api/gpu/stats")

    # 5. gpu/detect
    req("GET", "/api/gpu/detect")

    # 6. CORS preflight
    r = urllib.request.Request(BASE + "/api/status", method="OPTIONS")
    with urllib.request.urlopen(r, timeout=30) as resp:
        ps = [h for h in resp.headers.items() if h[0].lower().startswith("access-control")]
        if ps:
            print("[OK] OPTIONS /api/status -> CORS headers: " + "; ".join(f"{k}={v}" for k, v in ps))
            results.append(("OPTIONS", "/api/status", 200, True))
        else:
            print("[FAIL] OPTIONS /api/status -> no CORS headers")
            results.append(("OPTIONS", "/api/status", 200, False))

    # 7. root page
    req("GET", "/", expect=200)

    # 8. load model: find first .gguf from /api/models
    code, raw = req("GET", "/api/models")
    model_names = []
    try:
        models = json.loads(raw)
        if isinstance(models, list):
            model_names = [m.get("name", "") for m in models if m.get("name", "").endswith(".gguf")]
    except Exception:
        pass
    if model_names:
        name = model_names[0]
        print(f">>> loading model: {name}")
        # UI uses the plural route (/api/models/load)
        code, raw = req("POST", "/api/models/load", {"name": name}, expect=200)
        if code == 200:
            time.sleep(1)
            # 9. benchmark on cpu
            req("POST", "/api/benchmark", {"backend": "cpu", "max_tokens": 16})
            # 10. chat
            req("POST", "/api/chat", {"prompt": "What is a Raspberry Pi?", "max_tokens": 16})
            # 11. stop (safe even when nothing running)
            req("POST", "/api/stop")
            # 12. backend switch if GPU available
            req("GET", "/api/backends")
            code, raw = req("GET", "/api/gpu/detect")
            try:
                d = json.loads(raw)
                if d.get("compute_available") and d.get("runtime_available"):
                    print(">>> GPU runtime available — testing GPU backend")
                    req("POST", "/api/backend", {"backend": "gpu"})
                    req("POST", "/api/backend", {"backend": "both"})
                    req("POST", "/api/benchmark", {"backend": "both", "max_tokens": 16}, expect=200)
                    req("POST", "/api/backend", {"backend": "cpu"})
                else:
                    print(f">>> GPU runtime NOT available ({d.get('runtime_detail', '')}) — skipping GPU backend tests")
            except Exception as e:
                print(f">>> could not parse gpu/detect: {e}")
        else:
            print(">>> model load failed — skipping model-dependent tests")
    else:
        print(">>> no .gguf models found — skipping model-dependent tests")

    # 13. reconnect sanity after backend switching
    req("GET", "/api/status")

    # 14. unload (UI uses plural route)
    req("POST", "/api/models/unload")

    print("\n=== SUMMARY ===")
    failed = [r for r in results if not r[3]]
    passed = len(results) - len(failed)
    print(f"passed: {passed}/{len(results)}")
    for r in failed:
        print(f"  FAILED: {r[0]} {r[1]} (got {r[2]})")
    if failed:
        print("\nSOME TESTS FAILED — see above")
        sys.exit(1)
    else:
        print("ALL ENDPOINT TESTS PASSED")

if __name__ == "__main__":
    main()