import csv, io, json, os, re, sys, traceback, urllib.request
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ROOT = Path(__file__).resolve().parent.parent


def find(*names):
    for n in names:
        for base in (ROOT, Path.cwd(), Path(__file__).resolve().parent):
            if (base / n).exists():
                return base / n
    return None


# ---------------- Q10: students ----------------
def load_students():
    p = find("students.csv", "q-fastapi.csv")
    if not p:
        return []
    with open(p, encoding="utf-8-sig", newline="") as f:
        return [{"studentId": int(r["studentId"]), "class": r["class"]} for r in csv.DictReader(f)]

STUDENTS = load_students()


@app.get("/api")
def students(class_: Optional[List[str]] = Query(None, alias="class")):
    return {"students": [s for s in STUDENTS if not class_ or s["class"] in class_]}


# ---------------- Q11: batch sentiment ----------------
POS = {"love", "loved", "great", "happy", "amazing", "awesome", "excellent", "wonderful", "fantastic",
       "good", "joy", "delighted", "best", "glad", "thrilled", "enjoy", "excited", "pleased", "proud",
       "fun", "perfect", "beautiful", "brilliant", "nice", "smile", "grateful", "cheerful", "yay"}
NEG = {"sad", "terrible", "awful", "bad", "hate", "horrible", "worst", "angry", "upset", "cry", "crying",
       "depressed", "miserable", "disappointed", "unhappy", "lonely", "heartbroken", "sorry", "grief",
       "hurt", "worse", "disgusting", "annoyed", "frustrated", "pain", "lost", "gloomy", "tragic"}


class SentReq(BaseModel):
    sentences: List[str]


@app.post("/sentiment")
def sentiment(req: SentReq):
    out = []
    for t in req.sentences:
        w = re.findall(r"[a-z']+", t.lower())
        p = sum(x in POS for x in w)
        n = sum(x in NEG for x in w)
        if re.search(r"n't|\bnot\b|\bnever\b", t.lower()) and (p or n):
            p, n = n, p  # crude negation flip
        out.append({"sentence": t, "sentiment": "happy" if p > n else "sad" if n > p else "neutral"})
    return {"results": out}


# ---------------- Q5: code interpreter ----------------
class CodeReq(BaseModel):
    code: str


def execute_python_code(code: str) -> dict:
    from io import StringIO
    old = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {"__name__": "__main__"})
        return {"success": True, "output": sys.stdout.getvalue()}
    except Exception:
        return {"success": False, "output": traceback.format_exc()}
    finally:
        sys.stdout = old


def analyze_error(code: str, tb: str) -> List[int]:
    # deterministic: last frame that belongs to the submitted code ("<string>")
    lines = [int(x) for x in re.findall(r'File "<string>", line (\d+)', tb)]
    fallback = lines[-1:] if lines else []
    token = os.environ.get("AIPIPE_TOKEN")
    if not token:
        return fallback
    try:
        body = json.dumps({
            "model": "openai/gpt-4o-mini",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content":
                f'Return JSON {{"error_lines":[ints]}} with the line number(s) of the code where the error occurred.\nCODE:\n{code}\n\nTRACEBACK:\n{tb}'}],
        }).encode()
        rq = urllib.request.Request("https://aipipe.org/openrouter/v1/chat/completions", body,
                                    {"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
        res = json.load(urllib.request.urlopen(rq, timeout=15))
        ai = json.loads(res["choices"][0]["message"]["content"])["error_lines"]
        ai = [int(x) for x in ai]
        return ai or fallback
    except Exception:
        return fallback


@app.post("/code-interpreter")
def code_interpreter(req: CodeReq):
    r = execute_python_code(req.code)
    if r["success"]:
        return {"error": [], "result": r["output"]}
    return {"error": analyze_error(req.code, r["output"]), "result": r["output"]}


# ---------------- Q25: latency analytics ----------------
def load_telemetry():
    p = find("telemetry.json", "q-vercel-latency.json", "telemetry.csv")
    if not p:
        return []
    if p.suffix == ".json":
        d = json.load(open(p, encoding="utf-8"))
        if isinstance(d, dict):
            d = next((v for v in d.values() if isinstance(v, list)), [])
        return d
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))

TELEMETRY = load_telemetry()


def pick(rec, *keys):
    for k in keys:
        if k in rec:
            return rec[k]
    raise KeyError(keys)


def percentile(vals, q):  # same as numpy default (linear interpolation)
    v = sorted(vals)
    if not v:
        return 0.0
    k = (len(v) - 1) * q / 100
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


@app.post("/api/latency")
def latency(body: dict):
    out = {}
    thr = float(body.get("threshold_ms", 180))
    for region in body.get("regions", []):
        rec = [d for d in TELEMETRY if str(pick(d, "region", "Region")).lower() == region.lower()]
        lat = [float(pick(d, "latency_ms", "latency")) for d in rec]
        up = [float(pick(d, "uptime_pct", "uptime", "uptime_percent")) for d in rec]
        out[region] = {
            "avg_latency": sum(lat) / len(lat) if lat else 0,
            "p95_latency": percentile(lat, 95),
            "avg_uptime": sum(up) / len(up) if up else 0,
            "breaches": sum(l > thr for l in lat),
        }
    return out