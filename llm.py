"""LLM intelligence layer. Works with Gemini, Anthropic or Groq; falls back to rule-based logic if no key/provider."""
import json, re, requests
from config import C

def _call(prompt, system="", json_mode=False):
    try:
        if C.LLM_PROVIDER == "gemini" and C.GEMINI_API_KEY:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{C.GEMINI_MODEL}:generateContent?key={C.GEMINI_API_KEY}"
            body = {"contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2}}
            if system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            if json_mode:
                body["generationConfig"]["responseMimeType"] = "application/json"
            r = requests.post(url, json=body, timeout=30).json()
            return r["candidates"][0]["content"]["parts"][0]["text"]
        if C.LLM_PROVIDER == "anthropic" and C.ANTHROPIC_API_KEY:
            r = requests.post("https://api.anthropic.com/v1/messages",
                              headers={"x-api-key": C.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                                       "content-type": "application/json"},
                              json={"model": C.ANTHROPIC_MODEL, "max_tokens": 900, "system": system,
                                    "messages": [{"role": "user", "content": prompt}]}, timeout=30).json()
            return r["content"][0]["text"]
        if C.LLM_PROVIDER == "groq" and C.GROQ_API_KEY:
            messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
            body = {"model": C.GROQ_MODEL, "messages": messages, "temperature": 0.2, "max_tokens": 900}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {C.GROQ_API_KEY}", "Content-Type": "application/json"},
                              json=body, timeout=30).json()
            return r["choices"][0]["message"]["content"]
    except Exception as e:
        print("LLM error:", e)
    return None

def _json(text):
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        return None

HIGH = ["accident", "crash", "injur", "blood", "fire", "collision", "hit", "ambulance", "unconscious", "overturn"]
MED = ["breakdown", "broke", "stuck", "obstruction", "tree", "flood", "water", "pothole", "signal", "jam", "blocked"]

def classify_report(rtype, text):
    """Returns dict: category, priority, summary, key_info, action."""
    system = "You are a traffic-police dispatcher assistant. Reply ONLY with JSON."
    prompt = ("Analyse this public traffic report. Return JSON with keys: category (Accident|Vehicle Breakdown|"
              "Illegal Parking|Road Obstruction|Traffic Violation|Other), priority (High|Medium|Low), "
              "summary (max 25 words), key_info (object: vehicles, injuries, landmarks), action (max 20 words).\n"
              f"Reported type: {rtype}\nDescription: {text}")
    out = _json(_call(prompt, system, True))
    if out and out.get("priority") in ("High", "Medium", "Low"):
        out.setdefault("category", rtype)
        return out
    t = f"{rtype} {text}".lower()
    pr = "High" if any(k in t for k in HIGH) else "Medium" if any(k in t for k in MED) else "Low"
    if rtype == "Accident":
        pr = "High"
    actions = {"High": "Dispatch nearest officer immediately; call ambulance if injuries reported.",
               "Medium": "Send an officer to clear the obstruction and manage traffic.",
               "Low": "Officer to inspect and issue notice/challan if required."}
    return {"category": rtype, "priority": pr, "summary": (text or rtype)[:120],
            "key_info": {}, "action": actions[pr]}

def summarize_situation(m):
    d = m.get("density", {})
    if not d:
        return "No live data yet. Start a video source from the Source page."
    busiest = max(d, key=d.get)
    sig = m.get("signal", {})
    green = [k for k, v in sig.items() if v == "green"]
    fallback = (f"Busiest approach: {busiest} ({d[busiest]} vehicles/lane). "
                f"Green: {', '.join(green) or 'none (transition)'}. Reason: {m.get('signal_reason','-')}. "
                f"Violations so far: {sum(m.get('totals', {}).values())}. "
                f"Emergency: {'YES' if m.get('emergency_lane') or m.get('siren') else 'no'}.")
    out = _call("Write a 3-sentence natural-language traffic summary and explain the signal decision.\n"
                + json.dumps(m, default=str), "You are a concise traffic control room assistant.")
    return out or fallback

def assistant(question, ctx):
    system = ("You are the AI assistant for a traffic control room. Answer ONLY from the data given. "
              "Be concise. If data is missing, say so.")
    out = _call(f"DATA:\n{json.dumps(ctx, default=str)}\n\nQUESTION: {question}", system)
    if out:
        return out
    ql = question.lower()
    if "violation" in ql:
        return f"Violations in the last 24h: {ctx['day']['violations_total']}. By type: {ctx['day']['violations_by_type'] or 'none'}."
    if "pending" in ql or "task" in ql:
        return f"Pending/active tasks: {ctx['active_tasks']}. Completed today: {ctx['day']['tasks_completed']}."
    if "dens" in ql or "busy" in ql or "traffic" in ql:
        return f"Average density per lane (24h): {ctx['day']['avg_density'] or 'no data'}. Busiest: {ctx['day']['busiest_lane'] or 'n/a'}."
    if "emergency" in ql or "ambulance" in ql:
        return f"Emergency events in the last 24h: {ctx['day']['emergency_events']}."
    if "report" in ql:
        return f"Citizen reports (24h): {ctx['day']['reports_total']} - {ctx['day']['reports_by_status']}."
    return "No LLM key set, so I can only answer about: violations, tasks, density, emergency, reports. Set LLM_PROVIDER and an API key for full answers."

def narrative(period, stats):
    fallback = (f"{period.title()} report: {stats['violations_total']} violations recorded "
                f"(fines INR {stats['fines_total']}), most common: {stats['top_violation'] or 'n/a'}. "
                f"{stats['reports_total']} citizen reports received, {stats['tasks_completed']} tasks completed"
                f"{' (avg ' + str(stats['avg_completion_min']) + ' min)' if stats['avg_completion_min'] else ''}. "
                f"Busiest approach: {stats['busiest_lane'] or 'n/a'}. Emergency events: {stats['emergency_events']}.")
    out = _call(f"Write a short professional {period} traffic report (max 120 words) from:\n"
                + json.dumps(stats, default=str), "You write official traffic department reports.")
    return out or fallback
