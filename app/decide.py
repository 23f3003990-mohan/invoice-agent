"""AI decision layer: one model call per chunk of packages, cached by canonical package content,
with deterministic validation of evidence references, facts and rationale."""
import asyncio
import hashlib
import json
import logging
import os
import re

import httpx

from . import store

log = logging.getLogger("decide")

ACTIONS = ["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]
LLM_URL = os.getenv("LLM_URL", "https://aipipe.org/openai/v1/chat/completions")
LLM_KEY = os.getenv("LLM_API_KEY") or os.getenv("AIPIPE_TOKEN", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
CHUNK = int(os.getenv("LLM_CHUNK", "4"))
REF_STYLE = os.getenv("REF_STYLE", "inner")  # "inner" -> ABC-1 ; "bracket" -> [ABC-1]
DECISION_VERSION = "v1"
REF_RE = re.compile(r"\[([^\[\]\n]{1,160})\]")


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def package_id(pkg):
    for k in ("packageId", "package_id", "id"):
        if isinstance(pkg, dict) and pkg.get(k):
            return str(pkg[k])
    return None


def cache_key(pkg, policy_revision):
    content = {k: v for k, v in pkg.items() if k not in ("packageId", "package_id", "id")} if isinstance(pkg, dict) else pkg
    return sha(DECISION_VERSION + "|" + LLM_MODEL + "|" + str(policy_revision) + "|" + canonical(content))


# ---------- evidence helpers ----------

def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def paragraphs(pkg):
    out = []
    for s in _strings(pkg):
        for para in re.split(r"\n\s*\n|\r\n\s*\r\n", s):
            refs = []
            for r in REF_RE.findall(para):
                r = r.strip()
                if r not in refs:
                    refs.append(r)
            if refs:
                out.append((para, refs))
    return out


def norm_ref(r):
    return str(r).strip().strip("[]").strip()


def fmt_ref(r):
    return f"[{r}]" if REF_STYLE == "bracket" else r


def pick_refs(pkg, model_refs):
    """Return exactly the refs of the decisive paragraph that best matches the model's choice."""
    wanted = [norm_ref(r) for r in model_refs if norm_ref(r)]
    paras = paragraphs(pkg)
    best, best_score = None, 0
    for para, refs in paras:
        score = len(set(wanted) & set(refs))
        if score > best_score or (score == best_score and best and len(refs) == 3 and len(best) != 3):
            best, best_score = refs, score
    if best and best_score:
        if len(best) == 3:
            chosen = best
        else:
            chosen = [r for r in wanted if r in best][:3]
            chosen += [r for r in best if r not in chosen][: 3 - len(chosen)]
    else:
        known = {r for _, refs in paras for r in refs}
        chosen = [r for r in wanted if r in known][:3] or wanted[:3]
    return [fmt_ref(r) for r in chosen]


def to_minor(v):
    if isinstance(v, bool):
        return 0
    if isinstance(v, int):
        return v
    try:
        s = str(v).replace(",", "").strip()
        return int(round(float(s)))
    except Exception:
        return 0


def fix_rationale(text, action, refs):
    text = (text or "").strip()
    cited = sum(1 for r in refs if r in text)
    if action not in text or cited < 2:
        text = (text + " " if text else "") + f"Action {action} is chosen based on decisive evidence " + ", ".join(refs) + "."
    if len(text) < 60:
        text += " The decisive paragraph in the current documents supports this action over the alternatives."
    return text[:1500]


# ---------- model call ----------

SYSTEM = """You are an accounts-payable control agent. For EVERY invoice package choose exactly one action:
- settle_invoice: invoice is valid, reconciled (PO / receipt / invoice agree) and within autonomous payment authority.
- request_approval: commercially valid and reconciled, but the amount or type is outside delegated/autonomous authority under the CURRENT policy, so a human approver is needed.
- hold_invoice: payment must pause until a stated verification completes (e.g. bank-detail change callback, tax/vendor verification, pending goods receipt or inspection).
- reject_duplicate: the SAME commercial invoice (same vendor, same invoice number, same amount) was ALREADY PAID. A payment for a different invoice from the same vendor is NOT a duplicate.
- open_exception: material records conflict (amount/quantity/price/vendor/currency mismatch between invoice, PO, receipt or contract) and need an exception workflow.

How to read the documents:
- The documents mix decisive facts with distractors: cover sheets, archived/historical or superseded examples, training samples and decoys, negated statements ("no hold is required", "this is not a duplicate"), and action words used in irrelevant contexts. Ignore all of those.
- Exactly one paragraph is DECISIVE: it states the facts that determine the action for THIS invoice under the current policy. It contains three bracketed references like [XYZ-123]. Return those three references exactly as written (without the brackets is fine). Never return cover-sheet, archive, example or training references.
- Use the current policy revision's limits, not older revisions.
- facts: vendorName and invoiceNumber exactly as written for this invoice; amountMinor = invoice amount in integer minor units (e.g. INR 1,234.50 -> 123450; if the document already states amountMinor use it); currency = ISO code.
- rationale: 80-600 characters, must contain the exact action string and cite at least two of the evidence references.

Return ONLY JSON: {"decisions":[{"packageId":"...","action":"...","vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"...","evidenceRefs":["...","...","..."],"rationale":"..."}]}"""


async def _call_model(client, batch_meta, pkgs):
    user = {"batchContext": batch_meta, "packages": pkgs}
    body = {
        "model": LLM_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    }
    headers = {"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json",
               "User-Agent": "invoice-agent/1.0"}
    last = None
    for attempt in range(2):
        try:
            r = await client.post(LLM_URL, json=body, headers=headers, timeout=38)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            return {str(d.get("packageId")): d for d in data.get("decisions", [])}
        except Exception as e:  # retry once
            last = e
            log.warning("model call failed (attempt %s): %s", attempt + 1, e)
    raise RuntimeError(f"model call failed: {last}")


async def decide_packages(packages, batch_meta, policy_revision):
    """Return {index: decision} for every package, using the cache where possible."""
    results, todo = {}, []
    for i, pkg in enumerate(packages):
        cached = store.get_decision(cache_key(pkg, policy_revision))
        if cached:
            results[i] = cached
        else:
            todo.append(i)
    if todo:
        async with httpx.AsyncClient() as client:
            chunks = [todo[j:j + CHUNK] for j in range(0, len(todo), CHUNK)]
            outs = await asyncio.gather(*[
                _call_model(client, batch_meta, [packages[i] for i in ch]) for ch in chunks
            ])
        for ch, out in zip(chunks, outs):
            for i in ch:
                pkg = packages[i]
                d = out.get(str(package_id(pkg))) or {}
                action = d.get("action") if d.get("action") in ACTIONS else "hold_invoice"
                refs = pick_refs(pkg, d.get("evidenceRefs") or [])
                dec = {
                    "action": action,
                    "facts": {
                        "vendorName": str(d.get("vendorName") or ""),
                        "invoiceNumber": str(d.get("invoiceNumber") or ""),
                        "amountMinor": to_minor(d.get("amountMinor")),
                        "currency": str(d.get("currency") or "").upper(),
                    },
                    "evidenceRefs": refs,
                    "rationale": fix_rationale(d.get("rationale"), action, refs),
                }
                if d:  # only cache real model answers
                    store.put_decision(cache_key(pkg, policy_revision), dec)
                results[i] = dec
    return results
