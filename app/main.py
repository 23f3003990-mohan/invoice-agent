"""A2A 1.0 HTTP+JSON invoice action agent."""
import asyncio
import copy
import datetime as dt
import hashlib
import json
import logging
import os
import uuid
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from . import store
from .decide import canonical, decide_packages, package_id, sha

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("agent")

A2A_MT = "application/a2a+json"
BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MT = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS_MT = "application/vnd.ga5.invoice-action-receipts+json"
RESULTS_MT = "application/vnd.ga5.invoice-action-results+json"
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}
BASE_PATH = "/" + os.getenv("BASE_PATH", "a2a").strip("/")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()
MAX_BYTES = 500 * 1024
LOG_BATCHES = os.getenv("LOG_BATCHES", "1") == "1"

app = FastAPI(redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None)
_locks = defaultdict(asyncio.Lock)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def a2a(body, status=200):
    return Response(json.dumps(body, ensure_ascii=False), status_code=status, media_type=A2A_MT)


def err(status, code, message):
    return JSONResponse({"error": {"code": status, "status": code, "message": message}}, status_code=status)


def base_url(request: Request):
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "localhost")
    proto = request.headers.get("x-forwarded-proto", "https")
    return f"{proto}://{host}{BASE_PATH}/"


# ---------------- Agent Card (public) ----------------

@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    url = base_url(request)
    return JSONResponse({
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one evidence-backed action per invoice "
                       "and executes only grader-accepted proposals with durable, idempotent task state.",
        "version": "1.0.0",
        "url": url,
        "preferredTransport": "HTTP+JSON",
        "protocolVersion": "1.0",
        "supportedInterfaces": [{"url": url, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": False},
        "securitySchemes": {"bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}},
        "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
        "defaultInputModes": [BATCH_MT, RESULTS_MT, A2A_MT],
        "defaultOutputModes": [PROPOSALS_MT, RECEIPTS_MT],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice action agent",
            "description": "Chooses settle_invoice, request_approval, hold_invoice, reject_duplicate or "
                           "open_exception for each invoice package, cites the decisive evidence, and "
                           "executes only accepted proposals with receipt binding.",
            "tags": ["invoices", "accounts-payable", "approvals", "duplicates", "a2a"],
            "inputModes": [BATCH_MT, RESULTS_MT],
            "outputModes": [PROPOSALS_MT, RECEIPTS_MT],
        }],
    })


# ---------------- request gate ----------------

def gate(request: Request, needs_body: bool):
    """Return (principal, None) or (None, error_response)."""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or not auth[7:].strip() or auth[7:] != auth[7:].strip():
        return None, err(401, "UNAUTHENTICATED", "Missing or invalid bearer token")
    principal = hashlib.sha256(auth[7:].encode()).hexdigest()
    if request.headers.get("a2a-version", "").strip() != "1.0":
        return None, err(400, "VERSION_NOT_SUPPORTED", "A2A-Version 1.0 is required")
    if needs_body:
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype not in (A2A_MT, "application/json"):
            return None, err(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/a2a+json")
    accept = request.headers.get("accept", "")
    if accept and not any(t in accept for t in (A2A_MT, "application/json", "*/*", "application/*")):
        return None, err(406, "NOT_ACCEPTABLE", "Response media type is application/a2a+json")
    return principal, None


def view(task, history_length=None):
    t = copy.deepcopy(task)
    if history_length is not None:
        try:
            n = int(history_length)
            t["history"] = t["history"][-n:] if n > 0 else []
        except (TypeError, ValueError):
            pass
    return t


# ---------------- message:send ----------------

@app.post(BASE_PATH + "/message:send")
async def message_send(request: Request):
    principal, bad = gate(request, True)
    if bad:
        return bad
    try:
        body = await request.json()
        msg = body["message"]
        assert isinstance(msg, dict) and isinstance(msg.get("messageId"), str) and msg["messageId"]
        assert isinstance(msg.get("parts"), list)
    except Exception:
        return err(400, "INVALID_REQUEST", "Body must contain a message with messageId and parts")
    cfg = body.get("configuration") or {}
    hist_len = cfg.get("historyLength")
    msg_hash = sha(canonical(msg))

    async with _locks[("msg", principal, msg["messageId"])]:
        seen = store.get_message(principal, msg["messageId"])
        if seen:
            if seen[0] != msg_hash:
                return err(409, "IDEMPOTENCY_CONFLICT", "messageId was already used with different content")
            task, _ = store.get_task(principal, seen[1])
            return a2a({"task": view(task, hist_len)})
        if msg.get("taskId"):
            return await continue_task(principal, msg, msg_hash, hist_len)
        return await new_task(principal, msg, msg_hash, hist_len)


async def new_task(principal, msg, msg_hash, hist_len):
    part = next((p for p in msg["parts"] if isinstance(p, dict) and p.get("mediaType") == BATCH_MT), None)
    data = (part or {}).get("data")
    if msg.get("role") != "ROLE_USER" or not isinstance(data, dict) or not isinstance(data.get("packages"), list) \
            or not data.get("packages"):
        return err(400, "INVALID_REQUEST", "Expected a ROLE_USER message with an invoice-claim-batch part")
    packages = data["packages"]
    pids = [package_id(p) for p in packages]
    if any(p is None for p in pids) or len(set(pids)) != len(pids):
        return err(400, "INVALID_REQUEST", "Every package needs a unique packageId")

    task_id = str(uuid.uuid4())
    context_id = msg.get("contextId") or str(uuid.uuid4())
    if LOG_BATCHES:
        try:
            bdir = os.path.join(os.path.dirname(store.DB_PATH), "batches")
            os.makedirs(bdir, exist_ok=True)
            with open(os.path.join(bdir, f"{data.get('batchId', task_id)}.json"), "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    meta = {k: v for k, v in data.items() if k not in ("packages", "batchId")}
    try:
        decisions = await asyncio.wait_for(
            decide_packages(packages, meta, data.get("policyRevision")), timeout=42)
    except Exception as e:
        log.exception("decision step failed")
        return err(503, "UNAVAILABLE", "Decision service temporarily unavailable; retry the same message")

    proposals = []
    for i, pkg in enumerate(packages):
        d = decisions[i]
        proposals.append({
            "packageId": pids[i],
            "actionId": "act_" + sha(f"{task_id}|{pids[i]}")[:24],
            "action": d["action"],
            "facts": d["facts"],
            "evidenceRefs": d["evidenceRefs"],
            "rationale": d["rationale"],
        })
    hist_msg = dict(msg, taskId=task_id, contextId=context_id)
    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
        "artifacts": [{
            "artifactId": "proposals-" + task_id,
            "name": "invoice-action-proposals",
            "parts": [{"mediaType": PROPOSALS_MT,
                       "data": {"batchId": data.get("batchId"), "proposals": proposals}}],
        }],
        "history": [hist_msg],
    }
    private = {"batchId": data.get("batchId"), "proposals": {p["packageId"]: p for p in proposals}}
    store.save_task(principal, task, private, (msg["messageId"], msg_hash, task_id))
    return a2a({"task": view(task, hist_len)})


async def continue_task(principal, msg, msg_hash, hist_len):
    task_id = msg["taskId"]
    async with _locks[("task", task_id)]:
        task, private = store.get_task(principal, task_id)
        if not task:
            return err(404, "TASK_NOT_FOUND", "Task not found")
        if task["status"]["state"] in TERMINAL or task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
            return err(409, "TASK_NOT_CONTINUABLE", "Task is not awaiting input")
        if msg.get("contextId") != task["contextId"] or msg.get("role") != "ROLE_USER":
            return err(400, "INVALID_CONTINUATION", "Continuation does not match the task")
        part = next((p for p in msg["parts"] if isinstance(p, dict) and p.get("mediaType") == RESULTS_MT), None)
        data = (part or {}).get("data")
        if not isinstance(data, dict) or data.get("batchId") != private["batchId"] \
                or not isinstance(data.get("results"), list) or not data["results"]:
            return err(400, "INVALID_CONTINUATION", "Continuation does not match the task")
        executions, seen = [], set()
        for r in data["results"]:
            prop = private["proposals"].get(r.get("packageId")) if isinstance(r, dict) else None
            if not prop or r["packageId"] in seen or r.get("actionId") != prop["actionId"] \
                    or r.get("action") != prop["action"] or r.get("outcome") not in ("ACCEPTED", "REJECTED"):
                return err(400, "INVALID_CONTINUATION", "Continuation does not match the stored proposals")
            seen.add(r["packageId"])
            if r["outcome"] == "ACCEPTED":
                nonce = r.get("receiptNonce")
                if not isinstance(nonce, str) or not nonce:
                    return err(400, "INVALID_CONTINUATION", "Accepted result is missing its receipt")
                executions.append({
                    "packageId": prop["packageId"], "actionId": prop["actionId"], "action": prop["action"],
                    "receiptNonce": nonce, "facts": prop["facts"], "evidenceRefs": prop["evidenceRefs"],
                })
        task = copy.deepcopy(task)
        task["history"].append(dict(msg))
        task["artifacts"].append({
            "artifactId": "receipts-" + task_id,
            "name": "invoice-action-receipts",
            "parts": [{"mediaType": RECEIPTS_MT,
                       "data": {"batchId": private["batchId"], "executions": executions}}],
        })
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        store.save_task(principal, task, private, (msg["messageId"], msg_hash, task_id))
        return a2a({"task": view(task, hist_len)})


# ---------------- task reads / cancel ----------------

@app.post(BASE_PATH + "/tasks/{task_id}:cancel")
async def cancel(task_id: str, request: Request):
    principal, bad = gate(request, False)
    if bad:
        return bad
    async with _locks[("task", task_id)]:
        task, private = store.get_task(principal, task_id)
        if not task:
            return err(404, "TASK_NOT_FOUND", "Task not found")
        if task["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_CANCELABLE", "Task is already in a terminal state")
        task = copy.deepcopy(task)
        task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        store.save_task(principal, task, private)
        return a2a(task)


@app.get(BASE_PATH + "/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    principal, bad = gate(request, False)
    if bad:
        return bad
    task, _ = store.get_task(principal, task_id)
    if not task:
        return err(404, "TASK_NOT_FOUND", "Task not found")
    return a2a(view(task, request.query_params.get("historyLength")))


@app.get(BASE_PATH + "/tasks")
async def list_tasks(request: Request):
    principal, bad = gate(request, False)
    if bad:
        return bad
    hl = request.query_params.get("historyLength", "0")
    out, size = [], 100
    for t in store.list_tasks(principal):
        v = view(t, hl)
        s = len(json.dumps(v))
        if size + s > MAX_BYTES:
            break
        out.append(v)
        size += s
    return a2a({"tasks": out, "nextPageToken": ""})


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "ok", "agentCard": "/.well-known/agent-card.json", "base": BASE_PATH + "/"}
