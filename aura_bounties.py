"""Local curation-bounty ledger for Aura.

The database coordinates motion curation, while SOL payment is submitted by the
browser directly to Solana devnet. This module never holds keys or custody.
It is a local/demo coordinator, not a production escrow service.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from aura_g1_metrics import canonical_bytes, sha256
from aura_preference import add_preference, get_motion

ADDRESS_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
SIG_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{64,96}$")
SYSTEM_PROGRAM_ID = "11111111111111111111111111111111"
MEMO_PROGRAM_ID = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
DEFAULT_SOLANA_RPC = "https://api.devnet.solana.com"
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"



def _b58decode(value: str) -> bytes:
    n = 0
    for ch in value:
        try:
            n = n * 58 + _B58.index(ch)
        except ValueError as exc:
            raise ValueError("Invalid base58 transaction data") from exc
    raw = b"" if n == 0 else n.to_bytes((n.bit_length() + 7) // 8, "big")
    pad = len(value) - len(value.lstrip("1"))
    return b"\x00" * pad + raw


def _rpc(method: str, params: list[Any], rpc_url: str | None = None) -> Any:
    url = rpc_url or os.environ.get("AURA_SOLANA_RPC_URL") or DEFAULT_SOLANA_RPC
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=float(os.environ.get("AURA_SOLANA_RPC_TIMEOUT", "8"))) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise ValueError(f"Could not verify transaction with Solana RPC: {exc}") from exc
    if payload.get("error"):
        raise ValueError(f"Solana RPC rejected verification: {payload['error'].get('message', payload['error'])}")
    return payload.get("result")


def _transaction(signature: str, rpc_url: str | None = None) -> dict[str, Any]:
    result = None
    # A wallet can report confirmation a fraction of a second before every RPC node
    # serves getTransaction, so allow a few short retries before rejecting the claim.
    for attempt in range(4):
        result = _rpc("getTransaction", [signature, {"encoding": "json", "commitment": "confirmed", "maxSupportedTransactionVersion": 0}], rpc_url)
        if isinstance(result, dict):
            break
        if attempt < 3:
            time.sleep(0.35 * (attempt + 1))
    if not isinstance(result, dict):
        raise ValueError("Solana transaction was not found or is not confirmed yet")
    meta = result.get("meta")
    if not isinstance(meta, dict) or meta.get("err") is not None:
        raise ValueError("Solana transaction did not succeed")
    tx = result.get("transaction")
    if not isinstance(tx, dict) or signature not in (tx.get("signatures") or []):
        raise ValueError("Solana RPC returned an unexpected transaction")
    return result


def _message_parts(result: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], set[str]]:
    message = result["transaction"].get("message") or {}
    account_keys = message.get("accountKeys") or []
    if not all(isinstance(x, str) for x in account_keys):
        raise ValueError("Unsupported Solana transaction account encoding")
    loaded = (result.get("meta") or {}).get("loadedAddresses") or {}
    account_keys = list(account_keys) + list(loaded.get("writable") or []) + list(loaded.get("readonly") or [])
    header = message.get("header") or {}
    required = int(header.get("numRequiredSignatures", 0))
    signers = set(account_keys[:required])
    instructions = message.get("instructions") or []
    if not isinstance(instructions, list):
        raise ValueError("Unsupported Solana transaction instruction encoding")
    return account_keys, instructions, signers


def _decoded_memos(result: dict[str, Any]) -> list[dict[str, Any]]:
    keys, instructions, _ = _message_parts(result)
    out: list[dict[str, Any]] = []
    for ix in instructions:
        try:
            program = keys[int(ix["programIdIndex"])]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if program != MEMO_PROGRAM_ID or not isinstance(ix.get("data"), str):
            continue
        try:
            decoded = _b58decode(ix["data"]).decode("utf-8")
            obj = json.loads(decoded)
            if isinstance(obj, dict):
                out.append(obj)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            continue
    return out


def _has_transfer(result: dict[str, Any], source: str, destination: str, lamports: int) -> bool:
    keys, instructions, _ = _message_parts(result)
    for ix in instructions:
        try:
            program = keys[int(ix["programIdIndex"])]
            accounts = [keys[int(i)] for i in ix.get("accounts", [])]
            data = _b58decode(ix.get("data", ""))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        # SystemProgram.transfer uses enum variant 2 (u32 LE) followed by lamports (u64 LE).
        if program == SYSTEM_PROGRAM_ID and len(accounts) >= 2 and len(data) >= 12:
            variant = int.from_bytes(data[:4], "little")
            amount = int.from_bytes(data[4:12], "little")
            if variant == 2 and accounts[0] == source and accounts[1] == destination and amount == lamports:
                return True
    return False


def validate_bounty_post_transaction(result: dict[str, Any], *, requester: str, bounty: dict[str, Any]) -> dict[str, Any]:
    keys, _, signers = _message_parts(result)
    if not keys or keys[0] != requester or requester not in signers:
        raise ValueError("Bounty transaction was not signed and fee-paid by the requester wallet")
    expected = {
        "app": "AURA", "v": 1, "kind": "curation-bounty",
        "bountyId": bounty["id"], "bountySha256": bounty["bounty_sha256"],
        "rewardLamports": int(bounty["reward_lamports"]), "targetCurations": int(bounty["target_curations"]),
    }
    if expected not in _decoded_memos(result):
        raise ValueError("Bounty transaction does not contain the expected Aura terms memo")
    return {"slot": int(result.get("slot") or 0)}


def validate_curation_payment_transaction(result: dict[str, Any], *, curation: dict[str, Any]) -> dict[str, Any]:
    payer = curation["requester_wallet"]
    curator = curation["curator_wallet"]
    lamports = int(curation["reward_lamports"] )
    keys, _, signers = _message_parts(result)
    if not keys or keys[0] != payer or payer not in signers:
        raise ValueError("Payment transaction was not signed and fee-paid by the bounty requester")
    if not _has_transfer(result, payer, curator, lamports):
        raise ValueError("Payment transaction does not contain the expected SOL transfer")
    meta = result.get("meta") or {}
    pre = meta.get("preBalances")
    post = meta.get("postBalances")
    if not isinstance(pre, list) or not isinstance(post, list) or len(pre) != len(post) or len(pre) < len(keys):
        raise ValueError("Solana transaction is missing balance data required for payment verification")
    try:
        curator_index = keys.index(curator)
    except ValueError as exc:
        raise ValueError("Curator account is missing from the Solana transaction") from exc
    if int(post[curator_index]) - int(pre[curator_index]) != lamports:
        raise ValueError("Curator balance change does not match the expected payout")
    expected = {
        "app": "AURA", "v": 1, "kind": "curation-payout",
        "bountyId": curation["bounty_id"], "curationId": int(curation["id"]),
        "evidenceSha256": curation["evidence_sha256"], "paidLamports": lamports,
    }
    if expected not in _decoded_memos(result):
        raise ValueError("Payment transaction does not contain the expected Aura evidence memo")
    return {"slot": int(result.get("slot") or 0)}


def verify_bounty_post(signature: str, *, requester: str, bounty: dict[str, Any], rpc_url: str | None = None) -> dict[str, Any]:
    return validate_bounty_post_transaction(_transaction(signature, rpc_url), requester=requester, bounty=bounty)


def verify_curation_payment(signature: str, *, curation: dict[str, Any], rpc_url: str | None = None) -> dict[str, Any]:
    return validate_curation_payment_transaction(_transaction(signature, rpc_url), curation=curation)


def _now() -> int:
    return int(time.time())


def _wallet(value: Any, field: str) -> str:
    if not isinstance(value, str) or not ADDRESS_RE.fullmatch(value.strip()):
        raise ValueError(f"{field} must be a Solana wallet address")
    return value.strip()


def _text(value: Any, field: str, max_len: int, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    value = " ".join(value.strip().split())
    if required and not value:
        raise ValueError(f"{field} is required")
    if len(value) > max_len:
        raise ValueError(f"{field} is too long (max {max_len})")
    return value


def _signature(value: Any) -> str:
    if not isinstance(value, str) or not SIG_RE.fullmatch(value.strip()):
        raise ValueError("Invalid Solana transaction signature")
    return value.strip()


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS bounties (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        prompt TEXT NOT NULL,
        criterion TEXT NOT NULL,
        requester_wallet TEXT NOT NULL,
        reward_lamports INTEGER NOT NULL,
        target_curations INTEGER NOT NULL,
        status TEXT NOT NULL,
        prompt_sha256 TEXT NOT NULL,
        posted_signature TEXT,
        created_at INTEGER NOT NULL,
        closed_at INTEGER
    )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS curations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bounty_id TEXT NOT NULL,
        preference_id INTEGER NOT NULL,
        curator_wallet TEXT NOT NULL,
        left_id TEXT NOT NULL,
        right_id TEXT NOT NULL,
        winner_id TEXT NOT NULL,
        evidence_sha256 TEXT NOT NULL,
        payment_signature TEXT,
        paid_lamports INTEGER,
        created_at INTEGER NOT NULL,
        paid_at INTEGER,
        UNIQUE(bounty_id, curator_wallet, left_id, right_id),
        FOREIGN KEY(bounty_id) REFERENCES bounties(id)
    )"""
    )
    bounty_cols = {r[1] for r in conn.execute("PRAGMA table_info(bounties)")}
    if "posted_slot" not in bounty_cols:
        conn.execute("ALTER TABLE bounties ADD COLUMN posted_slot INTEGER")
    if "posted_verified_at" not in bounty_cols:
        conn.execute("ALTER TABLE bounties ADD COLUMN posted_verified_at INTEGER")
    curation_cols = {r[1] for r in conn.execute("PRAGMA table_info(curations)")}
    if "payment_slot" not in curation_cols:
        conn.execute("ALTER TABLE curations ADD COLUMN payment_slot INTEGER")
    if "payment_verified_at" not in curation_cols:
        conn.execute("ALTER TABLE curations ADD COLUMN payment_verified_at INTEGER")
    conn.commit()


def create_bounty(conn: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, Any]:
    ensure_tables(conn)
    title = _text(payload.get("title", ""), "title", 80)
    prompt = _text(payload.get("prompt", ""), "prompt", 480)
    criterion = _text(payload.get("criterion", ""), "criterion", 240)
    requester = _wallet(payload.get("requester_wallet"), "requester_wallet")
    try:
        reward = int(payload.get("reward_lamports"))
        target = int(payload.get("target_curations", 10))
    except (TypeError, ValueError):
        raise ValueError("reward_lamports and target_curations must be integers") from None
    if not 10_000 <= reward <= 10_000_000_000:
        raise ValueError("reward must be between 0.00001 and 10 SOL per accepted curation")
    if not 1 <= target <= 1000:
        raise ValueError("target_curations must be between 1 and 1000")
    bounty_id = "bounty_" + uuid.uuid4().hex[:16]
    created = _now()
    prompt_hash = sha256(canonical_bytes({"prompt": prompt, "criterion": criterion}))
    conn.execute(
        "INSERT INTO bounties(id,title,prompt,criterion,requester_wallet,reward_lamports,target_curations,status,prompt_sha256,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (bounty_id, title, prompt, criterion, requester, reward, target, "open", prompt_hash, created),
    )
    conn.commit()
    return get_bounty(conn, bounty_id)


def get_bounty(conn: sqlite3.Connection, bounty_id: str) -> dict[str, Any]:
    ensure_tables(conn)
    row = conn.execute("SELECT * FROM bounties WHERE id=?", (bounty_id,)).fetchone()
    if not row:
        raise ValueError("Bounty not found")
    item = dict(row)
    counts = conn.execute(
        "SELECT COUNT(*) n, SUM(CASE WHEN payment_signature IS NOT NULL THEN 1 ELSE 0 END) paid FROM curations WHERE bounty_id=?",
        (bounty_id,),
    ).fetchone()
    item["curation_count"] = int(counts["n"] or 0)
    item["paid_count"] = int(counts["paid"] or 0)
    item["bounty_sha256"] = sha256(canonical_bytes({k: item[k] for k in (
        "id", "title", "prompt", "criterion", "requester_wallet", "reward_lamports", "target_curations", "prompt_sha256", "created_at"
    )}))
    return item


def list_bounties(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    ensure_tables(conn)
    ids = [r[0] for r in conn.execute("SELECT id FROM bounties ORDER BY created_at DESC")]
    return [get_bounty(conn, x) for x in ids]


def attach_post_signature(conn: sqlite3.Connection, bounty_id: str, signature: Any, rpc_url: str | None = None) -> dict[str, Any]:
    sig = _signature(signature)
    bounty = get_bounty(conn, bounty_id)
    if bounty.get("posted_signature"):
        if bounty["posted_signature"] == sig:
            return bounty
        raise ValueError("This bounty already has verified on-chain terms")
    if conn.execute("SELECT 1 FROM bounties WHERE posted_signature=? AND id<>?", (sig, bounty_id)).fetchone() or conn.execute("SELECT 1 FROM curations WHERE payment_signature=?", (sig,)).fetchone():
        raise ValueError("This Solana transaction signature is already used by another Aura record")
    verified = verify_bounty_post(sig, requester=bounty["requester_wallet"], bounty=bounty, rpc_url=rpc_url)
    conn.execute("UPDATE bounties SET posted_signature=?,posted_slot=?,posted_verified_at=? WHERE id=?",
                 (sig, verified["slot"], _now(), bounty_id))
    conn.commit()
    return get_bounty(conn, bounty_id)


def curate(conn: sqlite3.Connection, base: Path, payload: dict[str, Any]) -> dict[str, Any]:
    ensure_tables(conn)
    bounty_id = _text(payload.get("bounty_id", ""), "bounty_id", 80)
    bounty = get_bounty(conn, bounty_id)
    if bounty["status"] != "open":
        raise ValueError("Bounty is not open")
    if not bounty.get("posted_signature"):
        raise ValueError("Publish the bounty terms on Solana devnet before accepting curations")
    curator = _wallet(payload.get("curator_wallet"), "curator_wallet")
    if curator == bounty["requester_wallet"]:
        raise ValueError("The bounty requester cannot curate their own bounty")
    if int(bounty["curation_count"]) >= int(bounty["target_curations"]):
        raise ValueError("This bounty has already reached its target number of curations")
    left_id = _text(payload.get("left_id", ""), "left_id", 120)
    right_id = _text(payload.get("right_id", ""), "right_id", 120)
    winner_id = _text(payload.get("winner_id", ""), "winner_id", 120)
    if left_id == right_id or winner_id not in (left_id, right_id):
        raise ValueError("Choose two different motions and one winner")
    # Ensure source motions exist and are G1 before writing anything.
    left_motion = get_motion(base, left_id)
    right_motion = get_motion(base, right_id)
    left_prompt = str(left_motion.get('name') or '').strip()
    right_prompt = str(right_motion.get('name') or '').strip()
    if left_prompt and right_prompt and left_prompt != right_prompt:
        raise ValueError('Bounty curations must compare motions generated from the same exact prompt')
    # The NVIDIA Kimodo adapter stores the prompt in the motion name (truncated to 100 chars).
    if left_prompt and left_prompt != str(bounty['prompt']).strip()[:100]:
        raise ValueError('Selected motions do not belong to this bounty prompt')
    ordered_left, ordered_right = sorted((left_id, right_id))
    # UNIQUE treats order canonically so A/B reversal cannot create a duplicate paid task.
    if conn.execute(
        "SELECT 1 FROM curations WHERE bounty_id=? AND curator_wallet=? AND ((left_id=? AND right_id=?) OR (left_id=? AND right_id=?))",
        (bounty_id, curator, ordered_left, ordered_right, ordered_right, ordered_left),
    ).fetchone():
        raise ValueError("This curator already evaluated this pair for the bounty")
    pref = add_preference(conn, base, left_id, right_id, winner_id, f"Bounty {bounty_id}: {bounty['criterion']}", f"wallet:{curator}")
    evidence = {
        "bounty_id": bounty_id,
        "preference_id": pref["id"],
        "curator_wallet": curator,
        "left_id": left_id,
        "right_id": right_id,
        "winner_id": winner_id,
        "preference_evidence_sha256": pref["evidence_sha256"],
        "reward_lamports": bounty["reward_lamports"],
    }
    evidence_hash = sha256(canonical_bytes(evidence))
    created = _now()
    cur = conn.execute(
        "INSERT INTO curations(bounty_id,preference_id,curator_wallet,left_id,right_id,winner_id,evidence_sha256,created_at) VALUES(?,?,?,?,?,?,?,?)",
        (bounty_id, pref["id"], curator, left_id, right_id, winner_id, evidence_hash, created),
    )
    conn.commit()
    return get_curation(conn, int(cur.lastrowid))


def get_curation(conn: sqlite3.Connection, curation_id: int) -> dict[str, Any]:
    ensure_tables(conn)
    row = conn.execute("SELECT * FROM curations WHERE id=?", (curation_id,)).fetchone()
    if not row:
        raise ValueError("Curation not found")
    item = dict(row)
    bounty = conn.execute("SELECT title,reward_lamports,requester_wallet FROM bounties WHERE id=?", (item["bounty_id"],)).fetchone()
    item["bounty_title"] = bounty["title"] if bounty else ""
    item["reward_lamports"] = int(bounty["reward_lamports"]) if bounty else 0
    item["requester_wallet"] = bounty["requester_wallet"] if bounty else ""
    return item


def list_curations(conn: sqlite3.Connection, bounty_id: str | None = None) -> list[dict[str, Any]]:
    ensure_tables(conn)
    if bounty_id:
        ids = [r[0] for r in conn.execute("SELECT id FROM curations WHERE bounty_id=? ORDER BY id DESC", (bounty_id,))]
    else:
        ids = [r[0] for r in conn.execute("SELECT id FROM curations ORDER BY id DESC LIMIT 500")]
    return [get_curation(conn, int(x)) for x in ids]


def record_payment(conn: sqlite3.Connection, payload: dict[str, Any], rpc_url: str | None = None) -> dict[str, Any]:
    ensure_tables(conn)
    try:
        curation_id = int(payload.get("curation_id"))
        paid_lamports = int(payload.get("paid_lamports"))
    except (TypeError, ValueError):
        raise ValueError("Invalid curation or payment amount") from None
    signature = _signature(payload.get("signature"))
    curation = get_curation(conn, curation_id)
    if curation.get("payment_signature"):
        raise ValueError("This curation is already marked paid")
    if conn.execute("SELECT 1 FROM curations WHERE payment_signature=?", (signature,)).fetchone() or conn.execute("SELECT 1 FROM bounties WHERE posted_signature=?", (signature,)).fetchone():
        raise ValueError("This Solana transaction signature is already used by another Aura record")
    if paid_lamports != int(curation["reward_lamports"]):
        raise ValueError("Payment amount does not match bounty reward")
    verified = verify_curation_payment(signature, curation=curation, rpc_url=rpc_url)
    conn.execute(
        "UPDATE curations SET payment_signature=?,paid_lamports=?,paid_at=?,payment_slot=?,payment_verified_at=? WHERE id=?",
        (signature, paid_lamports, _now(), verified["slot"], _now(), curation_id),
    )
    conn.commit()
    return get_curation(conn, curation_id)


def close_bounty(conn: sqlite3.Connection, bounty_id: str) -> dict[str, Any]:
    if not conn.execute("SELECT 1 FROM bounties WHERE id=?", (bounty_id,)).fetchone():
        raise ValueError("Bounty not found")
    conn.execute("UPDATE bounties SET status='closed',closed_at=? WHERE id=?", (_now(), bounty_id))
    conn.commit()
    return get_bounty(conn, bounty_id)
