"""腾讯混元供应商：使用腾讯云 TC3-HMAC-SHA256 签名调用 ChatCompletions。"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import json
import os
from typing import Dict, List

import requests

from ..adapter import ModelAdapter


def _sha256(s: bytes) -> str:
    return hashlib.sha256(s).hexdigest()


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


class HunyuanProvider(ModelAdapter):
    provider = "hunyuan"

    def __init__(
        self,
        model_name: str | None = None,
        secret_id: str | None = None,
        secret_key: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model_name or os.getenv("HUNYUAN_MODEL", "hunyuan-pro"), **kwargs)
        self.secret_id = secret_id or os.getenv("HUNYUAN_SECRET_ID", "")
        self.secret_key = secret_key or os.getenv("HUNYUAN_SECRET_KEY", "")
        self.host = "hunyuan.tencentcloudapi.com"
        self.service = "hunyuan"
        self.version = "2023-09-01"
        self.action = "ChatCompletions"

    def _sign(self, body: str, timestamp: int) -> str:
        date = datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
        content_type = "application/json; charset=utf-8"
        action_lower = self.action.lower()

        credential_scope = f"{date}/{self.service}/tc3_request"
        canonical_headers = (
            f"content-type:{content_type}\n"
            f"host:{self.host}\n"
            f"x-tc-action:{action_lower}\n"
        )
        signed_headers = "content-type;host;x-tc-action"
        payload_hash = _sha256(body.encode("utf-8"))
        canonical_request = "\n".join([
            "POST", "/", "",
            canonical_headers, signed_headers, payload_hash,
        ])
        string_to_sign = "\n".join([
            "TC3-HMAC-SHA256",
            str(timestamp),
            credential_scope,
            _sha256(canonical_request.encode("utf-8")),
        ])
        secret_date = _hmac(("TC3" + self.secret_key).encode("utf-8"), date)
        secret_service = _hmac(secret_date, self.service)
        secret_signing = _hmac(secret_service, "tc3_request")
        signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        return (
            f"TC3-HMAC-SHA256 Credential={self.secret_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

    def _raw_chat(self, messages: List[Dict[str, str]], response_format=None, **kwargs):
        payload: Dict = {
            "Model": self.model_name,
            "Messages": [{"Role": m["role"], "Content": m["content"]} for m in messages],
        }
        if response_format and response_format.get("type") == "json_object":
            payload["ResponseFormat"] = {"Type": "json_object"}

        body = json.dumps(payload, ensure_ascii=False)
        timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
        authorization = self._sign(body, timestamp)

        headers = {
            "Authorization": authorization,
            "Content-Type": "application/json; charset=utf-8",
            "Host": self.host,
            "X-TC-Action": self.action,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": self.version,
        }
        r = requests.post(f"https://{self.host}/", headers=headers, data=body, timeout=120)
        r.raise_for_status()
        data = r.json()
        resp = data.get("Response", {})
        text = resp.get("Choices", [{}])[0].get("Message", {}).get("Content", "")
        u = resp.get("Usage", {})
        in_tok = u.get("PromptTokens") or self.count_tokens(json.dumps(messages, ensure_ascii=False))
        out_tok = u.get("CompletionTokens") or self.count_tokens(text)
        return text, in_tok, out_tok
