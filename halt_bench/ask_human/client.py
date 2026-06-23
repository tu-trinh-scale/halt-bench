from __future__ import annotations

import json
import urllib.request


def ask_human(*, server_url: str, instance_id: str, question: str, timeout_s: int = 30) -> str:
    payload = {"instance_id": instance_id, "question": question}
    request = urllib.request.Request(
        url=server_url.rstrip("/") + "/ask",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("response", ""))
