"""Minimal same-gateway reproduction of Hermes identity prompt latency.

Run with CTX_TEST_LIVE_BASE_URL, CTX_TEST_LIVE_MODEL and
CTX_TEST_LIVE_CREDENTIAL_REF. Prints status/timing only, never credentials.
"""

import concurrent.futures
import json
import os
import subprocess
import time
import urllib.error
import urllib.request


HERMES_IDENTITY = (
    "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
    "You are helpful, knowledgeable, and direct. You assist users with a wide r"
)


def credential(reference):
    if reference.startswith("env:"):
        return os.environ[reference[4:]]
    if reference.startswith("keychain:"):
        return subprocess.run(["security", "find-generic-password", "-a", "default",
                               "-s", reference[9:], "-w"], capture_output=True,
                              text=True, check=True).stdout.strip()
    raise ValueError("credential must use env: or keychain:")


def main():
    base = os.environ["CTX_TEST_LIVE_BASE_URL"].rstrip("/")
    model = os.environ["CTX_TEST_LIVE_MODEL"]
    key = credential(os.environ["CTX_TEST_LIVE_CREDENTIAL_REF"])

    def probe(label, system):
        body = {"model": model, "messages": [{"role": "system", "content": system},
                                                {"role": "user", "content": "Reply exactly OK."}],
                "max_tokens": 64, "stream": True}
        request = urllib.request.Request(
            f"{base}/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                     "User-Agent": "curl/8.0"})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                for line in response:
                    if line.startswith(b"data:"):
                        return {"case": label, "status": response.status,
                                "first_data_seconds": round(time.monotonic() - start, 2)}
                return {"case": label, "status": response.status,
                        "first_data_seconds": None}
        except urllib.error.HTTPError as error:
            return {"case": label, "status": error.code,
                    "first_data_seconds": None}
        except (urllib.error.URLError, TimeoutError):
            return {"case": label, "status": "timeout_or_network_error",
                    "first_data_seconds": None}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda item: probe(*item),
                                [("generic", "You are a helpful assistant."),
                                 ("hermes_identity_prefix", HERMES_IDENTITY)]))
    print(json.dumps({"requested_model": model, "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
