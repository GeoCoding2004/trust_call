#!/usr/bin/env python3
import json
import os
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

OUT_DIR = Path('docs/demo_evidence/generated')

BACKEND_BASE_URL = os.getenv('BACKEND_BASE_URL', 'http://localhost:8080').rstrip('/')
RAWNET_BASE_URL = os.getenv('RAWNET_BASE_URL', 'http://localhost:8000').rstrip('/')
DISTILBERT_BASE_URL = os.getenv('DISTILBERT_BASE_URL', 'http://localhost:8002').rstrip('/')

CHECKS = [
    ('01_backend_docs.txt', f'{BACKEND_BASE_URL}/docs', 'txt'),
    ('02_backend_metrics.txt', f'{BACKEND_BASE_URL}/metrics', 'txt'),
    ('03_identity_config.json', f'{BACKEND_BASE_URL}/identity/config', 'json'),
    ('04_live_sessions.json', f'{BACKEND_BASE_URL}/identity/live/sessions', 'json'),
    ('05_rawnet_metrics.txt', f'{RAWNET_BASE_URL}/metrics', 'txt'),
    ('06_distilbert_metrics.txt', f'{DISTILBERT_BASE_URL}/metrics', 'txt'),
]


def fetch(url: str) -> tuple[bool, str]:
    req = Request(url, headers={'User-Agent': 'trust-call-demo-evidence/1.0'})
    with urlopen(req, timeout=10) as resp:
        return True, resp.read().decode('utf-8', errors='replace')


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for filename, url, kind in CHECKS:
        target = OUT_DIR / filename
        try:
            ok, payload = fetch(url)
            if kind == 'json':
                try:
                    parsed = json.loads(payload)
                    target.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
                except json.JSONDecodeError:
                    target.write_text(payload, encoding='utf-8')
            else:
                target.write_text(payload, encoding='utf-8')
            status = 'PASS' if ok else 'SKIPPED'
            detail = 'ok'
        except (URLError, HTTPError, TimeoutError, OSError, ValueError) as exc:
            status = 'FAIL'
            detail = str(exc)
            target.write_text(f'CHECK FAILED\nurl: {url}\nerror: {exc}\n', encoding='utf-8')
        results.append((filename, status, url, detail))

    print(f"{'FILE':35} {'STATUS':8} URL")
    print('-' * 90)
    for filename, status, url, _ in results:
        print(f'{filename:35} {status:8} {url}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
