"""Bounded image validation worker; reads no external metadata or model files."""
from __future__ import annotations

import json
import sys

from ..providers.vision import _validate_image


def main() -> None:
    try:
        data = sys.stdin.buffer.read(10_000_001)
        if not 0 < len(data) <= 10_000_000 or sys.argv[1] not in ('image/png', 'image/jpeg', 'image/webp'):
            raise ValueError('invalid image input')
        width, height = _validate_image(data, sys.argv[1], 20_000_000)
        result = {'width': width, 'height': height}
    except Exception:
        sys.stdout.buffer.write(b'{"error":"image cannot decode as its declared type; install scone-memory[images]"}')
        return
    sys.stdout.buffer.write(json.dumps(result).encode())


if __name__ == '__main__':
    main()
