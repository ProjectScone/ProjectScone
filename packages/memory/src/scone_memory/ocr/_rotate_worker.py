"""Isolated image turning for OCR. Decodes one PNG, turns it clockwise, writes a PNG."""
from __future__ import annotations

from io import BytesIO
import sys


def main() -> None:
    from PIL import Image

    degrees, max_pixels = int(sys.argv[1]), int(sys.argv[2])
    if degrees not in (90, 180, 270) or max_pixels < 1:
        raise SystemExit(2)
    data = sys.stdin.buffer.read(64 * 1024 * 1024 + 1)
    Image.MAX_IMAGE_PIXELS = max_pixels
    with Image.open(BytesIO(data)) as image:
        if image.format != 'PNG' or image.width * image.height > max_pixels:
            raise SystemExit(3)
        # PIL turns counter-clockwise; the detection says how far to turn clockwise.
        turned = image.rotate(-degrees, expand=True)
        output = BytesIO()
        turned.save(output, format='PNG')
    sys.stdout.buffer.write(output.getvalue())


if __name__ == '__main__':
    main()
