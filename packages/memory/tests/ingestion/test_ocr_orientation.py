"""A scanned page the wrong way up, turned before it is read.

A page scanned upside down or on its side is read by Tesseract as noise:
measured on a rendered page turned 180 degrees, its text came back as
"aun Ul 8dJJO INOqUeY". Page segmentation mode 1 turns a sideways page
but not an upside-down one. With ``orientation=True`` the engine first
asks Tesseract's orientation detection how far to turn the page, turns
it in a bounded child process when the detection is confident, reads the
turned page, and gives every box back in the page as it was given, so
the result still matches the frame it came from. The engine name says
what happened: turned, left because the detection saw nothing to turn,
left because it was unsure, or left because it could not tell.
"""

from __future__ import annotations

import io
import shutil

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ocr.tesseract import TesseractOcr, parse_osd, unrotated_box

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

OSD_180 = b"Page number: 0\nOrientation in degrees: 180\nRotate: 180\nOrientation confidence: 5.96\nScript: Latin\nScript confidence: 2.10\n"


def test_the_detection_says_how_far_to_turn_and_how_sure_it_is():
    assert parse_osd(OSD_180) == (180, 5.96)
    assert parse_osd(OSD_180.replace(b"Rotate: 180", b"Rotate: 90")) == (90, 5.96)
    assert parse_osd(b"Page number: 0\nOrientation confidence: 5.96\n") is None, "no turn, nothing to act on"
    assert parse_osd(OSD_180.replace(b"Rotate: 180", b"Rotate: 45")) is None, "only quarter turns"
    assert parse_osd(b"") is None


def turned(box, degrees):
    """Where a box of the page lands once the page is turned clockwise by ``degrees``."""
    x0, y0, x1, y1 = box
    if degrees == 90:
        return (1 - y1, x0, 1 - y0, x1)
    if degrees == 180:
        return (1 - x1, 1 - y1, 1 - x0, 1 - y0)
    return (y0, 1 - x1, y1, 1 - x0)


@pytest.mark.parametrize("degrees", [90, 180, 270])
def test_a_box_read_on_the_turned_page_is_given_back_where_it_was(degrees):
    box = (0.1, 0.2, 0.35, 0.3)
    assert unrotated_box(turned(box, degrees), degrees) == pytest.approx(box)


def test_orientation_is_off_unless_asked_for_and_the_threshold_is_checked():
    assert TesseractOcr().orientation is False
    with pytest.raises(InvalidInput):
        TesseractOcr(orientation=True, min_orientation_confidence=-1.0)


def page(angle: int = 0, *, blank: bool = False) -> tuple[bytes, Image.Image]:
    image = Image.new("RGB", (1400, 400), "white")
    if not blank:
        draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=40)
        draw.text((40, 40), "The harbour crane survey found rust on the jib.", font=font, fill="black")
        draw.text((40, 110), "The slew ring needed grease before the June inspection.", font=font, fill="black")
        draw.text((40, 180), "Invoice 20931 was paid by the harbour office in June.", font=font, fill="black")
    shown = image.rotate(angle, expand=True) if angle else image
    output = io.BytesIO()
    shown.save(output, format="PNG")
    return output.getvalue(), shown


needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract is not installed")


def words(result) -> str:
    return " ".join(region.text for region in result.regions)


@needs_tesseract
@pytest.mark.parametrize("angle, degrees", [(180, 180), (90, 90)], ids=["upside-down", "on-its-side"])
async def test_a_turned_page_is_read_upright_with_its_boxes_in_the_page_as_given(angle, degrees):
    upright_png, _ = page()
    turned_png, shown = page(angle)
    plain = await TesseractOcr().recognize(turned_png)
    assert "harbour crane survey" not in words(plain), "the fixture must actually defeat an unturned read"
    result = await TesseractOcr(orientation=True).recognize(turned_png)
    assert "harbour crane survey" in words(result)
    assert (result.width, result.height) == shown.size and result.engine.endswith(f":rotated{degrees}")
    upright = await TesseractOcr().recognize(upright_png)
    expected = next(region.box for region in upright.regions if region.text == "crane")
    found = next(region.box for region in result.regions if region.text == "crane")
    # PIL turns counter-clockwise, so the page as given is the upright one turned clockwise by 360 - angle.
    assert found == pytest.approx(turned(expected, (360 - angle) % 360), abs=0.02)


@needs_tesseract
async def test_an_upright_page_is_left_and_says_so():
    upright_png, _ = page()
    result = await TesseractOcr(orientation=True).recognize(upright_png)
    assert "harbour crane survey" in words(result) and result.engine.endswith(":osd-upright")


@needs_tesseract
async def test_a_page_the_detection_cannot_judge_is_read_as_given():
    blank_png, _ = page(blank=True)
    result = await TesseractOcr(orientation=True).recognize(blank_png)
    assert result.regions == () and result.engine.endswith(":osd-unknown")


@needs_tesseract
async def test_an_unsure_detection_leaves_the_page_as_given():
    turned_png, _ = page(180)
    result = await TesseractOcr(orientation=True, min_orientation_confidence=1000.0).recognize(turned_png)
    assert result.engine.endswith(":osd-unsure") and "harbour crane survey" not in words(result)


def fake_tesseract(tmp_path, languages: str):
    """A stand-in executable: orientation detection fails, the language list is ``languages``,
    and recognition finds no words."""
    script = tmp_path / "tesseract"
    header = "level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num\\tleft\\ttop\\twidth\\theight\\tconf\\ttext"
    script.write_text("#!/bin/sh\n"
                      'case "$*" in\n'
                      f'  *--list-langs*) printf "List of available languages (2):\\n{languages}\\n" ;;\n'
                      '  *"--psm 0"*) cat >/dev/null; exit 1 ;;\n'
                      f'  *) cat >/dev/null; printf "{header}\\n" ;;\n'
                      "esac\n")
    script.chmod(0o755)
    return str(script)


async def test_missing_orientation_data_is_said_rather_than_read_as_an_unjudgeable_page(tmp_path):
    png, _ = page(blank=True)
    with pytest.raises(InvalidInput, match="osd"):
        await TesseractOcr(executable=fake_tesseract(tmp_path, "eng"), orientation=True).recognize(png)
    result = await TesseractOcr(executable=fake_tesseract(tmp_path, "eng\\nosd"), orientation=True).recognize(png)
    assert result.engine.endswith(":osd-unknown"), "with the data installed, a failed detection is a page it cannot judge"


def test_a_language_list_too_long_to_name_with_orientation_is_refused_up_front():
    many = "+".join(["eng"] * 8)
    # 79 characters: the engine name fits in its 96 without the orientation suffix, not with it.
    long_name = "+".join(f"lang{n:02d}xxxxxxxxx" for n in range(5))
    assert len(long_name) == 79
    TesseractOcr(language=many, orientation=True)
    TesseractOcr(language=long_name)
    with pytest.raises(InvalidInput, match="name"):
        TesseractOcr(language=long_name, orientation=True)
