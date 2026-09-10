"""Direct child execution with bounded output, deadlines and cancellation cleanup."""
from __future__ import annotations

import asyncio
import math
import os
import signal
from collections.abc import Sequence

from ..core.errors import InvalidInput


async def run_bounded(argv: Sequence[str], data: bytes, *, timeout: float, max_output: int) -> bytes:
    if not math.isfinite(timeout) or timeout <= 0 or max_output < 1:
        raise InvalidInput('OCR process limits must be positive and finite')
    try:
        process = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            start_new_session=os.name == 'posix')
    except OSError as error:
        raise InvalidInput('OCR executable is unavailable; install and configure it explicitly') from error
    assert process.stdin is not None and process.stdout is not None
    stdin, stdout = process.stdin, process.stdout

    async def feed() -> None:
        try:
            stdin.write(data)
            await stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            stdin.close()

    async def read() -> bytes:
        output = bytearray()
        while block := await stdout.read(min(65536, max_output + 1 - len(output))):
            output.extend(block)
            if len(output) > max_output:
                raise InvalidInput('OCR process exceeded its output byte limit')
        return bytes(output)

    feeder = asyncio.create_task(feed())
    reader = asyncio.create_task(read())
    waiter = asyncio.create_task(process.wait())
    try:
        _, output, code = await asyncio.wait_for(asyncio.gather(feeder, reader, waiter), timeout)
        if code != 0:
            raise InvalidInput('OCR process failed; check the configured executable, language data and input')
        return output
    except asyncio.TimeoutError as error:
        raise InvalidInput('OCR process exceeded its wall time limit') from error
    finally:
        try:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGKILL)
            elif process.returncode is None:
                process.kill()
        except ProcessLookupError:
            pass
        for task in (feeder, reader, waiter):
            if not task.done():
                task.cancel()
        await asyncio.gather(feeder, reader, waiter, return_exceptions=True)
        await process.wait()
