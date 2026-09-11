"""Direct child execution with bounded output, deadlines and cancellation cleanup."""
from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
import signal
import sys
from collections.abc import Sequence

from ..core.errors import InvalidInput


def worker_environment() -> dict[str, str]:
    allowed = {'PATH', 'LANG', 'LC_ALL', 'LC_CTYPE', 'SYSTEMROOT', 'SystemRoot',
               'TMPDIR', 'TEMP', 'TMP', 'TESSDATA_PREFIX', 'OMP_THREAD_LIMIT',
               'SCONE_MEMORY_DOCUMENT_CONVERTER'}
    return {key: value for key, value in os.environ.items() if key in allowed}


def python_worker(module: str, *arguments: str) -> list[str]:
    """Use this installation's package root, never cwd or inherited PYTHONPATH."""
    package_root = str(Path(__file__).resolve().parents[2])
    bootstrap = ('import runpy,sys; '
                 f'sys.path.insert(0, {package_root!r}); '
                 'module=sys.argv.pop(1); runpy.run_module(module, run_name="__main__", alter_sys=True)')
    return [sys.executable, '-I', '-c', bootstrap, module, *arguments]


async def run_bounded(argv: Sequence[str], data: bytes, *, timeout: float, max_output: int,
                      label: str = 'document process') -> bytes:
    if not math.isfinite(timeout) or timeout <= 0 or max_output < 1:
        raise InvalidInput(f'{label} limits must be positive and finite')
    try:
        process = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            start_new_session=os.name == 'posix', env=worker_environment())
    except OSError as error:
        raise InvalidInput(f'{label} executable is unavailable; install and configure it explicitly') from error
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
                raise InvalidInput(f'{label} exceeded its output byte limit')
        return bytes(output)

    feeder = asyncio.create_task(feed())
    reader = asyncio.create_task(read())
    waiter = asyncio.create_task(process.wait())
    try:
        _, output, code = await asyncio.wait_for(asyncio.gather(feeder, reader, waiter), timeout)
        if code != 0:
            raise InvalidInput(f'{label} failed; check the configured executable, dependencies and input')
        return output
    except asyncio.TimeoutError as error:
        raise InvalidInput(f'{label} exceeded its wall time limit') from error
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
