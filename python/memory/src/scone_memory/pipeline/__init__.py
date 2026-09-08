"""Composable real-time stages.

A pipeline is an ordered list of stages. A stage is anything with
``handle(frame, emit)``: it reads a frame and sends whatever should
continue. Data goes exactly where a stage sends it, downstream through
``emit(frame)`` or back to the stages before it through ``emit.up(frame)``;
announcements (start, stop, interruption, failure) reach every stage,
because they are about the run rather than about a turn.

Interruption is a turn number, not a queue-jumping frame: every delivery
carries the turn it belongs to, and a frame of a turn that has been cut
off is dropped where it is found, including in a slow stage's backlog.
That is one integer instead of a priority queue in every stage.

A stage runs inline unless it sets ``buffered = True``, which gives it a
queue and a task of its own, so only the stages that wait on a network
or a model cost a context switch.
"""

from .core import (DOWN, DROPPED, FAILED, HANDLED, UP, Delivery, Emit, Failed, Feed, Interrupted,
                   Observer, Pipeline, Stage, Started, Stopped)

__all__ = ["DOWN", "DROPPED", "FAILED", "HANDLED", "UP", "Delivery", "Emit", "Failed", "Feed", "Interrupted",
           "Observer", "Pipeline", "Stage", "Started", "Stopped"]
