"""
Minimal shared messaging utilities for the standalone SS5 submission.

This file provides a lightweight version of the team's common.py so that
the integration-emulator scripts can run outside the full team repository.
"""

import inspect
from datetime import datetime, timezone
from queue import Empty


COLORS = {
    "system": "",
    "ss3": "",
    "ss4": "",
    "ss5": "",
    "alert": "",
}

RESET = ""
BOLD = ""


class SharedImage:
    """Compatibility shim for the team-level SharedImage object."""
    pass


def cprint(node_name, message, style=None):
    """Minimal console printer used by the standalone submission package."""
    prefix = f"[{node_name}]"
    print(f"{prefix} {message}", flush=True)


class Node:
    """
    Lightweight queue-backed message node.

    Each node owns one inbox queue and a dictionary of peer queues. Messages
    are dictionaries with type, sender, timestamp and data fields.
    """

    def __init__(self, name, inbox, peers):
        self.name = name
        self.inbox = inbox
        self.peers = peers
        self._handlers = {}

    def on(self, message_type, handler):
        """Register a handler for a message type."""
        self._handlers[message_type] = handler

    def send(self, target, message_type, data):
        """Send a typed message to another node."""
        if target not in self.peers:
            raise KeyError(f"Unknown target node: {target}")

        msg = {
            "type": message_type,
            "sender": self.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        self.peers[target].put(msg)

    async def poll(self):
        """
        Drain currently available messages from the inbox and dispatch them
        to registered handlers.
        """
        while True:
            try:
                msg = self.inbox.get_nowait()
            except Empty:
                break

            msg_type = msg.get("type")
            handler = self._handlers.get(msg_type)

            if handler is None:
                cprint(self.name, f"No handler registered for {msg_type}", "alert")
                continue

            result = handler(msg)
            if inspect.isawaitable(result):
                await result