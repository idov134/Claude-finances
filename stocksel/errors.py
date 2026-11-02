class LeakageError(RuntimeError):
    """Raised when an operation would use information not available at decision time."""


class HoldoutError(RuntimeError):
    """Raised when the one-shot holdout would be touched or re-opened."""


class KillSwitchTriggered(RuntimeError):
    """Raised by live components when the kill switch is active."""
