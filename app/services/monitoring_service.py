"""In-memory runtime metrics for prediction requests."""

from __future__ import annotations

from threading import Lock


class RuntimeMonitoringService:
    """Collect lightweight, process-local prediction endpoint metrics."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._prediction_request_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0
        self._last_request_latency_ms: float | None = None

    def record_prediction_request(self, latency_ms: float, is_error: bool) -> None:
        """Record the outcome and duration of one ``/predict`` request."""

        with self._lock:
            self._prediction_request_count += 1
            self._total_latency_ms += latency_ms
            self._last_request_latency_ms = latency_ms
            if is_error:
                self._error_count += 1

    def get_metrics(self) -> dict[str, int | float | None]:
        """Return a snapshot of the currently collected runtime metrics."""

        with self._lock:
            request_count = self._prediction_request_count
            average_latency = (
                self._total_latency_ms / request_count if request_count else 0.0
            )
            return {
                "prediction_request_count": request_count,
                "error_count": self._error_count,
                "average_request_latency_ms": round(average_latency, 3),
                "last_request_latency_ms": (
                    round(self._last_request_latency_ms, 3)
                    if self._last_request_latency_ms is not None
                    else None
                ),
            }


monitoring_service = RuntimeMonitoringService()
