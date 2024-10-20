# Dataclass for Processing Results
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ProcessingResult:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
