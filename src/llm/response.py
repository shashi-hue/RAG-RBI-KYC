import time
from typing import Optional, Dict
from pydantic import BaseModel, Field


class CitationSource(BaseModel):
    ref_num:      int             # [1], [2] as they appear in answer text
    source:       str             # "chunks" | "annex_iv" | "appendix"
    chapter:      str
    paragraph:    Optional[str]  = None
    citation:     str             # full citation string from chunk
    status:       str             # active | amended | inserted | deleted
    score:        float
    text_snippet: str             # first 200 chars — for API response / debug


class KYCResponse(BaseModel):
    query:                  str
    answer:                 str
    citations:              list[CitationSource]
    has_deleted_provisions: bool  = False
    has_amended_provisions: bool  = False
    chunks_used:            int   = 0
    elapsed_sec:            float = 0.0
    timestamp:              str   = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S")
    )
    llm_usage: Optional[Dict[str, int]] = None

    
    def to_terminal(self) -> str:
        """Clean formatted string for CLI / logging."""
        lines = ["", "━" * 60, self.answer, ""]

        if self.has_deleted_provisions:
            lines.append("  WARNING: One or more cited provisions have been DELETED.")
        if self.has_amended_provisions:
            lines.append(" NOTE: One or more cited provisions have been AMENDED.")
        if self.has_deleted_provisions or self.has_amended_provisions:
            lines.append("")

        lines.append("Sources:")
        for c in self.citations:
            tag = f" [{c.status.upper()}]" if c.status != "active" else ""
            lines.append(f"  [{c.ref_num}] {c.citation}{tag}")

        lines += [
            "",
            f"Chunks used: {self.chunks_used}  |  "
            f"Elapsed: {self.elapsed_sec:.2f}s  |  "
            f"{self.timestamp}",
            "━" * 60,
        ]
        return "\n".join(lines)