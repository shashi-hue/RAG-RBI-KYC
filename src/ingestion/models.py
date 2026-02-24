from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Footnote:
    fn_num:       str
    action:       str            # Inserted | Amended | Deleted | Substituted
    ref:          Optional[str]  # circular number OR gazette ref
    date:         Optional[str]  # raw date string e.g. "April 28, 2023"
    deleted_text: Optional[str]  # original text if deleted/amended
    shifted_to:   Optional[str]  # para number if "shifted to paragraph X"

    def to_citation(self) -> str:
        parts = [self.action]
        if self.ref:
            parts.append(f"vide {self.ref}")
        if self.date:
            parts.append(f"dated {self.date}")
        return " ".join(parts)

    def to_dict(self):
        return asdict(self)


@dataclass
class KYCChunk:
    chunk_id:       str
    chapter:        str          
    chapter_title:  str           
    part:           Optional[str] 
    paragraph:      Optional[str] 
    page:           int
    text:           str          
    embed_text:     str          
    status:         str          
    historical_text: Optional[str]
    footnotes:      list          
    citation:       str           

    def to_dict(self):
        d = asdict(self)
        return d


@dataclass
class TableChunk:
    chunk_id:    str
    source:      str   
    row_label:   str   
    row_data:    dict  
    embed_text:  str   
    citation:    str
    footnotes:   list  

    def to_dict(self):
        return asdict(self)