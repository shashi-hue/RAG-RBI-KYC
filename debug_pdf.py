# debug_annex.py — run from project root
import pdfplumber

PDF = "data/raw/KYC_Directions_2025.pdf"

in_annex_iv   = False
all_data_rows = []

with pdfplumber.open(PDF) as pdf:
    for pg_num, page in enumerate(pdf.pages, start=1):
        text = (page.extract_text() or "").lower()

        has_annex_iv  = "annex iv" in text
        has_appendix  = "appendix" in text
        has_fpi       = "eligible foreign investors" in text

        # Print EVERY page's trigger state
        if has_annex_iv or has_appendix or in_annex_iv:
            print(f"Page {pg_num:3d} | in_annex={in_annex_iv} | annex_iv={has_annex_iv} | appendix={has_appendix} | fpi={has_fpi}")

        if has_annex_iv:
            in_annex_iv = True

        if in_annex_iv and has_appendix and not has_annex_iv:
            print(f"  --> BREAK triggered on page {pg_num}")
            break

        if not in_annex_iv:
            continue

        if has_fpi:
            print(f"  --> SKIPPED (eligible foreign investors)")
            continue

        tables = page.extract_tables()
        print(f"  --> COLLECTING: {len(tables)} tables")
        for t in tables:
            if t:
                all_data_rows.extend(t)

print(f"\nTotal rows collected: {len(all_data_rows)}")
for i, row in enumerate(all_data_rows[:10]):
    print(f"  row[{i}]: {row}")

# Show if header is found
for i, row in enumerate(all_data_rows):
    joined = " ".join(str(c) for c in row if c).lower()
    if "category i" in joined and "category ii" in joined:
        print(f"\nHeader found at index {i}: {row}")
        break
else:
    print("\nNO HEADER FOUND")
