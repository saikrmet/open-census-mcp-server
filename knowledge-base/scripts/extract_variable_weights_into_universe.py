#!/usr/bin/env python
"""
Extract var→concept pairs from variable_links.ttl (hasConcept predicate)
and merge domain weights into the universe file.

Run from knowledge-base/scripts/
"""
import re, json, pathlib

HERE   = pathlib.Path(__file__).parent
KBROOT = HERE.parent

TTL    = KBROOT / "concepts/variable_links.ttl"
DEFS   = KBROOT / "concepts/subject_definitions_weighted.json"
UNI_IN = KBROOT / "2023_ACS_Enriched_Universe.json"
UNI_OUT= KBROOT / "2023_ACS_Enriched_Universe_weighted.json"

# --- 1 ▪ collect var → concept ---------------------------------
var2con = {}
with TTL.open() as fh:
    prev_var = None
    for line in fh:
        # variable line: starts with cendata:B19013_001E
        m_var = re.match(r'^cendata:(\w+)\s*$', line.strip())
        if m_var:
            prev_var = m_var.group(1)
            continue
        # concept line: indented, contains hasConcept
        m_con = re.search(r'cendata:hasConcept\s+cendata:(\w+)', line)
        if m_con and prev_var:
            var2con[prev_var] = m_con.group(1)
            prev_var = None

print(f"✅ Parsed {len(var2con):,} variable links")

# --- 2 ▪ load definition weights --------------------------------
defs = {d["concept_id"]: d["domain_weights"]
        for d in json.load(DEFS.open())["definitions"]}
print(f"✅ Loaded {len(defs):,} concepts with weights")

# --- 3 ▪ merge into universe ------------------------------------
uni = json.load(UNI_IN.open())
merged = 0
for vid, row in uni["variables"].items():
    w = defs.get(var2con.get(vid))
    if w:
        row["category_weights_linear"] = w
        merged += 1

UNI_OUT.write_text(json.dumps(uni, indent=2))
print(f"🎉 Merged weights for {merged:,} variables → {UNI_OUT}")
