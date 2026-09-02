"""Inspect the REAL export.xml (streamed from the zip) before writing the parser.

Reports:
  - distinct Record types + counts + a sample element for each
  - distinct units seen per type
  - Workout attributes + counts by type + a sample
  - the sleep category record shape (HKCategoryTypeIdentifierSleepAnalysis)
  - a couple of raw sample lines so we see real attribute strings/offsets
Reads incrementally with iterparse + elem.clear(); never loads the whole tree.
"""
import io
import sys
import zipfile
from collections import Counter, defaultdict
from xml.etree import ElementTree as ET

ZIP_PATH = sys.argv[1] if len(sys.argv) > 1 else "export.zip"
INNER = "apple_health_export/export.xml"

record_type_counts = Counter()
record_type_units = defaultdict(set)
record_samples = {}
workout_type_counts = Counter()
workout_attrs = set()
workout_sample = None
sleep_value_counts = Counter()
sleep_samples = []

zf = zipfile.ZipFile(ZIP_PATH)
stream = zf.open(INNER, "r")

n = 0
for event, elem in ET.iterparse(stream, events=("end",)):
    tag = elem.tag
    if tag == "Record":
        rtype = elem.get("type", "?")
        record_type_counts[rtype] += 1
        u = elem.get("unit")
        if u:
            record_type_units[rtype].add(u)
        if rtype not in record_samples:
            record_samples[rtype] = dict(elem.attrib)
        if rtype == "HKCategoryTypeIdentifierSleepAnalysis":
            sleep_value_counts[elem.get("value", "?")] += 1
            if len(sleep_samples) < 3:
                sleep_samples.append(dict(elem.attrib))
        elem.clear()
    elif tag == "Workout":
        workout_type_counts[elem.get("workoutActivityType", "?")] += 1
        workout_attrs.update(elem.attrib.keys())
        if workout_sample is None:
            # capture nested children names too
            children = Counter(c.tag for c in elem)
            workout_sample = (dict(elem.attrib), dict(children))
        elem.clear()
    n += 1
    if n % 2_000_000 == 0:
        print(f"... {n:,} elements parsed", file=sys.stderr)

stream.close()

print("=" * 70)
print(f"TOTAL elements parsed: {n:,}")
print(f"Distinct Record types: {len(record_type_counts)}")
print(f"Total Record rows: {sum(record_type_counts.values()):,}")
print(f"Total Workout rows: {sum(workout_type_counts.values()):,}")
print("=" * 70)

print("\n### RECORD TYPES (count, units) ###")
for rtype, cnt in record_type_counts.most_common():
    units = ",".join(sorted(record_type_units.get(rtype, {"-"}))) or "-"
    print(f"{cnt:>10,}  {rtype}   units=[{units}]")

print("\n### SAMPLE RECORD ELEMENTS (first seen of each type) ###")
for rtype, attrs in record_samples.items():
    print(f"\n-- {rtype}")
    for k, v in attrs.items():
        print(f"     {k} = {v!r}")

print("\n### WORKOUTS ###")
print("attrs:", sorted(workout_attrs))
print("by activity type:")
for wt, cnt in workout_type_counts.most_common():
    print(f"  {cnt:>6,}  {wt}")
if workout_sample:
    print("sample workout attrib:", workout_sample[0])
    print("sample workout child tags:", workout_sample[1])

print("\n### SLEEP (HKCategoryTypeIdentifierSleepAnalysis) ###")
print("value distribution:", dict(sleep_value_counts))
for s in sleep_samples:
    print("  sample:", s)
