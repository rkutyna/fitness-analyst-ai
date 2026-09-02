"""Dump the raw XML of the first few Workout elements to see nested structure
(WorkoutStatistics attrs, WorkoutRoute/FileReference, MetadataEntry)."""
import sys, zipfile
from xml.etree import ElementTree as ET

zf = zipfile.ZipFile("export.zip")
stream = zf.open("apple_health_export/export.xml", "r")

shown = 0
want = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for event, elem in ET.iterparse(stream, events=("end",)):
    if elem.tag == "Workout":
        print("=" * 60, f"WORKOUT #{shown+1}")
        # serialize compactly
        print("ATTR:", dict(elem.attrib))
        for child in elem:
            print(f"  <{child.tag}> {dict(child.attrib)}")
            for gc in child:
                print(f"      <{gc.tag}> {dict(gc.attrib)}")
        shown += 1
        if shown >= want:
            break
        elem.clear()
stream.close()
