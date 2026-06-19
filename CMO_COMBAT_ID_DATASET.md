# CMO combat-ID data extraction

This repository includes a two-stage extractor for building combat-ID training
examples from Command: Modern Operations (CMO) scenarios:

1. `cmo_scenario_export.lua` runs inside CMO and appends scenario contacts and
   units to JSONL snapshots.
2. `extract_cmo_combat_id_dataset.py` runs outside CMO and converts one or more
   snapshots into `train.jsonl`, `val.jsonl`, `test.jsonl`, `label_map.json`,
   `manifest.json`, and `all_examples.csv`.

## Detailed Lua exporter workflow

### 1. Pick a snapshot folder outside the repository

Create a writable folder for CMO exports, for example `C:/cmo_exports`. Keep
these raw scenario snapshots outside Git because large scenarios and repeated
time steps can produce many records.

### 2. Copy or reference the Lua script from CMO

CMO executes Lua through its built-in Lua console or event/action system. In
CMO, the standard Lua `dofile` helper may not be available in the console; if
you see `attempt to call a nil value (global 'dofile')`, use CMO's script
runner instead. `ScenEdit_RunScript` only loads scripts from CMO's installation
`Lua` folder, or from a subfolder beneath it, so first copy
`cmo_scenario_export.lua` into that folder. For example, copy it to a location
like:

```text
<CMO install directory>/Lua/cmo_scenario_export.lua
```

Then run it by relative Lua-folder path, not by absolute Windows path:

```lua
ScenEdit_RunScript('cmo_scenario_export.lua')
```

If you keep helper scripts in a subfolder under CMO's `Lua` folder, include only
that relative subfolder path, for example
`ScenEdit_RunScript('combat_id/cmo_scenario_export.lua')`. Do not pass the
export folder to the script runner; the export folder is configured separately
with `CMO_COMBAT_ID_EXPORT`, as shown below.

### 3. Export one scenario time step

Open the scenario, pause it at the time step you want to begin sampling, open
the Lua console, set `CMO_COMBAT_ID_EXPORT`, and run the exporter. The first run
exports immediately and installs/updates a repeatable event that reruns the
exporter every 60 seconds of scenario time. The exporter creates the CMO
`RegularTime` trigger with the documented interval string format first (for
example `60 sec`) and, if the active CMO build rejects that format, retries
other common parser formats such as `60sec`, `00:01:00`, and `60`. It updates
existing exporter triggers/actions instead of trying to add duplicates:

```lua
CMO_COMBAT_ID_EXPORT = 'C:/cmo_exports/scenario_001_t0000.jsonl'
CMO_COMBAT_ID_SCRIPT_PATH = 'cmo_scenario_export.lua'
ScenEdit_RunScript(CMO_COMBAT_ID_SCRIPT_PATH)
```

By default, every generated JSONL record is also printed to the Lua console as it is written. Set `CMO_COMBAT_ID_PRINT_JSONL = false` before running the script if you want file/key-value output without console mirroring.

When CMO exposes Lua file I/O, the exporter appends to the file named by
`CMO_COMBAT_ID_EXPORT`. If the file does not exist, CMO creates it. If it
already exists, new records are added to the end, so delete the file first when
you want a clean re-export. Some CMO Lua environments disable the standard Lua
`io` library; if you see `attempt to index a nil value (global 'io')`, the
exporter now falls back to CMO scenario key-values instead of crashing. Leave
the scenario running or advance scenario time and the installed event will
append new snapshots once per scenario minute. Set
`CMO_COMBAT_ID_AUTO_EVENT = false` before running the script if you only want a
one-off export.

### 4. Capture multiple time steps

For separate temporal runs, change the output filename and rerun the setup. The
recurring event will be updated to append subsequent one-minute snapshots to the
new file:

```lua
CMO_COMBAT_ID_EXPORT = 'C:/cmo_exports/scenario_001_t0010.jsonl'
CMO_COMBAT_ID_SCRIPT_PATH = 'cmo_scenario_export.lua'
ScenEdit_RunScript(CMO_COMBAT_ID_SCRIPT_PATH)
```

Recommended naming is `scenario_<id>_t<minutes-or-seconds>.jsonl` so each file
maps cleanly back to a scenario and time offset. Capturing separate files per
time step makes it easier to exclude bad snapshots, balance scenarios, and audit
training examples.

### 5. What the Lua exporter writes

For every side, the script attempts to export both:

- `contact` records from `ScenEdit_GetContacts(...)`, which are the preferred
  combat-ID observations because they represent what an observing side currently
  knows.
- `unit` records from `ScenEdit_GetUnits(...)`, which are useful for ground
  truth, debugging, label QA, and platform-level sensor state.

Each output line is one JSON object. Important fields include `record_kind`,
`side`, `guid`, `name`, `type`, `subtype`, `class_name`, `dbid`, `latitude`,
`longitude`, `altitude_m`, `speed_kts`, `heading_deg`, `course_deg`, `posture`,
`actual_side`, `identification_status`, `detected_by`, `scenario_time`,
`sensors`, `components`, `emissions`, `last_detections`, `doctrine`, `damage`,
`loadout`, `mounts`, `magazines`, `weapons`, and `wrapper_snapshot`. The exporter
also emits child records such as `unit_sensor`, `unit_component`,
`contact_emission`, and `contact_last_detection` so individual sensor/emitter
wrappers can be inspected directly. The nested sensor/component fields preserve
CMO-provided radar, ECM, ESM, and other emitter state when it is exposed by the
unit or contact wrapper.


### 6. Retrieve key-value fallback output when Lua `io` is unavailable

If the CMO console reports that JSONL records were stored in scenario key-values,
copy them from the console with:

```lua
local count = tonumber(ScenEdit_GetKeyValue('CMO_COMBAT_ID_EXPORT_count') or '0') or 0
for i = 1, count do
  print(ScenEdit_GetKeyValue('CMO_COMBAT_ID_EXPORT_' .. tostring(i)))
end
```

Paste the printed JSONL lines into a `.jsonl` file before running the Python
converter. Override the key prefix by setting `CMO_COMBAT_ID_KEY_PREFIX` before
running the exporter if you need multiple independent captures in the same
scenario.


### 7. Recover console-mirrored JSONL from a CMO Logs text file

If your CMO installation mirrors Lua console `print(...)` output into a `.txt`
file under its `Logs` directory, recover the console-mirrored JSONL records with:

```bash
python extract_cmo_jsonl_from_log.py \
  --input "C:/path/to/CMO/Logs/LuaHistory.txt" \
  --output C:/cmo_exports/recovered_from_log.jsonl \
  --unique
```

The scraper tolerates timestamps or other text before/after each JSON object and
only writes records that match this exporter's `cmo_combat_id_v1` schema or
`cmo_lua` source marker. Use the recovered `.jsonl` file as input to the normal
Python converter.

### 8. Verify the export before conversion

After running the Lua script, confirm that the CMO Lua console prints a message
like:

```text
CMO combat-ID export appended to C:/cmo_exports/scenario_001_t0000.jsonl
```

Open the JSONL file in a text editor and verify that it contains one JSON object
per line. Empty files usually mean the scenario has no sides/contacts visible at
that time step, the output folder is not writable, or the Lua console ran from a
different scenario state than expected.

### 9. Optional: automate snapshot collection in CMO

For larger data-generation runs, attach the same Lua command to a CMO event or
run it at regular manual pause points. Use a unique output path for each time
step or scenario branch. If you intentionally append multiple time steps to the
same file, keep an external log of which scenario times were appended because
the exporter only records the track/contact attributes it receives from CMO.

## Convert snapshots into combat-ID splits

Run the Python converter against one or more CMO JSONL exports:

```bash
python extract_cmo_combat_id_dataset.py \
  --input C:/cmo_exports/scenario_001_t0000.jsonl C:/cmo_exports/scenario_001_t0010.jsonl \
  --output-root C:/combat_id_dataset/scenario_001 \
  --drop-unknown-labels
```

By default, labels are selected from CMO fields in this priority order:
`posture`, `actual_side`, then `side`. Override that order when your combat-ID
pathway expects a different target, for example:

```bash
python extract_cmo_combat_id_dataset.py \
  --input C:/cmo_exports/*.jsonl \
  --output-root C:/combat_id_dataset/by_side \
  --label-field actual_side \
  --label-field posture
```

Each JSONL example contains:

- `label`: the combat-ID class selected from the configured label fields.
- `features`: normalized CMO attributes such as observer side, track type,
  class, DBID, latitude, longitude, altitude, speed, heading, course, and
  identification status.
- `raw`: the original exported CMO record for traceability and later feature
  expansion.

Use `label_map.json` to map class strings to integer model targets, and use the
split JSONL files as the direct inputs for the combat-ID training pathway.
