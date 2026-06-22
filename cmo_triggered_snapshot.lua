-- cmo_triggered_snapshot.lua
-- Command: Modern Operations (CMO) triggered-event snapshot helper.
--
-- Use this script directly as the Lua action body for a CMO event, or copy it
-- into CMO's Lua folder and call it from an event action with:
--   ScenEdit_RunScript('cmo_triggered_snapshot.lua')
--
-- The standard Lua io library is intentionally not used. Instead, the script
-- assigns the same combat-ID snapshot information captured by
-- cmo_scenario_export.lua to in-memory globals and CMO scenario key-values:
--   CMO_COMBAT_ID_TRIGGER_SNAPSHOT          -- table containing this event run
--   CMO_COMBAT_ID_TRIGGER_RECORDS           -- array of captured records
--   <prefix>_snapshot_count                 -- total snapshots captured
--   <prefix>_<snapshot>_record_count        -- records in this snapshot
--   <prefix>_<snapshot>_<record>            -- JSON record for retrieval/logging
--
-- Optional settings before running the event action:
--   CMO_COMBAT_ID_TRIGGER_KEY_PREFIX = 'CMO_COMBAT_ID_TRIGGER'
--   CMO_COMBAT_ID_TRIGGER_PRINT_JSONL = true

local keyvalue_prefix = CMO_COMBAT_ID_TRIGGER_KEY_PREFIX or 'CMO_COMBAT_ID_TRIGGER'
local print_jsonl_to_console = CMO_COMBAT_ID_TRIGGER_PRINT_JSONL ~= false

local function json_escape(value)
  if value == nil then return '' end
  value = tostring(value)
  value = value:gsub('\\', '\\\\')
  value = value:gsub('"', '\\"')
  value = value:gsub('\n', '\\n')
  value = value:gsub('\r', '\\r')
  value = value:gsub('\t', '\\t')
  return value
end

local function is_array(value)
  local max_index = 0
  local count = 0
  for key, _ in pairs(value) do
    if type(key) ~= 'number' or key < 1 or key % 1 ~= 0 then
      return false
    end
    if key > max_index then max_index = key end
    count = count + 1
  end
  return max_index == count
end

local function json_value(value, depth, seen)
  depth = depth or 0
  seen = seen or {}
  if value == nil then return 'null' end

  local value_type = type(value)
  if value_type == 'number' then return tostring(value) end
  if value_type == 'boolean' then return value and 'true' or 'false' end
  if value_type ~= 'table' then return '"' .. json_escape(value) .. '"' end

  if depth >= 4 then return '"' .. json_escape('<max-depth>') .. '"' end
  if seen[value] then return '"' .. json_escape('<cycle>') .. '"' end
  seen[value] = true

  local parts = {}
  if is_array(value) then
    for _, item in ipairs(value) do
      if type(item) ~= 'function' then
        table.insert(parts, json_value(item, depth + 1, seen))
      end
    end
    seen[value] = nil
    return '[' .. table.concat(parts, ',') .. ']'
  end

  for key, item in pairs(value) do
    if type(item) ~= 'function' then
      table.insert(parts, '"' .. json_escape(key) .. '":' .. json_value(item, depth + 1, seen))
    end
  end
  seen[value] = nil
  return '{' .. table.concat(parts, ',') .. '}'
end

local function field(object, ...)
  if type(object) ~= 'table' then return nil end
  for _, key in ipairs({...}) do
    local ok, value = pcall(function() return object[key] end)
    if ok and value ~= nil and value ~= '' then return value end
  end
  return nil
end

local function current_time()
  local ok, now = pcall(function() return ScenEdit_CurrentTime() end)
  if ok then return now end
  return nil
end

local function make_record(kind, side_name, object, extra)
  local record = {
    export_schema = 'cmo_combat_id_v1',
    source = 'cmo_triggered_lua',
    record_kind = kind,
    wrapper_snapshot = object,
    scenario_time = current_time(),
    side = side_name,
    guid = field(object, 'guid', 'Guid', 'GUID'),
    name = field(object, 'name', 'Name'),
    type = field(object, 'type', 'Type'),
    subtype = field(object, 'subtype', 'SubType', 'unitType'),
    class_name = field(object, 'class', 'Class', 'classname'),
    dbid = field(object, 'dbid', 'DBID'),
    latitude = field(object, 'latitude', 'Latitude', 'lat'),
    longitude = field(object, 'longitude', 'Longitude', 'lon'),
    altitude_m = field(object, 'altitude', 'Altitude', 'alt'),
    speed_kts = field(object, 'speed', 'Speed'),
    heading_deg = field(object, 'heading', 'Heading'),
    course_deg = field(object, 'course', 'Course'),
    posture = field(object, 'posture', 'Posture'),
    actual_side = field(object, 'actualside', 'ActualSide', 'actual_side'),
    identification_status = field(object, 'identificationstatus', 'IdentificationStatus', 'identstatus'),
    detected_by = field(object, 'detectedby', 'DetectedBy'),
    detection_by = field(object, 'detectionBy', 'DetectionBy'),
    last_detections = field(object, 'lastDetections', 'LastDetections'),
    emissions = field(object, 'emissions', 'Emissions'),
    sensors = field(object, 'sensors', 'Sensors'),
    components = field(object, 'components', 'Components'),
    signature = field(object, 'signature', 'Signature'),
    doctrine = field(object, 'doctrine', 'Doctrine'),
    weather = field(object, 'weather', 'Weather'),
    damage = field(object, 'damage', 'Damage'),
    loadout = field(object, 'loadout', 'Loadout'),
    mounts = field(object, 'mounts', 'Mounts'),
    magazines = field(object, 'magazines', 'Magazines'),
    weapons = field(object, 'weapons', 'Weapons', 'weapon', 'Weapon'),
    ai_evaluate_targets_enabled = field(object, 'AI_EvaluateTargets_enabled'),
    ai_determine_primary_target_enabled = field(object, 'AI_DeterminePrimaryTarget_enabled'),
    use_custom_intermittent_emission_only = field(object, 'UseCustomIntermittentEmissionOnly'),
  }
  if extra ~= nil then
    for key, value in pairs(extra) do record[key] = value end
  end
  return record
end

local function record_to_json(record)
  local parts = {}
  for key, value in pairs(record) do
    table.insert(parts, '"' .. json_escape(key) .. '":' .. json_value(value))
  end
  return '{' .. table.concat(parts, ',') .. '}'
end


local function list_items(value)
  if type(value) ~= 'table' then return {} end

  local items = {}
  local seen = {}
  local function append(item)
    if item ~= nil and type(item) ~= 'function' and seen[item] == nil then
      table.insert(items, item)
      seen[item] = true
    end
  end

  for _, item in pairs(value) do append(item) end

  local count = field(value, 'count', 'Count')
  if tonumber(count) ~= nil then
    for index = 0, tonumber(count) do
      local ok, item = pcall(function() return value[index] end)
      if ok then append(item) end
    end
  end

  for _, container_key in ipairs({'items', 'Items', 'units', 'Units', 'contacts', 'Contacts', 'sides', 'Sides'}) do
    local container = field(value, container_key)
    if type(container) == 'table' and container ~= value then
      for _, item in ipairs(list_items(container)) do append(item) end
    end
  end

  return items
end

local function context_value(context, ...)
  if type(context) ~= 'table' then return nil end
  for _, key in ipairs({...}) do
    local value = field(context, key)
    if value ~= nil then return value end
  end
  return nil
end


local function append_unique_record(records, appended, kind, side_name, object, extra)
  if type(object) ~= 'table' or appended[object] ~= nil then return false end
  table.insert(records, make_record(kind, side_name, object, extra))
  appended[object] = true
  return true
end

local function safe_call_global(function_name)
  local fn = _G[function_name]
  if type(fn) ~= 'function' then return nil end
  local ok, value = pcall(fn)
  if ok then return value end
  return nil
end

local function side_name_for_object(object, fallback)
  local side_value = field(object, 'side', 'Side')
  if type(side_value) == 'table' then side_value = field(side_value, 'name', 'Name') end
  return side_value or field(object, 'actualside', 'ActualSide', 'actual_side') or fallback or 'EventTrigger'
end

local function append_trigger_function_records(records)
  local appended = {}

  -- In CMO event actions, ScenEdit_UnitX()/UnitX() returns the activating unit
  -- and ScenEdit_UnitY()/UnitY() returns the detecting unit for detection-style
  -- triggers. These are available even when EventContext or side sweeps are not.
  local trigger_objects = {
    {kind='event_trigger_unit', value=safe_call_global('ScenEdit_UnitX') or safe_call_global('UnitX')},
    {kind='event_detecting_unit', value=safe_call_global('ScenEdit_UnitY') or safe_call_global('UnitY')},
  }

  for _, entry in ipairs(trigger_objects) do
    local object = entry.value
    append_unique_record(records, appended, entry.kind, side_name_for_object(object), object, {event_trigger_function = entry.kind})
  end
end

local function append_event_context_records(records)
  if type(EventContext) ~= 'table' then return end

  local side_name = context_value(EventContext, 'SideName', 'sideName', 'side', 'Side')
  if type(side_name) == 'table' then side_name = field(side_name, 'name', 'Name') end
  side_name = side_name or context_value(EventContext, 'detectorSideName', 'DetectorSideName') or 'EventContext'

  local context_objects = {
    {kind='event_context_unit', value=context_value(EventContext, 'Unit', 'unit', 'SubjectUnit', 'subjectUnit', 'DetectedUnit', 'detectedUnit', 'TargetUnit', 'targetUnit')},
    {kind='event_context_contact', value=context_value(EventContext, 'Contact', 'contact', 'DetectedContact', 'detectedContact', 'TargetContact', 'targetContact')},
    {kind='event_context_detector', value=context_value(EventContext, 'Detector', 'detector', 'DetectingUnit', 'detectingUnit')},
    {kind='event_context_weapon', value=context_value(EventContext, 'Weapon', 'weapon')},
  }

  local appended = {}
  for _, entry in ipairs(context_objects) do
    append_unique_record(records, appended, entry.kind, side_name, entry.value, {event_context = EventContext})
  end

  if next(appended) == nil then
    table.insert(records, make_record('event_context', side_name, EventContext, {event_context = EventContext}))
  end
end

local function append_child_records(records, side_name, parent_kind, parent, child_kind, children)
  if children == nil or type(children) ~= 'table' then return end
  for _, child in ipairs(list_items(children)) do
    if type(child) == 'table' then
      table.insert(records, make_record(child_kind, side_name, child, {
        parent_record_kind = parent_kind,
        parent_guid = field(parent, 'guid', 'Guid', 'GUID'),
        parent_name = field(parent, 'name', 'Name'),
      }))
    end
  end
end

local function append_platform_records(records, kind, side_name, object)
  table.insert(records, make_record(kind, side_name, object, nil))
  append_child_records(records, side_name, kind, object, kind .. '_sensor', field(object, 'sensors', 'Sensors'))
  append_child_records(records, side_name, kind, object, kind .. '_component', field(object, 'components', 'Components'))
  append_child_records(records, side_name, kind, object, kind .. '_emission', field(object, 'emissions', 'Emissions'))
  append_child_records(records, side_name, kind, object, kind .. '_last_detection', field(object, 'lastDetections', 'LastDetections'))
end

local function append_side_records(records, side)
  local side_name = field(side, 'name', 'Name') or tostring(side)

  local ok_units, units = pcall(function() return ScenEdit_GetUnits({side=side_name}) end)
  if ok_units and units ~= nil then
    for _, unit in ipairs(list_items(units)) do append_platform_records(records, 'unit', side_name, unit) end
  end

  local ok_contacts, contacts = pcall(function() return ScenEdit_GetContacts(side_name) end)
  if ok_contacts and contacts ~= nil then
    for _, contact in ipairs(list_items(contacts)) do append_platform_records(records, 'contact', side_name, contact) end
  end
end

local records = {}
append_trigger_function_records(records)
append_event_context_records(records)
local ok_sides, sides = pcall(function() return VP_GetSides() end)
if (not ok_sides or sides == nil) and type(ScenEdit_GetSides) == 'function' then
  ok_sides, sides = pcall(function() return ScenEdit_GetSides() end)
end
if ok_sides and sides ~= nil then
  for _, side in ipairs(list_items(sides)) do append_side_records(records, side) end
elseif #records == 0 then
  error('Unable to enumerate CMO sides with VP_GetSides()/ScenEdit_GetSides(), and no event trigger records were available')
else
  print('WARNING: Unable to enumerate CMO sides; storing event trigger records only')
end

local snapshot_count_key = keyvalue_prefix .. '_snapshot_count'
local snapshot_index = (tonumber(ScenEdit_GetKeyValue(snapshot_count_key) or '0') or 0) + 1
local snapshot_prefix = keyvalue_prefix .. '_' .. tostring(snapshot_index)

CMO_COMBAT_ID_TRIGGER_RECORDS = records
CMO_COMBAT_ID_TRIGGER_SNAPSHOT = {
  export_schema = 'cmo_combat_id_trigger_snapshot_v1',
  source = 'cmo_triggered_lua',
  scenario_time = current_time(),
  key_prefix = keyvalue_prefix,
  snapshot_index = snapshot_index,
  record_count = #records,
  records = records,
}

ScenEdit_SetKeyValue(snapshot_count_key, tostring(snapshot_index))
ScenEdit_SetKeyValue(snapshot_prefix .. '_record_count', tostring(#records))
ScenEdit_SetKeyValue(snapshot_prefix .. '_scenario_time', tostring(CMO_COMBAT_ID_TRIGGER_SNAPSHOT.scenario_time or ''))

for index, record in ipairs(records) do
  local line = record_to_json(record)
  ScenEdit_SetKeyValue(snapshot_prefix .. '_' .. tostring(index), line)
  if print_jsonl_to_console then print(line) end
end

print('CMO triggered combat-ID snapshot assigned ' .. tostring(#records) .. ' records to CMO_COMBAT_ID_TRIGGER_RECORDS and key-values under ' .. snapshot_prefix)
