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
local debug_enabled = CMO_COMBAT_ID_TRIGGER_DEBUG ~= false
local debug_messages = {}

local function debug(message)
  if not debug_enabled then return end
  local line = 'CMO_TRIGGER_DEBUG: ' .. tostring(message)
  table.insert(debug_messages, line)
  print(line)
end

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
  debug('ScenEdit_CurrentTime failed: ' .. tostring(now))
  return nil
end

local function table_count(value)
  if type(value) ~= 'table' then return 0 end
  local count = 0
  for _, _ in pairs(value) do count = count + 1 end
  return count
end

local function safe_get_key_value(key)
  if type(ScenEdit_GetKeyValue) ~= 'function' then
    debug('ScenEdit_GetKeyValue is unavailable while reading ' .. tostring(key))
    return nil
  end
  local ok, value = pcall(function() return ScenEdit_GetKeyValue(key) end)
  if not ok then
    debug('ScenEdit_GetKeyValue failed for ' .. tostring(key) .. ': ' .. tostring(value))
    return nil
  end
  return value
end

local function safe_set_key_value(key, value)
  if type(ScenEdit_SetKeyValue) ~= 'function' then
    debug('ScenEdit_SetKeyValue is unavailable while writing ' .. tostring(key))
    return false
  end
  local ok, err = pcall(function() return ScenEdit_SetKeyValue(key, value) end)
  if not ok then
    debug('ScenEdit_SetKeyValue failed for ' .. tostring(key) .. ': ' .. tostring(err))
    return false
  end
  return true
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

local function append_child_records(records, side_name, parent_kind, parent, child_kind, children)
  if children == nil or type(children) ~= 'table' then return end
  for _, child in pairs(children) do
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
  local before_count = #records
  debug('Inspecting side "' .. tostring(side_name) .. '"')

  if type(ScenEdit_GetUnits) ~= 'function' then
    debug('ScenEdit_GetUnits is unavailable for side "' .. tostring(side_name) .. '"')
  else
    local ok_units, units = pcall(function() return ScenEdit_GetUnits({side=side_name}) end)
    if ok_units and units ~= nil then
      debug('ScenEdit_GetUnits returned ' .. tostring(table_count(units)) .. ' units for side "' .. tostring(side_name) .. '"')
      for _, unit in pairs(units) do append_platform_records(records, 'unit', side_name, unit) end
    elseif ok_units then
      debug('ScenEdit_GetUnits returned nil for side "' .. tostring(side_name) .. '"')
    else
      debug('ScenEdit_GetUnits failed for side "' .. tostring(side_name) .. '": ' .. tostring(units))
    end
  end

  if type(ScenEdit_GetContacts) ~= 'function' then
    debug('ScenEdit_GetContacts is unavailable for side "' .. tostring(side_name) .. '"')
  else
    local ok_contacts, contacts = pcall(function() return ScenEdit_GetContacts(side_name) end)
    if ok_contacts and contacts ~= nil then
      debug('ScenEdit_GetContacts returned ' .. tostring(table_count(contacts)) .. ' contacts for side "' .. tostring(side_name) .. '"')
      for _, contact in pairs(contacts) do append_platform_records(records, 'contact', side_name, contact) end
    elseif ok_contacts then
      debug('ScenEdit_GetContacts returned nil for side "' .. tostring(side_name) .. '"')
    else
      debug('ScenEdit_GetContacts failed for side "' .. tostring(side_name) .. '": ' .. tostring(contacts))
    end
  end

  debug('Side "' .. tostring(side_name) .. '" added ' .. tostring(#records - before_count) .. ' records')
end

local function run_snapshot()
  debug('Triggered snapshot script started')
  debug('Configuration prefix=' .. tostring(keyvalue_prefix) .. ', print_jsonl=' .. tostring(print_jsonl_to_console))
  debug('API availability VP_GetSides=' .. tostring(type(VP_GetSides)) .. ', ScenEdit_GetUnits=' .. tostring(type(ScenEdit_GetUnits)) .. ', ScenEdit_GetContacts=' .. tostring(type(ScenEdit_GetContacts)) .. ', ScenEdit_SetKeyValue=' .. tostring(type(ScenEdit_SetKeyValue)))

  local records = {}
  local fatal_error = nil
  if type(VP_GetSides) ~= 'function' then
    fatal_error = 'VP_GetSides is unavailable'
    debug(fatal_error)
  else
    local ok_sides, sides = pcall(function() return VP_GetSides() end)
    if ok_sides and sides ~= nil then
      debug('VP_GetSides returned ' .. tostring(table_count(sides)) .. ' sides')
      for _, side in pairs(sides) do append_side_records(records, side) end
    elseif ok_sides then
      fatal_error = 'VP_GetSides returned nil'
      debug(fatal_error)
    else
      fatal_error = 'VP_GetSides failed: ' .. tostring(sides)
      debug(fatal_error)
    end
  end

  local snapshot_count_key = keyvalue_prefix .. '_snapshot_count'
  local snapshot_index = (tonumber(safe_get_key_value(snapshot_count_key) or '0') or 0) + 1
  local snapshot_prefix = keyvalue_prefix .. '_' .. tostring(snapshot_index)

  CMO_COMBAT_ID_TRIGGER_RECORDS = records
  CMO_COMBAT_ID_TRIGGER_DEBUG_LOG = debug_messages
  CMO_COMBAT_ID_TRIGGER_SNAPSHOT = {
    export_schema = 'cmo_combat_id_trigger_snapshot_v1',
    source = 'cmo_triggered_lua',
    scenario_time = current_time(),
    key_prefix = keyvalue_prefix,
    snapshot_index = snapshot_index,
    record_count = #records,
    fatal_error = fatal_error,
    debug_log = debug_messages,
    records = records,
  }

  safe_set_key_value(snapshot_count_key, tostring(snapshot_index))
  safe_set_key_value(snapshot_prefix .. '_record_count', tostring(#records))
  safe_set_key_value(snapshot_prefix .. '_scenario_time', tostring(CMO_COMBAT_ID_TRIGGER_SNAPSHOT.scenario_time or ''))
  if fatal_error ~= nil then
    safe_set_key_value(snapshot_prefix .. '_fatal_error', fatal_error)
  end

  for index, record in ipairs(records) do
    local line = record_to_json(record)
    safe_set_key_value(snapshot_prefix .. '_' .. tostring(index), line)
    if print_jsonl_to_console then print(line) end
  end

  safe_set_key_value(snapshot_prefix .. '_debug_count', tostring(#debug_messages))
  for index, message in ipairs(debug_messages) do
    safe_set_key_value(snapshot_prefix .. '_debug_' .. tostring(index), message)
  end

  debug('Triggered snapshot script finished with ' .. tostring(#records) .. ' records under ' .. snapshot_prefix)
  safe_set_key_value(snapshot_prefix .. '_debug_count', tostring(#debug_messages))
  safe_set_key_value(snapshot_prefix .. '_debug_' .. tostring(#debug_messages), debug_messages[#debug_messages] or '')
  print('CMO triggered combat-ID snapshot assigned ' .. tostring(#records) .. ' records to CMO_COMBAT_ID_TRIGGER_RECORDS and key-values under ' .. snapshot_prefix)
end

local ok, err = pcall(run_snapshot)
if not ok then
  debug('UNEXPECTED ERROR: ' .. tostring(err))
  CMO_COMBAT_ID_TRIGGER_DEBUG_LOG = debug_messages
  safe_set_key_value(keyvalue_prefix .. '_last_error', tostring(err))
  safe_set_key_value(keyvalue_prefix .. '_last_debug_count', tostring(#debug_messages))
  for index, message in ipairs(debug_messages) do
    safe_set_key_value(keyvalue_prefix .. '_last_debug_' .. tostring(index), message)
  end
  error(err)
end
