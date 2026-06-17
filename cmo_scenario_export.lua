-- cmo_scenario_export.lua
-- Command: Modern Operations (CMO) Lua export helper for combat-ID datasets.
--
-- Usage from CMO:
--   1. Open the scenario at the time step you want to sample.
--   2. Open the Lua console and run: ScenEdit_RunScript('C:/path/to/cmo_scenario_export.lua')
--   3. Optionally set CMO_COMBAT_ID_EXPORT before running this file:
--        CMO_COMBAT_ID_EXPORT = 'C:/path/to/export.jsonl'
--   4. The script exports immediately and installs/updates a repeatable CMO
--      event that reruns it every 60 seconds of scenario time. If this file is
--      not in CMO's Lua folder, set CMO_COMBAT_ID_SCRIPT_PATH to its full path
--      before step 2 so the recurring event can find it.
--
-- The script writes one JSON object per detected/contact-or-unit track. The companion
-- Python script extract_cmo_combat_id_dataset.py converts those JSONL snapshots into
-- train/validation/test splits for the combat ID training pathway.

local output_path = CMO_COMBAT_ID_EXPORT or 'cmo_combat_id_export.jsonl'
local script_path = CMO_COMBAT_ID_SCRIPT_PATH or 'cmo_scenario_export.lua'
local auto_event_enabled = CMO_COMBAT_ID_AUTO_EVENT ~= false
local auto_event_name = CMO_COMBAT_ID_EVENT_NAME or 'CMO combat-ID export every minute'
local auto_trigger_name = CMO_COMBAT_ID_TRIGGER_NAME or 'CMO combat-ID export 60s trigger'
local auto_action_name = CMO_COMBAT_ID_ACTION_NAME or 'CMO combat-ID export action'
local auto_interval_seconds = CMO_COMBAT_ID_INTERVAL_SECONDS or 60

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
    if type(key) ~= 'number' or key < 0 or key % 1 ~= 0 then
      return false
    end
    if key > max_index then max_index = key end
    count = count + 1
  end
  return max_index == count or max_index == count - 1
end

local function json_value(value, depth, seen)
  depth = depth or 0
  seen = seen or {}
  if value == nil then
    return 'null'
  end
  local value_type = type(value)
  if value_type == 'number' then
    return tostring(value)
  end
  if value_type == 'boolean' then
    return value and 'true' or 'false'
  end
  if value_type == 'table' then
    if depth >= 4 then
      return '"' .. json_escape('<max-depth>') .. '"'
    end
    if seen[value] then
      return '"' .. json_escape('<cycle>') .. '"'
    end
    seen[value] = true
    local parts = {}
    if is_array(value) then
      for _, item in pairs(value) do
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
  return '"' .. json_escape(value) .. '"'
end

local function field(object, ...)
  for _, key in ipairs({...}) do
    local ok, value = pcall(function() return object[key] end)
    if ok and value ~= nil and value ~= '' then
      return value
    end
  end
  return nil
end

local function current_time()
  local ok, now = pcall(function() return ScenEdit_CurrentTime() end)
  if ok then return now end
  return nil
end

local function emit_record_with_extra(handle, kind, side_name, object, extra)
  local record = {
    export_schema = 'cmo_combat_id_v1',
    source = 'cmo_lua',
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
    for key, value in pairs(extra) do
      record[key] = value
    end
  end

  local parts = {}
  for key, value in pairs(record) do
    table.insert(parts, '"' .. key .. '":' .. json_value(value))
  end
  handle:write('{' .. table.concat(parts, ',') .. '}\n')
end

local function emit_record(handle, kind, side_name, object)
  emit_record_with_extra(handle, kind, side_name, object, nil)
end

local function emit_child_records(handle, side_name, parent_kind, parent, child_kind, children)
  if children == nil or type(children) ~= 'table' then return end
  for _, child in pairs(children) do
    if type(child) == 'table' then
      emit_record_with_extra(handle, child_kind, side_name, child, {
        parent_record_kind = parent_kind,
        parent_guid = field(parent, 'guid', 'Guid', 'GUID'),
        parent_name = field(parent, 'name', 'Name'),
      })
    end
  end
end

local function emit_platform_records(handle, kind, side_name, object)
  emit_record(handle, kind, side_name, object)
  emit_child_records(handle, side_name, kind, object, kind .. '_sensor', field(object, 'sensors', 'Sensors'))
  emit_child_records(handle, side_name, kind, object, kind .. '_component', field(object, 'components', 'Components'))
  emit_child_records(handle, side_name, kind, object, kind .. '_emission', field(object, 'emissions', 'Emissions'))
  emit_child_records(handle, side_name, kind, object, kind .. '_last_detection', field(object, 'lastDetections', 'LastDetections'))
end

local function export_side(handle, side)
  local side_name = field(side, 'name', 'Name') or tostring(side)

  local ok_units, units = pcall(function() return ScenEdit_GetUnits({side=side_name}) end)
  if ok_units and units ~= nil then
    for _, unit in pairs(units) do
      emit_platform_records(handle, 'unit', side_name, unit)
    end
  end

  -- Contacts are the preferred combat-ID observations: they preserve the observer's
  -- current identification state instead of only the ground-truth unit attributes.
  local ok_contacts, contacts = pcall(function() return ScenEdit_GetContacts(side_name) end)
  if ok_contacts and contacts ~= nil then
    for _, contact in pairs(contacts) do
      emit_platform_records(handle, 'contact', side_name, contact)
    end
  end
end

local function install_minute_export_event()
  if not auto_event_enabled then return end
  local action_script = "CMO_COMBAT_ID_EXPORT = '" .. output_path:gsub("\\", "\\\\"):gsub("'", "\\'") .. "'\n" ..
    "CMO_COMBAT_ID_SCRIPT_PATH = '" .. script_path:gsub("\\", "\\\\"):gsub("'", "\\'") .. "'\n" ..
    "ScenEdit_RunScript(CMO_COMBAT_ID_SCRIPT_PATH)"

  pcall(function() ScenEdit_SetTrigger({mode='add', type='RegularTime', name=auto_trigger_name, interval=auto_interval_seconds}) end)
  pcall(function() ScenEdit_SetTrigger({mode='update', type='RegularTime', name=auto_trigger_name, interval=auto_interval_seconds}) end)
  pcall(function() ScenEdit_SetAction({mode='add', type='LuaScript', name=auto_action_name, ScriptText=action_script}) end)
  pcall(function() ScenEdit_SetAction({mode='update', type='LuaScript', name=auto_action_name, ScriptText=action_script}) end)
  pcall(function() ScenEdit_SetEvent(auto_event_name, {mode='add', isActive=true, isRepeatable=true}) end)
  pcall(function() ScenEdit_SetEvent(auto_event_name, {mode='update', isActive=true, isRepeatable=true}) end)
  pcall(function() ScenEdit_SetEventTrigger(auto_event_name, {mode='add', name=auto_trigger_name}) end)
  pcall(function() ScenEdit_SetEventAction(auto_event_name, {mode='add', name=auto_action_name}) end)
end

install_minute_export_event()

local handle, err = io.open(output_path, 'a')
if handle == nil then
  error('Unable to open CMO export path: ' .. tostring(err))
end

local ok_sides, sides = pcall(function() return VP_GetSides() end)
if ok_sides and sides ~= nil then
  for _, side in pairs(sides) do
    export_side(handle, side)
  end
else
  error('Unable to enumerate CMO sides with VP_GetSides()')
end

handle:close()
print('CMO combat-ID export appended to ' .. output_path .. '; recurring export event interval=' .. tostring(auto_interval_seconds) .. 's')
