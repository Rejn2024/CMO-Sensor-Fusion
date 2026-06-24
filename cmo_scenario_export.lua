-- cmo_scenario_export.lua
-- Command: Modern Operations (CMO) Lua export helper for combat-ID datasets.
--
-- Usage from CMO:
--   1. Open the scenario at the time step you want to sample.
--   2. Copy this file into CMO's Lua folder, then run from the Lua console:
--        ScenEdit_RunScript('cmo_scenario_export.lua')
--   3. Optionally set CMO_COMBAT_ID_EXPORT before running this file:
--        CMO_COMBAT_ID_EXPORT = 'C:/path/to/export.jsonl'
--   4. The script exports immediately and installs/updates a repeatable CMO
--      event that reruns it every 60 seconds of scenario time.
--
-- Important: ScenEdit_RunScript only loads files from CMO's Lua folder (or a
-- subfolder beneath it). CMO_COMBAT_ID_SCRIPT_PATH should therefore be a path
-- relative to that Lua folder, not an absolute Windows path.
--
-- The script writes one JSON object per detected/contact-or-unit track. The companion
-- Python script extract_cmo_combat_id_dataset.py converts those JSONL snapshots into
-- train/validation/test splits for the combat ID training pathway.

local output_path = CMO_COMBAT_ID_EXPORT or 'C:/Users/theon/CMO-Sensor-Fusion/CMO_Data_Exports/scenario_001_t0000.jsonl'
local keyvalue_prefix = CMO_COMBAT_ID_KEY_PREFIX or 'CMO_COMBAT_ID_EXPORT'
local script_path = CMO_COMBAT_ID_SCRIPT_PATH or 'cmo_scenario_export.lua'
local auto_event_enabled = CMO_COMBAT_ID_AUTO_EVENT ~= false
local auto_event_name = CMO_COMBAT_ID_EVENT_NAME or 'CMO combat-ID export every minute'
local auto_trigger_name = CMO_COMBAT_ID_TRIGGER_NAME or 'CMO combat-ID export 60s trigger'
local auto_action_name = CMO_COMBAT_ID_ACTION_NAME or 'CMO combat-ID export action'
local auto_interval_seconds = tonumber(CMO_COMBAT_ID_INTERVAL_SECONDS or 60) or 60
local print_jsonl_to_console = CMO_COMBAT_ID_PRINT_JSONL ~= false

local function is_absolute_windows_path(path)
  return type(path) == 'string' and (path:match('^%a:[/\\]') ~= nil or path:match('^[/\\][/\\]') ~= nil)
end

local function warn_if_runscript_path_is_absolute()
  if is_absolute_windows_path(script_path) then
    print('WARNING: CMO_COMBAT_ID_SCRIPT_PATH should be relative to CMO Lua folder for ScenEdit_RunScript: ' .. script_path)
  end
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

local function lua_file_io_available()
  return type(io) == 'table' and type(io.open) == 'function'
end

local function open_export_writer(path)
  if lua_file_io_available() then
    local file_handle, err = io.open(path, 'a')
    if file_handle == nil then
      error('Unable to open CMO export path: ' .. tostring(err))
    end
    return {
      mode = 'file',
      write = function(_, line)
        file_handle:write(line)
        if print_jsonl_to_console then print(line) end
      end,
      close = function() file_handle:close() end,
    }
  end

  local count_key = keyvalue_prefix .. '_count'
  local existing_count = tonumber(ScenEdit_GetKeyValue(count_key) or '0') or 0
  local next_index = existing_count
  print('WARNING: Lua io library is unavailable in this CMO environment; writing JSONL records to scenario key-values with prefix ' .. keyvalue_prefix)
  return {
    mode = 'keyvalue',
    write = function(_, line)
      next_index = next_index + 1
      ScenEdit_SetKeyValue(keyvalue_prefix .. '_' .. tostring(next_index), line)
      ScenEdit_SetKeyValue(count_key, tostring(next_index))
      if print_jsonl_to_console then print(line) end
    end,
    close = function() end,
  }
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

local function cmo_list_contains(list_value, expected_name)
  if type(list_value) ~= 'table' then return false end
  for _, item in pairs(list_value) do
    if type(item) == 'table' then
      local item_name = field(item, 'name', 'Name', 'description', 'Description')
      if item_name == expected_name then return true end
    elseif item == expected_name then
      return true
    end
  end
  return false
end

local function cmo_object_exists(list_function, object_name)
  local ok, list_value = pcall(list_function)
  return ok and cmo_list_contains(list_value, object_name)
end

local function cmo_regular_time_interval_candidates(interval_seconds)
  local seconds = math.floor(tonumber(interval_seconds) or 60)
  if seconds < 1 then seconds = 1 end

  local minutes = math.floor(seconds / 60)
  local remaining_seconds = seconds - (minutes * 60)
  local hours = math.floor(minutes / 60)
  minutes = minutes - (hours * 60)

  return {
    {display=tostring(seconds) .. ' (number)', value=seconds},
    {display=tostring(seconds) .. ' sec', value=tostring(seconds) .. ' sec'},
    {display=tostring(seconds) .. 'sec', value=tostring(seconds) .. 'sec'},
    {display=tostring(seconds) .. ' second', value=tostring(seconds) .. ' second'},
    {display=tostring(seconds) .. ' seconds', value=tostring(seconds) .. ' seconds'},
    {display=string.format('%02d:%02d:%02d', hours, minutes, remaining_seconds), value=string.format('%02d:%02d:%02d', hours, minutes, remaining_seconds)},
    {display=string.format('0.%02d:%02d:%02d', hours, minutes, remaining_seconds), value=string.format('0.%02d:%02d:%02d', hours, minutes, remaining_seconds)},
    {display=tostring(seconds), value=tostring(seconds)},
  }
end

local function cmo_upsert_trigger(trigger_name, interval_seconds)
  local mode = 'add'
  if cmo_object_exists(function() return ScenEdit_SetTrigger({mode='list'}) end, trigger_name) then
    mode = 'update'
  end

  local errors = {}
  for _, interval_candidate in ipairs(cmo_regular_time_interval_candidates(interval_seconds)) do
    local ok, trigger = pcall(function()
      return ScenEdit_SetTrigger({
        mode=mode,
        type='RegularTime',
        name=trigger_name,
        description=trigger_name,
        interval=interval_candidate.value,
      })
    end)
    if ok and trigger ~= nil then
      print('CMO combat-ID export recurring trigger interval set to ' .. interval_candidate.display)
      return trigger, interval_candidate.display
    end
    local error_text = ok and tostring(_errmsg_ or 'unknown error') or tostring(trigger)
    table.insert(errors, interval_candidate.display .. ' => ' .. error_text)
  end

  print('WARNING: Unable to install/update CMO combat-ID recurring trigger; tried intervals: ' .. table.concat(errors, '; '))
  return nil, nil
end

local function cmo_upsert_action(action_name, action_script)
  local mode = 'add'
  if cmo_object_exists(function() return ScenEdit_SetAction({mode='list'}) end, action_name) then
    mode = 'update'
  end
  return ScenEdit_SetAction({mode=mode, type='LuaScript', name=action_name, description=action_name, ScriptText=action_script})
end

local function cmo_upsert_event(event_name)
  local mode = 'add'
  local ok, existing_event = pcall(function() return ScenEdit_GetEvent(event_name, 4) end)
  if ok and existing_event ~= nil then mode = 'update' end
  return ScenEdit_SetEvent(event_name, {mode=mode, isActive=true, isRepeatable=true})
end

local function cmo_event_link_contains(event_value, level_name, expected_name)
  if type(event_value) ~= 'table' then return false end
  return cmo_list_contains(event_value[level_name] or event_value[level_name:sub(1, 1):upper() .. level_name:sub(2)], expected_name)
end

local function cmo_add_event_links_if_missing(event_name, trigger_name, action_name)
  local ok, event_details = pcall(function() return ScenEdit_GetEvent(event_name, 0) end)
  if not ok then event_details = nil end
  if not cmo_event_link_contains(event_details, 'triggers', trigger_name) then
    ScenEdit_SetEventTrigger(event_name, {mode='add', name=trigger_name, description=trigger_name})
  end
  if not cmo_event_link_contains(event_details, 'actions', action_name) then
    ScenEdit_SetEventAction(event_name, {mode='add', name=action_name, description=action_name})
  end
end

local function install_minute_export_event()
  if not auto_event_enabled then return end
  local action_script = "CMO_COMBAT_ID_EXPORT = '" .. output_path:gsub("\\", "\\\\"):gsub("'", "\\'") .. "'\r\n" ..
    "CMO_COMBAT_ID_KEY_PREFIX = '" .. keyvalue_prefix:gsub("\\", "\\\\"):gsub("'", "\\'") .. "'\r\n" ..
    "CMO_COMBAT_ID_INTERVAL_SECONDS = " .. tostring(auto_interval_seconds) .. "\r\n" ..
    "CMO_COMBAT_ID_PRINT_JSONL = " .. tostring(print_jsonl_to_console) .. "\r\n" ..
    "CMO_COMBAT_ID_SCRIPT_PATH = '" .. script_path:gsub("\\", "\\\\"):gsub("'", "\\'") .. "'\r\n" ..
    "ScenEdit_RunScript(CMO_COMBAT_ID_SCRIPT_PATH)"

  local trigger = cmo_upsert_trigger(auto_trigger_name, auto_interval_seconds)
  if trigger == nil then return end
  cmo_upsert_action(auto_action_name, action_script)
  cmo_upsert_event(auto_event_name)
  cmo_add_event_links_if_missing(auto_event_name, auto_trigger_name, auto_action_name)
end

warn_if_runscript_path_is_absolute()
install_minute_export_event()

local handle = open_export_writer(output_path)

local ok_sides, sides = pcall(function() return VP_GetSides() end)
if ok_sides and sides ~= nil then
  for _, side in pairs(sides) do
    export_side(handle, side)
  end
else
  error('Unable to enumerate CMO sides with VP_GetSides()')
end

handle:close()
if handle.mode == 'file' then
  print('CMO combat-ID export appended to ' .. output_path .. '; recurring export event interval=' .. tostring(auto_interval_seconds) .. 's')
else
  print('CMO combat-ID export stored in scenario key-values with prefix ' .. keyvalue_prefix .. '; recurring export event interval=' .. tostring(auto_interval_seconds) .. 's')
end
