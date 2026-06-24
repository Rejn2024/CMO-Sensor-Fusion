local unit = ScenEdit_GetUnit({name="Typhoon FGR.4"})
local unit_side = unit.side
local unit_guid = unit.guid
local contacts = ScenEdit_GetContacts(unit_side)
for key,value in pairs(contacts) do
  print(key)
  print(value)
  print(value.emissions)
  print(value.classificationlevel)
  print(value.detectionBy)

  local contacts_02 = ScenEdit_GetContact({side=unit.side, guid=value.guid})
  print(contacts_02.emissions)
end


name	string
age	number	Time detection held
solid	True/false	Precise type/id of emitter detected
sensor_dbid	number	Sensor detecting databse id
sensor_name	string	Sensor name
sensor_type	string	Sensor type
sensor_role	string	Sensor role
sensor_maxrange
