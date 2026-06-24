
-- Define the names of your three specific aircraft
-- "Red Fighter" GUID: Y01VZN-0HNME1NPAQTQH
-- "Blue Fighter" GUID: Y01VZN-0HNMGEHM1QV1R
-- "White Fighter" GUID: Y01VZN-0HNME1GL7A8AU

local target_aircraft = {}
local red_unit = ScenEdit_GetUnit({name="MiG-29KUB Fulcrum D"})
if red_unit then
    local red_guid = red_unit.guid
    target_aircraft[#target_aircraft+1]=red_guid
end

local blue_unit = ScenEdit_GetUnit({name="Typhoon FGR.4"})
if blue_unit then
    local blue_guid = blue_unit.guid
    target_aircraft[#target_aircraft+1]=blue_guid
end

local white_unit = ScenEdit_GetUnit({name="MiG-29 Fulcrum C"})
if white_unit then
    local white_guid = white_unit.guid
    target_aircraft[#target_aircraft+1]=white_guid
end

local function emissionString(em_table, sensor_aircraft, lat, long, head, alt, speed, target_type, classificationlevel)
    local sim_time = ScenEdit_CurrentTime()
    for _, emission in ipairs(em_table) do
        print(string.format("PY_CONTACT_LOG  Time : %s , Sensor_aircraft : %s , Emission_sensor_name : %s , Emission_age : %s , Emission_solid : %s , Emission_type : %s , Emission_role : %s , Emission_latitude : %s , Emission_longitude : %s , Emission_heading : %s , Emission_altitude : %s , Emission_speed : %s , Emission_target_type : %s , Emission_classificationlevel : %s",
                             sim_time,
                             sensor_aircraft,
                             --emission.name,
                             emission.sensor_name,
                             emission.age,
                             emission.solid,
                             emission.sensor_type,
                             emission.sensor_role,
                             lat,
                             long,
                             head,
                             alt,
                             speed,
                             target_type,
                             classificationlevel
                            ))

    end
end


-- FIX: Use the standard API function to get the current simulation date/time string
local sim_time = ScenEdit_CurrentTime()

for _, identifier in ipairs(target_aircraft) do
    local unit = ScenEdit_GetUnit({guid=identifier})

    if unit then
        local unit_side = unit.side
        local contacts = ScenEdit_GetContacts(unit_side)


        if contacts and type(contacts) == "table" and #contacts > 0 then
            for _, contact in ipairs(contacts) do
                if contact and contact.guid then
                    local sensor_aircraft = unit.name
                    local lat = contact.latitude
                    local long = contact.longitude
                    local head = contact.heading
                    local alt = contact.altitude
                    local speed = contact.speed
                    local target_type = contact.type_description
                    local classificationl = contact.classification
                    local classificationlevel = contact.classificationlevel
                    local cem = contact.emissions
                    if cem then
                    -- Format: CSV style prefixed with a unique keyword for Python to find
                        emissionString(contact.emissions, sensor_aircraft, lat, long, head, alt, speed, target_type, classificationlevel)

                    end
                end
            end
        else
            -- Clean log output when an aircraft has no sensor detections
            --print(string.format("PY_CONTACT_LOG,%s,%s,NO_CONTACTS,None,None,0", sim_time, unit.name))
        end
    else
        print("LOG_ERROR: Could not find aircraft: " .. identifier)
    end
end
