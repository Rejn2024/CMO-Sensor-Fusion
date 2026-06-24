-- Define the names of your three specific aircraft
-- "Red Fighter" GUID: Y01VZN-0HNME1NPAQTQH
-- "Blue Fighter" GUID: Y01VZN-0HNMGEHM1QV1R
-- "White Fighter" GUID: Y01VZN-0HNME1GL7A8AU
local target_aircraft = {"Y01VZN-0HNME1NPAQTQH", "Y01VZN-0HNMGEHM1QV1R", "Y01VZN-0HNME1GL7A8AU"}


-- FIX: Use the standard API function to get the current simulation date/time string
local sim_time = ScenEdit_CurrentTime()

for _, identifier in ipairs(target_aircraft) do
    local unit = ScenEdit_GetUnit({guid=identifier})

    if unit then
        local unit_side = unit.side
        local contacts = unit.contacts

        if contacts and type(contacts) == "table" and #contacts > 0 then
            for _, contact in ipairs(contacts) do
                if contact and contact.guid then
                    local c_details = ScenEdit_GetContact({side=unit.side, guid=contact.guid})

                    if c_details then
                        -- Format: CSV style prefixed with a unique keyword for Python to find
                        print(string.format("PY_CONTACT_LOG,%s,%s,%s,%s,%s,%s",
                            sim_time,
                            unit.name,
                            c_details.name or "Unknown",
                            c_details.type or "Unknown",
                            c_details.classification or "Unknown",
                            c_details.age or "0"
                        ))
                    end
                end
            end
        else
            -- Clean log output when an aircraft has no sensor detections
            print(string.format("PY_CONTACT_LOG,%s,%s,NO_CONTACTS,None,None,0", sim_time, unit.name))
        end
    else
        print("LOG_ERROR: Could not find aircraft: " .. identifier)
    end
end
