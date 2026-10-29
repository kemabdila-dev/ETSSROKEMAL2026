-- trail.lua
-- Menggambar jalur (garis biru) yang dilewati drone langsung di scene CoppeliaSim.
-- Pasang sebagai non-threaded child script pada dummy "TrailDrawer".

function sysCall_init()
    drone = sim.getObject('/Quadcopter')
    -- linestrip, tebal 4 px, tanpa toleransi duplikat, koordinat dunia, maks 100000 titik, warna biru
    trail = sim.addDrawingObject(sim.drawing_linestrip, 4, 0, -1, 100000, {0, 0, 1})
end

function sysCall_sensing()
    local p = sim.getObjectPosition(drone, -1)
    sim.addDrawingObjectItem(trail, p)
end

function sysCall_cleanup()
    sim.removeDrawingObject(trail)
end
