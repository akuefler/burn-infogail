using ZMQ
import JSON

using AutomotiveDrivingModels

include("multifeatureset.jl")
include("passive_aggressive.jl")

ctx = Context()
#s1=Socket(ctx, REP)
s2=Socket(ctx, REQ)

#ZMQ.bind(s1, "tcp://*:5555")
ZMQ.connect(s2, "tcp://localhost:5555")

# type simparams
# extractor
# scene record ## can I append to an existing scene record?
adm_context = IntegratedContinuous(0.1,1)
rec = SceneRecord(1, adm_context.Δt)
roadway = PASSIVE_AGGRESSIVE_ROADWAY

function create_feature_extractor()

    const EXTRACT_CORE = true
    const EXTRACT_TEMPORAL = false
    const EXTRACT_WELL_BEHAVED = true
    const EXTRACT_NEIGHBOR_FEATURES = false
    const EXTRACT_CARLIDAR_RANGERATE = true
    const CARLIDAR_NBEAMS = 20
    const ROADLIDAR_NBEAMS = 0
    const ROADLIDAR_NLANES = 2
    const CARLIDAR_MAX_RANGE = 100.0
    const ROADLIDAR_MAX_RANGE = 100.0

    # create a feature extractor
    extractor = MultiFeatureExtractor(
        EXTRACT_CORE,
        EXTRACT_TEMPORAL,
        EXTRACT_WELL_BEHAVED,
        EXTRACT_NEIGHBOR_FEATURES,
        EXTRACT_CARLIDAR_RANGERATE,
        CARLIDAR_NBEAMS,
        ROADLIDAR_NBEAMS,
        ROADLIDAR_NLANES,
        carlidar_max_range=CARLIDAR_MAX_RANGE,
        roadlidar_max_range=ROADLIDAR_MAX_RANGE,
        )

    extractor
end

function update_scene_record(D::Dict)
    n_vehicles = D["vehicle_defs"]["n_vehicles"]
    vds = D["vehicle_defs"]
    vss = D["vehicle_states"]

    println(vss)

    vehicles = Array(Vehicle, n_vehicles)
    for i in 1:n_vehicles
        x = vss["x"][i]
        y = vss["y"][i]
        @printf "x coord: %d ; y coord: %d \n" x y
        #(posG::VecSE2, roadway::Roadway, v::Float64) 
        vs = VehicleState(
                          VecSE2(
                                 vss["x"][i],
                                 vss["y"][i],
                                 vss["theta"][i]),
                          roadway,
                          vss["v"][i]
                         )
        vd = VehicleDef(
                        vds["id"][i],
                        vds["cls"][i],
                        vds["length"][i],
                        vds["width"][i]
                       )
        vehicle = Vehicle(vs, vd)
        vehicles[i] = vehicle
    end

    scene = Scene(vehicles)
    update!(rec, scene)
end

function main()
    extractor = create_feature_extractor()
    while true
        request = Vector{Float64}([1.0, 2.0, 3.0])
        println("Sending request...")
        ZMQ.send(s2, request)
        println("SENT!")

        println("Receiving reply...")
        msg = ZMQ.recv(s2)
        out = convert(IOStream, msg)
        seek(out,0)
        println("RECEIVED!")

        println(typeof(msg))
        jsonout = String(out)
        D = JSON.parse(jsonout::AbstractString, dicttype=Dict)

        #read_dict(D)
        update_scene_record(D)

        # pull features from scene record
        features = Array(Float64, length(extractor))
        veh_index = 1
        println("once ... ")
        print(rec)
        pull_features!(extractor, features, rec, roadway, veh_index)
        println(features)
    end

end

main()

