module Auto2D

using AutomotiveDrivingModels
using AutoViz
using Colors
using PDMats
using ForwardNets
using NGSIM

import Reel

include("load_environment_data.jl")
include("../pull_traces/multifeatureset.jl")
include("../pull_traces/passive_aggressive.jl")

export SimParams, GaussianMLPDriver, LatentEncoder, tick, step, gen_simparams,
    reset, observe, retrieve_image, retrieve_prime, get_classdict, get_obsdim,
    action_space_bounds, observation_space_bounds, isdone, get_info,
    get_norm_vecs, observe_state, pull_metrics, get_n_vehicles,
    reel_drive, rollout_ego_features, copy_simparams

##################################
# SimParams
# include("../validation/load_policy.jl")

type SimState
    frame::Int
    start_frame::Int
    egoid::Int
    egoids::Vector{Int}
    trajdata_index::Int

    scene::Scene
    rec::SceneRecord

    n_egos::Int
    prime_dict::Dict{String, Array{Float64,1}}

    playback_reactive_active_vehicle_ids::Set{Int}

    function SimState(context::IntegratedContinuous, rec_size::Int)
        retval = new()
        retval.scene = Scene()
        retval.rec = SceneRecord(rec_size, context.Δt)
        retval.egoids = Vector{Int}()

        retval.prime_dict = Dict{String, Array{Float64,1}}()

        # playbacks
        retval.playback_reactive_active_vehicle_ids = Set{Int}()
        retval
    end
end

type SimParams
    context::IntegratedContinuous
    prime_history::Int
    ego_action_type::DataType

    trajdatas::Dict{Int, Trajdata}
    segments::Vector{TrajdataSegment}
    classes::Vector{Int}
    mix::Bool
    valid::Bool

    nsteps::Int
    step_counter::Int
    simstates::Vector{SimState}
    features::Vector{Float64}
    extractor::MultiFeatureExtractor

    # playback / NGSIM fields
    use_playback_reactive::Bool
    playback_reactive_model::DriverModel{LatLonAccel, IntegratedContinuous}
    playback_reactive_threshold_brake::Float64
    playback_reactive_scene_buffer::Scene

    model_all::Bool
    ngsim::Bool

end

#SimParams(trajdatas, segments, classes, false,
#          nsimstates, prime_history, nsteps, ego_action_type, extractor,
#          use_playback_reactive, playback_reactive_threshold_brake,
#          model_all, true)

#Auto2D.SimParams(trajdatas, evaldata.segments, classes, mix_class, 1, EVAL_PRIME_STEPS, EVAL_DURATION_STEPS, AccelTurnrate, extractor)
function SimParams(trajdatas::Dict{Int, Trajdata}, segments::Vector{TrajdataSegment},
    classes::Vector{Int},
    mix::Bool,
    valid::Bool,
    nsimstates::Int,
    prime_history::Int,
    nsteps::Int,
    ego_action_type::DataType,
    extractor::MultiFeatureExtractor,
    use_playback_reactive = false,
    playback_reactive_threshold_brake = -2.0,
    model_all = false,
    ngsim = false,
    context = IntegratedContinuous(NGSIM_TIMESTEP,1),
    )

    simstates = Array(SimState, nsimstates)
    for i in 1 : length(simstates)
        simstates[i] = SimState(context, prime_history+1)
    end

    # playback model
    playback_reactive_model = LatLonSeparableDriver(
        context,
        ProportionalLaneTracker(σ=0.1, kp=3.0, kd=2.0),
        IntelligentDriverModel(σ=0.1, k_spd=1.0, T=0.5, s_min=1.0, a_max=3.0, d_cmf=2.5),
        )
    playback_reactive_scene_buffer = Scene()

    features = Array(Float64, length(extractor))
    #Auto2D.SimParams(trajdatas, segments, classes, mix_class, 1, EVAL_PRIME_STEPS, EVAL_DURATION_STEPS, AccelTurnrate, extractor)
    SimParams(context, prime_history, ego_action_type, trajdatas, segments,
              classes, mix, valid, nsteps, 0, simstates, features, extractor,
              use_playback_reactive, playback_reactive_model,
              playback_reactive_threshold_brake,
              playback_reactive_scene_buffer,
              model_all, ngsim)
end


#################################
# additional functions for creating, observing the simparams
#function AutoViz.render(simparams::SimParams, image::Matrix{UInt32}, batch_index::Int=1)
function reel_drive(
    gif_filename::AbstractString,
    actions::Matrix{Float64}, # columns are actions
    simparams::SimParams
    )
    framerate = 5.0
    frames = Reel.Frames(MIME("image/png"), fps=framerate)

    simstate = simparams.simstates[1]
    trajdata = simparams.trajdatas[simstate.trajdata_index]

    car_colors = colorize_cars(simparams)

    action = [NaN, NaN]
    push!(frames, render(simstate.scene,trajdata.roadway,
                             cam=CarFollowCamera(simstate.egoid, 5.0),
                        car_colors=car_colors,
                       #canvas_width=1200))
                       canvas_width=500))
    for frame_index in 1:size(actions,2)
        #step_forward!(simparams, action)
        #action = actions[:,frame_index]
        action[1] = actions[1,frame_index]
        action[2] = actions[2,frame_index]
        Base.step(simparams, action)
        #step_forward!(simstate, simparams, action)

        push!(frames, render(simstate.scene,trajdata.roadway,
                             car_colors=car_colors,
                             cam=CarFollowCamera(simstate.egoid, 5.0),
                             canvas_width=500))
    end

    Reel.write(gif_filename, frames) # Write to a gif file
end

function reel_drive(
    gif_filename::AbstractString,
    simparams::SimParams
    )
    framerate = 5.0
    frames = Reel.Frames(MIME("image/png"), fps=framerate)

    simstate = simparams.simstates[1]
    trajdata = simparams.trajdatas[simstate.trajdata_index]

    car_colors = colorize_cars(simparams)

    push!(frames, render(simstate.scene,trajdata.roadway,
                             cam=CarFollowCamera(simstate.egoid, 5.0),
                             car_colors=car_colors,
                             canvas_width=500))
    counter = 0
    while counter < simparams.nsteps
        simstate.frame += 1
        counter += 1
        get!(simstate.scene, trajdata, simstate.frame)
        update!(simstate.rec, simstate.scene)
        #step_forward!(simstate, simparams, action)
        push!(frames, render(simstate.scene,trajdata.roadway,
                             car_colors=car_colors,
                             cam=CarFollowCamera(simstate.egoid, 5.0),
                             canvas_width=500))
    end

    Reel.write(gif_filename, frames) # Write to a gif file
end

#simparams = deepcopy(simparams)
#
#simstate = simparams.simstates[1]
#trajdata = simparams.trajdatas[simstate.trajdata_index]
#veh_ego = get_vehicle(simstate.scene, simstate.egoid)
#
#counter = 0
##features = Vector{Dict{String,Float64}}()
#features = Vector{}()
#while counter < simparams.nsteps
#    simstate.frame += 1
#    counter += 1
#    # pull new frame from trajdata
#    get!(simstate.scene, trajdata, simstate.frame)
#
#    # move in propagated ego vehicle
#    veh_index = get_index_of_first_vehicle_with_id(simstate.scene,
#                                                   simstate.egoid)
#
#    # update record
#    update!(simstate.rec, simstate.scene)
#    dict = observe_metrics(simstate.rec, trajdata.roadway, veh_index)
#    push!(features,observe_metrics(simstate.rec, trajdata.roadway,
#                                     veh_index))
#end

function colorize_cars(simparams::SimParams)
    CAR_CLASS_COLORS = distinguishable_colors(6,colorant"blue")
    simstate = simparams.simstates[1]
    trajdata = simparams.trajdatas[simstate.trajdata_index]
    carcolors = Dict{Int,Colorant}()
    num_vehicles = length(trajdata.vehdefs)
    simstate = simparams.simstates[1] # ASSUME : only 1 simstate

    if ~simparams.ngsim
        #class_dict = load_classdict(simstate.trajdata_index, num_vehicles,
        #                            simparams.mix, simparams.valid)
        # WARNING: ASSUME only 10 trajdatas per domain
        if simparams.valid
            num_trajdatas = 5
        else
            num_trajdatas = 10
        end
        class_dict = load_classdict((simstate.trajdata_index % num_trajdatas)+1, num_vehicles, simparams.mix, simparams.valid)

        for j = 1:num_vehicles
            cls = class_dict[j]
            if simparams.model_all
                carcolors[j] = CAR_CLASS_COLORS[1]
            else
                carcolors[j] = CAR_CLASS_COLORS[cls]
            end
        end
    end

    if simparams.model_all
        tdframe = trajdata.frames[simstate.frame]
        for egoid in simstate.egoids # loop over each car
            carcolors[egoid] = CAR_CLASS_COLORS[end]
        end
        follow_car = simstate.egoids[1]
    else
        #carcolors[simstate.egoid] = CAR_CLASS_COLORS[end]
        follow_car = simstate.egoid
    end
    carcolors
end

function retrieve_image(simparams::SimParams, image::Matrix{UInt32})

    simstate = simparams.simstates[1]
    trajdata = simparams.trajdatas[simstate.trajdata_index]

    c = CairoImageSurface(image, Cairo.FORMAT_ARGB32, flipxy=false)
    ctx = CairoContext(c)
    carcolors = colorize_cars(simparams)

    if simparams.model_all
        follow_car = simstate.egoids[1]
    else
        follow_car = simstate.egoid
    end

    render(ctx, simstate.scene, trajdata.roadway,
           cam=CarFollowCamera(follow_car, 5.0), car_colors=carcolors)
    image
end

function retrieve_prime(simparams::SimParams)
    simstate = simparams.simstates[1] # ASSUME : only 1 simstate
    simstate.prime_dict
end

function observe(simparams::SimParams, batch_index::Int=1)
    if ~simparams.model_all
        simstate = simparams.simstates[batch_index]
        trajdata = simparams.trajdatas[simstate.trajdata_index]
        veh_index = get_index_of_first_vehicle_with_id(simstate.scene,simstate.egoid)
        #@printf "trajdata index: %d, veh_index: %d \n" simstate.trajdata_index veh_index

        pull_features!(simparams.extractor, simparams.features, simstate.rec, trajdata.roadway, veh_index)

    else
        simstate = simparams.simstates[1]
        trajdata = simparams.trajdatas[simstate.trajdata_index]

        # WARNING: This is probably expensive ...
        all_feats = []
        for egoid in simstate.egoids # loop over each car
            veh_index = get_index_of_first_vehicle_with_id(simstate.scene,egoid)
            features = zeros(length(simparams.extractor))
            pull_features!(simparams.extractor, features, simstate.rec, trajdata.roadway, veh_index)
            all_feats = vcat(all_feats, features)
        end

        all_feats
    end
end

function observe(extractor::MultiFeatureExtractor, rec::SceneRecord,
                 roadway::Roadway, veh_index::Int)
    # useful for observing during prime period
    features = Array(Float64, length(extractor))
    pull_features!(extractor, features, rec, roadway, veh_index)
end

#function pull_metrics!(
#    metrics::Vector{Float64},
#    rec::SceneRecord,
#    roadway::Roadway,
#    vehicle_index::Int
#   )
#    D = observe_metrics(rec, roadway, vehicle_index)
#    metric_headers = sort([key for key in keys(D)])
#
#    assert(length(metrics) == length(metric_headers))
#
#    for i in 1:length(metrics)
#        metrics[i] = D[metric_headers[i]]
#    end
#    metrics
#end
#
##pull_metrics!(metrics, rec, trajdata.roadway, vehicle_index)
#function observe_metrics(rec::SceneRecord, scene::Scene, roadway::Roadway, vehicle_index::Int)
#    veh_ego = scene[vehicle_index]
#
#    ittc = convert(Float64,
#                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_Inv_TTC(),rec, roadway, vehicle_index)
#                  )
#    timegap = convert(Float64,
#                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_Timegap(),rec, roadway, vehicle_index)
#                  )
#
#    speed = convert(Float64,
#                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_Speed(),rec, roadway, vehicle_index)
#                  )
#
#    turnRateG = convert(Float64,
#                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_TurnRateG(),rec, roadway, vehicle_index)
#                  )
#
#    jerk = convert(Float64,
#                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_Jerk(),rec, roadway, vehicle_index)
#                  )
#
#    desiredAng = convert(Float64,
#                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_DesiredAngle(),rec, roadway, vehicle_index)
#                  )
#
#    laneOffsetL = convert(Float64,
#                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_LaneOffsetLeft(),rec, roadway, vehicle_index)
#                  )
#
#    laneOffsetR = convert(Float64,
#                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_LaneOffsetRight(),rec, roadway, vehicle_index)
#                  )
#
#    timeConBrake = convert(Float64,
#                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_Time_Consecutive_Brake(),
#                                               rec, roadway, vehicle_index)
#                  )
#
#    timeConAcc = convert(Float64,
#                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_Time_Consecutive_Accel(),
#                                               rec, roadway, vehicle_index)
#                  )
#
#    Dict("measure/ittc" => ittc,
#         "measure/timegap" => timegap,
#         "measure/speed" => speed,
#         "measure/turnRateG" => turnRateG,
#         "measure/jerk" => jerk,
#         "measure/desiredAng" => desiredAng,
#         "measure/laneOffsetL" => laneOffsetL,
#         "measure/laneOffsetR" => laneOffsetR,
#         "measure/timeConBrake" => timeConBrake,
#         "measure/timeConAcc" => timeConAcc,
#         "measure/pos_x" => veh_ego.state.posG.x,
#         "measure/pos_y" => veh_ego.state.posG.y)
#end
function observe_metrics(simstate::SimState, roadway::Roadway, vehicle_index::Int)
    rec = simstate.rec
    veh_ego = simstate.scene[vehicle_index]

    ittc = convert(Float64,
                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_Inv_TTC(),rec, roadway, vehicle_index)
                  )
    timegap = convert(Float64,
                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_Timegap(),rec, roadway, vehicle_index)
                  )

    speed = convert(Float64,
                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_Speed(),rec, roadway, vehicle_index)
                  )

    turnRateG = convert(Float64,
                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_TurnRateG(),rec, roadway, vehicle_index)
                  )

    jerk = convert(Float64,
                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_Jerk(),rec, roadway, vehicle_index)
                  )

    desiredAng = convert(Float64,
                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_DesiredAngle(),rec, roadway, vehicle_index)
                  )

    laneOffsetL = convert(Float64,
                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_LaneOffsetLeft(),rec, roadway, vehicle_index)
                  )

    laneOffsetR = convert(Float64,
                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_LaneOffsetRight(),rec, roadway, vehicle_index)
                  )

    timeConBrake = convert(Float64,
                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_Time_Consecutive_Brake(),
                                               rec, roadway, vehicle_index)
                  )

    timeConAcc = convert(Float64,
                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_Time_Consecutive_Accel(),
                                               rec, roadway, vehicle_index)
                  )

    posFt = convert(Float64,
                   AutomotiveDrivingModels.get(AutomotiveDrivingModels.Feature_PosFt(),
                                               rec, roadway, vehicle_index)
                  )

    Dict("measure/ittc" => ittc,
         "measure/timegap" => timegap,
         "measure/speed" => speed,
         "measure/turnRateG" => turnRateG,
         "measure/jerk" => jerk,
         "measure/desiredAng" => desiredAng,
         "measure/laneOffsetL" => laneOffsetL,
         "measure/laneOffsetR" => laneOffsetR,
         "measure/timeConBrake" => timeConBrake,
         "measure/timeConAcc" => timeConAcc,
         "measure/pos_x" => veh_ego.state.posG.x,
         "measure/pos_y" => veh_ego.state.posG.y,
         "measure/posFt" => posFt
        )
end

function observe_state(simparams::SimParams, batch_index::Int=1)
    simstate = simparams.simstates[batch_index]
    trajdata = simparams.trajdatas[simstate.trajdata_index]
    veh_index = get_index_of_first_vehicle_with_id(simstate.scene, simstate.egoid)

    dict = observe_metrics(simstate, trajdata.roadway, veh_index)
    dict["trajdata_index"] = simstate.trajdata_index
    dict["n_vehicles"] = simstate.scene.n_vehicles
    dict["n_egos"] = simstate.n_egos
    dict
end

function get_n_vehicles(simparams::SimParams)
    simstate = simparams.simstates[1]
    simstate.scene.n_vehicles
end

function copy_simparams(simparams::SimParams)
    initial_simparams = deepcopy(simparams)
    initial_simparams
end

function get_info(simparams::SimParams)
    simstate = simparams.simstates[1] # ASSUME : only 1 simstate
    trajdata = simparams.trajdatas[simstate.trajdata_index]

    # Using NGSIM data
    if simparams.ngsim
        dom_ix = simstate.trajdata_index
        ego_cls = 0
    else
        # WARNING: ASSUME only 10 trajdatas per domain
        if simparams.valid
            num_trajdatas = 5
        else
            num_trajdatas = 10
        end
        assert(simstate.trajdata_index > 0)
        dom_ix = Int(
                     floor(
                           (simstate.trajdata_index - 1) / num_trajdatas
                          )
                    )

        #num_vehicles = length(trajdata.vehdefs)
        #class_dict = load_classdict((simstate.trajdata_index % num_trajdatas)+1, num_vehicles, simparams.mix, simparams.valid)

        #ego_index = get_index_of_first_vehicle_with_id(simstate.scene, simstate.egoid)
        #ego_cls = class_dict[ego_index]
    end

    # MODELING ALL
    if simparams.model_all
        simstate = simparams.simstates[1] # ASSUME: Only 1 simstate
        trajdata = simparams.trajdatas[simstate.trajdata_index]
        # veh_index = get_index_of_first_vehicle_with_id(simstate.scene,simstate.egoid)

        # WARNING: This is probably expensive ...
        tdframe = trajdata.frames[simstate.frame]
        all_feats = []

        collide = zeros(length(simstate.egoids))
        offroad = zeros(length(simstate.egoids))
        reverse = zeros(length(simstate.egoids))
        for (i, egoid) in enumerate(simstate.egoids) # loop over each car
            veh_index = get_index_of_first_vehicle_with_id(simstate.scene,
                                                           egoid)
            veh_ego = simstate.scene[veh_index]
            veh = simstate.scene[veh_index]

            d_ml = convert(Float64, get(MARKERDIST_LEFT, simstate.rec, trajdata.roadway, veh_index))
            d_mr = convert(Float64, get(MARKERDIST_RIGHT, simstate.rec, trajdata.roadway, veh_index))

            #collide = (collide || get_first_collision(simstate.scene,veh_index).is_colliding)
            #offroad = (offroad || (d_ml < -1.0 || d_mr < -1.0))
            #reverse = (reverse || (veh_ego.state.v < 0.0))
            collide[i] = Int(get_first_collision(simstate.scene,veh_index).is_colliding)
            offroad[i] = Int(d_ml < -1.0 || d_mr < -1.0)
            reverse[i] = Int(veh_ego.state.v < 0.0)
        end

        Dict("collision" => collide,
             "offroad" => offroad,
             "reverse" => reverse,
             "dom_ix" => dom_ix
            )

    else
        veh_index = get_index_of_first_vehicle_with_id(simstate.scene, simstate.egoid)
        veh_ego = simstate.scene[veh_index]

        d_ml = convert(Float64, get(MARKERDIST_LEFT, simstate.rec, trajdata.roadway, veh_index))
        d_mr = convert(Float64, get(MARKERDIST_RIGHT, simstate.rec, trajdata.roadway, veh_index))

        Dict("collision" => get_first_collision(simstate.scene, veh_index).is_colliding,
             "offroad" => (d_ml < -1.0 || d_mr < -1.0),
             "reverse" => (veh_ego.state.v < 0.0),
             "dom_ix" => dom_ix
             #"ego_cls" => ego_cls
            )
    end
end

function get_norm_vecs(simparams::SimParams)
    norm_actions = false
    if norm_actions
        feature_means = vec(h5read(filepath, "policy/obs_mean"))
        feature_std = vec(h5read(filepath, "policy/obs_std"))
        action_means = vec(h5read(filepath, "policy/act_mean"))
        action_std = vec(h5read(filepath, "policy/act_std"))
    else
        feature_means = vec(h5read(filepath, "initial_obs_mean"))
        feature_std = vec(h5read(filepath, "initial_obs_std"))
        action_means = zeros(2)
        action_std = ones(2)
    end
    Dict("mean" => feature_means, "std" => feature_std)
end

action_space_bounds(simparams::SimParams) = ([-5.0, -1.0], [3.0, 1.0])
observation_space_bounds(simparams::SimParams) = (fill(-Inf, length(simparams.extractor)), fill(Inf, length(simparams.extractor)))

# was Base.reset
function Base.reset(simparams::SimParams, n_egos::Int=2, seed::Int=-1)
    for state in simparams.simstates
        state.n_egos = n_egos
        if simparams.ngsim
            ngsim_restart!(state, simparams)
        else
            while true # using rejection sampling to throw out bad scenes.
                reset_simstate!(state, simparams, seed)
                if state.scene.n_vehicles > 0
                    break
                end
            end
        end
    end

    simparams.step_counter = 0
    simparams
end

function get_classdict(simparams::SimParams)
    simparams.classdict
end

function get_obsdim(simparams::SimParams)
    length(simparams.extractor)
end

function reset_simstate!(simstate::SimState, simparams::SimParams, seed::Int=-1)
    if seed >= 0
        srand(seed)
    end

    train_seg_index = rand(1:length(simparams.segments))
    seg = simparams.segments[train_seg_index]

    simstate.trajdata_index = seg.trajdata_index

    # pull the trajdata
    trajdata = simparams.trajdatas[simstate.trajdata_index]

    candidate_frame_lo = 1
    #candidate_frame_hi = 350 - simparams.nsteps - simparams.prime_history - 1
    candidate_frame_hi = 50 - simparams.prime_history - 1
    assert(candidate_frame_hi > candidate_frame_lo) # WARNING : Used to be >
    simstate.frame = rand(candidate_frame_lo:candidate_frame_hi-1)

    # clear rec and make first observation
    tdframe = trajdata.frames[simstate.frame]
    simstate.egoids = Vector{Int}()
    for i in shuffle(tdframe.lo : tdframe.hi)[1:simstate.n_egos]
        append!(simstate.egoids, trajdata.states[i].id)
    end
    assert(length(simstate.egoids)==simstate.n_egos)
    if simparams.model_all
        simstate.egoid = simstate.egoids[1]
    else
        simstate.egoid = seg.egoid
    end

    empty!(simstate.rec)
    a = zeros(2)
    pastframe = 1
    for i in 1 : simparams.prime_history + 1
        get!(simstate.scene, trajdata, simstate.frame)
        update!(simstate.rec, simstate.scene)

        simstate.frame += 1

        veh_index = get_index_of_first_vehicle_with_id(simstate.scene, simstate.egoid)

        a[1] = get(ACC, simstate.rec, trajdata.roadway, veh_index)
        a[2] = get(TURNRATEG, simstate.rec, trajdata.roadway, veh_index)
        pastframe += 1
        simstate.prime_dict["obs" * string(i)] = observe(simparams.extractor,
                                                         simstate.rec,
                                                         trajdata.roadway,
                                                         veh_index
                                                        )
        simstate.prime_dict["act" * string(i)] = a
    end

    simstate.frame -= 1
    simstate.start_frame = simstate.frame

    simstate
end

function ngsim_restart!(simstate::SimState, simparams::SimParams)

    # pick a random segment
    local train_seg_index
    for i in 1 : 100
        train_seg_index = rand(1:length(simparams.segments))
        seg = simparams.segments[train_seg_index]
        candidate_frame_lo = seg.frame_lo
        candidate_frame_hi = seg.frame_hi - simparams.nsteps - simparams.prime_history
        if candidate_frame_hi > candidate_frame_lo
        break
    end
    if i > 95
        assert(false)
        end
    end

    seg = simparams.segments[train_seg_index]
    simstate.egoid = seg.egoid
    simstate.trajdata_index = seg.trajdata_index

    # pull the trajdata
    trajdata = simparams.trajdatas[simstate.trajdata_index]

    # pick a random sub-trajectory
    candidate_frame_lo = seg.frame_lo
    candidate_frame_hi = seg.frame_hi - simparams.nsteps - simparams.prime_history
    simstate.frame = rand(candidate_frame_lo:candidate_frame_hi)

    # clear rec and make first observations
    empty!(simstate.rec)
    for i in 1 : simparams.prime_history + 1
        get!(simstate.scene, trajdata, simstate.frame)
        update!(simstate.rec, simstate.scene)
        simstate.frame += 1
    end
    simstate.frame -= 1
    simstate.start_frame = simstate.frame

    # empty playback reactive
    empty!(simstate.playback_reactive_active_vehicle_ids)

    # simparams.extractor.road_lidar_culling = simparams.roadway_cullers[simstate.trajdata_index]

    simstate
end

function gen_simparams(trajdata_indices::Vector,
    use_playback_reactive::Bool,
    playback_reactive_threshold_brake::Float64,
    nsimstates::Int, # AKA, batch size
    prime_history::Int,
    nsteps::Int,
    ego_action_type::DataType,
    extractor::MultiFeatureExtractor,
    model_all::Bool
    )
    ## "NGSIM roadway"
    println("loading trajdatas: ", trajdata_indices); tic()
    trajdatas = Dict{Int, Trajdata}()
    for i in trajdata_indices
        trajdatas[i] = load_trajdata(i)
    end
    toc()

    println("loading training segments"); tic()
    segments = get_train_segments(trajdatas, nsteps)
    toc()

    classes = Vector{Int}(1)
    SimParams(trajdatas, segments, classes, false,
              nsimstates, prime_history, nsteps, ego_action_type, extractor,
              use_playback_reactive, playback_reactive_threshold_brake,
              model_all, true)

end

function gen_simparams(batch_size::Int, args::Dict)
    ## "NGSIM Interface to Python"
    use_playback_reactive = convert(Bool, get(args, "use_playback_reactive", false))
    model_all = convert(Bool, get(args, "model_all", false))
    playback_reactive_threshold_brake = get(args, "playback_reactive_threshold_brake", -2.0) # [m/s²]
    prime_history = get(args, "prime_history", 2) # ASSUME : this right?
    nsteps = get(args, "nsteps", 20) #ASSUME : this right?

    extract_core = get(args, "extract_core", true)
    extract_temporal = get(args, "extract_temporal", false)
    extract_well_behaved = get(args, "extract_well_behaved", true)
    extract_neighbor_features = get(args, "extract_neighbor_features", false)
    extract_carlidar_rangerate = get(args, "extract_carlidar_rangerate", true)
    carlidar_nbeams = get(args, "carlidar_nbeams", 20)
    roadlidar_nbeams = get(args, "roadlidar_nbeams", 0)
    roadlidar_nlanes = get(args, "roadlidar_nlanes", 2)
    carlidar_max_range = get(args, "carlidar_max_range", 100.0) # [m]
    roadlidar_max_range = get(args, "roadlidar_max_range", 100.0) # [m]

    extractor = MultiFeatureExtractor(
        extract_core,
        extract_temporal,
        extract_well_behaved,
        extract_neighbor_features,
        extract_carlidar_rangerate,
        carlidar_nbeams,
        roadlidar_nbeams,
        roadlidar_nlanes,
        carlidar_max_range=carlidar_max_range,
        roadlidar_max_range=roadlidar_max_range,
        )

    ego_action_type = AccelTurnrate
    T = get(args, "action_type", "AccelTurnrate")
    if T == "AccelTurnrate"
        ego_action_type = AccelTurnrate
    elseif T == "LatLonAccel"
        ego_action_type = LatLonAccel
    end

    # generate the actual simparams
    if haskey(args, "trajdata_filepaths")
        assert(false)
    else
        trajdata_indices = get(args, "trajdata_indices", [1,2,3,4,5,6])
        gen_simparams(trajdata_indices,
            use_playback_reactive, playback_reactive_threshold_brake,
            batch_size, prime_history, nsteps, ego_action_type, extractor,
            model_all)
    end
end

function gen_simparams(args::Dict)
#:Dict{Int64,AutomotiveDrivingModels.AutoCore.Trajdata}, ::Array{Any,1}, ::Array{Int64,1}
    ## "Track roadway"
    domain_indices = get(args, "domain_indices", [0,1,2,3])
    domain_indices += 1 # NOTE : Python uses 0-indexing

    model_all = get(args, "model_all", false)
    mix_class = get(args, "mix_class", true)
    use_valid = get(args, "use_valid", false)

    if use_valid
        num_trajdata = 5
    else
        num_trajdata = 10
    end

    cars_per_domain = [24, 33, 48, 96]

    tds = Vector{AutomotiveDrivingModels.AutoCore.Trajdata}()
    segments = Vector{AutomotiveDrivingModels.TrajdataSegment}()
    classes = Vector{Int64}()

    for i in 1:length(domain_indices)
        cars = cars_per_domain[domain_indices[i]]
        idx = (i-1) * num_trajdata
        evaldata, cls, num_trajdata = load_envdata(mix_class, cars, idx, use_valid)
        append!(tds, evaldata.trajdatas)
        append!(classes, cls)
        append!(segments, evaldata.segments)
    end

    trajdatas = Dict(zip(collect(1:length(tds)), tds))

    const EVAL_PRIME_STEPS = 15
    const EVAL_DURATION_STEPS = 300 #50 #20

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

    # Construct extractor
    extractor = Auto2D.MultiFeatureExtractor(
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

    # Convert array of trajdatas to dict
    Auto2D.SimParams(trajdatas, segments, classes, mix_class, use_valid, 1, EVAL_PRIME_STEPS,
              EVAL_DURATION_STEPS, AccelTurnrate, extractor, false, -2.0,
              model_all, false)
end
isdone(simparams::SimParams) = simparams.step_counter ≥ simparams.nsteps

##################################
# Gaussian MLP Driver

type GaussianMLPDriver{A<:DriveAction, F<:Real, G<:Real, H<:Real, E<:AbstractFeatureExtractor, M<:MvNormal} <: DriverModel{A, IntegratedContinuous}
    net::ForwardNet
    rec::SceneRecord
    pass::ForwardPass
    input::Vector{F}
    output::Vector{G}
    latent_state::Vector{H}
    driver_class::Vector{H}
    extractor::E
    mvnormal::M
    context::IntegratedContinuous
    use_latent::Bool
    oracle::Bool
end

_get_Σ_type{Σ,μ}(mvnormal::MvNormal{Σ,μ}) = Σ
function GaussianMLPDriver{A <: DriveAction}(::Type{A}, net::ForwardNet, extractor::AbstractFeatureExtractor, context::IntegratedContinuous;
    input::Symbol = :input,
    output::Symbol = :output,
    Σ::Union{Real, Vector{Float64}, Matrix{Float64},  Distributions.AbstractPDMat} = 0.1,
    rec::SceneRecord = SceneRecord(2, context.Δt),
    use_latent::Bool = false,
    oracle::Bool = false,
    )

    pass = calc_forwardpass(net, [input], [output])
    input_vec = net[input].tensor
    output = net[output].tensor
    latent_state = output  # Initialize to whatever
    driver_class = zeros(4)
    mvnormal = MvNormal(Array(Float64, 2), Σ)
    GaussianMLPDriver{A, eltype(input_vec), eltype(output), eltype(output), typeof(extractor), typeof(mvnormal)}(net, rec, pass, input_vec, output, latent_state, driver_class, extractor, mvnormal, context, use_latent, oracle)
end

AutomotiveDrivingModels.get_name(::GaussianMLPDriver) = "GaussianMLPDriver"
AutomotiveDrivingModels.action_context(model::GaussianMLPDriver) = model.context

# Set driver class to one-hot vector
function set_driver_class!{A,F,G,H,E,P}(model::GaussianMLPDriver{A,F,G,H,E,P}, c::Int)
    model.driver_class = zeros(4)
    model.driver_class[c] = 1.0
end

# Set latent state at input to policy
function set_latent_state!{A,F,G,H,E,P}(model::GaussianMLPDriver{A,F,G,H,E,P}, z::Vector{Float64})
    model.latent_state = z
end

# Empty record
function AutomotiveDrivingModels.reset_hidden_state!(model::GaussianMLPDriver)
    empty!(model.rec)
    model
end

# Fill in input with observations and latent samples
function AutomotiveDrivingModels.observe!{A,F,G,H,E,P}(
                                            model::GaussianMLPDriver{A,F,G,H,E,P}, 
                                            simparams::SimParams, 
                                            scene::Scene, 
                                            roadway::Roadway, 
                                            egoid::Int)

    update!(model.rec, scene)
    vehicle_index = get_index_of_first_vehicle_with_id(scene, egoid)
    o = pull_features!(simparams.extractor, simparams.features, model.rec, roadway, vehicle_index)
    o_norm = (o - model.extractor.feature_means)./model.extractor.feature_std
    
    if !(model.use_latent)
        if model.oracle && (:MLP_0 in keys(model.net.name_to_index))
            model.net[:MLP_0].input = cat(1, o_norm, model.driver_class)
        elseif model.oracle
            model.net[:LSTM_0].input = cat(1, o_norm, model.driver_class)
        elseif (:MLP_0 in keys(model.net.name_to_index))
            model.net[:MLP_0].input = o_norm
        elseif (:LSTM_0 in keys(model.net.name_to_index))
            model.net[:LSTM_0].input = o_norm
        else
            model.net[:GAIL_0].input = o_norm
        end
    else
        if (:MLP_0 in keys(model.net.name_to_index))
            model.net[:MLP_0].input = cat(1, o_norm, model.latent_state)
        else
            model.net[:LSTM_0].input = cat(1, o_norm, model.latent_state)
        end
    end
    forward!(model.pass)
    copy!(model.mvnormal.μ, model.output[1:2])

    model
end

function Base.rand{A,F,G,H,E,P}(model::GaussianMLPDriver{A,F,G,H,E,P})
    a = rand(model.mvnormal)
    if model.extractor.norm_actions
        a = (a.*model.extractor.action_std + model.extractor.action_means)
    end
    convert(A, a)
end
Distributions.pdf{A,F,G,H,E,P}(model::GaussianMLPDriver{A,F,G,H,E,P}, a::A) = pdf(model.mvnormal, convert(Vector{Float64}, a))
Distributions.logpdf{A,F,G,H,E,P}(model::GaussianMLPDriver{A,F,G,H,E,P}, a::A) = logpdf(model.mvnormal, convert(Vector{Float64}, a))


##################################
# Latent Encoder
type LatentEncoder{F<:Real, G<:Real, E<:AbstractFeatureExtractor, M<:MvNormal}
    net::ForwardNet
    rec::SceneRecord
    pass::ForwardPass
    input::Vector{F}
    output::Vector{G}
    extractor::E
    mvnormal::M
    context::IntegratedContinuous
    z_dim::Int
end

_get_Σ_type{Σ,μ}(mvnormal::MvNormal{Σ,μ}) = Σ
function LatentEncoder(net::ForwardNet, extractor::AbstractFeatureExtractor, context::IntegratedContinuous;
    input::Symbol = :input,
    output::Symbol = :output,
    Σ::Union{Real, Vector{Float64}, Matrix{Float64},  Distributions.AbstractPDMat} = [0.1, 0.1],
    rec::SceneRecord = SceneRecord(2, context.Δt),
    z_dim::Int = 2,
    )

    pass = calc_forwardpass(net, [input], [output])
    input_vec = net[input].tensor
    output = net[output].tensor
    Σ = 0.1 * ones(z_dim)
    mvnormal = MvNormal(Array(Float64, z_dim), Σ)
    LatentEncoder{eltype(input_vec), eltype(output), typeof(extractor), typeof(mvnormal)}(net, rec, pass, input_vec, output, extractor, mvnormal, context, z_dim)
end

AutomotiveDrivingModels.get_name(::LatentEncoder) = "LatentEncoder"

function AutomotiveDrivingModels.reset_hidden_state!(model::LatentEncoder)
    empty!(model.rec)
    model
end

function AutomotiveDrivingModels.observe!{F,G,E,P}(
                                            model::LatentEncoder{F,G,E,P}, 
                                            simparams::SimParams, 
                                            scene::Scene, 
                                            roadway::Roadway, 
                                            egoid::Int)

    update!(model.rec, scene)
    vehicle_index = get_index_of_first_vehicle_with_id(scene, egoid)

    # Find and normalize features
    o = pull_features!(simparams.extractor, simparams.features, model.rec, roadway, vehicle_index)
    o_norm = (o - model.extractor.feature_means)./model.extractor.feature_std
    print("FEATURE MEAN: ")

    # Find and normalize vehicle actions
    a = zeros(2)
    a[1] = get(ACC, model.rec, roadway, vehicle_index, 0)
    a[2] = get(TURNRATEG, model.rec, roadway, vehicle_index, 0)
    a_norm = (a - model.extractor.action_means)./model.extractor.action_std

    model.net[:LSTM_0].input = cat(1, o_norm, a_norm)
    forward!(model.pass)
    copy!(model.mvnormal.μ, model.output[1:model.z_dim])
    copy!(model.mvnormal.Σ.diag, exp(model.output[(model.z_dim+1):2*model.z_dim]).^2)

    model
end
Base.rand{F,G,E,P}(model::LatentEncoder{F,G,E,P}) = rand(model.mvnormal)


########################################
# Propagate scenes
function rollout_ego_features(simparams::SimParams)
    simparams = deepcopy(simparams)

    simstate = simparams.simstates[1]
    trajdata = simparams.trajdatas[simstate.trajdata_index]
    veh_ego = get_vehicle(simstate.scene, simstate.egoid)

    counter = 0
    #features = Vector{Dict{String,Float64}}()
    features = Vector{}()
    while counter < simparams.nsteps
        simstate.frame += 1
        counter += 1
        # pull new frame from trajdata
        get!(simstate.scene, trajdata, simstate.frame)

        # move in propagated ego vehicle
        veh_index = get_index_of_first_vehicle_with_id(simstate.scene,
                                                       simstate.egoid)

        # update record
        update!(simstate.rec, simstate.scene)
        push!(features,observe_metrics(simstate, trajdata.roadway,
                                         veh_index))
    end

    features
end

# Step scene forward
function step_forward!(simstate::SimState, simparams::SimParams, action_ego::Vector{Float64})

    trajdata = simparams.trajdatas[simstate.trajdata_index]
    #if simparams.model_all
    #if length(action_ego) > 2

    if simparams.model_all
        assert(length(action_ego) == (2 * simstate.n_egos)) # WARNING : Beware magic number
        empty!(simparams.playback_reactive_scene_buffer)

        tdframe = trajdata.frames[simstate.frame]
        veh_ids_to_states = Dict{Int, AutomotiveDrivingModels.VehicleState}()

        # get all the propagated states for each ego vehicle
        for (x, egoid) in enumerate(simstate.egoids) # loop over each car
            veh_index = get_index_of_first_vehicle_with_id(simstate.scene, egoid)
            if veh_index != 0 # iscarinframe
                #x = (2 * (egoid - tdframe.lo + 1)) - 1
                #@printf "x: %d \n ; len: %d " x length(action_ego)
                act = action_ego[x:x+1]
                action = convert(simparams.ego_action_type, act)

                # Propagate scene
                veh = simstate.scene[veh_index]
                veh_state = propagate(veh, action, simparams.context, trajdata.roadway)
                veh_ids_to_states[egoid] = veh_state

            else
                assert(false)
            end
        end

        simstate.frame += 1
        get!(simstate.scene, trajdata, simstate.frame)

        # move in propagated ego vehicles
        for id in keys(veh_ids_to_states)
            veh_index = get_index_of_first_vehicle_with_id(simstate.scene, id)
            veh_state = veh_ids_to_states[id]

            simstate.scene[veh_index].state = veh_state
        end

    else
        veh_ego = get_vehicle(simstate.scene, simstate.egoid)
        action_ego = convert(simparams.ego_action_type, action_ego)

        ego_state = propagate(veh_ego, action_ego, simparams.context, trajdata.roadway)

        simstate.frame += 1

        #######
        # Use Playback Reactive here
        #######

        # pull new frame from trajdata
        get!(simstate.scene, trajdata, simstate.frame)

        # move in propagated ego vehicle
        veh_index = get_index_of_first_vehicle_with_id(simstate.scene,
                                                       simstate.egoid)
        simstate.scene[veh_index].state = ego_state

    end

    # update record
    update!(simstate.rec, simstate.scene)

    simstate
end

function tick(simparams::SimParams, u::Vector{Float64}, batch_index::Int=1)
    step_forward!(simparams.simstates[batch_index], simparams, u)
    simparams
end

function Base.step(simparams::SimParams, u::Vector{Float64}, batch_index::Int=1)
    tick(simparams, u, batch_index)
    simparams.step_counter += 1
    #@printf "simparams.step_counter: %d \n" simparams.step_counter
end

end # module

