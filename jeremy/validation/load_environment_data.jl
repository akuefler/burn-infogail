using AutomotiveDrivingModels
using NGSIM
using HDF5
using JLD

const TRAJDATA_FILEPATH = "/home/alex/stanford_dev/burn-infogail/jeremy/2d_drive_data/trajdata/"
const CAR_CLASS_FILEPATH = "/home/alex/stanford_dev/burn-infogail/jeremy/2d_drive_data/car_classes/"
const NUM_SEG = 3           # Number of segments to divide each trajdata into

export
    load_envdata, load_classdict

#####
# NGSIM Environment
#####
include("../validation/load_train_test_split.jl")
#include("../pull_traces/multifeatureset.jl")
include("../validation/RootDir.jl")

function get_train_segments(trajdatas::Dict{Int, Trajdata}, nsteps::Int)

    assert(false) # load_evaldata can't be right ..
    assignment = load_assignment()
    evaldata = load_evaldata()

    # pull all segments that are for training
    all_train_segments = evaldata.segments[find(assignment .== FOLD_TRAIN)]

    # only keep segments that are long enough and are not within 500 frames of start or end
    train_segments = TrajdataSegment[]
    for seg in all_train_segments
        if !haskey(trajdatas, seg.trajdata_index) # only keep valid segments
            continue
        end
        trajdata = trajdatas[seg.trajdata_index]
        frame_lo = max(seg.frame_lo, 500)
        frame_hi = min(seg.frame_hi, nframes(trajdata) - 1000)
        if frame_hi - frame_lo > nsteps # if it is long enough
            push!(train_segments, TrajdataSegment(seg.trajdata_index, seg.egoid, frame_lo, frame_hi))
        end
    end

    train_segments
end

#####
# Racing Environment
#####

# Load trajdata from given file
function load_track_trajdata(data_ix::Int, ncars::Int, mix::Bool, valid::Bool)

    #data_ix = (i % NUM_TRAJDATA) + 1
    if mix
        trajdata_name = "_mix_"
    else
        trajdata_name = "_single_"
    end
    if valid
        trajdata_name *= "VALID"
    else
        trajdata_name *= "TRAIN"
    end
    filepath = TRAJDATA_FILEPATH * "trajdata_ncars" * string(ncars) * trajdata_name * string(data_ix) * ".txt"
    td = open(io->read(io, Trajdata), filepath, "r")
    td.roadway = gen_stadium_roadway(3, length=250.0, radius=45.0)
    td
end

# Load dictionaries mapping vehicle to driving class
function load_classdict(data_ix::Int, ncars::Int, mix::Bool, valid::Bool)

    #data_ix = (i % NUM_TRAJDATA) + 1
    if mix
        trajdata_name = "_mix_"
    else
        trajdata_name = "_single_"
    end
    if valid
        trajdata_name *= "VALID"
    else
        trajdata_name *= "TRAIN"
    end
    filepath = CAR_CLASS_FILEPATH * "car_classes_ncars" * string(ncars) * trajdata_name * string(data_ix) * ".jld"
    cd = JLD.load(filepath, "classes")
    cd
end

function load_envdata(mix::Bool, ncars::Int, tid::Int, valid::Bool)
    # Extract trajdatas
    if valid
        num_trajdata = 5
    else
        num_trajdata = 10
    end
    trajdatas = map(i->load_track_trajdata((i % num_trajdata)+1, ncars, mix, valid), 1:(num_trajdata))
    classdicts = map(i->load_classdict((i % num_trajdata)+1, ncars, mix, valid), 1:(num_trajdata))

    # Loop over each trajdata, vehicle, and segment
    segments = Array(TrajdataSegment, num_trajdata*ncars*NUM_SEG)
    classes = Array(Int, num_trajdata*ncars*NUM_SEG)
    for i = 1:(num_trajdata)
        n_vehicles = length(trajdatas[i].vehdefs)
        for j = 1:n_vehicles
            for k = 1:NUM_SEG
                # Find index for individual segment
                idx = n_vehicles*NUM_SEG*(i-1) + NUM_SEG*(j-1) + k
                #segments[idx] = TrajdataSegment(i, j, (k-1)*100+1, k*100)
                segments[idx] = TrajdataSegment(tid + i, j, (k-1)*100+1, k*100)
                classes[idx] = classdicts[i][j]     # dict of dicts, need to index twice
            end
        end
    end

    EvaluationData(trajdatas, segments), classes, num_trajdata
end
