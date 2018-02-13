include("Auto2D.jl")
include("load_evaluation_data.jl")
using Auto2D

#println("Test NGSIM ...")
##X = zeros(UInt32, (500,500))
#simparams = gen_simparams(1, Dict("trajdata_indices" => [1]))
##println("observe 1")
##observe(simparams)
#
## I think I need to reset in order to set the trajdata_index ...
#simparams = reset(simparams)
#observe_state(simparams)
#observe(simparams)
##Xprime = retrieve_image(simparams, X)

println("Test track ...")
D = Dict("mix_class" => true,
         "model_all" => true,
         "domain_indices" => [0],
         "use_valid" => true)

simparams = gen_simparams(D)
simparams = reset(simparams)
rollout_ego_features(simparams)

assert(false)

# I think I need to reset in order to set the trajdata_index ...
n_egos = 5
simparams = reset(simparams, n_egos)
initial_simparams = deepcopy(simparams)
A = Matrix(randn((2 * n_egos,15)))

reel_drive("filename", A, initial_simparams)

#Xprime = retrieve_image(simparams, X)
