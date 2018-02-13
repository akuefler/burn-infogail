expert_data_paths = {
        "PP":"models/17-03-10/PP-EXPERT-G10-0",
        #"Racing-State":"expert_trajs/racing/Racing-State-0",
	"LL":"models/17-04-07/ParamEnv-LL-0",
	"RA":"models/17-04-07/RA-2",
	"RS":"models/17-04-10/RS-0",
    "JT":"expert_trajs/juliaTrack",
    "JTZM":"expert_trajs/juliaTrack_mix",
    "JTZS":"expert_trajs/juliaTrack_single",
    "JNGSIM":"expert_trajs/juliaNGSIM"
    }

policy_paths = {
        "VAE":"jeremy/train_vae/policy_vae.h5",
        }

normalizer_paths = {
        "JNGSIM":"expert_trajs/juliaNGSIM/norm.h5",
        "ORIG":"data/original_norm.h5"
        }

best_epochs = {
        "../data/models/17-06-13/CORL2-06130728-JTZM-4" : 97,
        "../data/models/17-06-13/CORL2-06130728-JTZM-7" : 99,
        "../data/models/17-06-19/CORL4-06182344-JTZM-1": 99
        }
