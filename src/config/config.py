class Config(object):
    def __init__(self):

        self.data_path_tudo = "./data/01_raw/TU_DO"
        self.data_path_duetal = "./data/01_raw/Du_et_al"
        self.data_path = "./data/01_raw"
        self.matrix_path_tudo = "./data/02_matrices/GNTK_TU_DO"
        self.matrix_path_duetal = "./data/02_matrices/GNTK_Du_et_al"
        self.matrix_path = "./data/02_matrices"
        self.exp_path = "./data/03_experiments"
        self.eval_path = "./data/04_evaluation"
        self.reporting_path = "./reporting"


class CVConfig(object):
    def __init__(self, exp):

        self.k_fold = 10
        self.inner_k_fold = 5
        self.max_iter = 10000

        if exp == "a.1":
            self.data_source = "Du_et_al"
            self.min_scale_mat = 1
            self.C_num = 120
            self.balanced = 0
            self.fold_files = 1
            self.n_iter = 1
            self.nested = 0
        elif exp == "a.2":
            self.data_source = "Du_et_al"
            self.min_scale_mat = 1
            self.C_num = 20
            self.balanced = 0
            self.fold_files = 1
            self.n_iter = 1
            self.nested = 1
        elif exp == "b.1":
            self.data_source = "TU_DO"
            self.min_scale_mat = 0
            self.C_num = 20
            self.balanced = 1
            self.fold_files = 0
            self.n_iter = 10
            self.nested = 1
        else:
            raise ValueError(f"Unknown experiment {exp}")


class GINConfig(object):
    def __init__(self, dataset):
        self.batch_size = 32
        self.iters_per_epoch = 50
        self.epochs = 350
        self.lr = 0.01
        self.device = 0
        self.hidden_dim = 64
        self.final_dropout = 0.5
        self.graph_pooling_type = "sum"

        if dataset == "MUTAG":
            self.num_layers = 1
            self.num_mlp_layers = 2
            self.neighbor_pooling_type = "sum"
        elif dataset == "PTC":
            self.num_layers = 14
            self.num_mlp_layers = 2
            self.neighbor_pooling_type = "average"
        elif dataset == "IMDBBINARY":
            self.num_layers = 1
            self.num_mlp_layers = 1
            self.neighbor_pooling_type = "average"
        elif dataset == "IMDBMULTI":
            self.num_layers = 1
            self.num_mlp_layers = 1
            self.neighbor_pooling_type = "average"
        elif dataset == "PROTEINS":
            self.num_layers = 13
            self.num_mlp_layers = 2
            self.neighbor_pooling_type = "average"
        elif dataset == "NCI1":
            self.num_layers = 12
            self.num_mlp_layers = 3
            self.neighbor_pooling_type = "average"
        else:
            raise ValueError(f"Unknown dataset specified: {dataset}")


class TimingExpConfig(object):
    def __init__(self):
        self.GNTK = {
            "IMDBBINARY": {
                "n_blocks": 1,
                "n_fc_layers": 2,
                "scale": "uniform",
                "jk": True,
                "normalize": True,
            },
            "IMDBMULTI": {
                "n_blocks": 1,
                "n_fc_layers": 1,
                "scale": "degree",
                "jk": True,
                "normalize": False,
            },
            "MUTAG": {
                "n_blocks": 1,
                "n_fc_layers": 2,
                "scale": "uniform",
                "jk": True,
                "normalize": False,
            },
            "NCI1": {
                "n_blocks": 11,
                "n_fc_layers": 3,
                "scale": "degree",
                "jk": False,
                "normalize": False,
            },
            "PROTEINS": {
                "n_blocks": 14,
                "n_fc_layers": 3,
                "scale": "degree",
                "jk": False,
                "normalize": True,
            },
            "PTC": {
                "n_blocks": 13,
                "n_fc_layers": 3,
                "scale": "degree",
                "jk": False,
                "normalize": True,
            },
        }
        self.WL = {
            "IMDBBINARY": 1,
            "IMDBMULTI": 1,
            "MUTAG": 2,
            "NCI1": 7,
            "PROTEINS": 4,
            "PTC": 7,
        }
