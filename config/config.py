# random seed
# TimesNet, TimeXer, PatchTST 2021
# iTransformer 2023

DEFAULT_DATASET_SETTINGS = {
    'ETTh1': {
        'root_path': '/data/pcw_workspace/Time-Series-Library/dataset/ETT-small/',
        'data_path': 'ETTh1.csv',
        'freq': 'h',
        'mark_in' : ['HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear'],
        'targets': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'PatchTST': {
            'e_layers': 3,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 128,
            'seq_len' : 336,
            'n_heads' : 4,
            'dropout' : 0.3,
            'd_ff': 128,
            'd_model': 16,
            'lradj' : 'type3',
            'random_seed' : 2021
        },
        'iTransformer': {
            'e_layers': 2,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 32,
            'random_seed' : 2023,
            'd_ff': {
                96: 128,
                192: 128,
                336: 512,
                720: 512
            },
            'd_model': {
                96: 128,
                192: 128,
                336: 512,
                720: 512
            }
        },
        'TimesNet': {
            'e_layers': 2,
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'batch_size' : 32,
            'd_ff': 32,
            'd_model': 16,
            'random_seed' : 2021
        },
        'TimeXer': {
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'random_seed' : 2021,
            'e_layers' : {
                96: 1,
                192: 2,
                336: 1,
                720: 1
            },
            'batch_size' : {
                96: 4,
                192: 4,
                336: 16,
                720: 16
            },
            'd_ff': {
                96: 2048,
                192: 2048,
                336: 1024,
                720: 1024
            },
            'd_model': {
                96: 256,
                192: 128,
                336: 512,
                720: 256
            }
        },
    },
        'ETTh2': {
        'root_path': '/data/pcw_workspace/Time-Series-Library/dataset/ETT-small/',
        'data_path': 'ETTh2.csv',
        'freq': 'h',
        'mark_in' : ['HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear'],
        'targets': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'PatchTST': {
            'e_layers': 3,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 128,
            'seq_len' : 336,
            'n_heads' : 4,
            'dropout' : 0.3,
            'd_ff': 128,
            'd_model': 16,
            'lradj' : 'type3',
            'random_seed' : 2021
        },
        'iTransformer': {
            'e_layers': 2,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 32,
            'd_ff': 128,
            'd_model': 128,
            'random_seed' : 2023
        },
        'TimesNet': {
            'e_layers': 2,
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'batch_size' : 32,
            'd_ff': 32,
            'd_model': 32,
            'random_seed' : 2021
        },
        'TimeXer': {
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'random_seed' : 2021,
            'e_layers' : {
                96: 1,
                192: 1,
                336: 2,
                720: 2
            },
            'batch_size' : {
                96: 16,
                192: 32,
                336: 4,
                720: 16
            },
            'd_ff': 1024,
            'd_model': {
                96: 256,
                192: 256,
                336: 512,
                720: 256
            }
        },
    },
        'ETTm1': {
        'root_path': '/data/pcw_workspace/Time-Series-Library/dataset/ETT-small/',
        'data_path': 'ETTm1.csv',
        'freq': 't',
        'mark_in' : ['MinuteOfHour', 'HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear'],
        'targets': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'PatchTST': {
            'e_layers': 3,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 128,
            'seq_len' : 336,
            'n_heads' : 16,
            'dropout' : 0.2,
            'd_ff': 256,
            'd_model': 128,
            'lradj' : 'TST',
            'pct_start' : 0.4,
            'random_seed' : 2021
        },
        'iTransformer': {
            'e_layers': 2,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 32,
            'd_ff': 128,
            'd_model': 128,
            'random_seed' : 2023
        },
        'TimesNet': {
            'e_layers': 2,
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'batch_size' : 32,
            'random_seed' : 2021,
            'd_ff': {
                96: 64,
                192: 64,
                336: 32,
                720: 32
            },
            'd_model': {
                96: 64,
                192: 64,
                336: 16,
                720: 16
            }
        },
        'TimeXer': {
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'e_layers' : 1,
            'batch_size' : 4,
            'random_seed' : 2021,
            'd_ff': {
                96: 2048,
                192: 256,
                336: 1024,
                720: 512
            },
            'd_model': 256
        },
    },
        'ETTm2': {
        'root_path': '/data/pcw_workspace/Time-Series-Library/dataset/ETT-small/',
        'data_path': 'ETTm2.csv',
        'freq': 't',
        'mark_in' : ['MinuteOfHour', 'HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear'],
        'targets': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'PatchTST': {
            'e_layers': 3,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 128,
            'seq_len' : 336,
            'n_heads' : 16,
            'dropout' : 0.2,
            'd_ff': 256,
            'd_model': 128,
            'lradj' : 'TST',
            'pct_start' : 0.4,
            'random_seed' : 2021
        },
        'iTransformer': {
            'e_layers': 2,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 32,
            'd_ff': 128,
            'd_model': 128,
            'random_seed' : 2023
        },
        'TimesNet': {
            'e_layers': 2,
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'batch_size' : 32,
            'd_ff': 32,
            'random_seed' : 2021,
            'd_model': {
                96: 32,
                192: 32,
                336: 32,
                720: 16
            }
        },
        'TimeXer': {
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'e_layers' : 1,
            'random_seed' : 2021,
            'batch_size' : {
                96: 32,
                192: 16,
                336: 32,
                720: 32
            },
            'd_ff': {
                96: 2048,
                192: 1024,
                336: 1024,
                720: 2048
            },
            'd_model': {
                96: 256,
                192: 256,
                336: 512,
                720: 512
            }
        },
    },
        'exchange_rate': {
        'root_path': '/data/pcw_workspace/Time-Series-Library/dataset/exchange_rate/',
        'data_path': 'exchange_rate.csv',
        'freq': 'd',
        'mark_in' : ['DayOfWeek', 'DayOfMonth', 'DayOfYear'],
        'targets': [str(num) for num in range(7)] + ['OT'],
        'iTransformer': {
            'e_layers': 2,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 32,
            'd_ff': 128,
            'd_model': 128,
            'random_seed' : 2023
        },
        'TimesNet': {
            'e_layers': 2,
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'batch_size' : 32,
            'random_seed' : 2021,
            'd_ff': {
                96: 64,
                192: 64,
                336: 32,
                720: 32
            },
            'd_model': {
                96: 64,
                192: 64,
                336: 32,
                720: 32
            }
        }, # TimeXer, PatchTST 논문X
    },
        'weather': {
        'root_path': '/data/pcw_workspace/Time-Series-Library/dataset/weather/',
        'data_path': 'weather.csv',
        'freq': 't',
        'mark_in' : ['MinuteOfHour', 'HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear'],
        'targets': ['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 
                    'VPact', 'VPdef', 'sh', 'H2OC', 'rho', 
                    'wv', 'max.wv', 'wd', 'rain', 'raining', 'SWDR', 
                    'PAR', 'max.PAR', 'Tlog', 'OT'],
        'PatchTST': {
            'e_layers': 3,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 128,
            'seq_len' : 336,
            'n_heads' : 16,
            'dropout' : 0.2,
            'd_ff': 256,
            'd_model': 128,
            'lradj' : 'type3',
            'random_seed' : 2021
        },
        'iTransformer': {
            'e_layers': 3,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 32,
            'd_ff': 512,
            'd_model': 512,
            'random_seed' : 2023
        },
        'TimesNet': {
            'e_layers': 2,
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'batch_size' : 32,
            'd_ff': 32,
            'd_model': 32,
            'random_seed' : 2021
        },
        'TimeXer': {
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'random_seed' : 2021,
            'e_layers' : {
                96: 1,
                192: 3,
                336: 1,
                720: 1
            },
            'batch_size' : 4,
            'd_ff': {
                96: 512,
                192: 1024,
                336: 2048,
                720: 2048
            },
            'd_model': {
                96: 256,
                192: 128,
                336: 256,
                720: 128
            }
        },
    },
        'ECL': {
        'root_path': '/data/pcw_workspace/Time-Series-Library/dataset/electricity/',
        'data_path': 'electricity.csv',
        'freq': 'h',
        'mark_in' : ['HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear'],
        'targets': [str(num) for num in range(320)] + ['OT'],
        'PatchTST': {
            'e_layers': 3,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 32,
            'seq_len' : 336,
            'n_heads' : 16,
            'dropout' : 0.2,
            'd_ff': 256,
            'd_model': 128,
            'lradj' : 'TST',
            'pct_start' : 0.2,
            'random_seed' : 2021
        },
        'iTransformer': {
            'e_layers': 3,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 16,
            'learning_rate' : 0.0005,
            'd_ff': 512,
            'd_model': 512,
            'random_seed' : 2023
        },
        'TimesNet': {
            'e_layers': 2,
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'batch_size' : 32,
            'd_ff': 512,
            'd_model': 256,
            'random_seed' : 2021
        },
        'TimeXer': {
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'random_seed' : 2021,
            'e_layers' : {
                96: 4,
                192: 3,
                336: 4,
                720: 3
            },
            'batch_size' : 4,
            'd_ff': {
                96: 512,
                192: 2048,
                336: 2048,
                720: 2048
            },
            'd_model': 512
        },
    },
        'traffic': {
        'root_path': '/data/pcw_workspace/Time-Series-Library/dataset/traffic/',
        'data_path': 'traffic.csv',
        'freq': 'h',
        'mark_in' : ['HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear'],
        'targets': [str(num) for num in range(861)] + ['OT'],
        'PatchTST': {
            'e_layers': 3,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 24,
            'seq_len' : 336,
            'n_heads' : 16,
            'dropout' : 0.2,
            'd_ff': 256,
            'd_model': 128,
            'lradj' : 'TST',
            'pct_start' : 0.2,
            'random_seed' : 2021
        },
        'iTransformer': {
            'e_layers': 4,
            'd_layers': 1,
            'factor': 1,
            'label_len': 0,
            'batch_size' : 16,
            'learning_rate' : 0.001,
            'd_ff': 512,
            'd_model': 512,
            'random_seed' : 2023
        },
        'TimesNet': {
            'e_layers': 2,
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'batch_size' : 32,
            'd_ff': 512,
            'd_model': 512,
            'random_seed' : 2021
        },
        'TimeXer': {
            'd_layers': 1,
            'factor': 3,
            'label_len': 48,
            'random_seed' : 2021,
            'e_layers' : {
                96: 1,
                192: 3,
                336: 1,
                720: 1
            },
            'batch_size' : 4,
            'd_ff': {
                96: 512,
                192: 1024,
                336: 2048,
                720: 2048
            },
            'd_model': {
                96: 256,
                192: 128,
                336: 256,
                720: 128
            }
        },
    },
}