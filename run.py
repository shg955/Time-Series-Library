import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from config.config import *
from torch import nn

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def main(args):
    params = {k:str(v) for k, v in vars(args).items()}
    # 실험 이름 생성
    experiment_name = f"{args.model}_{args.data}_{args.seq_len}_{args.pred_len}_{args.label_len}"
    #print(params)
    #exit()
    # 실험 생성 (artifact_location은 데이터셋 이름으로 설정)
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            #artifact_location=f"mlflow/mlruns/{args.data}",  # 데이터셋 이름으로 artifact_location 설정
            tags=params  # 입력받은 하이퍼파라미터를 tags로 설정
        )
    except Exception as e:
        # 이미 존재하는 실험일 경우, 해당 실험 ID를 가져옴
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        # 실험 복구
        if experiment.lifecycle_stage == "deleted":
            client = MlflowClient()
            client.restore_experiment(experiment_id)

    # 실험 가져오기
    experiment = mlflow.get_experiment(experiment_id)
    print(f"Experiment ID: {experiment.experiment_id}, Name: {experiment.name}")
    # exit()
    
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        print(f"Current artifact uri: {mlflow.get_artifact_uri()}")
        
        # MLflow로 하이퍼파라미터 로깅
        mlflow.log_params(params)  # 하이퍼파라미터 로깅
        # mlflow.log_artifact("/root/workspace/Time-Series-Library/test_results/short_term_forecast_m4_Monthly_DLinear_m4_ftM_sl36_ll18_pl18_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/0.png")
        # exit()
        
        if torch.cuda.is_available() and args.use_gpu:
            args.device = torch.device('cuda:{}'.format(args.gpu))
            print('Using GPU')
        else:
            if hasattr(torch.backends, "mps"):
                args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            else:
                args.device = torch.device("cpu")
            print('Using cpu or mps')

        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]

        print('Args in experiment:')
        print_args(args)

        if args.task_name == 'long_term_forecast' or args.task_name == 'paper':
            Exp = Exp_Long_Term_Forecast
        elif args.task_name == 'short_term_forecast':
            Exp = Exp_Short_Term_Forecast
        elif args.task_name == 'imputation':
            Exp = Exp_Imputation
        elif args.task_name == 'anomaly_detection':
            Exp = Exp_Anomaly_Detection
        elif args.task_name == 'classification':
            Exp = Exp_Classification
        
        if args.is_tsne_emb:
            exp = Exp(args)
            ii = 0
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            
            print('>>>>>>>save tsne : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.tsne_emb(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()

        elif args.is_training:
            for ii in range(args.itr):
                # setting record of experiments
                exp = Exp(args)  # set experiments
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.expand,
                    args.d_conv,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des, ii)

                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

                # make report csv file
                print('>>>>>>>summarizing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.summarize()
                    
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
        elif args.is_training == 0:
            exp = Exp(args)  # set experiments
            ii = 0
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
            

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://ubinetlab.iptime.org:15000")
    tracking_uri = mlflow.get_tracking_uri()
    print(f"Current tracking uri: {tracking_uri}")
    fix_seed = 2021
    # iTransformer 2023 TimesNet 2021 TimeXer 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='paper',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    # parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--dataset', type=str, required=True, default='ETTh1', help='dataset name')
    parser.add_argument('--is_tsne_emb', action='store_true', help='visualize embedding')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='SNP500', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/data/pcw_workspace/Quant_snp/dataset/DAN_Dataset_11_07/Input', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='Close', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='/data/pcw_workspace/Time-Series-Library/checkpoints', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--mark_in', type=str, nargs='+', default=None, help='date feature')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=32, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=150, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=str2bool, nargs='?', default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--augmentation_prob', type=float, default=0.0, help="Probability of applying augmentation per sample (0.0 ~ 1.0)")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    
    # iaaft
    parser.add_argument('--iaaft', default=False, action='store_true', help='Apply IAAFT surrogate augmentation')
    # timeGAN
    parser.add_argument('--timeGAN', default=False, action="store_true", help="Use generative model (e.g., TimeGAN) for augmentation")
    parser.add_argument('--n_generate', type=int, default=1, help='Number of synthetic samples per input when using generative model')

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # DeformableTST
    parser.add_argument('--stem_ratio', type=int, default=1, help='number of downsampling steps')
    parser.add_argument('--expansion', type=int, default=4, help='ffn ratio')

    parser.add_argument('--summarize_only', action='store_true', help='Only summarize feature loss')
    parser.add_argument('--reconstruction', action='store_true', help='reconstruction')
    parser.add_argument('--channelwise_embedding', action='store_true', help='use channel-wise embedding')
    parser.add_argument('--channelwise_projection', action='store_true', help='use channel-wise projection')
    parser.add_argument('--position_encoding_emb', action='store_true', help='use position encoding parameter embedding')
    parser.add_argument('--position_encoding_proj', action='store_true', help='use position encoding parameter projection')
    parser.add_argument('--position_embedding_emb', action='store_true', help='use position embedding parameter embedding')
    parser.add_argument('--position_embedding_proj', action='store_true', help='use position embedding parameter projection')
    parser.add_argument('--use_separate', action='store_true', help='use embedding projection different parameter')
    parser.add_argument('--position_embedding_weight', action='store_true', help='use positional embedding weight')
    parser.add_argument('--each_weight', action='store_true', help='all feature, each feature differenct weight')
    parser.add_argument('--pe_weight', type=float, default=0.1, help='positional embedidng weight')
    parser.add_argument('--pe_weight_activation', type=str, default='identity', choices=['identity', 'sigmoid', 'relu'], help='Activation function for positional embedding weights')

    # Parameters for PS Loss
    parser.add_argument('--use_ps_loss', action='store_true', help='use PSLoss True')
    parser.add_argument('--ps_lambda', type=float, default=3.0, help='weight for ps_loss')
    parser.add_argument('--patch_len_threshold', type=int, default=24, help='patch length threshold')
    parser.add_argument('--name', type=str, default="", help="information model name")

    # iTransformer에 STFT적용했을 때
    # parser.add_argument("--hop_length", type=int, default=1)
    # parser.add_argument("--n_fft", type=int, default=4)

    # store_true : 해당 인자가 존재하면 T/ 아니면 F

    args = parser.parse_args()

    args.data = args.dataset if args.dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'] else 'custom'
    args.freq = DEFAULT_DATASET_SETTINGS[args.dataset]['freq']
    args.mark_in = DEFAULT_DATASET_SETTINGS[args.dataset]['mark_in']
    args.root_path = DEFAULT_DATASET_SETTINGS[args.dataset]['root_path']
    args.data_path = DEFAULT_DATASET_SETTINGS[args.dataset]['data_path']

    # args.batch_size = DEFAULT_DATASET_SETTINGS[args.dataset][args.model]['batch_size']
    # args.e_layers = DEFAULT_DATASET_SETTINGS[args.dataset][args.model]['e_layers']
    args.d_layers = DEFAULT_DATASET_SETTINGS[args.dataset][args.model]['d_layers']
    args.factor = DEFAULT_DATASET_SETTINGS[args.dataset][args.model]['factor']
    args.label_len = DEFAULT_DATASET_SETTINGS[args.dataset][args.model]['label_len']
    args.learning_rate = DEFAULT_DATASET_SETTINGS[args.dataset][args.model]['learning_rate']
    
    # paper, recon, psloss
    # for experiment_type in [(False, False),(True, False),(True, True)]:
    # for experiment_type in [(False, False)]:
    experiment_type = (False, False)
    args.is_tsne_emb = False
    # args.position_embedding_weight = True
    # args.each_weight = True

    activation_map = {
    'identity': nn.Identity(),
    'sigmoid': nn.Sigmoid(),
    'relu': nn.ReLU(),
    'tanh' : nn.Tanh()
    }
    
    # channelwise Projection, embedding, projection+embedding
    # for position_setting in [(True, False), (False, True), (True, True)]:
    # for position_setting in [(False, False, True), (True, False, True), (False, True, True), (True, True, True), (True, True, False)]:
    for position_setting in [(False, False, True), (True, False, True)]:
        args.reconstruction, args.use_ps_loss = experiment_type
        # args.position_encoding_emb, args.position_encoding_proj = position_setting
        args.position_embedding_emb, args.position_embedding_proj, args.use_separate = position_setting

        # positional embedding weight 적용 (True, False, True)
        # for position_weights in [round(x * 0.1, 1) for x in range(1, 10)]:
        # for position_weights in [0.1]:
        #     args.pe_weight = position_weights
            
        #     for func in ['identity']:
        #         args.pe_weight_activation = activation_map[func]
      
        # M MS S
        # for feature_mode in ['M', 'MS', 'S']:
        for feature_mode in ['M']:
            args.features = feature_mode
            args.enc_in = len(DEFAULT_DATASET_SETTINGS[args.dataset]['targets']) if feature_mode != 'S' else 1
            args.dec_in = len(DEFAULT_DATASET_SETTINGS[args.dataset]['targets']) if feature_mode != 'S' else 1
            args.c_out = len(DEFAULT_DATASET_SETTINGS[args.dataset]['targets']) if feature_mode != 'S' else 1

            # target (MS, S일때만)
            for target in DEFAULT_DATASET_SETTINGS[args.dataset]['targets']:
                args.target = target
                # prediction length, d_model도 달라지게 d_ff
                for pred_len in [96, 192, 336, 720]:
                    args.pred_len = pred_len
                    args.d_ff = DEFAULT_DATASET_SETTINGS[args.dataset][args.model]['d_ff'][pred_len]
                    args.d_model = DEFAULT_DATASET_SETTINGS[args.dataset][args.model]['d_model'][pred_len]
                    args.e_layers = DEFAULT_DATASET_SETTINGS[args.dataset][args.model]['e_layers'][pred_len]
                    args.batch_size = DEFAULT_DATASET_SETTINGS[args.dataset][args.model]['batch_size'][pred_len]

                    args.model_id = f'{args.model}_{args.dataset}_{args.seq_len}_{args.pred_len}_{args.features}'
                    args.model_id = args.model_id + f'_{args.target}' if args.features != 'M' else args.model_id
                    args.model_id = args.model_id + '_recon' if args.reconstruction else args.model_id
                    args.model_id = args.model_id + '_ps' if args.use_ps_loss else args.model_id
                    args.model_id = args.model_id + '_posiemb_emb' if args.position_embedding_emb else args.model_id
                    args.model_id = args.model_id + '_posiemb_proj' if args.position_embedding_proj else args.model_id
                    args.model_id = args.model_id + '_share' if not args.use_separate else args.model_id
                    args.model_id = args.model_id + '_posienc_emb' if args.position_encoding_emb else args.model_id
                    args.model_id = args.model_id + '_posienc_proj' if args.position_encoding_proj else args.model_id
                    # args.model_id = args.model_id + f'_weight_{args.pe_weight}' if args.position_embedding_weight else args.model_id
                    # args.model_id = args.model_id + '_all' if not args.each_weight else args.model_id
                    # args.model_id = args.model_id + f'_{func}' if func != 'identity' else args.model_id

                    suffix = ""
                    if args.reconstruction:
                        suffix = "_recon_ps" if args.use_ps_loss else "_recon"
                    elif args.use_ps_loss:
                        suffix = "_ps" if args.use_ps_loss else ""
                    elif args.position_embedding_emb or args.position_embedding_proj:
                        suffix = "_posiemb"
                        if args.position_embedding_weight:
                            suffix += "_weight"
                    elif args.position_encoding_emb or args.position_encoding_proj:
                        suffix = "_posienc"
                    elif args.channelwise_embedding or args.channelwise_projection:
                        suffix = "_channelwise"
                    
                    current_path = f'/data/pcw_workspace/Time-Series-Library/checkpoints/{args.model}/{args.dataset}{suffix}/{args.model_id}'
                    
                    if args.is_tsne_emb:
                        if not os.path.exists(current_path):
                            print(f'{args.model_id} not found. Skip TSNE.')
                            continue
                    elif args.is_training == 1:
                        if os.path.exists(current_path):
                            print(f'{args.model_id} experiments already run!')
                            continue

                    main(args)
                
                if feature_mode == 'M':
                    break