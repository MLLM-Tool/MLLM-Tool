from header import *
from dataset import *
from model import *
from config import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str)
    # parser.add_argument('--data_path', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--log_path', type=str)
    # model configurations
    # parser.add_argument('--image_root_path', type=str) # the directory that stores all images
    parser.add_argument('--imagebind_ckpt_path', type=str) # the path that stores the imagebind checkpoint
    parser.add_argument('--llm_ckpt_path', type=str) # the path that stores the llm checkpoint
    parser.add_argument('--delta_ckpt_path', type=str) # the delta parameters trained in stage 1
    parser.add_argument('--max_tgt_len', type=int) # the maximum sequence length
    parser.add_argument('--stage', type=int) # the maximum sequence length
    parser.add_argument('--epochs', type=int) # the number of training epochs
    parser.add_argument('--version', type=str) # the name of ckpt file
    return parser.parse_args()

def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def config_env(args):
    args['root_dir'] = '../'
    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])

def build_directory(path):
    if os.path.exists(path):
        pass
    else: # recursively construct directory
        os.makedirs(path, exist_ok=True)

def main(**args):
    config_env(args)
    args['ds_config_path'] = f'dsconfig/{args["model"]}_stage_{args["stage"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    build_directory(args['save_path'])
    build_directory(args['log_path'])

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', 
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )
    train_data, train_iter, sampler = load_dataset(args, args['dataset_name_list'])

    train_num = max([_cur_dataset.__len__() for _cur_dataset in train_data.datasets.datasets]) * len(train_data.datasets.datasets)
    length = args['epochs'] * train_num // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps = args['epochs'] * train_num // dschf.config['train_batch_size']
    args['total_steps'] = total_steps
    agent = load_model(args)
    torch.distributed.barrier()

    # begin to train
    pbar = tqdm(total=length)    # maximum total number
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        for batch in train_iter:
            
            # torch.distributed.barrier()
    
            # agent.save_model(args['save_path'], 0)
            agent.train_model(
                batch, 
                current_step=current_step, 
                pbar=pbar
            )
            current_step += 1
            if current_step%200==0:
                torch.distributed.barrier()
                agent.save_model(args['save_path'], current_step)
    # save at the end of the training
    torch.distributed.barrier()
    
    agent.save_model(args['save_path'], 0)

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
