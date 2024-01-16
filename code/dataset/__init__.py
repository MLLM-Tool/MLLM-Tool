from header import *
from .samplers import DistributedBatchSampler, DistributedMultiDatasetBatchSampler
from .catalog import DatasetCatalog
from torch.utils.data import ConcatDataset
from .concat_dataset import MyConcatDataset

'''
def get_tokenizer(model):
    tokenizer = LlamaTokenizer.from_pretrained(model)
    tokenizer.bos_token_id, tokenizer.eos_token_id = 1, 2
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
'''

def load_dataset(args, dataset_name_list):
    '''
    tokenizer = get_tokenizer(args['model_path'])
    dataset_name = args['models'][args['model']]['stage1_train_dataset'] # SupervisedDataset, str
    data_path = args["data_path"]
    data = globals()[dataset_name](data_path, tokenizer, args['max_length']) #SupervisedDataset
    '''
    concat_data = MyConcatDataset(dataset_name_list)

    sampler = torch.utils.data.RandomSampler(concat_data)
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedMultiDatasetBatchSampler(dataset=concat_data,
                                                        sampler=sampler,
                                                        batch_size=batch_size,
                                                        drop_last=True,
                                                        rank=rank,
                                                        world_size=world_size)
    iter_ = DataLoader(
        concat_data, 
        batch_sampler=batch_sampler, 
        num_workers=1,
        collate_fn=concat_data.collate, 
        pin_memory=True
    )
    return concat_data, iter_, sampler
