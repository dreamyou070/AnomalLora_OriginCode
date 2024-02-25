import os
from model.tokenizer import load_tokenizer
from data.mvtec import MVTecDRAEMTrainDataset
from data.mvtec_cropping import MVTecDRAEMTrainDataset_Cropping
import torch


def call_dataset(args) :

    tokenizer = load_tokenizer(args)

    # [1] set root data
    if args.reference_check:
        root_dir = os.path.join(args.data_path, f'{args.obj_name}/test')
    else:
        root_dir = os.path.join(args.data_path, f'{args.obj_name}/train')

    if args.cropping_test:
        root_dir = os.path.join(args.data_path, f'{args.obj_name}/train_cropping')


    # [2] set anomaly source path
    if args.use_small_anomal:
        args.anomal_source_path = os.path.join(args.data_path, f"anomal_source_{args.obj_name}")

    data_class = MVTecDRAEMTrainDataset
    if args.cropping_test :
        data_class = MVTecDRAEMTrainDataset_Cropping
        print(f'cropping_test clss = {data_class.__class__.__name__}')
    dataset = data_class(root_dir=root_dir,
                                     anomaly_source_path=args.anomal_source_path,
                                     resize_shape=[512, 512],
                                     tokenizer=tokenizer,
                                     caption=args.trigger_word,
                                     use_perlin=True,
                                     anomal_only_on_object=args.anomal_only_on_object,
                                     anomal_training=True,
                                     latent_res=args.latent_res,
                                     kernel_size=args.kernel_size,
                                     beta_scale_factor=args.beta_scale_factor,
                                     reference_check = args.reference_check,
                                     do_anomal_sample =args.do_anomal_sample,)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    return dataloader

