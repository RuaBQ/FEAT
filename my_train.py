import json

import numpy as np
import torch

from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (get_command_line_parser, postprocess_args, pprint,
                         set_gpu)

# from ipdb import launch_ipdb_on_exception


if __name__ == '__main__':

    parser = get_command_line_parser()
    args = parser.parse_args()

    for dataset in ['TieredImageNet', 'MiniImageNet']:
        for shot in ['1', '5']:

            with open('args/' + dataset + ' shot-'+str(shot) + '.txt', 'r') as f:
                args.__dict__ = json.load(f)

            # with launch_ipdb_on_exception():
            pprint(vars(args))

            set_gpu(args.gpu)
            trainer = FSLTrainer(args)
            trainer.train()
            trainer.evaluate_test()
            trainer.final_record()
            print(args.save_path)
