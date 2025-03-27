import json
import os
import torch
import argparse
import wandb
from torch.utils.data import DataLoader
from filelock import FileLock

from utils import set_random_seeds
from model import get_model
from trainer import Trainer
from data import get_datasets

HF_MODELS = {
    'spectra': "AstroMLCore/AstroM3-CLIP-spectra",
    'meta': "AstroMLCore/AstroM3-CLIP-meta",
    'photo': "AstroMLCore/AstroM3-CLIP-photo",
    'all': "AstroMLCore/AstroM3-CLIP-all"
}

run_ids = {
    'spectra': {
        False: {
            'sub10': {42: 'jx5u1ynn', 0: 'o7ouxm7r', 66: 'o9josicx', 12: '4sf41lk2', 123: '16cey292'},
            'sub25': {42: 'rzx4dh71', 0: 'yd1na38z', 66: 'nto5uvs0', 12: 'py8c95cx', 123: 'p3ec2bdk'},
            'sub50': {42: '56979zxw', 0: '66b5ui6r', 66: '8e34m91t', 12: 'mjnc2b8z', 123: '1iuysc4s'},
            'full': {42: 'bjn836x0', 0: 'nmtdyve6', 66: 'vh85spjt', 12: '8w4hrn1q', 123: '5nyk8b93'},
        },
        True: {
            'sub10': {42: 'gr1kgco1', 0: 'mh837xql', 66: 'xg66aech', 12: 'xxw1jffs', 123: 'nwtp6yst'},
            'sub25': {42: '5sgu488f', 0: 'ns0scmtf', 66: 'ra4zobzu', 12: 'y4u681fc', 123: 'pysl0zra'},
            'sub50': {42: '2t6hw3b4', 0: 'db5ldmjy', 66: 'e158eh1h', 12: 'cus5xgfn', 123: 'a1dmvdty'},
            'full': {42: '7duigpu6', 0: 'gurys2pl', 66: 'cbhn5crj', 12: 'ae0xnmfq', 123: 'rint4ydd'},
        },
    },
    'meta': {
        False: {
            'sub10': {42: 'fpjfk551', 0: 'd3e31vjn', 66: '585g4my3', 12: '85j8j6de', 123: 'ywn0sm8n'},
            'sub25': {42: 'u5vseua7', 0: '5fbphtzv', 66: 'xj98adhf', 12: 'lupductk', 123: 'p7shnlkp'},
            'sub50': {42: 'fhl7csvr', 0: 'ewpds0d1', 66: 'dyrexi1i', 12: 'dep22smn', 123: 'yk3wru2y'},
            'full': {42: 'u7t749qr', 0: 'hdrgyv4y', 66: 'hmj98puz', 12: 'ni9ax3w3', 123: 'msulxi2n'},
        },
        True: {
            'sub10': {42: '4xuh4aze', 0: '0c0atjhy', 66: 'sajeqh2k', 12: 'x01zvz66', 123: 'l0ujfyx3'},
            'sub25': {42: '6h0ixmga', 0: 'gv163g1x', 66: 'x99psdxg', 12: 'mwbygeh2', 123: 'gdmtiygk'},
            'sub50': {42: 'eno09dld', 0: 'hai59hhx', 66: 'h1354ufd', 12: 'gr0mwjen', 123: 'vv2ai62i'},
            'full': {42: '88hknhsy', 0: 'vxkivnzo', 66: '8bqoelj0', 12: 'jqag2igo', 123: '8mked9w4'},
        },
    },
    'photo': {
        False: {
            'sub10': {42: 'hk8krcio', 0: 'lf755tq8', 66: '8swaytm0', 12: 'y1klyh8z', 123: 'p5q1fa6m'},
            'sub25': {42: 'wlxezh97', 0: '9ry0rsyn', 66: '9j9zv1ug', 12: '01re4u82', 123: 'xtdsdo5e'},
            'sub50': {42: '8yumoyk1', 0: 'nq6xly6y', 66: 'i375sb01', 12: 'bxayy09a', 123: 'm9yg7qiq'},
            'full': {42: 'aqvg9uh0', 0: '1globuaf', 66: '4kyf19vy', 12: 'v5c79w91', 123: 'sr1ku1u8'},
        },
        True: {
            'sub10': {42: 'kct2gb3w', 0: '0mv2m0lj', 66: 'v1kmhqio', 12: '4mpksdjv', 123: '13ur9kfa'},
            'sub25': {42: 'cyfssbgc', 0: '70ls17co', 66: 'p6ruroj9', 12: 's5owyu0a', 123: 'nlr8r80b'},
            'sub50': {42: '6i2ou7zk', 0: 's51qryof', 66: 'ehqx5g8c', 12: 'uftgqvad', 123: 'dnquxqwi'},
            'full': {42: 'ml94r2gd', 0: 'mbtdlsm4', 66: 'jiqwgp7w', 12: 'dv19s5dy', 123: 'ypu0sp4g'},
        },
    },
    'all': {
        False: {
            'sub10': {42: 'jntorvdx', 0: 'sj8v8sxv', 66: '0ufmjrn3', 12: 's565gfi3', 123: '7qh2w6po'},
            'sub25': {42: 'yi4nv6b9', 0: 'vf2aih6r', 66: 'xuffthzq', 12: 'o2219rkv', 123: 'm61m1o7c'},
            'sub50': {42: 'v8t9fns1', 0: 'zsz1lvk0', 66: 'r9gm0ss1', 12: 'a2zyqejb', 123: 'mqmiv18q'},
            'full': {42: 'i41odbdf', 0: 'n09w1h8n', 66: 'qtpid3rl', 12: 'yjdkxkxp', 123: '4w7rm33n'},
        },
        True: {
            'sub10': {42: 'hj83x7iw', 0: 'sj0iltjt', 66: 'ethypik8', 12: 'ow9hbuqk', 123: '73w23l9u'},
            'sub25': {42: 'u9uc6zf6', 0: '7ypbck9s', 66: '327kgfiq', 12: 'm91t1xjo', 123: '4icx8q35'},
            'sub50': {42: 'sdgdczsy', 0: 'xq0z0un8', 66: 'e1z0cdv8', 12: 'jlx82igo', 123: 'vzc0fgyi'},
            'full': {42: 'yyhltoac', 0: '1f908r4p', 66: 'm2ehfvyz', 12: 'szo6ap13', 123: 'j5sw2xzi'},
        },
    }
}


def str_to_bool_or_none(value):
    """Convert string input to boolean or None."""
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value.lower() == 'none':
        return None
    raise argparse.ArgumentTypeError('Pretrain must be "true", "false", or "none".')


def eval_run(run, use_hf=False):
    """Evaluate a given WandB run or Hugging Face model and return accuracy."""
    config = run.config
    config['use_wandb'] = False

    set_random_seeds(config['random_seed'])

    _, _, test = get_datasets(config['dataset'],
                              name=f'{config["data_sub"]}_{config["random_seed"]}',
                              seq_len=config['seq_len'])
    test_dataloader = DataLoader(test, batch_size=config['batch_size'], shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(config)

    if use_hf:
        model = model.from_pretrained(HF_MODELS[config['mode']])
    else:
        weights_path = os.path.join(config['weights_path'] + '-' + run.id, 'weights-best.pth')
        model.load_state_dict(torch.load(weights_path, weights_only=True))

    model = model.to(device)

    trainer = Trainer(model=model, optimizer=None, scheduler=None, warmup_scheduler=None, criterion=criterion,
                      device=device, config=config)
    acc = trainer.evaluate(test_dataloader, test.features['label'].names, plot=False)

    return acc


def update_results(res_path, mode, pretrain, sub, seed, acc):
    """Safely updates results file using a file lock, ensuring nested structure with defaultdict."""
    lock = FileLock(res_path + '.lock')
    with lock:
        if os.path.exists(res_path):
            with open(res_path, 'r') as f:
                results = json.load(f)  # Load existing results
        else:
            print(f'{res_path} does not exist, creating a new one')
            results = {}

        # Convert pretrain and seed to str for json
        pretrain = str(pretrain)
        seed = str(seed)

        # Ensure nested dictionary structure
        if mode not in results:
            results[mode] = {}
        if pretrain not in results[mode]:
            results[mode][pretrain] = {}
        if sub not in results[mode][pretrain]:
            results[mode][pretrain][sub] = {}

        # Save accuracy
        results[mode][pretrain][sub][seed] = acc

        # Write back the updated results
        with open(res_path, 'w') as f:
            json.dump(results, f, indent=4)


def main(mode=None, pretrain=None, sub=None, seed=None, run_id=None, use_hf=False, res_path='results.json'):
    """Main function to evaluate models and update results.json."""
    api = wandb.Api()

    # Evaluate a single run if `run_id` is specified
    if run_id:
        print(f'Evaluating single run: {run_id}')
        run = api.run(f'AstroM3/{run_id}')
        mode = run.config['mode']
        pretrain = True if run.config['use_pretrain'] is True else False
        sub = run.config['data_sub']
        seed = run.config['random_seed']

        acc = eval_run(run)
        print(f'Run {run_id} {mode} {pretrain} {sub} {seed} - Accuracy: {acc}')

        # Update results.json for a single run
        update_results(res_path, mode, pretrain, sub, seed, acc)
        return

    # HF models can be evaluated only with these parameters
    if use_hf:
        assert pretrain is True or pretrain is None, "Hugging Face models can only be evaluated with pretrain=True."
        assert sub == 'full' or sub is None, ("Hugging Face models can only be evaluated on the full dataset "
                                              "(sub='full').")
        assert seed == 42 or seed is None, "Hugging Face models can only be evaluated with seed=42."

        pretrain = True
        sub = 'full'
        seed = 42

    # Otherwise, evaluate based on parameters
    modes = [mode] if mode else run_ids.keys()
    for m in modes:
        pretrains = [pretrain] if pretrain is not None else run_ids[m].keys()
        for p in pretrains:
            subs = [sub] if sub else run_ids[m][p].keys()
            for s in subs:
                seeds = [seed] if seed else run_ids[m][p][s].keys()
                for se in seeds:
                    run_id = run_ids[m][p][s][se]
                    run = api.run(f'AstroM3/{run_id}')

                    # Evaluate the run
                    acc = eval_run(run, use_hf=use_hf)

                    # Update the results
                    update_results(res_path, m, p, s, se, acc)

                    print(f'Evaluated {m}, {p}, {s}, {se} - Accuracy: {acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate models and store results.')
    parser.add_argument('--mode', type=str, default=None,
                        help='Mode to evaluate (e.g., spectra, meta, photo, all)')
    parser.add_argument('--pretrain', type=str_to_bool_or_none, default=None,
                        help='Pretrain setting (true/false/none)')
    parser.add_argument('--sub', type=str, default=None,
                        help='Subset to evaluate (e.g., sub10, sub25, sub50, full)')
    parser.add_argument('--seed', type=int, default=None, help='Seed to evaluate')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Evaluate a specific WandB run ID directly')
    parser.add_argument('--use_hf', action='store_true',
                        help='Load weights from Hugging Face')
    parser.add_argument('--res_path', type=str, default='results.json', help='Path to store results')

    args = parser.parse_args()
    main(mode=args.mode, pretrain=args.pretrain, sub=args.sub, seed=args.seed, run_id=args.run_id, use_hf=args.use_hf,
         res_path=args.res_path)
