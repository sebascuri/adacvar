"""Python Script Template."""
import argparse
from experiments.util import run_multiple, post_process, plot_task, print_task

parser = argparse.ArgumentParser(description='Run Task.')
parser.add_argument('task', type=str)
parser.add_argument('--job', type=str, default='run')
parser.add_argument(
        '--algorithms',
        nargs='+',
        type=str,
        default=['adacvar', 'mean', 'trunc_cvar', 'soft_cvar']
)
parser.add_argument('--num-threads', type=int, default=2)
parser.add_argument('--use-gpu', action='store_true', default=False)
parser.add_argument('--wall-time', type=int, default=None)
parser.add_argument('--memory', type=int, default=None)
parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
parser.add_argument('--alphas', nargs='+', type=float, default=[0.01, 0.02, 0.05, 0.1])
parser.add_argument('--shifts', nargs='+', type=str, default=[None, 'train'])
parser.add_argument('--datasets', nargs='+', type=str, default=None)
parser.add_argument('--networks', nargs='+', type=str, default=None)

parser.add_argument('--p-value', type=float, default=0.05)

args = parser.parse_args()
params = dict(num_threads=args.num_threads, use_gpu=args.use_gpu,
              wall_time=args.wall_time, memory=args.memory)

if args.job == 'run':
    run_multiple(args.task, args.algorithms, args.alphas, args.shifts, args.seeds,
                 params, datasets=args.datasets)
elif args.job == 'post-process':
    post_process(args.task, args.algorithms, args.alphas, args.shifts, args.datasets,
                 args.networks)
elif args.job == 'plot':
    plots = [('test', 'accuracy'), ('test', 'precision'), ('test', 'f1'),
             ('test', 'cvar'), ('test', 'var'), ('test', 'loss'),
             ('train', 'cvar'), ('train', 'var'), ('train', 'loss')]
    plot_task(args.task, args.algorithms, args.alphas, args.shifts, plots,
              args.datasets)

elif args.job == 'print':
    plots = [('test', 'cvar'), ('test', 'loss')]
    print_task(args.task, args.algorithms, args.alphas, args.shifts, plots,
               args.datasets, args.networks, p_value=args.p_value)
