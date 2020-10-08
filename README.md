# Adaptive Sampling for Stochastic Risk-Averse Learning

This package is the companion of the paper `Adaptive Sampling for Stochastic Risk-Averse Learning' by Sebastian Curi, Kfir. Y. Levy, Stefanie Jegelka, Andreas Krause.

To install the package run:
```bash
$ pip install .
$ pip install .[test]
```
To check that everything runs fine run the testing script.
```bash
$ bash scripts/test_code.sh
```

To run an experiment run 
```bash
$ python cvar/run.py
```
To see the arguments run  
```bash
$ python cvar/run.py --help 
```

To reproduce experiments run from the main directory the following commands.
```bash
$ python experiments/run_task.py classification 
$ python experiments/run_task.py regression --shifts None
$ python experiments/run_task.py mnist --num-threads 4 --alphas 0.1 0.01 --seeds 0 1 2
$ python experiments/run_task.py fashion-mnist --num-threads 4 --alphas 0.1 0.01 --seeds 0 1 2
$ python experiments/run_task.py cifar-10 --num-threads 6 --use-gpu --alphas 0.1 0.01 --seeds 0 1 2 --memory 4096
$ python experiments/run_task.py trade-off --num-threads 6 --use-gpu --seeds 0 --memory 4096
$ python experiments/classification-shift --job run --alpha 0.1 --shifts train test both double --num-threads 1
$ python experiments/classification-upsample --job run --alpha 0.1 --shifts train test both double --num-threads 1
```

To post-process the experiments run.
```bash
$ python experiments/run_task.py classification --job post-process
$ python experiments/run_task.py regression --job post-process --shifts None
$ python experiments/run_task.py vision --job post-process --datasets mnist-augmented fashion-mnist-augmented cifar-10-augmented
$ python experiments/datashift_process_runs.py
```

To plot the experiments run

```bash
$ python experiments/run_task.py classification --job plot
$ python experiments/run_task.py regression --job plot --shifts None
$ python experiments/run_task.py vision --job plot --shifts train --datasets mnist-augmented fashion-mnist-augmented --alphas 0.01 0.1
$ python experiments/
```

