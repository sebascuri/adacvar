"""Definition of all runner classes."""

import multiprocessing
import os
from abc import ABC, abstractmethod
from .util import *
import time


class AbstractRunner(ABC):
    """Abstract runner class.

    Parameters
    ----------
    num_threads: int, optional
        Number of threads to use.
    use_gpu: bool, optional
        Flag to indicate GPU usage.
    wall_time: int, optional
        Required time, in minutes, to run the process.
    memory: int, optional
        Required memory, in MB, to run run the process.

    """

    def __init__(self, name, num_threads=1, use_gpu=False, wall_time=None, memory=None):
        self.num_workers = max(1, (multiprocessing.cpu_count() - 1) // num_threads)
        self.num_threads = num_threads
        self.num_gpu = get_gpu_count()
        self.gpu_idx = 0
        self.use_gpu = use_gpu
        self.wall_time = wall_time
        self.memory = memory
        self.name = name

    @abstractmethod
    def run(self, cmd_list):
        """Run commands in list.

        Parameters
        ----------
        cmd_list: list

        """
        raise NotImplementedError

    @abstractmethod
    def run_batch(self, cmd_list):
        """Run commands in list in batch mode.

        Parameters
        ----------
        cmd_list: list

        """
        raise NotImplementedError

    def _add_device(self, cmd):
        """Add device keyword to a command."""
        if self.num_gpu == 0 or (not self.use_gpu):
            cmd += ' --device cpu'
        else:
            cmd += ' --device cuda:{}'.format(self.gpu_idx)
            self.gpu_idx = (self.gpu_idx + 1) % self.num_gpu

        return cmd


class LeonhardRunner(AbstractRunner):
    """Runner in Leonhard Cluster."""

    def run(self, cmd_list):
        """See `AbstractRunner.run'."""
        tasks = cmd_list[:]
        try:
            os.makedirs('logs/')
        except FileExistsError:
            pass

        for i, cmd in enumerate(tasks):
            bsub_cmd = 'bsub '
            bsub_cmd += '-o {} '.format('logs/lsf.'
                                        + cmd.split('dataset=')[1].split(' ')[0])

            if self.wall_time is not None:
                bsub_cmd += '-W {} '.format(self.wall_time)
            if self.memory is not None:
                bsub_cmd += '-R "rusage[mem={}]" '.format(self.memory)
            if self.use_gpu:
                bsub_cmd += '-R "rusage[ngpus_excl_p=1]" '
            if self.name is not None:
                bsub_cmd += '-J "{}-{}"'.format(self.name, i)

            bsub_cmd += '-n {} '.format(self.num_threads)
            os.system(bsub_cmd + '"{}"'.format(cmd))

    def run_batch(self, cmd_list):
        """Run jobs in batch mode."""
        try:
            os.makedirs('logs/')
        except FileExistsError:
            pass

        batch_size = len(cmd_list)
        cmd_file = 'logs/{}_cmd'.format(self.name)
        with open(cmd_file, 'w') as f:
            for cmd in cmd_list:
                f.write(cmd + '\n')

        bsub_cmd = 'bsub '
        bsub_cmd += '-o {} '.format('logs/lsf.' + self.name)

        if self.wall_time is not None:
            bsub_cmd += '-W {} '.format(self.wall_time)
        if self.memory is not None:
            bsub_cmd += '-R "rusage[mem={}]" '.format(self.memory)
        if self.use_gpu:
            bsub_cmd += '-R "rusage[ngpus_excl_p=1]" '

        bsub_cmd += '-n {} '.format(self.num_threads)

        # bsub -J "jobname[1-20]"
        bsub_cmd += '-J "{}[1-{}]"'.format(self.name, batch_size)

        bsub_cmd += ' "awk -v jindex=\\$LSB_JOBINDEX \'NR==jindex\' {} | bash"'.format(
            cmd_file)
        os.system(bsub_cmd)


class SingleRunner(AbstractRunner):
    """Runner in a Single Machine."""

    def run(self, cmd_list):
        """See `AbstractRunner.run'."""
        workers_idle = [False] * self.num_workers
        pool = [start_process(lambda: None) for _ in range(self.num_workers)]
        tasks = cmd_list[:]

        while not all(workers_idle):
            for i in range(self.num_workers):
                if not pool[i].is_alive():
                    pool[i].terminate()
                    if len(tasks) > 0:
                        cmd = self._add_device(tasks.pop(0))
                        pool[i] = start_process(lambda x: os.system(x), (cmd,))
                        time.sleep(5)
                    else:
                        workers_idle[i] = True

    def run_batch(self, cmd_list):
        """Run batch."""
        self.run(cmd_list)
