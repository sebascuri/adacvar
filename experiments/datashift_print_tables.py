"""Print datashift tables."""
import pandas as pd
from itertools import product
from collections import OrderedDict

path = 'data_frames_cvar'
column_names = OrderedDict({
    'adacvar': "\\adacvar",
    'trunc_cvar': "\\trunkcvar",
    'soft_cvar': "\\softcvar",
    'mean': "\\mean"
})

index_names = OrderedDict({
    'adult': 'Adult',
    'Australian': 'Australian',
    'german.numer': 'German',
    'monks-problems-1': 'Monks',
    'phoneme': 'Phoneme',
    'spambase': 'Spambase',
    'splice': 'Splice',
    'Titanic': 'Titanic'
})

for data_shift, resample in product(['both', 'train', 'test', 'double'],
                                    ['', 'upsample']):
    df = pd.read_pickle(f"{path}/{data_shift}_{resample}.pkl")

    table = df.groupby(['dataset', 'algorithm']).apply(
        lambda x: ' & '.join(
            [fr"{x.accuracy.mean():.2f} $\pm$ {max(0, x.accuracy.std()):.1f}",
             fr"{x.loss.mean():.2f} $\pm$ {max(0, x.loss.std()):.1f}"])
    ).unstack()

    table = table[column_names.keys()]
    table = table.reindex(index_names.keys())

    table.rename(columns=column_names, index=index_names, inplace=True)
    table.to_latex(
        f"{path}/{data_shift}_{resample}.tex", escape=False,
        label=f"table:{data_shift}:{resample}",
        caption=(fr"Test accuracy/loss (mean $\pm$ s.d.) over five independent"
                 f" data splits with {data_shift} distribution shift"
                 f" with {'upsampling' if resample == 'upsample' else 'no resampling'}"
                 f" In shaded bold we indicate the best algorithms.")
    )
