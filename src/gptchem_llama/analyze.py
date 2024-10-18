import pandas as pd
from fastcore.xtras import load_pickle
import os
from glob import glob
from scipy.stats import sem
from scipy.constants import golden
import matplotlib.pyplot as plt
from datetime import datetime
import matplotx
import ast

MODEL_DICT = {'meta-llama/Meta-Llama-3.1-8B-Instruct':'Llama',
              'EleutherAI/gpt-j-6b': 'GPT-J',
              'mistralai/Mistral-7B-Instruct-v0.3': 'mistral',
              'RandomForestClassifier': 'RF',
            'XGBClassifier' : 'XGBoost'
              }


ANALYSIS_COLS = ['train_size', 'representation', 'target', 'accuracy', 'f1_macro', 'f1_micro', 'kappa', 'n_epochs', 'lr', 'n_bins', 'test_size']

INFO_COLS = ['file'] #to do

class OutFolder():
    def __init__(self, folder:str) -> None:
        if os.path.exists(folder):
            self.folder = folder
        else:
            raise NotADirectoryError(f'{folder}')
        
        self.n_files = len(os.listdir(self.folder))
        self.pickles:list = glob(f'{self.folder}/*.pkl')

    def get_df(self, df_type = 'all', grouped = True, exclude = {}):
        '''
        param:
        ------
        df_type: all, analysis or info
            return type of data frame
        grouped: bool
            group metrics if True
        exclude: dict
            excludes the given values (dict values) from the given col (dict keys),
            e.g., {'train size': [5]} excludes all entries with training size 5
        '''
        df = pd.DataFrame([OutPickle(p).__dict__ for p in self.pickles])
        for col, values in exclude.items():
            df = df.loc[~df[col].isin(values)]
        if df_type == 'all':
            return df
        elif df_type == 'analysis':
            reduced_df = df[ANALYSIS_COLS]
            if grouped:
                return reduced_df.groupby(['n_bins', 'n_epochs', 'representation', 'target', 'train_size']).agg(['mean', 'sem'])
            else:
                return reduced_df
        elif df_type == 'info':
            return df[INFO_COLS]
        else:
            return df 
        
    def plot_analysis(self, 
                      bins = None, 
                      representation = None,
                      target = None,
                      ax = None, 
                      n_epochs = None,
                      onlyModelLabel = True,
                      exclude = {}):
        df = self.get_df(df_type='analysis', grouped = True, exclude=exclude)
        df_all = self.get_df(df_type='all', grouped=False, exclude= exclude)
        
        ONE_COL_WIDTH_INCH = 5
        TWO_COL_WIDTH_INCH = 7.2

        ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
        TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

        if bins is None:
            bins = df_all['n_bins'].unique()[0]
        else:
            if bins not in df_all['n_bins'].unique():
                raise ValueError(f"Givin number of bins ({bins}) not in experiment. Valid \'bins\': {df_all['n_bins'].unique()}")

        if representation is None:
            representation = df_all['representation'].unique()[0]
        else:
            if representation not in df_all['representation'].unique():
                raise ValueError(f"Givin representation ({representation}) not in experiment. Valid \'representation\': {df_all['representation'].unique()}")
        
        if target is None:
            target = df_all['target'].unique()[0]
        else:
            if target not in df_all['target'].unique():
                return ValueError(f"Givin target ({target}) not in experiment. Valid \'target\': {df_all['target'].unique()}")
        
        if n_epochs is None:
            n_epochs = df_all['n_epochs'].unique()
        else:
            for i in n_epochs:
                if i not in df_all['n_epochs'].unique():
                    return ValueError(f"Givin n_epochs ({i}) not in experiment. Valid \'n_epochs\': {df_all['n_epochs'].unique()}")
        
        if ax is None:
            fig, ax = plt.subplots(4, 1,figsize=(ONE_COL_WIDTH_INCH, TWO_COL_GOLDEN_RATIO_HEIGHT_INCH),constrained_layout = True , sharex=True)
        else:
            if len(ax.flatten()) != 4:
                raise ValueError(f'4 axes needs to be available for plotting all metrics. {len(ax)} are given.')
        for num_epochs in n_epochs:
            for i, metric in enumerate(['accuracy', 'f1_macro', 'f1_micro', 'kappa']):
                if onlyModelLabel:
                    plot_label = f"{df_all['model_short'].unique()[0]}"
                else: 
                    plot_label = f"{df_all['model_short'].unique()[0]} {num_epochs} epochs"
                ax[i].plot(
                        df.loc[bins, num_epochs,representation, target].index, 
                        df.loc[bins, num_epochs,representation, target][metric]['mean'],
                        marker='o',
                        label=plot_label
                                )
                ax[i].fill_between(
                    df.loc[bins, num_epochs,representation,target].index,
                    df.loc[bins, num_epochs,representation, target][metric]['mean'] - df.loc[bins, num_epochs,representation, target][metric]['sem'],
                    df.loc[bins, num_epochs,representation, target][metric]['mean'] + df.loc[bins, num_epochs,representation, target][metric]['sem'],
                    alpha=0.2
                )

                ax[i].set_ylim(0,1)
                #ax[i].set_title(metric)
                ax[i].set_xticks((list(df_all['train_size'].unique())))
                ax[i].set_title(metric)
                #ylabel_top('accuracy', ax=ax[0])
                #ylabel_top(r'F$_1$ macro', ax=ax[1])
                #ylabel_top(r'F$_1$ micro', ax=ax[2])
                #ylabel_top(r'$\kappa$', ax=ax[3])
        ax[-1].set_xlabel('training set size')
        #ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

        #matplotx.line_labels(ax[0])
        now = datetime.now().strftime('%Y%m%d_%H%M') 
        #fig.suptitle('Hydrides - {} - {}'.format(REPRESENTATION, 'binary'), fontsize=16)
        #fig.savefig(f'{now}_AdhesiveFreeE-{REPRESENTATION}-binary-{BINS}bin-classification-results.pdf', bbox_inches='tight')

    def add_zero_rule(self, ax, color = 'gray', exclude = {}):
        train_sizes = self.get_df(df_type='all', exclude = exclude)['train_size'].unique()
        ax[0].plot(train_sizes, [0.5] * len(train_sizes), label = 'zero-rule', linestyle = '--', color = color, alpha = 0.3)

class OutPickle():
    def __init__(self, file:str) -> None:
        self.file = file
        self.content:dict = load_pickle(file)

        self.trues = self.content['true']
        self.preds = self.content['predictions']

        self.result_dict = self.content['results']

        self.accuracy = self.content['results']['accuracy']
        self.acc_macro = self.content['results']['acc_macro']
        self.racc = self.content['results']['racc']
        self.kappa = self.content['results']['kappa']
        self.f1_macro = self.content['results']['f1_macro']
        self.f1_micro = self.content['results']['f1_micro']
        self.frac_valid = self.content['results']['frac_valid']
        self.all_y_true = self.content['results']['all_y_true']
        self.valid_indices = self.content['results']['valid_indices']

        self.train_size = self.content['train_size']
        self.test_size = len(self.preds)

        self.config_dict = self.content['config']
        self.property_name = self.content['config']['property_name']
        self.tune_settings_dict = self.content['config']['tune_settings']
        self.n_epochs = self.content['config']['tune_settings']['num_train_epochs']
        self.lr = self.content['config']['tune_settings']['learning_rate']
        self.per_device_train_batch_size = self.content['config']['tune_settings']['per_device_train_batch_size']

        self.tokenizer_cutoff = self.content['config']['tokenizer_kwargs']['cutoff_len']
        self.model = self.content['config']['base_model']
        try:
            self.model_short = MODEL_DICT[self.model]
        except:
            self.model_short = None
        self.interference_batch_size = self.content['config']['inference_batch_size']

        self.data_summary_dict = self.content['data_summary']
        self.datafile = self.content['data_summary']['datafile']
        self.target = self.content['data_summary']['target']
        self.representation = self.content['data_summary']['representation']

        self.n_bins = len(list(set(self.trues)))


def add_traditional_ml(ax, file, 
                       modeltypes = ['RandomForestClassifier', 'XGBClassifier'], 
                       target = 'y_bin', 
                       num_epochs = 4,
                       onlyModelLabel = True
                       ):
    df_ml_all = pd.read_csv(file)
    
    df_ml_all['target'] = [ ast.literal_eval(l)[0] for l in df_ml_all['target']]
    for modeltype in modeltypes:
        df_ml = df_ml_all.loc[df_ml_all['modeltype'] == modeltype][['modeltype','target','train_size','accuracy','f1_macro','f1_micro','kappa']]
        df_ml_grouped =df_ml.groupby(['modeltype', 'target', 'train_size']).agg(['mean', 'sem'])
        display(df_ml_grouped)
        if onlyModelLabel:
            plot_label = MODEL_DICT[modeltype]
        else:
            plot_label = f'{modeltype} {num_epochs} epochs'

        for i, metric in enumerate(['accuracy', 'f1_macro', 'f1_micro', 'kappa']):
            ax[i].plot(
                df_ml_grouped.loc[modeltype,target].index, 
                df_ml_grouped.loc[modeltype,target][metric]['mean'],
                marker='o',
                label=plot_label,
                color = 'gray',
                alpha = 0.3
            )
            ax[i].fill_between(
                df_ml_grouped.loc[modeltype,target].index,
                df_ml_grouped.loc[modeltype,target][metric]['mean'] - df_ml_grouped.loc[modeltype,target][metric]['sem'],
                df_ml_grouped.loc[modeltype,target][metric]['mean'] + df_ml_grouped.loc[modeltype,target][metric]['sem'],
                color = 'k',
                alpha=0.05
            )
    return ax

