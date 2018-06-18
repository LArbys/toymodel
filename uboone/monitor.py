import sys,os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from time import gmtime, strftime

def make_plot(ax, plane):

        
    df_test  = pd.read_csv('/data/dayajun/toymodel/uboone/test_csv/plane%s/test_plane%s.csv'%(plane,plane))
    df_train = pd.read_csv('/data/dayajun/toymodel/uboone/test_csv/plane%s/train_plane%s.csv'%(plane,plane))
    
    t=strftime("%Y-%m-%d %H:%M:%S", gmtime())

    ax.plot(df_train.iter.values, df_train.acc.values, '-*',color='blue', label='Train_sample Acc')
    ax.plot(df_test.iter.values, df_test.acc.values, '-*',color='red' ,label='Test_sample Acc')
    ax1=ax.twinx()
    ax1.plot(df_train.iter.values, df_train.loss.values, '-*',color='orange', label='Train_sample Loss')

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Step')
    ax1.set_ylabel('Loss')

    ax.grid()

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1))
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=20)
    ax1.legend(bbox_to_anchor=(1.05, 0.1), loc=2, borderaxespad=0., fontsize=20)
    ax.set_title("Traning Status %s"%t)


def main():
    fig, ax = plt.subplots(1,1,figsize=(8,6))

    plane=0

    F_test_file = os.path.isfile('/data/dayajun/toymodel/uboone/test_csv/plane%s/test_plane%s.csv'%(plane,plane))
    F_train_file = os.path.isfile('/data/dayajun/toymodel/uboone/test_csv/plane%s/train_plane%s.csv'%(plane,plane))
    
    if (not F_test_file*F_train_file):
        ax.axis([0,10,0,10])
        
        if (not F_test_file): ax.text(2,3,'No test csv',fontsize=30)
        if (not F_train_file): ax.text(2,7,'No training csv',fontsize=30)
        
    make_plot(ax, plane)

    '''
    fig, (ax_1,ax_2,ax_3) = plt.subplots(3,1,figsize=(10,24))
    axes=[ax_1,ax_2,ax_3]
    plane=0
    for ax in axes:
        F_test_file = os.path.isfile('/data/dayajun/toymodel/uboone/test_csv/plane%s/test_plane%s.csv'%(plane,plane))
        F_train_file = os.path.isfile('/data/dayajun/toymodel/uboone/test_csv/plane%s/train_plane%s.csv'%(plane,plane))
        plane+=1
        if (not F_test_file*F_train_file):
            ax.axis([0,10,0,10])

            if (not F_test_file): ax.text(2,3,'No test csv',fontsize=30)
            if (not F_train_file): ax.text(2,7,'No training csv',fontsize=30)

            continue
        print plane-1
        make_plot(ax, plane-1)
    '''
    fig.savefig('Rui_monitor.png',bbox_inches="tight")

    
if __name__ == '__main__':
    main()
