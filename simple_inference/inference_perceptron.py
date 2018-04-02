import os, sys, gc
import ROOT as root
from larcv import larcv
import numpy as np
import tensorflow as tf

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)
sys.path.insert(0,os.path.join(BASE_PATH,".."))

from lib.config import config_loader
from lib.rootdata_p import ROOTData

def main(MPID_FILE,OUT_DIR, CFG):

    cfg = config_loader(CFG)
    assert cfg.batch==1

    rd = ROOTData()

    file_path = MPID_FILE
    print file_path
    f=root.TFile.Open(file_path)
    print f
    tree=f.Get("multipid_tree")

    from toynet import toy_perceptron
    input_data = tf.placeholder("float", [None, 5])
    p_net = toy_perceptron.multilayer_perceptron(input_data)

    NUM = int(os.path.basename(MPID_FILE).split(".")[0].split("_")[-1])
    FOUT = os.path.join(OUT_DIR,"perceptron_out_%d.root" % NUM)
    tfile = root.TFile.Open(FOUT,"RECREATE")
    tfile.cd()

    out_tree = root.TTree("perceptron_tree","")
    rd.init_tree(out_tree)
    rd.reset()

    for entry in tree:
        rd.run[0] = entry.run
        rd.subrun[0] = entry.subrun
        rd.event[0] = entry.event
        rd.vtxid[0] = entry.vtxid
        rd.num_vertex[0] = entry.num_vertex

        for plane in xrange(3):
            if plane !=2 : continue
            rd.inferred[0] = 1
            eminus_score  = entry.eminus_score[plane]
            gamma_score   = entry.gamma_score[plane]
            muon_score    = entry.muon_score[plane]
            pion_score    = entry.pion_score[plane]
            proton_score  = entry.proton_score[plane]
            five_scores = np.array([[eminus_score,gamma_score,muon_score,pion_score,proton_score]])
        
            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            reader = tf.train.Saver()
            
            p_weight_file = ""
            exec("p_weight_file = cfg.weight_perceptron")
        
            reader.restore(sess,p_weight_file)
        
            ####Perceptron for Plane2 ####
        
            softmax = None
            softmax = tf.nn.softmax(p_net)
            out = sess.run(softmax, feed_dict={input_data: five_scores})
            print out
            inter_type =np.argmax(out)
            print inter_type
            rd.perceptron[plane]=inter_type
        
        out_tree.Fill()
        rd.reset_vertex()
    tfile.cd()
    out_tree.Write()
    tfile.Close()

                
if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print
        print "\tMPID_FILE = str(sys.argv[1])"
        print "\tOUT_DIR  = str(sys.argv[2])"
        print 
        sys.exit(1)
    
    MPID_FILE = str(sys.argv[1])
    OUT_DIR  = str(sys.argv[2])

    CFG = os.path.join(BASE_PATH,"cfg","simple_config.cfg")

    with tf.device('/cpu:0'):
        main(MPID_FILE,OUT_DIR,CFG)

    sys.exit(0)
