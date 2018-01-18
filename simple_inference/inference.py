import os, sys, gc
import ROOT
from larcv import larcv
import numpy as np
import tensorflow as tf

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)
sys.path.insert(0,os.path.join(BASE_PATH,".."))

from lib.config import config_loader
from lib.rootdata import ROOTData

def main(VTX_FILE,OUT_DIR,CFG):
    #
    # initialize
    #
    cfg  = config_loader(CFG)
    assert cfg.batch == 1

    rd = ROOTData()

    NUM = int(VTX_FILE.split(".")[0].split("_")[-1])
    FOUT = os.path.join(OUT_DIR,"multipid_out_%d.root" % NUM)
    tfile = ROOT.TFile.Open(FOUT,"RECREATE")
    tfile.cd()
    print "OPEN %s"%FOUT

    tree  = ROOT.TTree("multipid_tree","")
    rd.init_tree(tree)
    rd.reset()

    #
    # initialize TF
    #    
    image_dim = np.array([1,1,cfg.xdim,cfg.ydim])
    data_tensor    = tf.placeholder(tf.float32, [None, image_dim[2] * image_dim[3]],name='x')
    data_tensor_2d = tf.reshape(data_tensor, [-1,image_dim[2],image_dim[3],1])
    
    from toynet import toy_pid

    net = toy_pid.build(data_tensor_2d,cfg.num_class)
    
    # Define accuracy
    sigmoid = None
    with tf.name_scope('sigmoid'):
        sigmoid = tf.nn.sigmoid(net)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    sess = tf.InteractiveSession(config=session_conf)
    sess.run(tf.global_variables_initializer())
    reader=tf.train.Saver()
    reader.restore(sess,cfg.weight_file)

    #
    # initialize iomanager
    #

    #oiom = larcv.IOManager(larcv.IOManager.kWRITE)
    #oiom.set_out_file("trash.root")
    #oiom.initialize()

    iom  = larcv.IOManager(larcv.IOManager.kREAD)
    iom.add_in_file(VTX_FILE)
    iom.initialize()

    for entry in xrange(iom.get_n_entries()):
        print "@entry={}".format(entry)

        iom.read_entry(entry)

        ev_pgr = iom.get_data(larcv.kProductPGraph,"test")
        ev_pix = iom.get_data(larcv.kProductPixel2D,"test_super_img")

        print ev_pix.run(),ev_pix.subrun(),ev_pix.event()

        rd.run[0]    = int(ev_pix.run())
        rd.subrun[0] = int(ev_pix.subrun())
        rd.event[0]  = int(ev_pix.event())

        rd.plane[0]  = int(cfg.plane)

        pixel2d_vv = ev_pix.Pixel2DClusterArray()
        if pixel2d_vv.empty()==True:
            rd.num_vertex[0] = 0
            tree.Fill()
            rd.reset_event()
            continue

        pixel2d_v = pixel2d_vv.at(cfg.plane)
        
        rd.num_vertex[0] = int(ev_pgr.PGraphArray().size())
        
        for ix,pgraph in enumerate(ev_pgr.PGraphArray()):
            
            rd.vtxid[0] = int(ix)

            print "@pgid=%d" % ix
            
            parid = pgraph.ClusterIndexArray().front()
            pixel2d = pixel2d_v.at(parid)
            
            if pixel2d.empty() == True:
                rd.inferred[0] = 0
                tree.Fill()
                rd.reset_vertex()
                continue

            rd.inferred[0] = 1

            img = larcv.cluster_to_image2d(pixel2d,cfg.xdim,cfg.ydim)
            
            img_arr  = np.array(img.as_vector())
            img_arr = np.where(img_arr<cfg.adc_lo,         0,img_arr)
            img_arr = np.where(img_arr>cfg.adc_hi,cfg.adc_hi,img_arr)

            img_arr = img_arr.reshape(cfg.batch,img_arr.size).astype(np.float32)
            
            score_vv = sess.run(sigmoid,feed_dict={data_tensor: img_arr})
            score_v = score_vv[0]

            rd.eminus_score[0] = score_v[0]
            rd.gamma_score[0]  = score_v[1]
            rd.muon_score[0]   = score_v[2]
            rd.pion_score[0]   = score_v[3]
            rd.proton_score[0] = score_v[4]
        
            tree.Fill()
            rd.reset_vertex()

        #ev_img = oiom.get_data(larcv.kProductImage2D,"out")
        #ev_img.Append(img)        
        #oiom.set_id(ev_pix.run(),ev_pix.subrun(),ev_pix.event())
        #oiom.save_entry()
        
        if entry==10: break
        
    tfile.cd()
    tree.Write()
    tfile.Close()
    iom.finalize()
    #oiom.finalize()

if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print
        print "\tVTX_FILE = str(sys.argv[1])"
        print "\tOUT_DIR  = str(sys.argv[2])"
        print 
        sys.exit(1)
    
    VTX_FILE = str(sys.argv[1])
    OUT_DIR  = str(sys.argv[2])

    CFG = os.path.join(BASE_PATH,"cfg","simple_config.cfg")

    with tf.device('/cpu:0'):
        main(VTX_FILE,OUT_DIR,CFG)

    sys.exit(0)


