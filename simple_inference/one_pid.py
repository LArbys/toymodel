import os, sys, gc
import ROOT
from larcv import larcv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)
sys.path.insert(0,os.path.join(BASE_PATH,".."))

from lib.config import config_loader
from lib.rootdata_pid import ROOTData

larcv.LArbysLoader()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

p_type = {0:"eminus", 1:"gamma", 2:"muon", 3:"piminus",4:"proton"}

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def image_modify (img, cfg):
    img_arr = np.array(img.as_vector())
    img_arr = np.where(img_arr<cfg.adc_lo,         0,img_arr)
    img_arr = np.where(img_arr>cfg.adc_hi,cfg.adc_hi,img_arr)
    img_arr = img_arr.reshape(cfg.batch,img_arr.size).astype(np.float32)
    
    return img_arr

def main(IMAGE_FILE,OUT_DIR,CFG,ENTRY):
    #
    # initialize
    #
    cfg  = config_loader(CFG)
    assert cfg.batch == 1

    rd = ROOTData()

    FOUT = os.path.join(OUT_DIR,"multipid_out_one_event.root")
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

    net       = toy_pid.build(data_tensor_2d,cfg.num_class,trainable=False,keep_prob=1)
    
    # Define accuracy
    sigmoid              = None
    
    sigmoid = tf.nn.sigmoid(net)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    sess = tf.InteractiveSession(config=session_conf)
    sess.run(tf.global_variables_initializer())
    reader = tf.train.Saver()


    #
    # initialize iomanager
    #

    # oiom = larcv.IOManager(larcv.IOManager.kWRITE)
    # oiom.set_out_file("trash.root")
    # oiom.initialize()

    iom  = larcv.IOManager(larcv.IOManager.kREAD)
    iom.add_in_file(IMAGE_FILE)
    iom.initialize()

    for entry in xrange(iom.get_n_entries()):
        if (entry!=ENTRY): continue
        print "@entry={}".format(entry)

        iom.read_entry(entry)

        #ev_pgr = iom.get_data(larcv.kProductPGraph,"test")
        #ev_pix = iom.get_data(larcv.kProductPixel2D,"test_super_img")
        ev_img = iom.get_data(larcv.kProductImage2D,"wire")
        
        print '========================>>>>>>>>>>>>>>>>>>>>'
        print 'run, subrun, event',ev_img.run(),ev_img.subrun(),ev_img.event()

        rd.run[0]    = int(ev_img.run())
        rd.subrun[0] = int(ev_img.subrun())
        rd.event[0]  = int(ev_img.event())
        rd.entry[0]  = int(iom.current_entry())

        #rd.num_vertex[0] = int(ev_pgr.PGraphArray().size())
        
        for plane in xrange(3):
        
            if plane == 0: continue
            
            if plane == 1: continue
            
            print "@plane=%d" % plane

            ### Get 2D vertex Image                
                
            whole_img = ev_img.at(plane)
            
            ###
            weight_file = ""
            exec("weight_file = cfg.weight_file_pid%d" % plane)
            
            reader.restore(sess,weight_file)

            '''
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (8, 6))
            #img_vtx_nd = larcv.as_ndarray(img_vtx)
            img_vtx_nd = larcv.as_ndarray(img_vtx)
            ax.imshow(img_vtx_nd)
            plt.savefig("image/vtx/%i_%i_%i_vertex_plane_%i"%(ev_pix.run(), ev_pix.subrun(), ev_pix.event(),plane))
            
                test_meta =  img_vtx.meta()
                
                print test_meta.tl().x,test_meta.tl().y
                print test_meta.bl().x,test_meta.bl().y
                print test_meta.tr().x,test_meta.tr().y
                print test_meta.br().x,test_meta.br().y
            
                img_vtx = larcv.as_ndarray(img_vtx)

                img_vtx = np.pad(img_vtx, 128, pad_with)
                img_vtx = np.where(img_vtx<cfg.adc_lo,         0,img_vtx)
                img_vtx = np.where(img_vtx>cfg.adc_hi,cfg.adc_hi,img_vtx)
                img_vtx_arr = img_vtx.reshape(cfg.batch,img_vtx.size).astype(np.float32)
            '''
            img_vtx_arr=larcv.as_ndarray(whole_img)
            img_vtx_arr = np.where(img_vtx_arr<cfg.adc_lo,         0,img_vtx_arr)
            img_vtx_arr = np.where(img_vtx_arr>cfg.adc_hi,cfg.adc_hi,img_vtx_arr)

            #img_vtx_arr=img_vtx_arr.T

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (8, 6))
            ax.imshow(img_vtx_arr)
            plt.savefig("image/vtx/%i_%i_%i_vertex_plane_%i"%(ev_img.run(), ev_img.subrun(), ev_img.event(),plane))

            
            
            img_vtx_arr = img_vtx_arr.reshape(cfg.batch,img_vtx_arr.size).astype(np.float32)

            
            print img_vtx_arr.shape
            score_vv_vtx = sess.run(sigmoid,feed_dict={data_tensor: img_vtx_arr})
            score_v_vtx  = score_vv_vtx[0]
            print 'score_v_vtx',score_v_vtx

                
                #for x in xrange(5): print p_type[x], score_v_vtx[x]
                    
                #rd.eminus_score_vtx[plane] = score_v_vtx[0]
                #rd.gamma_score_vtx[plane]  = score_v_vtx[1]
                #rd.muon_score_vtx[plane]   = score_v_vtx[2]
                #rd.pion_score_vtx[plane]   = score_v_vtx[3]
                #rd.proton_score_vtx[plane] = score_v_vtx[4]

                ######
                
            tree.Fill()
            rd.reset_vertex()
        
        #ev_img = oiom.get_data(larcv.kProductImage2D,"out")
        #ev_img.Append(img)        
        #oiom.set_id(ev_pix.run(),ev_pix.subrun(),ev_pix.event())
        #oiom.save_entry()
        
    tfile.cd()
    tree.Write()
    tfile.Close()
    iom.finalize()
    #oiom.finalize()

if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        print
        print "\tIMAGE_FILE = str(sys.argv[1])"
        print "\tOUT_DIR  = str(sys.argv[2])"
        print "\tentry  = int(sys.argv[3])"
        print 
        sys.exit(1)
    
    IMAGE_FILE = str(sys.argv[1]) 
    OUT_DIR  = str(sys.argv[2])
    ENTRY = int(sys.argv[3])

    CFG = os.path.join(BASE_PATH,"cfg","simple_config.cfg")

    with tf.device('/cpu:0'):
        main(IMAGE_FILE,OUT_DIR,CFG,ENTRY)

    sys.exit(0)


