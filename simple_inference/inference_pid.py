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

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="2"

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

def main(IMAGE_FILE,VTX_FILE,OUT_DIR,CFG):
    #
    # initialize
    #
    cfg  = config_loader(CFG)
    assert cfg.batch == 1

    rd = ROOTData()

    NUM = int(os.path.basename(VTX_FILE).split(".")[0].split("_")[-1])
    FOUT = os.path.join(OUT_DIR,"multipid_out_%04d.root" % NUM)
    #FOUT = os.path.join(OUT_DIR,"multipid_out_04.root")
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
    iom.add_in_file(VTX_FILE)
    iom.initialize()

    for entry in xrange(iom.get_n_entries()):
        #if (entry!=26): continue
        print "@entry={}".format(entry)

        iom.read_entry(entry)

        ev_pgr = iom.get_data(larcv.kProductPGraph,"inter_par")
        ev_par = iom.get_data(larcv.kProductPixel2D,"inter_par_pixel")
        ev_pix = iom.get_data(larcv.kProductPixel2D,"inter_img_pixel")
        ev_int = iom.get_data(larcv.kProductPixel2D,"inter_int_pixel")
        ev_img = iom.get_data(larcv.kProductImage2D,"wire")
        
        print '========================>>>>>>>>>>>>>>>>>>>>'
        print 'run, subrun, event',ev_pix.run(),ev_pix.subrun(),ev_pix.event()

        rd.run[0]    = int(ev_pix.run())
        rd.subrun[0] = int(ev_pix.subrun())
        rd.event[0]  = int(ev_pix.event())
        rd.entry[0]  = int(iom.current_entry())

        rd.num_vertex[0] = int(ev_pgr.PGraphArray().size())

        print 'num of vertices, ',rd.num_vertex[0]
        print 'pgrapgh size, ',int(ev_pgr.PGraphArray().size())
        
        for ix,pgraph in enumerate(ev_pgr.PGraphArray()):
            print "@pgid=%d" % ix
            #if (ix != 2): continue
            rd.vtxid[0] = int(ix)
            pgr = ev_pgr.PGraphArray().at(ix)
            cindex_v = np.array(pgr.ClusterIndexArray())
            pixel2d_par_vv = ev_par.Pixel2DClusterArray()
            pixel2d_pix_vv = ev_pix.Pixel2DClusterArray()
            pixel2d_int_vv = ev_int.Pixel2DClusterArray()
            cluster_vv_int = ev_int.ClusterMetaArray()
            cluster_vv_pix = ev_pix.ClusterMetaArray()
            
            print 'There are ', pgraph.ParticleArray().size(), 'clusters.'
            if (not pgraph.ParticleArray().size()) : continue
            parid = pgraph.ClusterIndexArray().front()
            
            roi0 = pgraph.ParticleArray().front()
    
            x = roi0.X()
            y = roi0.Y()
            z = roi0.Z()

            #y_2d_plane_0 = ROOT.Double()

            for plane in xrange(3):
                print "@plane=%d" % plane
                
                if pixel2d_par_vv.size() != 0:
                    rd.npar[plane] = 0
                    pixel2d_par_v = pixel2d_par_vv.at(plane)
                    for cidx in cindex_v:
			pixel2d_par = pixel2d_par_v.at(cidx)
			if pixel2d_par.size()>0:
			    rd.npar[plane] += 1;
				                  
                ### Get 2D vertex Image
                
                #meta = roi0.BB(plane)

                x_2d = ROOT.Double()
                y_2d = ROOT.Double()
                
                whole_img = ev_img.at(plane)
                
                larcv.Project3D(whole_img.meta(), x, y, z, 0.0, plane, x_2d, y_2d)
                y_2d = y_2d*6+2400

                #print 'x2d, ', x_2d, 'y2d, ',y_2d
                
                #if (plane == 0) : y_2d_plane_0 = y_2d
                #else : y_2d = y_2d_plane_0
                
                ###
                weight_file = ""
                exec("weight_file = cfg.weight_file_pid%d" % plane)
                
                reader.restore(sess,weight_file)

                # nothing
                if pixel2d_int_vv.empty()==True: continue

                pixel2d_pix_v = pixel2d_pix_vv.at(plane)
                pixel2d_pix = pixel2d_pix_v.at(ix)
                cluster_v_pix = cluster_vv_pix.at(plane)
                cluster_pix   = cluster_v_pix.at(ix)
		
                pixel2d_int_v = pixel2d_int_vv.at(plane)
                pixel2d_int = pixel2d_int_v.at(ix)
                cluster_v_int = cluster_vv_int.at(plane)
                cluster_int   = cluster_v_int.at(ix)
                # nothing on this plane
                #if pixel2d_pix.empty() == True: continue

                rd.inferred[0] = 1
                
                int_img=larcv.Image2D(512,512)

                for px in pixel2d_int : 
                    posx = cluster_int.pos_x(px.Y());
                    posy = cluster_int.pos_y(px.X());
                    row  = whole_img.meta().row(posy);
                    col  = whole_img.meta().col(posx);
                    
                    col = (int)(posx - x_2d + 0.5)
                    if(col < -256 or col > 255): continue

                    row = (int)(posy - y_2d - 0.5)
                    if(row < -256 or row > 255): continue
                    
                    #test_img.set_pixel(row+256,col+256,px.Intensity());
                    int_img.set_pixel(col+256,row+256,px.Intensity());

                pix_img=larcv.Image2D(512,512)

                for px in pixel2d_pix : 
                    posx = cluster_pix.pos_x(px.Y());
                    posy = cluster_pix.pos_y(px.X());
                    row  = whole_img.meta().row(posy);
                    col  = whole_img.meta().col(posx);
                    
                    col = (int)(posx - x_2d + 0.5)
                    if(col < -256 or col > 255): continue

                    row = (int)(posy - y_2d - 0.5)
                    if(row < -256 or row > 255): continue
                    
                    #test_img.set_pixel(row+256,col+256,px.Intensity());
                    pix_img.set_pixel(col+256,row+256,px.Intensity());
                
                
                #img_pix = larcv.cluster_to_image2d(pixel2d_pix,cfg.xdim,cfg.ydim)
                #img_int = larcv.cluster_to_image2d(pixel2d_int,cfg.xdim,cfg.ydim)
                img_pix = pix_img
                img_int = int_img

                #Plot the image from pgraph
                '''
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (8, 6))
                img_vtx_nd = larcv.as_ndarray(img)
                ax.imshow(img_vtx_nd)
                plt.savefig("image/%i_%i_%i_graph_plane_%i"%(ev_pix.run(), ev_pix.subrun(), ev_pix.event(), plane))
                '''
                img_pix_arr = image_modify(img_pix, cfg)
                img_int_arr = image_modify(img_int, cfg)
                
                ######## Occlusion Analysis Start
                #do_occlusion = True
                do_occlusion = cfg.do_occlusion
                if (do_occlusion):
                
                    stride = 3
                
                    occlusion_scores_5par = np.zeros(shape = [5, cfg.xdim, cfg.ydim])
                    
                    img_ndarray = larcv.as_ndarray(img)

                    y = 0
                    #for y in xrange(cfg.ydim - stride +1):
                    while y < 500:
                        print 'y',y,'/',cfg.ydim - stride
                        #for x in xrange(cfg.xdim - stride +1):
                        x = 0
                        while x < 500:
                            print 'y',y,'/', cfg.ydim - stride ,'x',x,'/', cfg.xdim - stride
                            test = img_ndarray.copy()
                            #for s in xrange(stride):
                            #print test[y:y+stride,x:x+stride].shape
                            #print np.zeros((stride, stride)).shape
                            test[y:y+stride,x:x+stride]  = np.zeros((stride, stride))
                            test = larcv.as_image2d(test)
                            test = image_modify(test,cfg)
                            print np.mean(test), np.mean(img_arr)
                            
                            test_ = test.reshape(512,512)
                            test_ = test_.T
                            '''#test_ = np.fliplr(test_)
                            #dump image
                            fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (18, 6))
                            img_ = img_arr.reshape(512,512)
                            
                            ax[0].imshow(test_)
                            ax[1].imshow(img_)
                            plt.savefig("image/y%i_x%i"%(y,x))
                            '''
                            test = test_.reshape(1,512*512)
                            score_vv = sess.run(sigmoid,feed_dict={data_tensor: test})
                            print score_vv[0]
                            #score_vv = sess.run(sigmoid,feed_dict={data_tensor: img_arr})
                            #print score_vv[0]
                            sys.stdout.flush()
                            x+=1
                            for idx in xrange(5):
                                occlusion_scores_5par[idx,x, y]  = score_vv[0][idx]
                        y+=1
                    for x in xrange(5):
                        np.savetxt("output_txt/occlusion_graph_%s_%i_%i_%i_%i.txt"%(p_type[x],ev_pix.run(), ev_pix.subrun(), ev_pix.event(), plane), occlusion_scores_5par[x], fmt="%.6f")
                        
                    '''
                    fig, axes = plt.subplots(nrows=5, ncols=1, figsize = (8, 6))
                    axes[0].imshow(occlusion_scores_eminus)
                    plt.savefig("image/%i_%i_%i_occlusion_plane_eminus_%i"%(ev_pix.run(), ev_pix.subrun(), ev_pix.event(), plane))
                    '''
                    
                #img_arr = np.array(img.as_vector())
                #img_arr = np.where(img_arr<cfg.adc_lo,         0,img_arr)
                #img_arr = np.where(img_arr>cfg.adc_hi,cfg.adc_hi,img_arr)
                #img_arr = img_arr.reshape(cfg.batch,img_arr.size).astype(np.float32)

                #score
                score_pix_vv = sess.run(sigmoid,feed_dict={data_tensor: img_pix_arr})
                score_pix_v  = score_pix_vv[0]

                score_int_vv = sess.run(sigmoid,feed_dict={data_tensor: img_int_arr})
                score_int_v  = score_int_vv[0]

                #print 'scores are ',score_pix_v
                
                rd.eminus_pix_score[plane] = score_pix_v[0]
                rd.gamma_pix_score[plane]  = score_pix_v[1]
                rd.muon_pix_score[plane]   = score_pix_v[2]
                rd.pion_pix_score[plane]   = score_pix_v[3]
                rd.proton_pix_score[plane] = score_pix_v[4]

                rd.eminus_int_score[plane] = score_int_v[0]
                rd.gamma_int_score[plane]  = score_int_v[1]
                rd.muon_int_score[plane]   = score_int_v[2]
                rd.pion_int_score[plane]   = score_int_v[3]
                rd.proton_int_score[plane] = score_int_v[4]

                '''
                rd.eminus_score[plane] = np.max(multiplicity_v[0:5])
                rd.gamma_score[plane]  = np.max(multiplicity_v[0:5])score_v[1]
                rd.muon_score[plane]   = np.max(multiplicity_v[0:5])score_v[2]
                rd.pion_score[plane]   = np.max(multiplicity_v[0:5])score_v[3]
                rd.proton_score[plane] = np.max(multiplicity_v[0:5])score_v[4]
                '''
                 
                continue

                ###### Adding scores for vertex images
                #print x_2d, y_2d , 'x2d, y2d'
                meta_crop = larcv.ImageMeta(256,256*6,
                                            256,256,
                                            0,8448,
                                            plane)
                #print meta_crop.dump()
                
                '''
                meta_origin_x = max(x_2d-256, 0)
                if (plane == 0): meta_origin_x = min(meta_origin_x, 3456-256)
                if (plane == 1): meta_origin_x = min(meta_origin_x, 3456-256)
                if (plane == 2): meta_origin_x = min(meta_origin_x, 3456-256)
                meta_origin_y = max(y_2d*6+2400 + 256 *6, 5472)
                if (plane == 0): meta_origin_y = min(meta_origin_y, 8448-256*6)
                if (plane == 1): meta_origin_y = min(meta_origin_y, 8448-256*6)
                if (plane == 2): meta_origin_y = min(meta_origin_y, 8448-256*6)
                '''
                meta_origin_x = max(x_2d-128, 256)
                if (plane == 0): meta_origin_x = min(meta_origin_x, 3456-256)
                if (plane == 1): meta_origin_x = min(meta_origin_x, 3456-256)
                if (plane == 2): meta_origin_x = min(meta_origin_x, 3456-256)

                meta_origin_y = max(y_2d*6+2400 + 128 *6, 5472)
                if (plane == 0): meta_origin_y = min(meta_origin_y, 8448-128*6-1)
                if (plane == 1): meta_origin_y = min(meta_origin_y, 8448-128*6-1)
                if (plane == 2): meta_origin_y = min(meta_origin_y, 8448-128*6-1)
                meta_crop.reset_origin(meta_origin_x, meta_origin_y)

                #Plot the image from vertex
                '''
                print 'Vertex Meta'
                print meta_crop.tl().x,meta_crop.tl().y
                print meta_crop.bl().x,meta_crop.bl().y
                print meta_crop.tr().x,meta_crop.tr().y
                print meta_crop.br().x,meta_crop.br().y
                '''
                #print meta_crop.dump()
                #print ev_img.at(plane).meta().dump()

                img_vtx = ev_img.at(plane).crop(meta_crop)
        
                #img_vtx.overlay(image)
                
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
                '''

                img_vtx = larcv.as_ndarray(img_vtx)

                img_vtx = np.pad(img_vtx, 128, pad_with)
                img_vtx = np.where(img_vtx<cfg.adc_lo,         0,img_vtx)
                img_vtx = np.where(img_vtx>cfg.adc_hi,cfg.adc_hi,img_vtx)
                img_vtx_arr = img_vtx.reshape(cfg.batch,img_vtx.size).astype(np.float32)

                #print img_vtx_arr.shape
                #score_vv_vtx = sess.run(sigmoid,feed_dict={data_tensor: img_vtx_arr})
                #score_v_vtx  = score_vv_vtx[0]
                #print 'score_v_vtx',score_v_vtx

                
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
    
    if len(sys.argv) != 5:
        print
        print "\tIMAGE_FILE = str(sys.argv[1])"
        print "\tVTX_FILE   = str(sys.argv[2])"
        print "\tOUT_DIR    = str(sys.argv[3])"
        print "\tCFG        = str(sys.argv[4])"
        print 
        sys.exit(1)
    
    IMAGE_FILE = str(sys.argv[1]) 
    VTX_FILE   = str(sys.argv[2])
    OUT_DIR    = str(sys.argv[3])
    CFG        = str(sys.argv[4])

    #CFG = os.path.join(BASE_PATH,"cfg","simple_config.cfg")

    with tf.device('/cpu:0'):
        main(IMAGE_FILE,VTX_FILE,OUT_DIR,CFG)
    
    print "DONE!"
    sys.exit(0)
