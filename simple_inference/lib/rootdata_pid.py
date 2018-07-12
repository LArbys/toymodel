import ROOT
from array import array

kINVALID_INT    = ROOT.std.numeric_limits("int")().lowest()
kINVALID_FLOAT  = ROOT.std.numeric_limits("float")().lowest()
kINVALID_DOUBLE = ROOT.std.numeric_limits("double")().lowest()

class ROOTData(object):

    def __init__(self):
        self.run    = array( 'i', [ kINVALID_INT ] )
        self.subrun = array( 'i', [ kINVALID_INT ] )
        self.event  = array( 'i', [ kINVALID_INT ] )
        self.entry  = array( 'i', [ kINVALID_INT ] )
        self.vtxid  = array( 'i', [ kINVALID_INT ] )
        self.num_vertex = array( 'i', [ kINVALID_INT ] )
        
        self.inferred = array( 'i', [ kINVALID_INT ] )

        self.eminus_pix_score  = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.gamma_pix_score   = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.muon_pix_score    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.pion_pix_score    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.proton_pix_score  = ROOT.std.vector("float")(3,kINVALID_FLOAT)

        self.eminus_int_score  = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.gamma_int_score   = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.muon_int_score    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.pion_int_score    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.proton_int_score  = ROOT.std.vector("float")(3,kINVALID_FLOAT)

	self.npar = ROOT.std.vector("float")(3,kINVALID_FLOAT)

        self.eminus_score_vtx  = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.gamma_score_vtx   = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.muon_score_vtx    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.pion_score_vtx    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.proton_score_vtx  = ROOT.std.vector("float")(3,kINVALID_FLOAT)

    def reset_event(self):
        self.run[0]     = kINVALID_INT
        self.subrun[0]  = kINVALID_INT
        self.event[0]   = kINVALID_INT
        self.entry[0]   = kINVALID_INT

        self.num_vertex[0] = kINVALID_INT
        
    def reset_vertex(self):
        self.vtxid[0]   = kINVALID_INT

        self.inferred[0] = kINVALID_INT


        for pl in xrange(3):

	    self.npar[pl] = kINVALID_INT;

            self.eminus_pix_score[pl] = kINVALID_FLOAT
            self.gamma_pix_score[pl]  = kINVALID_FLOAT
            self.muon_pix_score[pl]   = kINVALID_FLOAT
            self.pion_pix_score[pl]   = kINVALID_FLOAT
            self.proton_pix_score[pl] = kINVALID_FLOAT 
            
            self.eminus_int_score[pl] = kINVALID_FLOAT
            self.gamma_int_score[pl]  = kINVALID_FLOAT
            self.muon_int_score[pl]   = kINVALID_FLOAT
            self.pion_int_score[pl]   = kINVALID_FLOAT
            self.proton_int_score[pl] = kINVALID_FLOAT

            self.eminus_score_vtx[pl] = kINVALID_FLOAT
            self.gamma_score_vtx[pl]  = kINVALID_FLOAT
            self.muon_score_vtx[pl]   = kINVALID_FLOAT
            self.pion_score_vtx[pl]   = kINVALID_FLOAT
            self.proton_score_vtx[pl] = kINVALID_FLOAT 
    def reset(self):
        self.reset_event()
        self.reset_vertex()

    def init_tree(self,tree):
        
        tree.Branch("run"   , self.run   , "run/I")
        tree.Branch("subrun", self.subrun, "subrun/I")
        tree.Branch("event" , self.event , "event/I")
        tree.Branch("entry" , self.entry , "entry/I")

        tree.Branch("vtxid" , self.vtxid, "vtxid/I")

        tree.Branch("num_vertex" , self.num_vertex, "num_vertex/I")

        tree.Branch("inferred"   , self.inferred  , "inferred/I")

        tree.Branch("eminus_pix_score", self.eminus_pix_score)
        tree.Branch("gamma_pix_score" , self.gamma_pix_score) 
        tree.Branch("muon_pix_score"  , self.muon_pix_score) 
        tree.Branch("pion_pix_score"  , self.pion_pix_score)
        tree.Branch("proton_pix_score", self.proton_pix_score)

        tree.Branch("eminus_int_score", self.eminus_int_score)
        tree.Branch("gamma_int_score" , self.gamma_int_score)
        tree.Branch("muon_int_score"  , self.muon_int_score)
        tree.Branch("pion_int_score"  , self.pion_int_score)
        tree.Branch("proton_int_score", self.proton_int_score)
	
        tree.Branch("npar", self.npar);
        

        
