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

        self.eminus_multiplicity_pix  = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.gamma_multiplicity_pix   = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.muon_multiplicity_pix    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.pion_multiplicity_pix    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.proton_multiplicity_pix  = ROOT.std.vector("float")(3,kINVALID_FLOAT)

        self.eminus_multiplicity_score_pix  = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.gamma_multiplicity_score_pix   = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.muon_multiplicity_score_pix    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.pion_multiplicity_score_pix    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.proton_multiplicity_score_pix  = ROOT.std.vector("float")(3,kINVALID_FLOAT)

        self.eminus_multiplicity_int  = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.gamma_multiplicity_int   = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.muon_multiplicity_int    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.pion_multiplicity_int    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.proton_multiplicity_int  = ROOT.std.vector("float")(3,kINVALID_FLOAT)

        self.eminus_multiplicity_score_int  = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.gamma_multiplicity_score_int   = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.muon_multiplicity_score_int    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.pion_multiplicity_score_int    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.proton_multiplicity_score_int  = ROOT.std.vector("float")(3,kINVALID_FLOAT)

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
            self.eminus_multiplicity_pix[pl]       = kINVALID_FLOAT
            self.gamma_multiplicity_pix[pl]        = kINVALID_FLOAT
            self.muon_multiplicity_pix[pl]         = kINVALID_FLOAT
            self.pion_multiplicity_pix[pl]         = kINVALID_FLOAT
            self.proton_multiplicity_pix[pl]       = kINVALID_FLOAT 
            self.eminus_multiplicity_score_pix[pl] = kINVALID_FLOAT
            self.gamma_multiplicity_score_pix[pl]  = kINVALID_FLOAT
            self.muon_multiplicity_score_pix[pl]   = kINVALID_FLOAT
            self.pion_multiplicity_score_pix[pl]   = kINVALID_FLOAT
            self.proton_multiplicity_score_pix[pl] = kINVALID_FLOAT 

            self.eminus_multiplicity_int[pl]       = kINVALID_FLOAT
            self.gamma_multiplicity_int[pl]        = kINVALID_FLOAT
            self.muon_multiplicity_int[pl]         = kINVALID_FLOAT
            self.pion_multiplicity_int[pl]         = kINVALID_FLOAT
            self.proton_multiplicity_int[pl]       = kINVALID_FLOAT 
            self.eminus_multiplicity_score_int[pl] = kINVALID_FLOAT
            self.gamma_multiplicity_score_int[pl]  = kINVALID_FLOAT
            self.muon_multiplicity_score_int[pl]   = kINVALID_FLOAT
            self.pion_multiplicity_score_int[pl]   = kINVALID_FLOAT
            self.proton_multiplicity_score_int[pl] = kINVALID_FLOAT 

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

        tree.Branch("eminus_multiplicity_pix", self.eminus_multiplicity_pix)
        tree.Branch("gamma_multiplicity_pix" , self.gamma_multiplicity_pix) 
        tree.Branch("muon_multiplicity_pix"  , self.muon_multiplicity_pix) 
        tree.Branch("pion_multiplicity_pix"  , self.pion_multiplicity_pix)
        tree.Branch("proton_multiplicity_pix", self.proton_multiplicity_pix)

        tree.Branch("eminus_multiplicity_score_pix", self.eminus_multiplicity_score_pix)
        tree.Branch("gamma_multiplicity_score_pix" , self.gamma_multiplicity_score_pix) 
        tree.Branch("muon_multiplicity_score_pix"  , self.muon_multiplicity_score_pix) 
        tree.Branch("pion_multiplicity_score_pix"  , self.pion_multiplicity_score_pix)
        tree.Branch("proton_multiplicity_score_pix", self.proton_multiplicity_score_pix)

        tree.Branch("eminus_multiplicity_int", self.eminus_multiplicity_int)
        tree.Branch("gamma_multiplicity_int" , self.gamma_multiplicity_int) 
        tree.Branch("muon_multiplicity_int"  , self.muon_multiplicity_int) 
        tree.Branch("pion_multiplicity_int"  , self.pion_multiplicity_int)
        tree.Branch("proton_multiplicity_int", self.proton_multiplicity_int)

        tree.Branch("eminus_multiplicity_score_int", self.eminus_multiplicity_score_int)
        tree.Branch("gamma_multiplicity_score_int" , self.gamma_multiplicity_score_int) 
        tree.Branch("muon_multiplicity_score_int"  , self.muon_multiplicity_score_int) 
        tree.Branch("pion_multiplicity_score_int"  , self.pion_multiplicity_score_int)
        tree.Branch("proton_multiplicity_score_int", self.proton_multiplicity_score_int)


