import ROOT
from array import array

kINVALID_INT    = ROOT.std.numeric_limits("int")().lowest()
kINVALID_FLOAT  = ROOT.std.numeric_limits("float")().lowest()
kINVALID_DOUBLE = ROOT.std.numeric_limits("double")().lowest()

class ROOTData:

    def __init__(self):
        self.run    = array( 'i', [ kINVALID_INT ] )
        self.subrun = array( 'i', [ kINVALID_INT ] )
        self.event  = array( 'i', [ kINVALID_INT ] )
        self.vtxid  = array( 'i', [ kINVALID_INT ] )
        self.plane  = array( 'i', [ kINVALID_INT ] )

        self.num_vertex = array( 'i', [ kINVALID_INT ] )
        
        self.inferred = array( 'i', [ kINVALID_INT ] )

        self.eminus_score  = array( 'f', [ kINVALID_FLOAT ] )
        self.gamma_score   = array( 'f', [ kINVALID_FLOAT ] )
        self.muon_score    = array( 'f', [ kINVALID_FLOAT ] )
        self.pion_score    = array( 'f', [ kINVALID_FLOAT ] )
        self.proton_score  = array( 'f', [ kINVALID_FLOAT ] )

    def reset_event(self):
        self.run[0]     = kINVALID_INT
        self.subrun[0]  = kINVALID_INT
        self.event[0]   = kINVALID_INT
        self.plane[0]   = kINVALID_INT

        self.num_vertex[0] = kINVALID_INT
        
    def reset_vertex(self):
        self.vtxid[0]   = kINVALID_INT

        self.inferred[0] = kINVALID_INT

        self.eminus_score[0] = kINVALID_FLOAT
        self.gamma_score[0]  = kINVALID_FLOAT
        self.muon_score[0]   = kINVALID_FLOAT
        self.pion_score[0]   = kINVALID_FLOAT
        self.proton_score[0] = kINVALID_FLOAT 
        
    def reset(self):
        self.reset_event()
        self.reset_vertex()

    def init_tree(self,tree):
        
        tree.Branch("run"   , self.run   , "run/I")
        tree.Branch("subrun", self.subrun, "subrun/I")
        tree.Branch("event" , self.event , "event/I")

        tree.Branch("vtxid" , self.vtxid, "vtxid/I")
        tree.Branch("plane" , self.plane, "plane/I") 

        tree.Branch("num_vertex" , self.num_vertex, "num_vertex/I")

        tree.Branch("inferred"   , self.inferred  , "inferred/I")

        tree.Branch("eminus_score", self.eminus_score, "eminus_score/F")
        tree.Branch("gamma_score" , self.gamma_score , "gamma_score/F")
        tree.Branch("muon_score"  , self.muon_score  , "muon_score/F")
        tree.Branch("pion_score"  , self.pion_score  , "pion_score/F")
        tree.Branch("proton_score", self.proton_score, "proton_score/F")
