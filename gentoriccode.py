import numpy as np
from typing import List, Tuple, Iterable, DefaultDict, Optional
from dataclasses import dataclass
import stim
from bposd.css import css_code
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

from parallelogram import Parallelogram
from utils import *

#get distance testing functions
import pysat
from pysat.examples.rc2 import RC2 
from pysat.formula import WCNF


from dataclasses import dataclass
from typing import Iterable, List




@dataclass 
class GTCode:
    """
    BBcode specifications

    Parameters
    ----------
    v1 : Iterable[float]
        First parallelogram basis vector defining the periodic geometry 
        Needs to be even and along x axis

    v2 : Iterable[float]
        Second parallelogram basis vector defining the periodic geometry

    basis : str
        'X' or 'Z'. Note X has not been fully implemented

    fpoly : List
        A polynomial for BB codes

    gpoly : List
        B polynomial for BB codes

    pphys : float
        Physical error rate 

    addwidth : float, optional (default=0.1)
        Wiggle room for cylinder trick

    addvlog : bool, optional (default=False)
        Add cylinder trick vertical logical operators 

    addhlog : bool, optional (default=False)
        Add cylinder trick horizontal logical operators

    d : int, optional (default=25)
        Code distance. Defaults to a large value as an upper bound

    coi : bool, optional (default=True)
        "Code of interest" flag. If True add generating set of logical operators
    """
    v1: Iterable[float]
    v2: Iterable[float]
    basis: str
    fpoly: List
    gpoly: List
    pphys : float
    addwidth : float = 0.1
    addvlog : bool = False
    addhlog : bool  = False
    d : int = 25
    coi : bool = True 

    def __post_init__(self):
        self.v1 = np.array(self.v1)
        self.v2 = np.array(self.v2)
        self.parallelogram = Parallelogram(self.v1,self.v2)
        
        assert self.basis == 'X' or self.basis == 'Z', "Initialization must be X or Z"
        assert self.v1[1] == 0, "One axis of parallelogram must be along x axis"
        assert self.v1[0] % 2 == 0, "v1 needs to be even"

        self.ell = self.v1[0] #base length of parallelogram
        self.em = self.v2[1] #height of parallelogram

        #create parity check matrices
        self.init_stim_circuit()
        self.add_pm_stabs()

        #create css code
        nldpc=css_code(hx=self.hx,hz=self.hz)
        nldpc.test()
        self.ldpccode = nldpc

        #get z symmetries
        _,cmatz = row_echelon_form(self.hz)
        zSymmetries = np.array(cmatz[(self.N//2)-(self.ldpccode.K)//2:,:].copy())
        print("Extracted Z symmetries: ", len(zSymmetries), 
              "symmetries of lengths", np.unique(np.sum(zSymmetries, axis=1)))
        self.z_syms = zSymmetries

        self.horz_zlogs = defaultdict(int)
        self.vert_zlogs = defaultdict(int)

        if self.addvlog or self.coi:
            self.get_strip_logicals(addwidth=self.addwidth, strip='v')
        if self.addhlog or self.coi:
            self.get_strip_logicals(addwidth=self.addwidth, strip='h')
        

        
    def add_pm_stabs(self) -> None:
        "Add generating set of stabilizers"

        hz_idxs_to_coord_dict = defaultdict(tuple)
        hz_coord_to_idx_dict = defaultdict(int)

        h_idx = 0
        pmx = []
        pmz = []
        n = len(self.parallelogram.all_points) //2

        for pt in self.parallelogram.all_points:
            xt, yt = pt[0], pt[1]
            if xt % 2 == 0 and yt % 2 == 0: #anchor to bottom left data qubit of square

                #X stabs
                xstab  = self.get_xstab(pt)
                targs = ([self.data_coord_to_idx_dict[subpt] for subpt in xstab])
                h_row = np.zeros(n)
                h_row[np.array(targs).astype(int)] = 1
                pmx.append(h_row)
                self.circ.append(self.add_stim_stab(stab=xstab,
                                                    xorz='X',
                                                    det=(self.basis=='X')))
                self.circ.append("TICK")

                #Z stabs
                zstab  = self.get_zstab(pt)
                targs = ([self.data_coord_to_idx_dict[tuple(np.rint(subpt))] for subpt in zstab])
                h_row = np.zeros(n)
                h_row[np.array(targs).astype(int)] = 1
                pmz.append(h_row)
                self.circ.append(self.add_stim_stab(stab=zstab,
                                                    xorz='Z',
                                                    det=(self.basis=='Z')))
                hz_coord_to_idx_dict[self.parallelogram.wrap_point([xt,yt-1])] = h_idx
                hz_idxs_to_coord_dict[h_idx] = self.parallelogram.wrap_point([xt,yt-1])
                h_idx+=1
                self.circ.append("TICK")
                

        self.hz_coord_to_idx_dict = dict(hz_coord_to_idx_dict)
        self.hz_idxs_to_coord_dict = dict(hz_idxs_to_coord_dict)
            
        self.hx = np.array(np.array(pmx))
        self.hz = np.array(np.array(pmz))
        self.N = len(self.hx[0])
        print("Added parity check matrices")


    def get_xstab(self,pt) -> List:
        "get X stabilizer anchored to bottom left data qubit point"

        targs = []

        anchorx, anchory = pt[0], pt[1]
        # go through A type qubits
        for fmono in self.fpoly:
            newx, newy = anchorx+2*fmono[0], anchory+2*fmono[1]
            targs.append(self.parallelogram.wrap_point([newx,newy]))

        anchorx, anchory = pt[0]+1, pt[1]-1
        # go through B type qubits
        for gmono in self.gpoly:
            newx, newy = anchorx+2*gmono[0], anchory+2*gmono[1]
            targs.append(self.parallelogram.wrap_point([newx,newy]))

        return targs
    
    
    def get_zstab(self,pt) -> List:
        "get Z stabilizer anchored to bottom left data qubit point"

        targs = []

        anchorx, anchory = pt[0], pt[1]
        # go through A type qubits
        for gmono in self.gpoly:
            newx, newy = anchorx+2*(-gmono[0]), anchory+2*(-gmono[1])
            targs.append(self.parallelogram.wrap_point([newx,newy]))

        anchorx, anchory = pt[0]+1, pt[1]-1
        # go through B type qubits
        for fmono in self.fpoly:
            newx, newy = anchorx+2*(-fmono[0]), anchory+2*(-fmono[1])
            targs.append(self.parallelogram.wrap_point([newx,newy]))

        return targs



    def coord_to_stim_idx(self, coord:Tuple) -> int:
        try: 
            self.data_coord_to_idx_dict[tuple(coord)]
            barrier = len(self.parallelogram.all_points) 
            return (barrier * coord[0]) + (coord[1])
        except:
            print("Error: ", coord, "is not a data qubit")
    
        
        
    def init_stim_circuit(self) -> stim.Circuit:
        "place coords and create dicts"

        data_coord_to_idx_dict = defaultdict(int)
        data_idxs_to_coord_dict = defaultdict(tuple)
        d_idx = 0

        #define qubit dicts
        for q_count, pt in enumerate(self.parallelogram.all_points):
            xt, yt = pt[0], pt[1]
            if (xt+yt) % 2 == 0:
                inttuple = tuple(map(int, tuple(self.parallelogram.all_points[q_count])))
                data_coord_to_idx_dict[inttuple] = d_idx
                data_idxs_to_coord_dict[d_idx] = inttuple
                d_idx+=1
                
        self.data_idxs_to_coord_dict = dict(data_idxs_to_coord_dict)
        self.data_coord_to_idx_dict = dict(data_coord_to_idx_dict)

        #place qubit coords
        circ = stim.Circuit()
        for q_count, pt in enumerate(self.parallelogram.all_points):
            xt, yt = pt[0], pt[1]
            if (xt+yt) % 2 == 0:
                stim_pt_idx = int(self.coord_to_stim_idx(pt))
                circ.append("QUBIT_COORDS", stim_pt_idx, (xt,yt,0) )

        if self.basis == 'X':
            for pt in self.parallelogram.all_points:
                if (pt[0] + pt[1]) % 2 == 0:
                    circ.append("H", self.coord_to_stim_idx(pt) )

        #insert noise
        for pt in self.parallelogram.all_points:
            xt, yt = pt[0], pt[1]
            if (xt+yt) % 2 == 0:
                if self.basis == 'Z': circ.append("X_ERROR", self.coord_to_stim_idx(pt), arg=self.pphys)
                if self.basis == 'X': circ.append("Z_ERROR", self.coord_to_stim_idx(pt), arg=self.pphys)
        circ.append("TICK")

        print("Initialized STIM circuit")
        self.circ = circ


    def add_stim_stab(self,stab:List,xorz:str,det:bool) -> stim.Circuit:
        "add one X or Z stabilizer"

        circ = stim.Circuit()
        targs = np.rint([self.coord_to_stim_idx(subpt) for subpt in stab]).astype(int)
        mpp_string = "MPP "

        for i in targs: mpp_string += (f"{xorz}{i}*")
        circ.append(stim.Circuit(mpp_string[:-1]))
        if det: circ.append("DETECTOR", stim.target_rec(-1))
                
        return circ
    

    
    def plotter(self, datapts, ancillapts, pretty=False, name='temp'):
        "Plot code grid with specified qubit lists"

        pts = np.array(self.parallelogram.all_points)
        if not pretty:
            plt.scatter(pts[:, 0], pts[:, 1],
                    color='lightgrey', s=20, linewidths=0)
        if pretty:
            for xpt in range(0,self.v1[0]+self.v2[0],2):
                plt.axvline(xpt, color='black')
            for ypt in range(0,self.v2[1],2):
                plt.axhline(ypt,color='black')

        for pt in datapts:
            coords = self.data_idxs_to_coord_dict[pt]
            plt.scatter(coords[0], coords[1], color='purple')

        for pt in ancillapts:
            coords = self.hz_idxs_to_coord_dict[pt]
            if not pretty:
                plt.scatter(coords[0], coords[1], color='blue')
            else:
                plt.scatter(coords[0], coords[1]-1, color='#FF7F7F', s=100, alpha=0.9)
                
        if pretty:
            plt.axis('off')
            plt.xlim((0,self.v1[0]+self.v2[0]))
            plt.ylim((0, self.v2[1]))
            
        plt.gca().set_aspect('equal')

        if pretty:
            plt.savefig(f'{name}.pdf', bbox_inches='tight')
        plt.show()

        
    def vertical_cyl_trick(self, sym, addwidth:float=0.1):
        """
        Finds vertical strip on code
        """

        midpt = (abs(self.v1[0])/2)
        fxs = [i[0] for i in self.fpoly]
        gxs = [i[0] for i in self.gpoly]
        maxfgw = max(max(fxs),max(gxs))

        stripstartpt = 2*maxfgw - addwidth
        stripendpt = self.v1[0] - 2*maxfgw + addwidth

        baseptstart = [stripstartpt, 0]
        lineptstart = [stripstartpt + self.v2[0], self.v2[1]]

        baseptmid = [midpt, 0]
        lineptmid = [midpt + self.v2[0],  self.v2[1]]

        baseptend= [stripendpt, 0]
        lineptend = [stripendpt + self.v2[0], self.v2[1]]

        num_checks = len(sym)
        testee = np.zeros(len(sym))
        for i in range(num_checks):
            icoords = self.hz_idxs_to_coord_dict[i]
            valstart = cross_prod(baseptstart, lineptstart, icoords)
            valend = cross_prod(baseptend, lineptend, icoords)
            if valstart<0 and valend>0: 
                if sym[i]: testee[i] = 1

        logopcombo = (testee @ self.hz)  % 2

        potentiallogop = np.zeros(2*num_checks)
        for i in range(int(num_checks*2)):
                p_idx = self.data_idxs_to_coord_dict[i]
                valmid = cross_prod(baseptmid, lineptmid, p_idx)
                if valmid>0:
                    potentiallogop[i] = logopcombo[i]

        return potentiallogop
    
    
    def horizontal_cyl_trick(self, sym, addwidth:float=0.1):
        """
        Finds horizontal strip on code
        """

        fys = [i[1] for i in self.fpoly]
        gys = [i[1] for i in self.gpoly]
        maxfgw = max(max(fys),max(gys))
        minfgw = min(min(fys),min(gys))
        dfgw = maxfgw-minfgw
        midpt = (abs(self.v2[1])/2)

        stripstartpt = dfgw - addwidth
        stripendpt = abs(self.v2[1]) - dfgw + addwidth

        baseptstart = [0,stripstartpt]
        lineptstart = [self.v1[0] + self.v2[0], stripstartpt]

        baseptmid = [0,midpt]
        lineptmid =  [self.v1[0] + self.v2[0], midpt]

        baseptend= [0,stripendpt]
        lineptend = [self.v1[0] + self.v2[0], stripendpt]

        num_checks = len(sym)
        testee = np.zeros(len(sym))

        for i in range(num_checks):
            icoords = self.hz_idxs_to_coord_dict[i]
            valstart = cross_prod(baseptstart, lineptstart, icoords)
            valend = cross_prod(baseptend, lineptend, icoords)
            if valstart>0 and valend<0: 
                if sym[i]: testee[i] = 1

        logopcombo = (testee @ self.hz)  % 2

        potentiallogop = np.zeros(2*num_checks)
        for i in range(int(num_checks*2)):
                p_idx = self.data_idxs_to_coord_dict[i]
                valmid = cross_prod(baseptmid, lineptmid, p_idx)
                if valmid<0:
                    potentiallogop[i] = logopcombo[i]

        return potentiallogop
    

    def get_strip_logicals(self, addwidth:float=0.1, strip:str = 'v'):
        "Start using cylinder trick to get logical set"

        assert self.basis == 'Z', "X basis strip logicals not implemented"

        templogs = defaultdict(int)
        stackermat = self.hz
        lidx = len(self.vert_zlogs) + len(self.horz_zlogs)

        for sidx, sym in enumerate(self.z_syms):
            flag = 0

            if strip == 'v':
                ltest = self.vertical_cyl_trick(sym, addwidth)
            elif strip == 'h':
                ltest = self.horizontal_cyl_trick(sym, addwidth)


            if len (np.flatnonzero((self.hx @ ltest) % 2)) == 0: #if leaves no syndromes
                mat, _ = row_echelon_form(np.vstack((stackermat,ltest)))
                rowsums = np.sum(mat, axis=1).astype(int)
                stabmat, _ = row_echelon_form(np.vstack((self.hz,ltest)))
                rowstabsums = np.sum(stabmat, axis=1).astype(int)
                if rowsums[-1-((self.ldpccode.K)//2)] != 0:
                    templogs[sidx] = np.array(ltest).astype(int)
                    stackermat = np.vstack((stackermat, ltest))
                else:
                    print(f"Sym {sidx}: operator is a product of stabilizers or previous logicals. Not adding.")
                    flag = 1
                    if rowstabsums[-1-((self.ldpccode.K)//2)] != 0:
                        print("(It's previous logicals)")
            else:
                print(f"Error at sym {sidx}: operator leaves syndromes. Adjust width")
                flag =1

            if not flag:
                lmeascirc = stim.Circuit()

                # add observables
                if self.basis == 'Z':
                    meas = np.flatnonzero(ltest)
                    logmeas = [self.data_idxs_to_coord_dict[pt] for pt in meas]
                    logmeasstim = [f"Z{self.coord_to_stim_idx(pt)}" for pt in logmeas]
                    lmeascirc.append(stim.Circuit(f"OBSERVABLE_INCLUDE({lidx}) "+" ".join(logmeasstim)))
                    lidx+=1

                self.circ.append(lmeascirc)
                print(f"Added {strip}  logical for sym {sidx}")

        if len(templogs)  != 0:
            if strip == 'v': self.vert_zlogs = templogs
            if strip == 'h': self.horz_zlogs = templogs

        if self.coi:
            if len(self.vert_zlogs) + len(self.horz_zlogs) == self.ldpccode.K:
                print("We have a generating set of logicals")
                self.bcode_hlogs = None
                self.dcode = None
                self.dist_test()

            else:
                print("We DO NOT have a generating set of logicals yet.")

                # if this is after trying h logs, try mapping to a bigger code
                if strip == 'h':
                    print("Trying to map to bigger code")
                    self.code_mapper()

                    for lkey, llog in self.bcode_hlogs.items():
                        if self.basis == 'Z':
                            meas = np.flatnonzero(llog)
                            logmeas = [self.data_idxs_to_coord_dict[pt] for pt in meas]
                            logmeasstim = [f"Z{self.coord_to_stim_idx(pt)}" for pt in logmeas]
                            lmeascirc = (stim.Circuit(f"OBSERVABLE_INCLUDE({lidx}) "+" ".join(logmeasstim)))

                            lidx+=1
                            self.circ.append(lmeascirc)
                            print(f"Added {strip}  logical for dcode sym {lkey}")

                    if len(self.vert_zlogs) + len(self.bcode_hlogs) == self.ldpccode.K:
                        print("We have a generating set of logicals")
                        self.dist_test()
                    else:
                        print("Somehow still don't have a gen set of logicals. Count:", 
                              len(self.vert_zlogs) , len(self.bcode_hlogs) ,self.ldpccode.K)
                    

    def dist_test(self):
        "Find distance of code if code not too large"

        if self.ldpccode.N < 190:
            s2 = self.circ.shortest_error_sat_problem()
            wcnf = WCNF(from_string=s2)

            with RC2(wcnf) as rc2:
                (rc2.compute())
                print("The shortest logical operator is: ", rc2.cost)
                self.d = rc2.cost


    def get_zsymmetry_pcm(self, whichSym):
        "Get code symmetries from Z PCM"

        sym = np.flatnonzero(whichSym)
        pcheck = []
        
        for indCh in (sym): #check in symmetry
                pcheck.append(self.hz[indCh]) #qubit connected to check in symmetry
        
        return np.array(pcheck).astype(int)
    

    def rlonly_cm(self, whichway, type='Z'):
        "Get check matrix that assumes only whichway errors"
        gd = {'h':1, 'v':0}

        if type == 'Z':
            pot_subsym = self.hz.copy()
        elif type == 'X':
            pot_subsym = self.hx.copy()
        for key, val in self.data_idxs_to_coord_dict.items():
            if val[1] % 2 == gd[whichway]:
                pot_subsym[:,key] = 0

        return pot_subsym
    

    def get_subsym(self, whichway, type='Z'):
        "Get subsymmetry matrix"

        pot_subsym = self.rlonly_cm(whichway=whichway,
                                    type=type)
        ref,cmat = row_echelon_form(pot_subsym)
        rowsum = np.sum(ref,axis=1)
        idx = ((rowsum == 0).argmax())
        return cmat[idx:]
    

    def convert_xanc_xerr(self, anccoords, type='h'):
        "Shift coords one unit left"
        anccoords = [self.hz_idxs_to_coord_dict[int(i)] for i in np.flatnonzero(anccoords)]
        if type == 'h':
            datacoords = [self.data_coord_to_idx_dict[self.parallelogram.wrap_point([v[0], v[1]+1])] for v in anccoords]
        elif type == 'v':
            datacoords = [self.data_coord_to_idx_dict[self.parallelogram.wrap_point([v[0]+1, v[1]])] for v in anccoords]
        err0s = np.zeros((self.N))
        err0s[datacoords] = 1
        return err0s
    
 
    def code_mapper(self):
        "Go to bigger code and extract h logicals"
        dcode = GTCode(v1 = self.v1,
                                v2=[2*self.v2[0], 2*self.v2[1]],
                                basis=self.basis,
                                fpoly=self.fpoly,
                                gpoly=self.gpoly,
                                pphys=self.pphys,
                                addwidth=-1,
                                addhlog=True,
                                coi=False
                                ) #doubled codes
        
        print("Testing what Dcode hlogicals translate to on Bcode")
        stackermat = self.hz
        templogs = defaultdict(int)

        for key, hlog in dcode.horz_zlogs.items():

            ltest = np.zeros(self.N)
            ltest[[self.data_coord_to_idx_dict[dcode.data_idxs_to_coord_dict[i]] for i in np.flatnonzero(hlog)]] = 1


            if len (np.flatnonzero((self.hx @ ltest) % 2)) == 0: #if leaves no syndromes
                mat, _ = row_echelon_form(np.vstack((stackermat,ltest)))
                rowsums = np.sum(mat, axis=1).astype(int)
                stabmat, _ = row_echelon_form(np.vstack((self.hz,ltest)))
                rowstabsums = np.sum(stabmat, axis=1).astype(int)
                if rowsums[-1-((self.ldpccode.K)//2)] != 0:
                    templogs[key] = np.array(ltest).astype(int)
                    stackermat = np.vstack((stackermat, ltest))
                else:
                    print(f"Sym {key}: operator is a product of stabilizers or previous logicals. Not adding.")
                    flag = 1
                    if rowstabsums[-1-((self.ldpccode.K)//2)] != 0:
                        print("(It's a product of previous logicals)", flush=True)
            else:
                print(f"Error at sym {key}: operator leaves syndromes. Adjust width")
                flag =1

        self.bcode_hlogs = templogs
        self.dcode = dcode


    def map_synd(self, synd):
        "doubles syndrome of base code onto a larger code"

        dcode = self.dcode

        stest = np.zeros(dcode.N // 2)
        stest[[dcode.hz_coord_to_idx_dict[self.hz_idxs_to_coord_dict[i]] for i in np.flatnonzero(synd)]] = 1
        stest[[dcode.hz_coord_to_idx_dict[self.hz_idxs_to_coord_dict[i][0]+self.v2[0], self.hz_idxs_to_coord_dict[i][1] + self.v2[1]] for i in np.flatnonzero(synd)]] = 1

        return stest


    def map_dq_weights_todcode(self, weights):
        "get weights for doubled code"

        dcode = self.dcode

        stest = np.zeros(dcode.N )

        for bid,bcoord in self.data_idxs_to_coord_dict.items():
            stest[dcode.data_coord_to_idx_dict[bcoord]] = weights[bid]
            stest[dcode.data_coord_to_idx_dict[bcoord[0]+self.v2[0], bcoord[1]+self.v2[1]]] = weights[bid]

        return stest

