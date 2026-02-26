import numpy as np
from dem_to_matrices import detector_error_model_to_check_matrices
import stim
import ldpc
from scipy.special import expit
import pymatching
from utils import *
from itertools import product
from typing import Optional, Dict, Any

from line_profiler import profile
from dataclasses import dataclass
from gentoriccode import GTCode

print("Test: Importing symatch_prep")

@dataclass
class SyMatch:
    gtcode: GTCode
    model: stim.DetectorErrorModel
    bp: bool = True
    predecode: bool = True
    simplex: bool = True
    logs: Optional[Dict[str, Any]] = None 
    min_prior: float = 0
    ms_scaling_factor: float = 0
    """
    Prepare a code for a particular symatch decoder

    Parameters:
    -------
    gtcode : GTCode
        the code object
    model : stim DEM
        extract the code object's circ's dem
    predcode: bool
        L/R only decoding with BP
    simplex: bool
        apply over-matching 
    logs: list of logs to be decoded

    """

        
    def __post_init__(self):
        f"""Class for decoding stim circuits using belief propagation and symatch+simplex."""

        model = self.model

        self._matrices = detector_error_model_to_check_matrices(
            model, allow_undecomposed_hyperedges=True
        )
        self.num_detectors = model.num_detectors
        self.num_errors = model.num_errors
        self.num_obs = model.num_observables

        self.vobservables =  [item for _, item in sorted(self.gtcode.vert_zlogs.items())]
        if self.gtcode.bcode_hlogs is not None:
            self.hobservables =  [item for _, item in sorted(self.gtcode.bcode_hlogs.items())]
        else:
            self.hobservables =  [item for _, item in sorted(self.gtcode.horz_zlogs.items())]
        self.observables = np.array(self.vobservables + self.hobservables).astype(np.uint8) 

        self.hz_subsyms = np.array(self.gtcode.get_subsym('h', type='Z'))
        self.vz_subsyms = np.array(self.gtcode.get_subsym('v', type='Z'))



    def decode(self, syndrome: np.ndarray, 
               verbose: bool = False) -> np.ndarray:
        """
        Decode the syndrome and return a prediction of which observables were flipped

        Parameters
        ----------
        syndrome : np.ndarray
            A single shot of syndrome data. This should be a binary array with a length equal to the
            number of detectors

        Returns
        -------
        np.ndarray
            A binary numpy array `predictions` which predicts which observables were flipped.
        """

        pcm = self.gtcode.hz
        #print("Decoding.... ", end="")

        ### BP preprocessing step
        if self.bp:
            bpObject = ldpc.BpDecoder(pcm, 
                                      error_rate=max(float(self._matrices.priors[0]),self.min_prior),
                                      max_iter=1000, #the maximum number of iterations for BP)
                                      ms_scaling_factor=self.ms_scaling_factor, #min sum scaling factor. If 0 the variable scaling factor method is used
                                      bp_method='minimum_sum')
            bpDcodeword = bpObject.decode(syndrome)

            if bpObject.converge:
                if verbose: print("BP converged", np.flatnonzero(bpDcodeword))
                return (self.observables @ bpDcodeword) % 2
            else:
                if verbose: print("BP ran but did not converge")
            
            llrs = bpObject.log_prob_ratios
            ps_e = expit(-llrs) #inserted to avoid overflow
            eps = 1e-11
            ps_e[ps_e > 1 - eps] = 1 - eps
            ps_e[ps_e < eps] = eps
            try:
                weights=-(10*np.log(ps_e)).astype(int)
            except:
                print("WARNING: BP returned nan for ", np.flatnonzero(syndrome))
                weights= np.ones(pcm.shape[1])
        else:
            weights= np.ones(pcm.shape[1])

        ### R/L only preprocessing with BP ###
        whichway = 'o'
        if self.predecode:
            if check_all_subsyms0(self.hz_subsyms, syndrome=syndrome):
                whichway = 'h'
            elif check_all_subsyms0(self.vz_subsyms, syndrome=syndrome):
                whichway = 'v'
        
        if whichway!='o':
            #RL only pattern detected
            rlbpobj = ldpc.BpDecoder(self.gtcode.rlonly_cm(whichway, type='Z'), 
                                    error_rate=float(self.gtcode.d/self.num_errors),
                                    max_iter=6*self.num_errors, #arbitrarily chosen
                                    ms_scaling_factor=0, 
                                    bp_method='min_sum')
            
            bpDcodeword = rlbpobj.decode(syndrome)
            if rlbpobj.converge:
                if np.sum(bpDcodeword) < self.gtcode.d // 2:
                    return (self.observables @ bpDcodeword) % 2
            else:
                pass #print("RLonly BP did not converge:", np.flatnonzero(bpDcodeword) )
        
    
    
        #we need the synd, weights and the logs for each sym
        vsyms = self.gtcode.z_syms
        liv_logs = self.vobservables #assumes vertical logicals per symmetry

        hsyms = self.gtcode.z_syms if self.gtcode.dcode is None else self.gtcode.dcode.z_syms
        lih_logs = self.hobservables if self.gtcode.dcode is None else [item for key, item in sorted(self.gtcode.dcode.horz_zlogs.items())]
        hsynd = syndrome if self.gtcode.dcode is None else self.gtcode.map_synd(syndrome)
        hweights = weights if self.gtcode.dcode is None else self.gtcode.map_dq_weights_todcode(weights)
        

        ## MATCHING: Bare or simplex
        if self.simplex:
            vpred = self.simplex_symatch(vsyms,liv_logs, syndrome, weights, verbose=verbose)
        else:
            vpred = self.li_symatch(vsyms,liv_logs, syndrome, weights)

        if self.simplex:
            hpred = self.simplex_symatch(hsyms,lih_logs, hsynd, hweights, verbose=verbose)
        else:
            hpred = self.li_symatch(hsyms,lih_logs, hsynd, hweights)

        #print(vpred, hpred)
        if self.gtcode.dcode is None:
            return list(vpred) + list(hpred)
        else:
            return list(vpred) + [hpred[k] for k, v in self.gtcode.bcode_hlogs.items()]
        

    @profile        
    def simplex_symatch(self,
                        syms,
                        liv_logs,
                        syndrome,
                        weights,
                        verbose: bool=True):

        """
        Matches on an overcomplete set of symmetries 

        Parameters
        -----

        syms: the z check symmetries
        liv_logs: the logicals associated with these symmetries, in order
        syndrome: the syndrome
        weights: the weights assigned to every column of the PCM
        verbose: print out steps

        Returns
        ------
        
        np.ndarray of the predicted logical status of each liv_log
        """
        
        pcm = self.gtcode.hz if len(syms[0]) == len(self.gtcode.hz) else self.gtcode.dcode.hz
        #print("\nStarting simplex symatch")
        
        
        ## MATCHING on all possible symmetry combinations
        li_dim = len(syms)
        li_2uple =  (2,) * li_dim
        syms_prediction = np.zeros(li_2uple, dtype=bool)

        for item in product([0, 1], repeat=li_dim):
            if (sum(item) > 0):
                sym = np.sum([idx * sym for idx, sym in zip(item, syms)],axis=0) % 2
                log = np.sum([idx * sym for idx, sym in zip(item, liv_logs)],axis=0) % 2

                #build syndrome and pcm
                symchecks = np.flatnonzero(sym)
                symerrsynd = np.take(syndrome, symchecks)

                symloc = np.flatnonzero(sym)
                pcheck = []
                for indCh in (symloc): pcheck.append(pcm[indCh]) #qubit connected to check in symmetry
                sympcm = (np.array(pcheck).astype(int))

                #match
                pred, w = pymatching.Matching(sympcm, weights=weights).decode(symerrsynd, return_weight=True)
                syms_prediction[item] = np.abs(np.sum(pred * log) % 2).astype(bool)#

                if verbose and sum(item) == 1: 
                    print(f" Symmetry at  {item} predicts error {np.flatnonzero(pred)} \
                    with weight {w} and log status {syms_prediction[item]} ")
                    

        ###now check corrections for parity checks formed by simplex code
        correction_update = np.zeros(li_2uple, dtype=bool)
        if verbose: print("Building simplex code with dimension", li_dim)

        # Initial parity check-based correction
        for ninv_fixed_idxs in range(li_dim-2): 
            for idx in product([0,1], repeat=li_dim):
                if sum(idx) == li_dim - ninv_fixed_idxs - 2 :
                    ranges = [range(i, 2) for i in idx]
                    site_correction = False
                    for sub_idx in product(*ranges):
                        site_correction ^= (syms_prediction[sub_idx] ^ correction_update[sub_idx])
                    correction_update[idx] ^= site_correction
                        
        
        #check global parity value to confirm other checks
        global_parity_val = bool(np.bitwise_xor(syms_prediction, correction_update).sum() % 2)

        if global_parity_val:
            if verbose: print("global parity flipped")
            for idx in product([0, 1], repeat=li_dim):
                if sum(idx) > 0:
                    correction_update[idx] = not correction_update[idx]

        if verbose: print("Finding least weight soln")
        #now find least weight correction
        least_corr_weight = li_dim**2

        best_corr_vec = np.zeros(li_dim, dtype=bool)

        for idx in product([0,1],repeat=li_dim):
            new_corr_weight = 0
            for sub_idx in product([0,1], repeat=li_dim):
                flip = (np.dot(sub_idx, idx) & 1) 
                if correction_update[sub_idx] ^ flip:
                    new_corr_weight += 1 

            if new_corr_weight < least_corr_weight:
                least_corr_weight = new_corr_weight
                best_corr_vec = idx
            
        #integrate best correction vector
        for idx in product([0,1], repeat=li_dim):
            flip = np.bitwise_and(idx , best_corr_vec).sum() % 2
            correction_update[idx] ^= flip

        if verbose: print("final correction update", np.ravel(correction_update).astype(int))

        #update sym predictions
        corrected_syms_predictions = np.bitwise_xor(syms_prediction, correction_update)
        if verbose: print("final syms predictions", np.ravel(corrected_syms_predictions).astype(int))

        #check all stabs are ok
        if verbose: print("Checking all simplex stabs")
        checksum = sum(np.ravel(corrected_syms_predictions).astype(int)) % 2
        if checksum: print("WARNING: found a violated stab")

        axis_pred = []
        for idx in product([0,1], repeat=li_dim):
            if sum(idx) == 1:
                axis_pred.append(corrected_syms_predictions[idx])

        return np.array(axis_pred).astype(int)[::-1]
 
    

    def li_symatch(self,
                   syms,
                   liv_logs,
                   syndrome,
                   weights):
        """
        Matches on a spanning linearly independent set of symmetries 

        Parameters
        -----

        syms: the z check symmetries
        liv_logs: the logicals associated with these symmetries, in order
        syndrome: the syndrome
        weights: the weights assigned to every column of the PCM

        Returns
        ------
        
        np.ndarray of the predicted logical status of each liv_log
        """
        
        pcm = self.gtcode.hz if len(syms[0]) == len(self.gtcode.hz) else self.gtcode.dcode.hz
        #print("Starting basic symatch")

        li_dim = len(syms)
        syms_prediction = np.zeros(li_dim, dtype=bool)
        for item in range(li_dim):
            sym = syms[item]
            log = (liv_logs[item]).T
            
            #create synd and pcm
            symchecks = np.flatnonzero(sym)
            symerrsynd = np.take(syndrome, symchecks)

            symloc = np.flatnonzero(sym)
            pcheck = []
            for indCh in (symloc): pcheck.append(pcm[indCh].T) #qubit connected to check in symmetry
            sympcm =  np.squeeze(np.array(pcheck).astype(int))

            #match
            pred, _ = pymatching.Matching(sympcm, weights=weights).decode(symerrsynd, return_weight=True)
            syms_prediction[item] = (np.sum(pred * log) % 2).astype(bool)#

        return np.array(syms_prediction).astype(int)
    

    def decode_batch(
        self,
        shots: np.ndarray,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
    ) -> np.ndarray:
        """
        Decode a batch of shots of syndrome data. This is just a helper method, equivalent to iterating over each
        shot and calling `BPOSD.decode` on it.

        Parameters
        ----------
        shots : np.ndarray
            A binary numpy array of dtype `np.uint8` or `bool` with shape `(num_shots, num_detectors)`, where
            here `num_shots` is the number of shots and `num_detectors` is the number of detectors 

        Returns
        -------
        np.ndarray
            A 2D numpy array `predictions` of dtype bool, where `predictions[i, :]` is the output of
            `self.decode(shots[i, :])`.
        """
        if bit_packed_shots:
            shots = np.unpackbits(shots, axis=1, bitorder="little")[
                :, : self.num_detectors
            ]
        predictions = np.zeros(
            (shots.shape[0], self.observables.shape[0]), dtype=bool
        )
        for i in range(shots.shape[0]):
            predictions[i, :] = self.decode(shots[i, :])
        if bit_packed_predictions:
            predictions = np.packbits(predictions, axis=1, bitorder="little")
        return predictions