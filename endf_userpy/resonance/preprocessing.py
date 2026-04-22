import sys
from endf_parserpy import EndfParserFactory
from physics import *

#==================================================================================
# Wrapper Class  ModelDataClass
#
# -----------------------------------------------------------
# model_type  model      f(data, E)    data
# -----------------------------------------------------------
#     0       Special    special       Special_DataClass
#     1       SLBW       slbw          BW_DataClass
#     2       MLBW       mlbw          BW_DataClass
#     3       R-M        reich_moore   RM_DataClass
#     4       A-A        adler_adler   AA_DataClass
#     5       RML        rml           RML_DataClass
#     6       URR        urrxs         URR_DataClass
#     7       Dummy      dummy         Dummy_DataClass
#
# Parameters:
#     model_type
#     data
#
# Methods:
#     build_jax_pure_function(self):      -> pointer to function f(data, E)
#     build_jax_pure_vmap_function(self): -> pointer to function f(data, E_array)
#     build_jax_function(self)            -> pointer to function f(E)
#     build_jax_vmap_function(self)       -> pointer to function f(E_array)
#
#==================================================================================

class ModelDataWrapper:
    """
    Wrapper around scalar model implementations.

    This class exposes two interfaces:

    1. A pure interface:
       - f(data, E_scalar)
       - f(data, E_array) via vmap

    2. A bounded interface to self.data:
       - f(E_scalar)
       - f(E_array) via vmap

    Notes
    -----
    - No jax.jit is applied at this level
    - It is expected a JIT-compilation at the higher-level caller function.
    - If data changes during optimization, use the pure interfaces
    - The bound interface f(E) is convenient, but should be used only
      when relying on object state is acceptable in the calling code.
    - The scalar functions are defined in the physics module
    """

    # Scalar implementations: expected signature is impl(data, E_scalar)

    _SCALAR_FUNCS = {
        0: special,
        1: slbw,
        2: mlbw,
        3: reich_moore,
        4: adler_adler,
        5: rml,
        6: urrxs,
        7: dummy,
    }

    def __init__(self, model_type, data):
        """
        Parameters
        ----------
        model_type : int
            Integer identifier of the model implementation.
        data : any
            Model-specific data structure passed as the first argument
            to the underlying implementation.
        """
        if model_type not in self._SCALAR_FUNCS:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model_type = model_type
        self.data = data

    def _get_scalar_impl(self):
        """
        Return the scalar implementation associated with this model type.

        Returns
        -------
        callable function
            Low-level function with signature impl(data, E_scalar).
        """

        return self._SCALAR_FUNCS[self.model_type]

    # ------------------------------------------------------------------
    # Pure interface: explicit data argument
    # ------------------------------------------------------------------

    def build_jax_pure_function(self):
        """
        Build a pure scalar-energy function.

        Returns
        -------
        callable function
            Function with signature f(data, E), where E is a scalar.

        Notes
        -----
        This is the recommended interface for optimization, grad, and jit,
        because it does not depend on mutable Python object state.
        """
        impl = self._get_scalar_impl()

        def f(data, E):
            return impl(data, E)

        return f

    def build_jax_pure_vmap_function(self):
        """
        Build a pure vmapped energy function.

        Returns
        -------
        callable function
            Function with signature f(data, E_array), where axis 0 of
            E_array is mapped over.

        Notes
        -----
        This is equivalent to:
            f_vmapped = jax.vmap(f_scalar, in_axes=(None, 0))
        where data is shared across the batch and E_array is mapped.
        """

        f_scalar = self.build_jax_pure_function()
        return jax.vmap(f_scalar, in_axes=(None, 0))

    # ------------------------------------------------------------------
    # Bound interface: uses self.data
    # ------------------------------------------------------------------

    def build_jax_function(self):
        """
        Build a scalar-energy function bound to the current `self.data`.

        Returns
        -------
        callable function
            Function with signature f(E), where E is a scalar.

        Notes
        -----
        The returned function reads self.data at call time, so if
        self.data is updated outside the jitted region, the new value
        is used automatically.
        """
        f_pure = self.build_jax_pure_function()

        def f(E):
            return f_pure(self.data, E)

        return f

    def build_jax_vmap_function(self):
        """
        Build a vmapped energy function bound to the current self.data.

        Returns
        -------
        callable function
            Function with signature f(E_array), where axis 0 is mapped over.

        Notes
        -----
        This is a convenience wrapper around the pure vmapped function:
            f_vmapped(self.data, E_array)

        If self.data changes during optimization inside a jitted workflow,
        prefer build_jax_pure_vmap_function() instead.
        """
        f_pure_vmapped = self.build_jax_pure_vmap_function()

        def f(E_batch):
            return f_pure_vmapped(self.data, E_batch)

        return f

#==================================================================================
# ENDF data preprocessing
#==================================================================================

def endf_preprocessor(endf_file):
    """
    Read ENDF-6 formatted file and prepare data for
    vectorized calculation of resonant cross
    sections using jax, jax.jit and jax.vmap
    Input parameters:
        endf_file: endf-6 formatted file
    Return:
        el: low energy boundaries vector  (N_TOTAL,)
        eh: high energy boundaries vector (N_TOTAL,)
        res_data: list of resonance data by model (len == N_TOTAL)
        bkg_data: background cross section data class
    Note:
        N_TOTAL = MAX_ISO * MAX_RANGES (imported from physics)
    """

    #--------------------------------
    # Read the ENDF-6 formatted file
    #--------------------------------
    parser = EndfParserFactory.create()
    d_endf = parser.parsefile(endf_file, include=[1, 2, 3])
    d_endf.pop(0, None)
    for key in range(4, 41):
        d_endf.pop(key, None)

    #---------------------
    # MF1 data processing
    #---------------------
    d1 = d_endf[1][451]
    lrp  = d1['LRP']
    temp = d1['TEMP']
    emax = d1['EMAX']
    izai = int(d1['NSUB']) // 10
    d_endf.pop(1, None)

    m , s = particle_data[izai] # particle mass and spin
    awi = m/mn

    #----------
    # MF3 data
    #----------

    d3 = d_endf[3]
    bkg_data = mf3_preprocessor(emax, d3)
    d_endf.pop(3, None)

    #----------
    # MF2 data
    #----------
    d2_151 = d_endf[2][151]
    el, eh, res_data = mf2_preprocessor(awi, s, emax, d2_151)
    del d_endf

    return el, eh, res_data, bkg_data

#==================================================================================
# mf3 preprocessor: endf data preprocessing and jax conversion
#==================================================================================

def mf3_preprocessor(emax, d3):
    """
    Process mf3 file to get background data
    Input parameters:
        emax: maximum energy
        d3: EndfDict[3] dictionary
    Return:
        bkg_data: background data
    """

    # Background MT numbers
    # Total, Elastic, Radiative capture, fission
    mts = [1, 2, 102, 18]
    tabs = []

    # All background TAB1 tables are padded to the fixed size NP_TAB1 so that
    # JAX sees identical shapes across all ENDF files (no recompilation).
    available_mts = [mt for mt in mts if mt in d3]
    if available_mts:
        actual_max = max(len(d3[mt]['xstable']['E']) for mt in available_mts)
        assert actual_max <= NP_TAB1, (
            f"MF3 TAB1 has {actual_max} points, "
            f"exceeds NP_TAB1={NP_TAB1}; "
            f"increase NP_TAB1 in physics.py"
        )

    NPSIZE = NP_TAB1

    # Block and pad TAB1 data for mts
    for mt in mts:
        if mt in d3:
            sig = d3[mt]['xstable']
            E = sig['E']
            xs = sig['xs']
            nbt = sig['NBT']
            intp = sig['INT']
        else:
            E = [1.0e-5, emax]
            xs = [0.0, 0.0]
            nbt = [2]
            intp = [1]

        tab1 = pad_tab1(E, xs, nbt, intp, NPSIZE, NR_TAB1)
        tabs.append(tab1)

    # Create the background data class
    return BKG_DataClass(
        tot=tabs[0],
        sct=tabs[1],
        cap=tabs[2],
        fis=tabs[3],
    )

#==================================================================================
# mf2 preprocessor: endf data preprocessing and jax conversion
#==================================================================================

def mf2_preprocessor(awi, s, emax, d2_151):
    """
    Preprocess MF2 data for jax compatibility and vectorization
    Input parameters:
        awi: incident particle m/mn
        s: incident particle intrinsic spin
        d2_151: EndfDict[2][151] dictionary
    Return range information:
        el: low energy boundaries vector  (N_TOTAL,)
        eh: high energy boundaries vector (N_TOTAL,)
        res_data: list of resonance data by range (len == N_TOTAL)
    Note:
        N_TOTAL = MAX_ISO * MAX_RANGES (imported from physics)
    """

    # MF2 resonance data processing by isotope and range
    ir = 0
    el_arr = np.zeros(N_TOTAL, dtype=np.float64)
    eh_arr = np.zeros(N_TOTAL, dtype=np.float64)
    res_data = []
    za = d2_151['ZA']
    NIS = d2_151['NIS']
    d_iso = d2_151['isotope']
    # Isotope loop
    for i in range(1, NIS+1):
        d_i = d_iso[i]
        zai = d_i['ZAI']
        abn = d_i['ABN']
        lfw = d_i['LFW']
        NER = d_i['NER']
        d_range = d_i['range']
        # Energy range loop
        for r in range(1, NER+1):
            d_r = d_range[r]
            lru = d_r['LRU']
            lrf = d_r['LRF']
            nro = d_r['NRO']
            naps = d_r['NAPS']
            el_arr[ir] = d_r['EL']
            eh_arr[ir] = d_r['EH']
            if lru==0 and lrf==0 and nro==0 and naps==0 and lfw==0 and NER==1:
                ap = d_r.get('AP', 0.0)
                if (za==zai and abn==1 and NIS==1):
                    # Special case for a single isotope
                    data = dummy_preprocessor(ap)
                    dummy_data = ModelDataWrapper(7, data)
                    res_data.append(dummy_data)
                else:
                    # Special case for a multi-isotope material
                    data = special_preprocessor(abn, ap)
                    special_data = ModelDataWrapper(0, data)
                    res_data.append(special_data)
            elif lru==1:
                if lrf==1 or lrf==2:
                    # lrf = 1 : SLBW, lrf = 2: MLBW
                    data = bw_preprocessor(awi, s, emax, abn, naps, d_r)
                    bw_data = ModelDataWrapper(lrf, data)
                    res_data.append(bw_data)
                elif lrf==3:
                    # Reich-Moore
                    data = rm_preprocessor(awi, s, emax, abn, naps, d_r)
                    rm_data = ModelDataWrapper(3, data)
                    res_data.append(rm_data)
                elif lrf==4:
                    # Adler_Adler
                    data = aa_preprocessor(awi, s, emax, abn, naps, d_r)
                    aa_data = ModelDataWrapper(4, data)
                    res_data.append(aa_data)
                elif lrf==7:
                    # R-Matrix Limited
                    data = rml_preprocessor(awi, s, emax, abn, naps, d_r)
                    rml_data = ModelDataWrapper(5, data)
                    res_data.append(rml_data)
                else:
                    # RRR formalism not allowed
                    print(' lru=',lru,'  lrf=',lrf,' ==> RRR: Fatal error, lrf not allowed')
                    sys.exit()
            elif lru==2:
                if lrf==1 and lfw==0:
                    # URR Case A (const)
                    data = urra_preprocessor(awi, s, emax, abn, naps, d_r)
                elif lrf==1 and lfw==1:
                    # URR Case B (only fission)
                    data = urrb_preprocessor(awi, s, emax, abn, naps, d_r)
                elif lrf==2:
                    # URR Case C (all)
                    data = urrc_preprocessor(awi, s, emax, abn, naps, d_r)
                else:
                    # URR formalism not allowed
                    print(' lru=',lru,'  lrf=',lrf,'  lfw=',lfw,' ==> URR: Fatal error, lfw/lrf not allowed')
                    sys.exit()
                urr_data = ModelDataWrapper(6, data)
                res_data.append(urr_data)
            else:
                # Resonance formalism not allowed
                print(' lru=',lru,'  lrf=',lrf,' ==> Fatal error: lru or lrf not allowed')
                sys.exit()
            ir += 1

        #Pad energy range information with dummy data if require
        if (NER < MAX_RANGES):
            n = MAX_RANGES - NER + 1
            for r in range(1, n):
                el_arr[ir] = r * emax
                eh_arr[ir] = (r + 1) * emax
                ap = 0.0
                data = dummy_preprocessor(ap)
                dummy_data = ModelDataWrapper(7, data)
                res_data.append(dummy_data)
                ir += 1

    # Pad the range information with dummy data if required
    if NIS < MAX_ISO:
        n = (MAX_ISO - NIS) * MAX_RANGES + 1
        for i in range(1, n):
            el_arr[ir] = i * emax
            eh_arr[ir] = (i + 1) * emax
            ap = 0.0
            data = dummy_preprocessor(ap)
            dummy_data = ModelDataWrapper(7, data)
            res_data.append(dummy_data)
            ir += 1

    # Checking size of arrays and data list for consistency
    if len(res_data) != N_TOTAL:
        raise ValueError(
            f"Expected len(res_data) == N_TOTAL = {N_TOTAL}, got {len(res_data)}"
        )

    if el_arr.shape[0] != N_TOTAL or eh_arr.shape[0] != N_TOTAL:
        raise ValueError(
            f"Expected el_arr and eh_arr with shape ({N_TOTAL},), "
            f"got {el_arr.shape} and {eh_arr.shape}"
        )

    # Conversion to jax arrays
    el = jnp.array(el_arr, dtype=jnp.float64)
    eh = jnp.array(eh_arr, dtype=jnp.float64)

    return el, eh, res_data

#==================================================================================
# mf2 dummy preprocessor: Used for no-resonance and jax.jit optimization
#==================================================================================

def dummy_preprocessor(ap):
    ap = jnp.array(ap, dtype=jnp.float64)
    return Dummy_DataClass(ap)

#==================================================================================
# Special case (LRU=0, LRF=0): endf data preprocessing and jax conversion
#==================================================================================

def special_preprocessor(abn, ap):
    abn = jnp.array(abn, dtype=jnp.float64)
    ap = jnp.array(ap, dtype=jnp.float64)
    return Special_DataClass(abn, ap)

#==================================================================================
# Breit-Wigner preprocessor: (LRU=1, LRF=1 (SLBW) or LRF=2 (MLBW)
#==================================================================================

def bw_preprocessor(awi, s, emax, abn, naps, d_r):
    """
    Preprocess an isotope/range data
    Input parameters:
        awi: incident particle m/mn
        s: incident particle intrinsic spin
        emax: evaluation maximum energy
        abn: isotope abundance
        naps: channel/scattering radius flag
        d_r: EndfDict[2][151]['isotope'][i]['range'][r] dict
    Return:
        BW_Dataclass for SLBW and MLBW reconstruction
    """

    # target spin, ap, ap(E), nls
    spi = d_r['SPI']
    ap = d_r.get('AP', 0.0)
    ape = d_r.get('AP_table', None)
    nls = d_r['NLS']
    d_grp = d_r['l_group']

    awri_arr  = np.zeros(nls, dtype=np.float64)
    qx_arr = np.zeros(nls, dtype=np.float64)
    l_arr = np.zeros(nls, dtype=np.int32)
    lrx_arr = np.zeros(nls, dtype=np.int32)
    nrs_arr = np.zeros(nls, dtype=np.int32)

    # lgroup preliminary scanning
    nch_tot = 0
    for l in range(nls):
        d_l = d_grp[l+1]
        awri_arr[l] = d_l['AWRI']
        qx_arr[l] = d_l['QX']
        l_arr[l] = d_l['L']
        lrx_arr[l] = d_l['LRX']
        nrs_arr[l] = d_l['NRS']
        nch_l, _ , _ = get_J_spins(spi, s, l_arr[l])
        nch_tot += nch_l

    # Compute qx (CM system)
    qx_arr = qx_arr * (awri_arr + awi) / awri_arr

    mask = (lrx_arr <= 0)
    qx_arr[mask] = 0.0

    qx0 = qx_arr[0]
    if np.allclose(qx_arr, qx0):
        qx = qx0
    else:
        mask = (qx_arr > 0.95 * qx0) & (qx_arr < 1.05 * qx0)
        qx = np.mean(qx_arr[mask])
        print(' warning: Different QX found for different L values')
        print('          Averaged QX used')

    # Get awri
    awr0 = awri_arr[0]
    if np.allclose(awri_arr, awr0):
        awri = awr0
    else:
        mask = (awri_arr > 0.95 * awr0) & (awri_arr < 1.05 * awr0)
        awri = np.mean(awri_arr[mask])
        print(' warning: AWRI inconsistency found')
        print('          Averaged AWRI value used')

    # Compute ki
    ki = kn * jnp.sqrt(awi) * awri / (awri + awi)

    # Prepare TAB1 for the channel and scattering radii
    if ape==None:
        x_ap = [0.0, emax]
        y_ap = [ap, 0.0]
        nbt_ap = [2]
        int_ap = [1]
    else:
        x1 = ape['Eint'][0]
        xn = ape['Eint'][-1]
        if x1 > 0.0:
            x0 = 0.0
            x_ap = [x0] + ape['Eint']
            y_ap = [ape['AP'][0]] + ape['AP']
            nbt_ap = [nb + 1 for nb in ape['NBT']]
        else:
            x_ap = ape['Eint']
            y_ap = ape['AP']
            nbt_ap = ape['NBT']
        int_ap = ape['INT']
        if xn < emax:
            x_ap = x_ap + [emax]
            y_ap = y_ap + [ape['AP'][-1]]
            nbt_ap[-1] += 1

    if naps==0 or naps==2:
        if naps==0:
            mwri = awri * mn
            a = 0.123 * mwri**(1.0/3.0) + 0.08
        else:
            a = ap
        x_a = [0.0, emax]
        y_a = [a, 0.0]
        nbt_a = [2]
        int_a = [1]
    else:
        x_a = x_ap
        y_a = y_ap
        nbt_a = nbt_ap
        int_a = int_ap

    assert len(x_a) <= NP_TAB1 and len(x_ap) <= NP_TAB1, \
        f"Radius TAB1 exceeds NP_TAB1={NP_TAB1}; increase NP_TAB1 in physics.py"
    assert len(nbt_a) <= NR_TAB1 and len(nbt_ap) <= NR_TAB1, \
        f"Radius TAB1 interpolation regions exceed NR_TAB1={NR_TAB1}; increase NR_TAB1 in physics.py"

    r_ap = pad_tab1(x_ap, y_ap, nbt_ap, int_ap, NP_TAB1, NR_TAB1)
    r_a = pad_tab1(x_a, y_a, nbt_a, int_a, NP_TAB1, NR_TAB1)

    # Process resonance data
    nrs_tot = np.sum(nrs_arr) + nls

    l_res =  np.zeros(nrs_tot, dtype=np.int32)
    lj_res = np.zeros(nrs_tot, dtype=np.int32)
    er_arr = np.zeros(nrs_tot, dtype=np.float64)
    gn_arr = np.zeros(nrs_tot, dtype=np.float64)
    gg_arr = np.zeros(nrs_tot, dtype=np.float64)
    gf_arr = np.zeros(nrs_tot, dtype=np.float64)
    gx_arr = np.zeros(nrs_tot, dtype=np.float64)
    ch_arr = np.zeros(nch_tot, dtype=np.int32)
    gj_arr = np.zeros(nch_tot, dtype=np.float64)

    gj_den = (2.0 * s + 1.0) * (2.0 * spi + 1.0)

    # lgroup loop
    ir=0
    ic=0
    for l in range(nls):
        nrs = nrs_arr[l]
        lg = l_arr[l]
        if nrs > 0:
            # Process resonance data
            d_l = d_grp[l+1]
            nr = ir + nrs
            l_res[ir:nr] = lg
            x = np.array(list(d_l['AJ'].values()), dtype=np.float64)
            j2 = np.rint(2 * x).astype(np.int32)
            idx = np.argsort(j2)
            j2 = j2[idx]
            x = np.array(list(d_l['ER'].values()), dtype=np.float64)
            er_arr[ir:nr] = x[idx]
            x = np.array(list(d_l['GN'].values()), dtype=np.float64)
            gn_arr[ir:nr] = x[idx]
            x = np.array(list(d_l['GG'].values()), dtype=np.float64)
            gg_arr[ir:nr] = x[idx]
            x = np.array(list(d_l['GF'].values()), dtype=np.float64)
            gf_arr[ir:nr] = x[idx]
            x = np.array(list(d_l['GT'].values()), dtype=np.float64)
            gt = x[idx]
            if (lrx_arr[l] > 0):
                gxc = gt - gn_arr[ir:nr] - gg_arr[ir:nr] - gf_arr[ir:nr]
                mask = (gxc < 0.0) | (gxc < ( 1.0e-6 * gt ))
                gxc[mask] = 0.0
                gx_arr[ir:nr] = gxc
            else:
                gx_arr[ir:nr] = 0.0
            lj_res[ir:nr] = CHN_STRIDE * lg + CHN_OFFSET + j2
            j2_unique = np.unique(j2)
            nch = j2_unique.shape[0]
            gj_wgt = (np.abs(j2_unique) + 1.0) / gj_den
            gj_sum = np.sum(gj_wgt)
            gj_dif = (2.0 * lg + 1.0) - gj_sum
            nc = ic + nch
            ch_arr[ic:nc] = CHN_STRIDE * lg + CHN_OFFSET + j2_unique
            gj_arr[ic:nc] = gj_wgt
            if gj_dif > 1.0e-30:
                # Missing channels
                # Add a dummy channel for potential
                l_res[nr] = lg
                er_arr[nr] = ER_VAL
                gn_arr[nr] = 0.0
                gg_arr[nr] = 0.0
                gf_arr[nr] = 0.0
                gx_arr[nr] = 0.0
                lj_res[nr] = CHN_STRIDE * lg + CHN_OFFSET + MISSED_2J
                ch_arr[nc] = lj_res[nr]
                gj_arr[nc] = gj_dif
                nr += 1
                nc += 1
        else:
            # l-group without resonance data
            # Add dummy channel for potential
            l_res[nr] = lg
            er_arr[nr] = ER_VAL
            gn_arr[nr] = 0.0
            gg_arr[nr] = 0.0
            gf_arr[nr] = 0.0
            gx_arr[nr] = 0.0
            lj_res[nr] = CHN_STRIDE * lg + CHN_OFFSET + MISSED_2J
            ch_arr[nc] = lj_res[nr]
            gj_arr[nc] = 2.0 * lg + 1.0
            nr += 1
            nc += 1
        ir = nr
        ic = nc

    # Data padding and jax conversion
    nrsmax = np.maximum(NRS_SIZE, nr)
    nchmax = np.maximum(MAX_CHN, nc)

    ch = np.full(nchmax, CHN_PADDED, dtype=np.int32)
    gj = np.zeros(nchmax, dtype=np.float64)
    lr = np.zeros(nrsmax, dtype=np.int32)
    lj = np.full(nrsmax, CHN_PADDED, dtype=np.int32)
    er = np.full(nrsmax, ER_VAL, dtype=np.float64)
    gn = np.zeros(nrsmax, dtype=np.float64)
    gg = np.zeros(nrsmax, dtype=np.float64)
    gf = np.zeros(nrsmax, dtype=np.float64)
    gx = np.zeros(nrsmax, dtype=np.float64)

    ch[:nc] = ch_arr[:nc]
    gj[:nc] = gj_arr[:nc]
    lr[:nr] = l_res[:nr]
    lj[:nr] = lj_res[:nr]
    er[:nr] = er_arr[:nr]
    gn[:nr] = gn_arr[:nr]
    gg[:nr] = gg_arr[:nr]
    gf[:nr] = gf_arr[:nr]
    gx[:nr] = gx_arr[:nr]

    abn = jnp.array(abn, dtype=jnp.float64)
    spi = jnp.array(spi, dtype=jnp.float64)
    ki  = jnp.array(ki, dtype=jnp.float64)
    qx  = jnp.array(qx, dtype=jnp.float64)
    nch = jnp.array(nc, dtype=jnp.int32)
    ch  = jnp.array(ch, dtype=jnp.int32)
    gch = jnp.array(gj, dtype=jnp.float64)
    lr  = jnp.array(lr, dtype=jnp.int32)
    lj  = jnp.array(lj, dtype=jnp.int32)
    er  = jnp.array(er, dtype=jnp.float64)
    gn  = jnp.array(gn, dtype=jnp.float64)
    gg  = jnp.array(gg, dtype=jnp.float64)
    gf  = jnp.array(gf, dtype=jnp.float64)
    gx  = jnp.array(gx, dtype=jnp.float64)

    ich = jnp.searchsorted(ch, lj)

    # Create BW_DataClass
    return BW_DataClass(
        abn, spi, ki, qx,
        r_a, r_ap,
        nch, ch, gch, ich,
        lr, er, gn, gg, gf, gx
    )

#==================================================================================
# Reich-Moore preprocessor: (LRU=1, LRF=3, reich_moore)
#==================================================================================

def rm_preprocessor(awi, s, emax, abn, naps, d_r):
    abn = jnp.array(abn, dtype=jnp.float64)
    return RM_DataClass(abn)

#==================================================================================
# Adler-Adler preprocessor: (LRU=1, LRF=4, adler_adler)
#==================================================================================

def aa_preprocessor(awi, s, emax, abn, naps, d_r):
    # TODO
    abn = jnp.array(abn, dtype=jnp.float64)
    return AA_DataClass(abn)

#==================================================================================
# R-Matrix Limited preprocessor: (LRU=1, LRF=7, rml)
#==================================================================================

def rml_preprocessor(awi, s, emax, abn, naps, d_r):
    # TODO
    abn = jnp.array(abn, dtype=jnp.float64)
    return RML_DataClass(abn)

#==================================================================================
# URR-Case A: (LRU=2, LRF=1, LFW=0)
#==================================================================================

def urra_preprocessor(awi, s, emax, abn, naps, d_r):
    # TODO
    abn = jnp.array(abn, dtype=jnp.float64)
    return URR_DataClass(abn)

#==================================================================================
# URR-Case B: (LRU=2, LRF=1, LFW=1)
#==================================================================================

def urrb_preprocessor(awi, s, emax, abn, naps, d_r):
    # TODO
    abn = jnp.array(abn, dtype=jnp.float64)
    return URR_DataClass(abn)

#==================================================================================
# URR-Case C: (LRU=2, LRF=2, LFW=0/1)
#==================================================================================

def urrc_preprocessor(awi, s, emax, abn, naps, d_r):
    # TODO
    abn = jnp.array(abn, dtype=jnp.float64)
    return URR_DataClass(abn)

