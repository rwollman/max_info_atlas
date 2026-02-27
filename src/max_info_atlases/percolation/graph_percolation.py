"""
Core percolation analysis for spatial organization scoring.

This module implements bond percolation analysis on spatial graphs with type information.
It computes entropy curves for real vs permuted type distributions to quantify
spatial organization.

The code is adapted from the original implementation, kept mostly as-is due to its
quality and correctness.
"""

import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

# Import ConnectedComponentEntropy from the cython module
try:
    from ..cython.ConnectedComponentEntropy import ConnectedComponentEntropy
except ImportError:
    # Fallback for development/testing with v1 code
    try:
        from ConnectedComponentEntropy import ConnectedComponentEntropy
    except ImportError:
        import sys
        sys.path.insert(0, '/purpledata/RoyTemp/max_info_atlases/code')
        from ConnectedComponentEntropy import ConnectedComponentEntropy


class EdgeListManager:
    """
    Manages computation, storage, and retrieval of k-nearest-neighbor edge lists.
    
    This class is designed to precompute and store edge lists for different XY coordinates,
    which can then be used by GraphPercolation instances.
    """
    
    def __init__(self, base_dir: str = "elk_data"):
        """
        Initialize EdgeListManager.
        
        Args:
            base_dir: Base directory for storing edge list data
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def compute_edge_list(self, XY: np.ndarray, maxK: int) -> np.ndarray:
        """
        Compute k-nearest-neighbors edge list for a set of coordinates.
        
        Args:
            XY: Array of shape (n, 2) containing x,y coordinates
            maxK: Maximum number of neighbors to consider
            
        Returns:
            Edge list with shape (n*maxK, 3) where each row contains [source, target, k]
        """
        # Ensure maxK is not larger than possible neighbors
        maxK = min(XY.shape[0] - 1, maxK)
        
        # Use NearestNeighbors to find maxK+1 neighbors (including self)
        nbrs = NearestNeighbors(n_neighbors=maxK + 1, algorithm='ball_tree').fit(XY)
        distances, indices = nbrs.kneighbors(XY)
        
        # Remove self indices and distances
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        # Create edge list
        ix_ks, ix_rows = np.meshgrid(np.arange(1, maxK + 1), np.arange(XY.shape[0]))
        ix_rows = ix_rows.T.flatten()
        ix_ks = ix_ks.T.flatten()
        ix_cols = indices.T.flatten()
        
        # Combine into edge list with [source, target, k]
        ELK = np.hstack((ix_rows[:, np.newaxis], ix_cols[:, np.newaxis], ix_ks[:, np.newaxis]))
        
        return ELK
    
    def save_edge_list(self, XY: np.ndarray, maxK: int, name: str, 
                       ELKfull: np.ndarray = None) -> str:
        """
        Save edge list for a set of coordinates.
        
        Args:
            XY: Array of shape (n, 2) containing x,y coordinates
            maxK: Maximum number of neighbors used
            name: Identifier for this XY dataset
            ELKfull: Precomputed edge list (if None, it will be computed)
            
        Returns:
            Path to the saved edge list file
        """
        if ELKfull is None:
            ELKfull = self.compute_edge_list(XY, maxK)
        
        filename = os.path.join(self.base_dir, f"{name}_k{maxK}.npz")
        
        np.savez(filename, XY=XY, maxK=maxK, ELKfull=ELKfull)
        
        return filename
    
    def load_edge_list(self, name: str, maxK: int) -> np.ndarray:
        """
        Load edge list for a set of coordinates.
        
        Args:
            name: Identifier for the XY dataset
            maxK: Maximum number of neighbors
            
        Returns:
            Edge list if found, None otherwise
        """
        filename = os.path.join(self.base_dir, f"{name}_k{maxK}.npz")
        
        if os.path.exists(filename):
            data = np.load(filename)
            return data['ELKfull']
        else:
            return None
    
    def get_edge_list(self, XY: np.ndarray, maxK: int, name: str, 
                      compute_if_missing: bool = True) -> np.ndarray:
        """
        Get edge list for a set of coordinates, computing it if necessary.
        
        Args:
            XY: Array of shape (n, 2) containing x,y coordinates
            maxK: Maximum number of neighbors
            name: Identifier for this XY dataset
            compute_if_missing: Whether to compute the edge list if not found
            
        Returns:
            Edge list if found or computed, None if not found and not computed
        """
        ELKfull = self.load_edge_list(name, maxK)
        
        if ELKfull is None and compute_if_missing:
            ELKfull = self.compute_edge_list(XY, maxK)
            self.save_edge_list(XY, maxK, name, ELKfull)
        
        return ELKfull


class GraphPercolation:
    """
    Bond percolation analysis for spatial organization scoring.
    
    This class performs percolation analysis on a spatial graph with type information,
    computing entropy curves for both real and permuted type distributions.
    """
    
    def __init__(self, XY: np.ndarray, type_vec: np.ndarray, maxK: int = None,
                 pbond_vec: np.ndarray = None, edge_list: np.ndarray = None):
        """
        Initialize GraphPercolation with spatial coordinates and type information.
        
        Args:
            XY: Array of shape (n, 2) containing x,y coordinates of points
            type_vec: Array of shape (n,) containing type labels for each point
            maxK: Maximum number of neighbors to consider (defaults to len(XY)-1)
            pbond_vec: Bond probability values to evaluate (default: 101 values from 0 to 1)
            edge_list: Precomputed edge list from EdgeListManager
        """
        if pbond_vec is None:
            pbond_vec = np.linspace(0, 1, 101)
            
        self.N = XY.shape[0]
        self.XY = XY
        self.type_vec = type_vec
        self.type_vec_perm = np.random.permutation(self.type_vec)
        self.maxK = maxK
        self.unq_types = np.unique(type_vec)
        self.n_types = len(np.unique(type_vec))

        if maxK is None:
            self.maxK = self.N - 1
            
        # Set default values for attributes that will be populated during percolation
        self.pbond_vec = pbond_vec
        self.pbond_vec_len = len(self.pbond_vec)
        self.ent_real = None
        self.ent_perm = None
        self.hz_min = None
        
        self.ELKfull = edge_list

    def calc_ELKexp(self, permute: bool = False) -> np.ndarray:
        """
        Calculate probability-weighted edge list based on type vector.

        Returns:
            np.ndarray of shape (m, 3) with columns [source, target, pbond_scale]
            Only includes same-type edges. pbond_scale is a deterministic "edge activation"
            value used for threshold sweeps (not i.i.d. bond percolation).
        """
        if self.ELKfull is None:
            raise ValueError("Edge list not provided, call percolation with edge_list_manager")

        # Use permuted or original type vector
        tvec = self.type_vec_perm if permute else self.type_vec

        # Extract types for each edge in the precomputed kNN list
        src = self.ELKfull[:, 0].astype(int)
        dst = self.ELKfull[:, 1].astype(int)
        k_raw = self.ELKfull[:, 2].astype(np.float64)

        type_left = tvec[src]
        type_right = tvec[dst]

        # Type fractions per node (fraction of nodes with the same type as node i)
        _, ix_inv, type_counts = np.unique(tvec, return_inverse=True, return_counts=True)
        type_frac = type_counts[ix_inv] / len(tvec)

        # Keep only same-type edges
        same = (type_left == type_right)
        if not np.any(same):
            return np.empty((0, 3), dtype=np.float64)

        src_s = src[same]
        dst_s = dst[same]
        k_s = k_raw[same]

        # Fraction associated with each edge (use source node's type fraction as in your code)
        frac_s = type_frac[src_s]

        # Build ELKexp: [src, dst, k, frac]
        ELKexp = np.column_stack((src_s, dst_s, k_s, frac_s)).astype(np.float64)

        # ---- FIX: split into blocks by k using ELKexp (filtered), not ELKfull ----
        # Sort by k so blocks are contiguous
        ELKexp = ELKexp[np.argsort(ELKexp[:, 2])]

        # Split where k changes in ELKexp
        split_ix = np.where(np.diff(ELKexp[:, 2]))[0] + 1
        blocks = np.split(ELKexp, split_ix)

        blocks_mod = []

        for block in blocks:
            if block.shape[0] == 0:
                continue

            # Shuffle edges within same-k block (keeps original behavior)
            np.random.shuffle(block)

            # Make k "continuous" within the block (keeps original behavior)
            # NOTE: this produces values in (k-1, k] approximately.
            k_cont = block[:, 2] + np.arange(block.shape[0]) / block.shape[0] - 1.0

            # Deterministic "activation scale" for this edge
            pbond = 1.0 - np.exp(-k_cont * block[:, 3])

            # Sort edges by pbond ascending (so lower pbond merges earlier)
            ordr = np.argsort(pbond)

            # Keep only [src, dst, pbond]
            blocks_mod.append(np.column_stack((block[ordr, 0], block[ordr, 1], pbond[ordr])))

        if not blocks_mod:
            return np.empty((0, 3), dtype=np.float64)

        return np.vstack(blocks_mod).astype(np.float64)

    def bond_percolation(self, permute: bool = False) -> np.ndarray:
        uf = ConnectedComponentEntropy(self.N)
        ent = np.full(len(self.pbond_vec) - 1, np.nan)
        
        # --- 1. EDGE DISCOVERY ---
        if permute:
            ELP = self._generate_monte_carlo_permuted_edges()
        else:
            ELP_local = self.calc_ELKexp(permute=False)
            ELP_macro = self._generate_macroscopic_edges(ELP_local)
            
            if len(ELP_macro) > 0:
                ELP = np.vstack([ELP_local, ELP_macro])
            else:
                ELP = ELP_local

            # Cache this to use for the normalization denominator!
            self.ELP_real = ELP
                
        # Sort combined edges by probability
        if len(ELP) > 0:
            ELP = ELP[np.argsort(ELP[:, 2])]
            
        # --- 2. THE SWEEP (Identical to your original logic) ---
        for i in range(len(self.pbond_vec) - 1):
            ix_to_merge = np.logical_and(
                ELP[:, 2] > self.pbond_vec[i],
                ELP[:, 2] <= self.pbond_vec[i + 1]
            )
            
            if not ix_to_merge.any():
                if i > 0:
                    ent[i] = ent[i - 1]
                else:
                    ent[i] = np.log2(self.N)
            else:
                ent[i] = uf.merge_all(ELP[ix_to_merge, :2].astype(int))[-1]
                
        if permute:
            self.ent_perm = ent
        else:
            self.ent_real = ent
            
        return ent

    def percolation(self, edge_list_manager: EdgeListManager = None, 
                    xy_name: str = None) -> None:
        """
        Perform percolation analysis for both real and permuted type distributions.
        
        Args:
            edge_list_manager: Manager to get edge list from if not already available
            xy_name: Identifier for the XY dataset, required if edge_list_manager is provided
        """
        # Only get ELKfull if it hasn't been provided
        if self.ELKfull is None:
            if edge_list_manager is not None:
                if xy_name is None:
                    raise ValueError("xy_name must be provided when using edge_list_manager")
                self.ELKfull = edge_list_manager.get_edge_list(self.XY, self.maxK, xy_name)
            else:
                # Auto-compute edge list (backward-compatible behavior)
                self.ELKfull = EdgeListManager().compute_edge_list(self.XY, self.maxK)
        
        # Shorten bond_vec by one to deal with edge case of 0
        self.pbond_vec_org = self.pbond_vec
        self.pbond_vec = self.pbond_vec[1:]

        # First do the real entropy calculation
        self.bond_percolation(permute=False)
        # Then do the permuted entropy calculation
        self.bond_percolation(permute=True)
        # Compute and store the lower bound curve
        self.hz_min = self._hz_min_curve_from_real_edges()

    def raw_score(self) -> float:
        """
        Calculate the raw (unnormalized) percolation score.

        Returns:
            Integrated absolute entropy difference between permuted and real curves.
        """
        dent = self.ent_perm - self.ent_real

        # Make sure arrays are compatible in length
        min_len = min(len(dent), len(self.pbond_vec))

        # Use only valid data points for integration
        return np.trapz(np.abs(dent[:min_len]), x=self.pbond_vec[:min_len], axis=0)

    def score(self) -> float:
        """
        Backward-compatible alias for the raw (unnormalized) percolation score.
        """
        return self.raw_score()
    
    def save(self, filename: str) -> None:
        """
        Save percolation results to file.
        
        Args:
            filename: Path to output .npz file
        """
        np.savez(
            filename,
            XY=self.XY,
            type_vec=self.type_vec,
            type_vec_perm=self.type_vec_perm,
            ent_real=self.ent_real,
            ent_perm=self.ent_perm,
            hz_min=self.hz_min if self.hz_min is not None else np.array([]),
            pbond_vec=self.pbond_vec,
            maxK=self.maxK
        )
        
    def load(self, filename: str) -> None:
        """
        Load percolation results from file.
        
        Args:
            filename: Path to input .npz file
        """
        dump = np.load(filename)
        self.XY = dump['XY']
        self.type_vec = dump['type_vec']
        self.type_vec_perm = dump['type_vec_perm']
        self.ent_real = dump['ent_real']
        self.ent_perm = dump['ent_perm']
        hz_min_loaded = dump['hz_min'] if 'hz_min' in dump else np.array([])
        self.hz_min = hz_min_loaded if hz_min_loaded.size > 0 else None
        self.pbond_vec = dump['pbond_vec']
        self.maxK = dump['maxK']
        self.unq_types = np.unique(self.type_vec)
        self.n_types = len(np.unique(self.type_vec))
        self.N = len(self.type_vec)

    def _hz_min_curve_from_real_edges(self) -> np.ndarray:
        """
        Compute H_min(P) using a graph-theory lower bound on component count,
        based on ALL AVAILABLE same-type edges (local + macroscopic) under the REAL labeling.
        """
        # 1. Retrieve the combined $K \to \infty$ real edge list
        if not hasattr(self, 'ELP_real') or self.ELP_real is None:
            raise ValueError("Run bond_percolation(permute=False) first to generate real edges.")
            
        ELP = self.ELP_real
        N = self.N

        n_bins = len(self.pbond_vec) - 1
        hz_min = np.zeros(n_bins, dtype=np.float64)

        # 2. Use REAL type vector and counts
        tvec = self.type_vec
        unq, inv, counts = np.unique(tvec, return_inverse=True, return_counts=True)
        T = len(unq)
        n_t = counts.astype(np.int64)  
        type_idx = inv

        # Sort ELP by pbond ascending
        if ELP.shape[0] > 0:
            order = np.argsort(ELP[:, 2])
            ELP = ELP[order]

        E_t = np.zeros(T, dtype=np.int64)
        j = 0
        m = ELP.shape[0]

        sing_term = 0.0
        if N > 0:
            p1 = 1.0 / N
            sing_term = -p1 * np.log2(p1)

        # 3. Sweep and count edges
        for i in range(n_bins):
            lo = self.pbond_vec[i]
            hi = self.pbond_vec[i + 1]

            while j < m and ELP[j, 2] <= hi:
                if ELP[j, 2] > lo:
                    src = int(ELP[j, 0])
                    # Because macroscopic edges use root IDs, they correctly map back to the root's cell type
                    t = type_idx[src] 
                    E_t[t] += 1
                j += 1

            # Graph bound: K_t >= max(1, n_t - E_t)
            K_t = n_t - E_t
            K_t[K_t < 1] = 1

            a = n_t - K_t + 1  
            H = 0.0
            
            for tt in range(T):
                pa = a[tt] / float(N)
                if pa > 0:
                    H += -pa * np.log2(pa)
                if K_t[tt] > 1:
                    H += (K_t[tt] - 1) * sing_term

            hz_min[i] = H

        return hz_min
    def normalized_score(self, return_curve: bool = False):
        """
        Normalized 0–1 coherence score.

        Uses:
            numerator(P)   = H_perm(P) - H_real(P)
            denominator(P) = H_perm(P) - H_min_perm(P)

        where H_min_perm(P) is a graph-theory lower envelope computed from permuted
        type counts and available same-type edges up to scale P.

        Returns:
            float score in [0,1] (clipped), and optionally the curve c(P).
        """
        if self.ent_real is None or self.ent_perm is None:
            raise ValueError("Run percolation() first to compute ent_real and ent_perm.")

        hz_min = self.hz_min if self.hz_min is not None else self._hz_min_curve_from_real_edges()

        # Align lengths
        L = min(len(self.ent_real), len(self.ent_perm), len(hz_min), len(self.pbond_vec) - 1)
        H_real = self.ent_real[:L]
        H_perm = self.ent_perm[:L]
        H_min = hz_min[:L]

        num = H_perm - H_real
        den = H_perm - H_min

        # Avoid divide-by-zero and negative denominators (can happen numerically)
        c = np.zeros(L, dtype=np.float64)
        ok = den > 0
        c[ok] = num[ok] / den[ok]

        # Clip for numerical noise
        c = np.clip(c, 0.0, 1.0)

        # Integrate/average over pbond axis (pbond_vec bins)
        x = self.pbond_vec[:L]  # note: pbond_vec was shortened (no 0), consistent with ent arrays
        # Score as mean value over P range
        area = np.trapz(c, x=x)
        span = float(x[-1] - x[0]) if L > 1 else 1.0
        score = area / span if span > 0 else float(c.mean())

        if return_curve:
            return score, c, hz_min
        return score

    def _generate_macroscopic_edges(self, ELP_local: np.ndarray) -> np.ndarray:
        from scipy.spatial.distance import pdist, squareform
        
        # 1. Temporary Union-Find to discover the local zones
        uf_temp = ConnectedComponentEntropy(self.N)
        if len(ELP_local) > 0:
            uf_temp.merge_all(ELP_local[:, :2].astype(int), calculate_entropy=False)
            
        # 2. Extract zones and compute their centroids
        roots = uf_temp.get_roots()
        unique_roots, root_inv, zone_sizes = np.unique(roots, return_inverse=True, return_counts=True)
        
        # Blazing fast O(N) centroid calculation
        sum_x = np.bincount(root_inv, weights=self.XY[:, 0])
        sum_y = np.bincount(root_inv, weights=self.XY[:, 1])
        centroids = np.column_stack((sum_x / zone_sizes, sum_y / zone_sizes))
        
        zone_types = self.type_vec[unique_roots]
        macroscopic_edges = []
        
        # 3. All-vs-All macroscopic connections within each cell type
        for t in self.unq_types:
            type_mask = (zone_types == t)
            type_zone_idx = np.where(type_mask)[0] 
            n_type_zones = len(type_zone_idx)
            
            if n_type_zones < 2:
                continue # Only one zone for this type, nothing to connect
                
            # Global fraction of this cell type
            f = zone_sizes[type_zone_idx].sum() / self.N
            
            t_centroids = centroids[type_zone_idx]
            t_sizes = zone_sizes[type_zone_idx]
            t_roots = unique_roots[type_zone_idx]
            
            # Physical distances between zones (super fast because Z << N)
            dist_matrix = squareform(pdist(t_centroids))
            
            # Convert macroscopic distance back to information rank (k)
            for i in range(n_type_zones):
                for j in range(i + 1, n_type_zones):
                    d_ab = dist_matrix[i, j]
                    
                    # k_rank is the number of same-type cells closer to 'i' than 'j' is
                    closer_mask = dist_matrix[i, :] <= d_ab
                    k_rank = np.sum(t_sizes[closer_mask]) - t_sizes[i] 
                    k_rank = max(1, k_rank) # Ensure minimum rank of 1
                    
                    p_bond = 1.0 - np.exp(-k_rank * f)
                    macroscopic_edges.append([t_roots[i], t_roots[j], p_bond])
                    
        if len(macroscopic_edges) > 0:
            return np.array(macroscopic_edges, dtype=np.float64)
        return np.empty((0, 3), dtype=np.float64)
    
    def _generate_monte_carlo_permuted_edges(self, max_m: int = 10) -> np.ndarray:
        """
        Analytically simulate the K -> infinity permuted edge list 
        using the Negative Binomial distribution, bypassing spatial search.
        """
        edges = []
        tvec = self.type_vec_perm
        unique_types, type_counts = np.unique(tvec, return_counts=True)
        
        for t, n_t in zip(unique_types, type_counts):
            if n_t < 2:
                continue
                
            f = n_t / self.N
            t_nodes = np.where(tvec == t)[0]
            
            # For each cell, simulate finding 'm' same-type neighbors
            for m in range(1, max_m + 1):
                # Negative Binomial models failures (other types) before m successes
                failures = np.random.negative_binomial(m, f, size=n_t)
                k_ranks = failures + m
                
                p_bonds = 1.0 - np.exp(-k_ranks * f)
                
                # Assign a random destination node of the same type
                dst_nodes = np.random.choice(t_nodes, size=n_t)
                
                m_edges = np.column_stack((t_nodes, dst_nodes, p_bonds))
                edges.append(m_edges)
                
        if len(edges) > 0:
            return np.vstack(edges)
        return np.empty((0, 3), dtype=np.float64)