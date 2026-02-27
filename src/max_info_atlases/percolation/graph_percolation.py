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
        
        # USE THE EXACT SAME PIPELINE FOR BOTH REAL AND PERMUTED
        ELP_local = self.calc_ELKexp(permute=permute)
        ELP_macro = self._generate_macroscopic_edges(ELP_local, permute=permute)
        
        if len(ELP_macro) > 0:
            ELP = np.vstack([ELP_local, ELP_macro])
        else:
            ELP = ELP_local
            
        # Sort combined edges by probability
        if len(ELP) > 0:
            ELP = ELP[np.argsort(ELP[:, 2])]
            
        # --- THE SWEEP ---
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
        self.pbond_vec = dump['pbond_vec']
        self.maxK = dump['maxK']
        self.unq_types = np.unique(self.type_vec)
        self.n_types = len(np.unique(self.type_vec))
        self.N = len(self.type_vec)

    def normalized_score(self) -> float:
        """
        Normalized percolation score.

        Returns raw_score() divided by log2(N), where N is the number of points.
        """
        if self.ent_real is None or self.ent_perm is None:
            raise ValueError("Run percolation() first to compute ent_real and ent_perm.")
        return self.raw_score() / np.log2(self.N)

    def _generate_macroscopic_edges(self, ELP_local: np.ndarray, permute: bool = False) -> np.ndarray:
        from scipy.spatial.distance import cdist
        
        # 1. Temporary Union-Find to discover the local zones
        uf_temp = ConnectedComponentEntropy(self.N)
        if len(ELP_local) > 0:
            uf_temp.merge_all(ELP_local[:, :2].astype(int))
            
        # 2. Extract ALL zones and compute their centroids
        roots = uf_temp.get_roots()
        unique_roots, root_inv, zone_sizes = np.unique(roots, return_inverse=True, return_counts=True)
        
        # Blazing fast O(N) centroid calculation for all zones
        sum_x = np.bincount(root_inv, weights=self.XY[:, 0])
        sum_y = np.bincount(root_inv, weights=self.XY[:, 1])
        all_centroids = np.column_stack((sum_x / zone_sizes, sum_y / zone_sizes))
        Z_total = len(all_centroids)
        
        tvec = self.type_vec_perm if permute else self.type_vec
        zone_types = tvec[unique_roots]
        
        macroscopic_edges = []
        
        # 3. All-vs-All macroscopic connections within each cell type
        for t in self.unq_types:
            type_mask = (zone_types == t)
            type_zone_idx = np.where(type_mask)[0] 
            n_type_zones = len(type_zone_idx)
            
            if n_type_zones < 2:
                continue 
                
            f = zone_sizes[type_zone_idx].sum() / self.N
            t_centroids = all_centroids[type_zone_idx]
            t_roots = unique_roots[type_zone_idx]
            source_sizes = zone_sizes[type_zone_idx]
            
            # Pre-allocate the complete rank matrix for this cell type
            k_matrix = np.zeros((n_type_zones, n_type_zones), dtype=np.float64)
            
            # BATCHED VECTORIZATION: Process a few thousand zones at a time 
            # This keeps the RAM footprint under 4GB even for massive Z_total
            batch_size = 2000 
            for start_i in range(0, n_type_zones, batch_size):
                end_i = min(start_i + batch_size, n_type_zones)
                
                # Distance from this batch of zones to ALL zones in the tissue
                D_batch = cdist(t_centroids[start_i:end_i], all_centroids)
                
                # Sort distances to establish relative physical rank
                sort_idx = np.argsort(D_batch, axis=1) 
                
                # Cumulative sum of zone sizes gives the absolute cell count (k-rank)
                cum_ranks = np.cumsum(zone_sizes[sort_idx], axis=1) 
                
                # Find exactly where our target same-type zones landed in the sorted order
                rank_idx = np.empty_like(sort_idx)
                np.put_along_axis(rank_idx, sort_idx, np.arange(Z_total)[None, :], axis=1)
                
                # Extract the cumulative cell count exactly at the target zones
                target_cum_ranks = np.take_along_axis(cum_ranks, rank_idx[:, type_zone_idx], axis=1)
                
                # Subtract the source zone's own size to get strict distance
                k_matrix[start_i:end_i, :] = target_cum_ranks - source_sizes[start_i:end_i, None]

            # Enforce physical symmetry (take the most optimistic local connection)
            k_sym = np.minimum(k_matrix, k_matrix.T)
            k_sym = np.maximum(1, k_sym) 
            
            # Vectorized bond probability calculation
            p_matrix = 1.0 - np.exp(-k_sym * f)
            
            # Extract upper triangle indices (unique undirected edges)
            i_idx, j_idx = np.triu_indices(n_type_zones, k=1)
            
            # Instantly stack the millions of edges into the output format
            edges = np.column_stack((t_roots[i_idx], t_roots[j_idx], p_matrix[i_idx, j_idx]))
            macroscopic_edges.append(edges)
            
        if len(macroscopic_edges) > 0:
            return np.vstack(macroscopic_edges)
        return np.empty((0, 3), dtype=np.float64)