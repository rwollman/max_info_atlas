import numpy as np
cimport numpy as np

# Add language level directive
# cython: language_level=3

# Make sure to include these declarations for C type conversion
ctypedef np.int64_t DTYPE_t
ctypedef np.float64_t DTYPE_float_t

# Import math functions for log2 from C library (faster than numpy)
from libc.math cimport log2

cdef class ConnectedComponentEntropy:
    """
    Class for tracking connected components and calculating their entropy efficiently.
    
    This class implements a union-find data structure optimized for computing entropy
    across all sections in the dataset, not just the current section.
    
    The entropy calculation considers the global context, where probabilities are
    scaled by the section-to-total ratio for proper normalization.
    """
    # Core data structure for union-find
    cdef int n_section          # Number of nodes in this section
    cdef int n                  # Total number of nodes across all sections
    cdef np.ndarray parent      # Parent array for union-find (parent[i] = parent of node i)
    cdef np.ndarray size        # Size array for union-find (size[i] = size of component rooted at i)
    cdef dict component_sizes   # Cache mapping component sizes to counts: {size: count}
    cdef public double entropy  # Make entropy directly public - this matches old interface
    
    def __init__(self, int n_section, int n_total = -1):
        """
        Initialize the union-find data structure with n_section singleton components,
        but calculate entropy based on n_total nodes across all sections.
        
        Parameters:
        -----------
        n_section : int
            Number of nodes in this section
        n_total : int, optional
            Total number of nodes across all sections. If -1, same as n_section.
        """
        if n_total == -1:
            n_total = n_section
            
        self.n_section = n_section
        self.n = n_total
        
        # Initially, each node is its own parent (n_section singletons)
        self.parent = np.arange(n_section, dtype=np.int64)
        # Initially, each component has size 1
        self.size = np.ones(n_section, dtype=np.int64)
        
        # Initialize component size cache - each node is its own component initially
        # {1: n_section} means "n_section components of size 1"
        self.component_sizes = {1: n_section}
        
        # Initial entropy calculation - scaled by section-to-total ratio
        # This matches the old version's initialization approach
        self.entropy = (n_section / n_total) * log2(n_total)
    
    cdef int find(self, int x):
        """
        Find the root of the component containing node x with path compression.
        
        Parameters:
        -----------
        x : int
            Node index
            
        Returns:
        --------
        int
            Root node of the component containing x
        """
        if x < 0 or x >= self.n_section:
            # Bounds checking to prevent segmentation faults
            raise IndexError(f"Node index {x} is out of bounds (0 to {self.n_section-1})")
            
        cdef int root = x
        # Find the root (keep following parent pointers until we reach a self-loop)
        while self.parent[root] != root:
            root = self.parent[root]
        
        # Path compression: make every node on path point directly to root
        cdef int next_node
        while x != root:
            next_node = self.parent[x]
            self.parent[x] = root  # Point directly to root
            x = next_node
            
        return root
    
    def merge(self, int x, int y, bint calculate_entropy=True):
        """
        Merge the components containing nodes x and y.
        
        Parameters:
        -----------
        x, y : int
            Nodes to merge
        calculate_entropy : bool, optional
            Whether to calculate entropy after merging (default True)
            
        Returns:
        --------
        float
            Entropy after merging
        """
        # Bounds checking
        if x < 0 or x >= self.n_section or y < 0 or y >= self.n_section:
            raise IndexError(f"Node indices ({x}, {y}) out of bounds (0 to {self.n_section-1})")
            
        # Find the roots (representatives) of both components
        cdef int root_x = self.find(x)
        cdef int root_y = self.find(y)
        
        # If already in the same component, no need to merge
        if root_x == root_y:
            return self.entropy
        
        # Get sizes before merging
        cdef int size_x = self.size[root_x]
        cdef int size_y = self.size[root_y]
        
        # Update component size cache: remove the old component sizes
        self.component_sizes[size_x] -= 1
        if self.component_sizes[size_x] == 0:
            del self.component_sizes[size_x]
            
        self.component_sizes[size_y] -= 1
        if self.component_sizes[size_y] == 0:
            del self.component_sizes[size_y]
        
        # Union by size: attach smaller tree under root of larger tree
        cdef int new_size = size_x + size_y
        
        if size_x < size_y:
            self.parent[root_x] = root_y
            self.size[root_y] = new_size
        else:
            self.parent[root_y] = root_x
            self.size[root_x] = new_size
        
        # Add the new combined component size to cache
        if new_size in self.component_sizes:
            self.component_sizes[new_size] += 1
        else:
            self.component_sizes[new_size] = 1
        
        # Update entropy if requested
        if calculate_entropy:
            self.entropy = self._calculate_entropy()
        
        return self.entropy
    
    cdef double _calculate_entropy(self):
        """
        Calculate entropy based on individual components, normalized by total nodes.
        
        This uses the approach from the old version, where probabilities are calculated
        relative to the total number of nodes across all sections.
        
        Returns:
        --------
        double
            The entropy value in bits
        """
        # Validate cache consistency by counting total nodes in this section
        cdef int total_nodes = 0
        cdef int s, c  # s=size, c=count
        
        # First pass: calculate total nodes and check consistency
        for s, c in self.component_sizes.items():
            total_nodes += s * c
        
        # If cache is inconsistent, rebuild it
        if total_nodes != self.n_section:
            self._rebuild_cache()
        
        # Calculate entropy correctly accounting for individual components
        # and scaling by the section-to-total ratio
        cdef double entropy = 0.0
        cdef double prob, contribution
        
        # For each unique component size
        for s, c in self.component_sizes.items():
            # For a component of size s:
            # p = s/n_total   (probability of a node being in this specific component)
            # We have c such components, so we add c times this contribution
            prob = s / <double>self.n  # Using n_total for global context
            
            # For each component of size s, add: -p * log2(p)
            # Since we have c identical terms, we can multiply by c
            contribution = -prob * log2(prob)
            entropy += c * contribution
        
        return entropy
    
    cdef void _rebuild_cache(self):
        """
        Rebuild the component size cache from the union-find structure.
        """
        # Find all component roots and their sizes
        cdef dict components = {}
        cdef int i, root
        
        # Count each node in its respective component
        for i in range(self.n_section):
            root = self.find(i)
            if root in components:
                components[root] += 1
            else:
                components[root] = 1
        
        # Clear and rebuild the component size cache
        self.component_sizes = {}
        cdef int sz
        
        # Convert from {root: size} to {size: count}
        for root, sz in components.items():
            if sz in self.component_sizes:
                self.component_sizes[sz] += 1
            else:
                self.component_sizes[sz] = 1
            
            # Update the size array for consistency
            self.size[root] = sz
    
    def entropy_value(self):
        """
        Return the current entropy value.
        
        Returns:
        --------
        float
            Entropy in bits
        """
        return self.entropy
    
    def merge_all(self, np.ndarray[np.int64_t, ndim=2] edges, np.ndarray[np.float64_t, ndim=1] p_values=None):
        """
        Merge all pairs of nodes specified in the edges array.
        
        Parameters:
        -----------
        edges : np.ndarray
            Array of shape (m, 2) where each row is a pair of nodes to merge
        p_values : np.ndarray, optional
            Probability values associated with each edge (unused, for compatibility)
            
        Returns:
        --------
        np.ndarray
            Entropy after each merge
        """
        cdef int m = edges.shape[0]
        
        # Handle empty edge arrays
        if m == 0:
            return np.array([], dtype=np.float64)
            
        # Initialize entropy array - will store entropy after each merge
        cdef np.ndarray[np.float64_t, ndim=1] entropies = np.zeros(m, dtype=np.float64)
        # Start with current entropy
        entropies[0] = self.entropy
        
        # Determine sampling frequency - for large edge lists, we only calculate
        # entropy at ~1% of the points to improve performance
        cdef int sample_freq = max(1, m // 100)
        
        cdef int i
        cdef int last_calculated = 0
        
        for i in range(m):
            # Calculate entropy only at sample points or the very end
            if i % sample_freq == 0 or i == m-1:
                # For sample points, calculate entropy after merging
                entropies[i] = self.merge(edges[i, 0], edges[i, 1], True)
                last_calculated = i
            else:
                # For non-sample points, just merge without calculating entropy
                self.merge(edges[i, 0], edges[i, 1], False)
                # Use the last calculated entropy for intermediate points
                entropies[i] = entropies[last_calculated]
        
        return entropies
    
    def get_roots(self):
        """
        Force path compression for all nodes and return the root ID for each cell.
        
        This extracts the current condensed state of the map (the "zones"), 
        allowing the Python layer to identify super-nodes for phase 2 macroscopic 
        percolation without breaking the C-level state.
        
        Returns:
        --------
        np.ndarray
            1D array of length n_section containing the root ID for each node.
        """
        # Allocate the output array with the specific C-type
        cdef np.ndarray[np.int64_t, ndim=1] roots = np.zeros(self.n_section, dtype=np.int64)
        cdef int i
        
        # Iterate through all nodes, forcing path compression via find()
        for i in range(self.n_section):
            roots[i] = self.find(i)
            
        return roots