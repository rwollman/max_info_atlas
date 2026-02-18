"""Tests for clustering module."""

import pytest
import numpy as np

from max_info_atlases.clustering.base import ClusteringMethod, get_resolution_values, get_k_values
from max_info_atlases.clustering.leiden import LeidenClustering
from max_info_atlases.clustering.kmeans import KMeansClustering
from max_info_atlases.clustering.lda import LDAClustering


class TestResolutionValues:
    """Tests for resolution value generation."""
    
    def test_default_resolutions(self):
        """Test default resolution array."""
        resolutions = get_resolution_values()
        
        assert len(resolutions) == 50
        assert resolutions[0] < resolutions[-1]  # Should be ascending
        assert resolutions[0] == pytest.approx(0.1, rel=0.1)  # 10^-1
    
    def test_custom_resolutions(self):
        """Test custom resolution parameters."""
        resolutions = get_resolution_values(n_resolutions=10, log_min=0, log_max=1)
        
        assert len(resolutions) == 10
        assert resolutions[0] == pytest.approx(1.0, rel=0.1)  # 10^0
        assert resolutions[-1] == pytest.approx(10.0, rel=0.1)  # 10^1


class TestKValues:
    """Tests for k value generation."""
    
    def test_default_k_values(self):
        """Test default k value array."""
        k_values = get_k_values()
        
        assert len(k_values) > 0
        assert k_values[0] >= 10
        assert k_values[-1] <= 1000
        assert all(isinstance(k, (int, np.integer)) for k in k_values)
    
    def test_unique_k_values(self):
        """Test that k values are unique."""
        k_values = get_k_values()
        
        assert len(k_values) == len(set(k_values))


class TestLeidenClustering:
    """Tests for Leiden clustering."""
    
    def test_init_with_resolution(self):
        """Test initialization with explicit resolution."""
        clustering = LeidenClustering(resolution=1.5)
        
        assert clustering.resolution == 1.5
        assert clustering.resolution_idx is None
    
    def test_init_with_resolution_idx(self):
        """Test initialization with resolution index."""
        clustering = LeidenClustering(resolution_idx=25)
        
        assert clustering.resolution_idx == 25
        assert clustering.resolution > 0
    
    def test_get_params(self):
        """Test parameter retrieval."""
        clustering = LeidenClustering(resolution=1.0)
        params = clustering.get_params()
        
        assert params['method'] == 'leiden'
        assert params['resolution'] == 1.0
    
    @pytest.mark.skipif(
        True,  # Skip by default as it requires igraph
        reason="Requires igraph"
    )
    def test_fit(self):
        """Test fitting on simple graph."""
        # Create a simple 2-community graph
        edges = np.array([
            [0, 1], [1, 2], [2, 0],  # First community
            [3, 4], [4, 5], [5, 3],  # Second community
            [2, 3],  # Bridge
        ])
        
        clustering = LeidenClustering(resolution=1.0)
        assignments = clustering.fit(edges)
        
        assert len(assignments) == 6
        assert len(np.unique(assignments)) >= 1


class TestKMeansClustering:
    """Tests for K-means clustering."""
    
    def test_init(self):
        """Test initialization."""
        clustering = KMeansClustering(n_clusters=5, n_pcs=15)
        
        assert clustering.n_clusters == 5
        assert clustering.n_pcs == 15
    
    def test_get_params(self):
        """Test parameter retrieval."""
        clustering = KMeansClustering(n_clusters=10)
        params = clustering.get_params()
        
        assert params['method'] == 'kmeans'
        assert params['n_clusters'] == 10
    
    def test_fit(self):
        """Test fitting on simple data."""
        np.random.seed(42)
        
        # Create 3 clear clusters
        cluster1 = np.random.randn(50, 10) + [5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cluster2 = np.random.randn(50, 10) + [0, 5, 0, 0, 0, 0, 0, 0, 0, 0]
        cluster3 = np.random.randn(50, 10) + [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]
        features = np.vstack([cluster1, cluster2, cluster3])
        
        clustering = KMeansClustering(n_clusters=3)
        assignments = clustering.fit(features)
        
        assert len(assignments) == 150
        assert len(np.unique(assignments)) == 3


class TestLDAClustering:
    """Tests for LDA clustering."""
    
    def test_init(self):
        """Test initialization."""
        clustering = LDAClustering(n_topics=10)
        
        assert clustering.n_topics == 10
    
    def test_get_params(self):
        """Test parameter retrieval."""
        clustering = LDAClustering(n_topics=20)
        params = clustering.get_params()
        
        assert params['method'] == 'lda'
        assert params['n_topics'] == 20
    
    def test_fit(self):
        """Test fitting on count data."""
        np.random.seed(42)
        
        # Create count-like data
        features = np.random.poisson(5, size=(100, 20)).astype(float)
        
        clustering = LDAClustering(n_topics=5)
        assignments = clustering.fit(features)
        
        assert len(assignments) == 100
        assert len(np.unique(assignments)) <= 5
    
    def test_topic_probabilities(self):
        """Test that topic probabilities are available after fit."""
        np.random.seed(42)
        features = np.random.poisson(5, size=(50, 10)).astype(float)
        
        clustering = LDAClustering(n_topics=3)
        clustering.fit(features)
        
        probs = clustering.get_topic_probabilities()
        
        assert probs is not None
        assert probs.shape == (50, 3)
        # Probabilities should sum to 1 (approximately)
        np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(50), decimal=5)
