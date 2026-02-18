"""Tests for path_utils module."""

import pytest
from max_info_atlases.path_utils import (
    parse_path, build_path, ClusteringMetadata, extract_metadata_from_path
)


class TestParsePath:
    """Tests for parse_path function."""
    
    def test_parse_leiden_celltype(self):
        """Test parsing Leiden clustering on cell type features."""
        path = "/results/LeidenRawCorrelation/res_5/C57BL6J-638850.40.npy"
        base_dir = "/results"
        
        metadata = parse_path(path, base_dir)
        
        assert metadata.method == 'leiden'
        assert metadata.section == 'C57BL6J-638850.40'
        assert metadata.feature_type == 'celltype'
        assert metadata.data_type == 'raw'
        assert metadata.distance == 'correlation'
        assert metadata.resolution_idx == 5
    
    def test_parse_leiden_env_weighted(self):
        """Test parsing Leiden clustering on weighted environment features."""
        path = "/results/LeidenRawCorrelation_envenv_w_k_20/res_2/C57BL6J-638850.40.npy"
        base_dir = "/results"
        
        metadata = parse_path(path, base_dir)
        
        assert metadata.method == 'leiden'
        assert metadata.section == 'C57BL6J-638850.40'
        assert metadata.feature_type == 'env'
        assert metadata.data_type == 'raw'
        assert metadata.distance == 'correlation'
        assert metadata.weighted == True
        assert metadata.env_k == 20
        assert metadata.resolution_idx == 2
    
    def test_parse_leiden_pca(self):
        """Test parsing Leiden with PCA features."""
        path = "/results/LeidenPCA50Euclidean/res_10/section1.npy"
        base_dir = "/results"
        
        metadata = parse_path(path, base_dir)
        
        assert metadata.method == 'leiden'
        assert metadata.data_type == 'pca50'
        assert metadata.distance == 'euclidean'
    
    def test_parse_kmeans(self):
        """Test parsing K-means clustering."""
        path = "/results/Kmeans_env_k_50/k_100/section1.npy"
        base_dir = "/results"
        
        metadata = parse_path(path, base_dir)
        
        assert metadata.method == 'kmeans'
        assert metadata.feature_type == 'env'
        assert metadata.env_k == 50
        assert metadata.clustering_k == 100
    
    def test_parse_lda(self):
        """Test parsing LDA clustering."""
        path = "/results/LDA/k_50/k_100/section1.npy"
        base_dir = "/results"
        
        metadata = parse_path(path, base_dir)
        
        assert metadata.method == 'lda'
        assert metadata.env_k == 50
        assert metadata.clustering_k == 100


class TestBuildPath:
    """Tests for build_path function."""
    
    def test_build_leiden_celltype(self):
        """Test building path for Leiden cell type clustering."""
        metadata = ClusteringMetadata(
            method='leiden',
            section='C57BL6J-638850.40',
            feature_type='celltype',
            data_type='raw',
            distance='correlation',
            resolution_idx=5,
        )
        
        path = build_path(metadata, '/results')
        
        assert 'LeidenRawCorrelation' in path
        assert 'res_5' in path
        assert 'C57BL6J-638850.40.npy' in path
    
    def test_build_leiden_env(self):
        """Test building path for Leiden environment clustering."""
        metadata = ClusteringMetadata(
            method='leiden',
            section='C57BL6J-638850.40',
            feature_type='env',
            data_type='raw',
            distance='correlation',
            env_k=20,
            weighted=True,
            resolution_idx=2,
        )
        
        path = build_path(metadata, '/results')
        
        assert 'envenv' in path
        assert '_w_' in path
        assert 'k_20' in path
        assert 'res_2' in path


class TestExtractMetadataFromPath:
    """Tests for backward compatibility wrapper."""
    
    def test_compatibility_format(self):
        """Test that old format is returned for compatibility."""
        path = "/results/LeidenRawCorrelation_envenv_w_k_20/res_2/C57BL6J-638850.40.npy"
        base_dir = "/results"
        
        metadata = extract_metadata_from_path(path, base_dir)
        
        # Should have old-format keys
        assert 'algorithm' in metadata
        assert 'method' in metadata
        assert 'section_name' in metadata
        assert 'file_path' in metadata
        assert metadata['method'] == 'Leiden'
        assert metadata['section_name'] == 'C57BL6J-638850.40'
