import numpy as np
import pandas as pd
from ase import Atoms
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, Voronoi
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from dscribe.descriptors import SOAP
from dscribe.descriptors import ACSF
from scipy.stats import skew
from ase.visualize import view
from dscribe.descriptors import MBTR
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


class Au20FeatureExtractor:
    """超级增强型金团簇特征提取器"""

    def __init__(self):
        self.feature_names = []

    def parse_xyz_file(self, file_path):
        """解析XYZ文件"""
        with open(file_path, 'r') as f:
            lines = f.readlines()

        num_atoms = int(lines[0].strip())
        energy = float(lines[1].strip())

        coordinates = []
        symbols = []

        for i in range(2, 2 + num_atoms):
            parts = lines[i].split()
            symbols.append(parts[0])
            coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

        return np.array(coordinates), energy, symbols

    def extract_all_features(self, coordinates):
        """提取所有特征"""
        features = {}

        features.update(self._extract_geometric_features(coordinates))
        features.update(self._extract_distance_features(coordinates))
        features.update(self._extract_angle_features(coordinates))
        features.update(self._extract_dihedral_features(coordinates))
        features.update(self._extract_electronic_features(coordinates))
        features.update(self._extract_local_density(coordinates))
        features.update(self._extract_cluster_distances(coordinates))
        features.update(self._extract_convex_hull_features(coordinates))
        features.update(self._extract_neighbor_graph_features(coordinates))
        features.update(self._extract_voronoi_features(coordinates))
        features.update(self._extract_rdf_features(coordinates))
        features.update(self._extract_coulomb_matrix(coordinates ))
        features.update(self._extract_soap_features(coordinates ))
        features.update(self._extract_acsf_features(coordinates ))
        features.update(self._extract_mbtr_features(coordinates ))
        return features


    
    def _extract_coulomb_matrix(self, coordinates):
            """提取库仑矩阵特征"""
            features = {}
            try:
                n_atoms = len(coordinates)
                M = np.zeros((n_atoms, n_atoms))
                Z = 79  # 金原子序数
                for i in range(n_atoms):
                    for j in range(n_atoms):
                        if i == j:
                            M[i, j] = 0.5 * Z ** 2.4
                        else:
                            dist = np.linalg.norm(coordinates[i] - coordinates[j])
                            M[i, j] = Z * Z / (dist + 1e-8)
                eigvals = np.linalg.eigvalsh(M)
                for k, v in enumerate(np.sort(eigvals)):
                    features[f'coulomb_eig_{k}'] = v
            except Exception as e:
                print(f"库仑矩阵特征提取失败: {e}")
            return features
    
    def _extract_soap_features(self, coordinates, rcut=5.0, nmax=8, lmax=6):
        """提取SOAP特征"""
        features = {}
        try:
            # 创建ASE Atoms对象
            atoms = Atoms(symbols=['Au'] * len(coordinates), positions=coordinates)
            
            soap = SOAP(
                species=['Au'],
                r_cut=rcut,
                n_max=nmax,
                l_max=lmax,
                average="inner",
                periodic=False
            )
            soap_vec = soap.create(atoms)
            for i, v in enumerate(soap_vec.flatten()):
                features[f'soap_{i}'] = v
        except Exception as e:
            print(f"SOAP特征提取失败: {e}")
        return features

    def _extract_acsf_features(self, coordinates, rcut=6.0):
        """提取ACSF特征"""
        features = {}
        try:
            # 创建ASE Atoms对象
            atoms = Atoms(symbols=['Au'] * len(coordinates), positions=coordinates)
            
            acsf = ACSF(
                species=['Au'],
                r_cut=rcut,
                g2_params=[[1, 1]],
                g4_params=[[1, 1, 1]],
                periodic=False
            )
            acsf_vec = acsf.create(atoms)
            acsf_mean = np.mean(acsf_vec, axis=0)
            for i, v in enumerate(acsf_mean):
                features[f'acsf_{i}'] = v
        except Exception as e:
            print(f"ACSF特征提取失败: {e}")
        return features

    def _extract_mbtr_features(self, coordinates):
        """提取MBTR特征"""
        features = {}
        try:
            # 创建ASE Atoms对象
            atoms = Atoms(symbols=['Au'] * len(coordinates), positions=coordinates)
            
            mbtr = MBTR(
                species=['Au'],
                geometry={"function": "atomic_number"},
                grid={"min": 0, "max": 100, "n": 100, "sigma": 0.1},
                weighting={"function": "unity", "r_cut": 10, "threshold": 1e-3},
                periodic=False,
                normalization="l2"  # 修改这里：使用有效的归一化选项
            )
            mbtr_vec = mbtr.create(atoms)
            
            # 如果需要展平结果，可以手动处理
            if hasattr(mbtr_vec, 'flatten'):
                mbtr_vec = mbtr_vec.flatten()
            elif isinstance(mbtr_vec, (list, tuple)):
                # 如果是列表或元组，转换为numpy数组并展平
                mbtr_vec = np.array(mbtr_vec).flatten()
            
            for i, v in enumerate(mbtr_vec):
                features[f'mbtr_{i}'] = v
        except Exception as e:
            print(f"MBTR特征提取失败: {e}")
        return features
                
   
    # ======================

    # ====================
    # 几何特征
    # ====================
    def _extract_geometric_features(self, coordinates):
        features = {}
        center = np.mean(coordinates, axis=0)

        inertia_tensor = np.zeros((3, 3))
        for coord in coordinates:
            r = coord - center
            inertia_tensor[0, 0] += r[1]**2 + r[2]**2
            inertia_tensor[1, 1] += r[0]**2 + r[2]**2
            inertia_tensor[2, 2] += r[0]**2 + r[1]**2
            inertia_tensor[0, 1] -= r[0] * r[1]
            inertia_tensor[0, 2] -= r[0] * r[2]
            inertia_tensor[1, 2] -= r[1] * r[2]

        inertia_tensor[1, 0] = inertia_tensor[0, 1]
        inertia_tensor[2, 0] = inertia_tensor[0, 2]
        inertia_tensor[2, 1] = inertia_tensor[1, 2]

        eigvals = np.linalg.eigvalsh(inertia_tensor)
        features['inertia_major'] = eigvals[2]
        features['inertia_middle'] = eigvals[1]
        features['inertia_minor'] = eigvals[0]
        features['inertia_ratio1'] = eigvals[2] / eigvals[0] if eigvals[0] > 0 else 0
        features['inertia_ratio2'] = eigvals[2] / eigvals[1] if eigvals[1] > 0 else 0
        features['inertia_ratio3'] = eigvals[1] / eigvals[0] if eigvals[0] > 0 else 0

        gyration_radius = np.sqrt(np.mean(np.sum((coordinates - center)**2, axis=1)))
        features['gyration_radius'] = gyration_radius

        max_extent = np.max(coordinates, axis=0) - np.min(coordinates, axis=0)
        features['max_extent_x'] = max_extent[0]
        features['max_extent_y'] = max_extent[1]
        features['max_extent_z'] = max_extent[2]
        features['max_extent_avg'] = np.mean(max_extent)
        features['max_extent_ratio_xy'] = max_extent[0] / max_extent[1] if max_extent[1] > 0 else 0
        features['max_extent_ratio_xz'] = max_extent[0] / max_extent[2] if max_extent[2] > 0 else 0
        features['max_extent_ratio_yz'] = max_extent[1] / max_extent[2] if max_extent[2] > 0 else 0
        features['bbox_volume'] = np.prod(max_extent)
        features['gyration_to_bbox'] = gyration_radius / (features['bbox_volume']**(1/3) + 1e-8)

        return features

    # ====================
    # 距离特征
    # ====================
    def _extract_distance_features(self, coordinates):
        features = {}
        dist_matrix = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dist_matrix, 0)

        triu_indices = np.triu_indices_from(dist_matrix, k=1)
        distances = dist_matrix[triu_indices]

        features['dist_mean'] = np.mean(distances)
        features['dist_std'] = np.std(distances)
        features['dist_min'] = np.min(distances)
        features['dist_max'] = np.max(distances)
        features['dist_skew'] = pd.Series(distances).skew()
        features['dist_kurtosis'] = pd.Series(distances).kurtosis()

        qs = [0.1, 0.25, 0.5, 0.75, 0.9]
        for q in qs:
            features[f'dist_q_{int(q*100)}'] = np.quantile(distances, q)
        features['dist_q75_q25'] = features['dist_q_75'] / (features['dist_q_25'] + 1e-8)
        features['dist_q90_q10'] = features['dist_q_90'] / (features['dist_q_10'] + 1e-8)

        nn_distances = np.sort(dist_matrix, axis=1)[:, 1:6]
        features['nn_dist_mean'] = np.mean(nn_distances)
        features['nn_dist_std'] = np.std(nn_distances)
        features['nn_dist_min'] = np.min(nn_distances)
        features['nn_dist_max'] = np.max(nn_distances)

        features['dist_sq_mean'] = np.mean(distances**2)
        features['dist_cu_mean'] = np.mean(distances**3)

        return features

    # ====================
    # 角度特征
    # ====================
    def _extract_angle_features(self, coordinates):
        features = {}
        n_atoms = len(coordinates)
        angles = []

        max_angles = 1000
        for _ in range(max_angles):
            i, j, k = np.random.choice(n_atoms, 3, replace=False)
            vec1 = coordinates[j] - coordinates[i]
            vec2 = coordinates[k] - coordinates[j]

            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)

        if angles:
            angles = np.array(angles)
            features['angle_mean'] = np.mean(angles)
            features['angle_std'] = np.std(angles)
            features['angle_min'] = np.min(angles)
            features['angle_max'] = np.max(angles)
            features['angle_skew'] = pd.Series(angles).skew()
            features['angle_kurtosis'] = pd.Series(angles).kurtosis()
            features['angle_cos_mean'] = np.mean(np.cos(angles))
            features['angle_cos_std'] = np.std(np.cos(angles))

        return features

    # ====================
    # 二面角特征
    # ====================
    def _extract_dihedral_features(self, coordinates):
        features = {}
        n_atoms = len(coordinates)
        dihedrals = []

        max_dihedrals = 500
        for _ in range(max_dihedrals):
            i, j, k, l = np.random.choice(n_atoms, 4, replace=False)
            b1 = coordinates[j] - coordinates[i]
            b2 = coordinates[k] - coordinates[j]
            b3 = coordinates[l] - coordinates[k]
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            m1 = np.cross(n1, b2/np.linalg.norm(b2))
            x = np.dot(n1, n2)
            y = np.dot(m1, n2)
            angle = np.arctan2(y, x)
            dihedrals.append(angle)

        if dihedrals:
            dihedrals = np.array(dihedrals)
            features['dihedral_mean'] = np.mean(dihedrals)
            features['dihedral_std'] = np.std(dihedrals)
            features['dihedral_min'] = np.min(dihedrals)
            features['dihedral_max'] = np.max(dihedrals)

        return features

    # ====================
    # 电子特征
    # ====================
    def _extract_electronic_features(self, coordinates):
        features = {}
        center = np.mean(coordinates, axis=0)
        dist_to_center = np.linalg.norm(coordinates - center, axis=1)

        features['center_dist_mean'] = np.mean(dist_to_center)
        features['center_dist_std'] = np.std(dist_to_center)
        features['center_dist_min'] = np.min(dist_to_center)
        features['center_dist_max'] = np.max(dist_to_center)
        features['center_dist_median'] = np.median(dist_to_center)
        features['center_dist_skew'] = pd.Series(dist_to_center).skew()

        volume = np.prod(np.max(coordinates, axis=0) - np.min(coordinates, axis=0))
        features['approx_density'] = len(coordinates) / volume if volume > 0 else 0

        return features

    # ====================
    # 局部密度特征
    # ====================
    def _extract_local_density(self, coordinates, radius_factor=1.5):
        features = {}
        dist_matrix = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)

        mean_dist = np.mean(dist_matrix, axis=1)
        radius = radius_factor * np.mean(mean_dist)
        local_counts = np.sum(dist_matrix < radius, axis=1)

        features['local_density_mean'] = np.mean(local_counts)
        features['local_density_std'] = np.std(local_counts)
        features['local_density_min'] = np.min(local_counts)
        features['local_density_max'] = np.max(local_counts)

        return features

    # ====================
    # 聚类中心距离特征
    # ====================
    def _extract_cluster_distances(self, coordinates, n_clusters=3):
        features = {}
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coordinates)
            centers = kmeans.cluster_centers_
            dist_matrix = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            triu_idx = np.triu_indices_from(dist_matrix, k=1)
            cluster_dists = dist_matrix[triu_idx]
            features['cluster_dist_mean'] = np.mean(cluster_dists)
            features['cluster_dist_std'] = np.std(cluster_dists)
            features['cluster_dist_min'] = np.min(cluster_dists)
            features['cluster_dist_max'] = np.max(cluster_dists)
        except:
            features['cluster_dist_mean'] = 0
            features['cluster_dist_std'] = 0
            features['cluster_dist_min'] = 0
            features['cluster_dist_max'] = 0

        return features

    # ====================
    # 凸包特征
    # ====================
    def _extract_convex_hull_features(self, coordinates):
        features = {}
        try:
            hull = ConvexHull(coordinates)
            features['hull_volume'] = hull.volume
            features['hull_area'] = hull.area
            features['hull_vertex_count'] = len(hull.vertices)
            features['hull_area_to_volume'] = hull.area / (hull.volume + 1e-8)
        except:
            features['hull_volume'] = 0
            features['hull_area'] = 0
            features['hull_vertex_count'] = 0
            features['hull_area_to_volume'] = 0

        return features

    # ====================
    # Voronoi 特征
    # ====================
    def _extract_voronoi_features(self, coordinates):
        features = {}
        try:
            vor = Voronoi(coordinates)
            region_volumes = []
            for region_idx in vor.point_region:
                vertices = vor.regions[region_idx]
                if -1 not in vertices and len(vertices) >= 4:
                    pts = vor.vertices[vertices]
                    try:
                        hull = ConvexHull(pts)
                        region_volumes.append(hull.volume)
                    except:
                        continue
            if region_volumes:
                region_volumes = np.array(region_volumes)
                features['voronoi_vol_mean'] = np.mean(region_volumes)
                features['voronoi_vol_std'] = np.std(region_volumes)
                features['voronoi_vol_min'] = np.min(region_volumes)
                features['voronoi_vol_max'] = np.max(region_volumes)
            else:
                features['voronoi_vol_mean'] = 0
                features['voronoi_vol_std'] = 0
                features['voronoi_vol_min'] = 0
                features['voronoi_vol_max'] = 0
        except:
            features['voronoi_vol_mean'] = 0
            features['voronoi_vol_std'] = 0
            features['voronoi_vol_min'] = 0
            features['voronoi_vol_max'] = 0

        return features

    # ====================
    # 径向分布函数 (RDF) 特征
    # ====================
    def _extract_rdf_features(self, coordinates, bin_width=0.1, max_dist=10.0):
        features = {}
        dist_matrix = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=2)
        triu_idx = np.triu_indices_from(dist_matrix, k=1)
        distances = dist_matrix[triu_idx]

        bins = np.arange(0, max_dist + bin_width, bin_width)
        hist, _ = np.histogram(distances, bins=bins, density=True)

        for i, val in enumerate(hist):
            features[f'rdf_bin_{i}'] = val

        return features

    # ====================
    # 邻居图拓扑特征
    # ====================
    def _extract_neighbor_graph_features(self, coordinates, n_neighbors=4):
        features = {}
        try:
            nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(coordinates)
            distances, indices = nbrs.kneighbors(coordinates)
            distances = distances[:, 1:]

            # 构建图
            G = nx.Graph()
            for i, neighbors in enumerate(indices):
                for j, dist in zip(neighbors, distances[i]):
                    G.add_edge(i, j, weight=dist)

            # 图拓扑指标
            features['graph_avg_degree'] = np.mean([d for n, d in G.degree()])
            features['graph_degree_std'] = np.std([d for n, d in G.degree()])
            features['graph_avg_clustering'] = nx.average_clustering(G)
            path_lengths = []
            if nx.is_connected(G):
                lengths = dict(nx.all_pairs_shortest_path_length(G))
                for src in lengths:
                    path_lengths.extend(lengths[src].values())
            if path_lengths:
                features['graph_path_mean'] = np.mean(path_lengths)
                features['graph_path_std'] = np.std(path_lengths)
            else:
                features['graph_path_mean'] = 0
                features['graph_path_std'] = 0

            features['neighbor_mean_dist'] = np.mean(distances)
            features['neighbor_std_dist'] = np.std(distances)
            features['neighbor_min_dist'] = np.min(distances)
            features['neighbor_max_dist'] = np.max(distances)
            features['neighbor_skew'] = pd.Series(distances.flatten()).skew()
        except:
            features['graph_avg_degree'] = 0
            features['graph_degree_std'] = 0
            features['graph_avg_clustering'] = 0
            features['graph_path_mean'] = 0
            features['graph_path_std'] = 0
            features['neighbor_mean_dist'] = 0
            features['neighbor_std_dist'] = 0
            features['neighbor_min_dist'] = 0
            features['neighbor_max_dist'] = 0
            features['neighbor_skew'] = 0

        return features
