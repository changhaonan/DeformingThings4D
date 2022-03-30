"""
Created by Haonan Chang, 03/28/2022
The idea of this file is to sample deformation graph node across different time-step.
The algorithm is bascially:
- Sample N_{max} node; Sampling T time-frame.
- Mask a random number of node during each time-frame.
- Logging position & speed in each time frame.
"""
import numpy as np
import json
import random
import scipy.spatial.transform as T

# Reading binary data
def anime_read(filename):
    """
    Author: DeformingThings4D
    filename: path of .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: riangle face data of the 1st frame
        offset_data: 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    '''check data consistency'''
    if len(offset_data) != (nf - 1) * nv * 3:
        raise ("data inconsistent error!", filename)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data


class SpatialFeature:
    """ Data is np.ndarry, use this class for operation
        The structure is:
        [pos(x, y, z), velocity(vx, vy, vz), time(t), mask(1/0)]
    """
    def __init__(self, data_array) -> None:
        self.data = data_array

    def time(self):
        return self.data[:, :, 6:7]
    
    def mask(self):
        return self.data[:, :, 7:8]
    
    def pos(self):
        return self.data[:, :, 0:3]
    
    def velocity(self):
        return self.data[:, :, 3:6]

    def rotate(self, R):
        """ R is a scipy rotation transform
        """
        # Pos is rotated
        rot_vec =  R.apply(self.data[:, :, 0:3].reshape([-1, 3]))
        self.data[:, :, 0:3] = rot_vec.reshape(self.data[:, :, 0:3].shape)
        # Velocity is rotated
        rot_vec =  R.apply(self.data[:, :, 3:6].reshape([-1, 3]))
        self.data[:, :, 3:6] = rot_vec.reshape(self.data[:, :, 3:6].shape)
        return self

    def translate(self, t):
        """ t is (3) vector
        """
        # Pos is translated
        self.data[:, :, 0:3] = self.data[:, :, 0:3] + t.reshape([1, 1, 3])
        return self

    def transform(self, M):
        """ M is a 4*4 matrix
        """
        R = T.Rotation.from_matrix(M[:3, :3])
        t = M[3, :3]
        self.rotate(R)
        self.translate(t)
        return self

    def toOrigin(self):
        """ Set the position mean in each time step to (0, 0, 0)
        """
        origin_offset = np.mean(self.data[:, :, 0:3], axis=1, keepdims=True)
        self.data[:, :, 0:3] = self.data[:, :, 0:3] - origin_offset
        return self

class SingleDeformGraphSampler:
    """ There is only one active animation
    """
    def __init__(self, anime_file, dum_path, config_file):
        self.nf, self.nv, self.nt, self.vert_data, self.face_data, self.offset_data = anime_read(anime_file)
        with open(config_file, "r") as f:
            self.config = json.loads(f.read())
        # Verification
        assert(self.config["max_sample_num"] < self.nv)
        assert(self.config["frame_sample_num"] < self.nf - 1)

    def sample(self):
        """
        Output should be:
        [T, Nmax, D], where D has
        pos, velocity, t, mask_status
        """
        n_sample_v = self.config["max_sample_num"]
        n_sample_t = self.config["frame_sample_num"]
        # Sample index in the first frame & track them all the way
        sampled_v_idx = random.sample(list(range(self.nv)), n_sample_v)
        sampled_frame = random.sample(list(range(self.nf - 2)), n_sample_t)  # The last frame won't be sampled
        # sampled_frame = list(range(n_sample_t))
        dim_feature = 3 + 3 + 1 + 1
        output = np.zeros([n_sample_t, n_sample_v, dim_feature], dtype=np.float32)
        sampled_v = self.vert_data[sampled_v_idx, :]
        for i, t in enumerate(sampled_frame):
            # pos
            sampled_offset = self.offset_data[t, sampled_v_idx, :]
            sampled_pos = sampled_v + sampled_offset
            # velocity
            sampled_velocity = self.offset_data[t + 1, sampled_v_idx, :] - sampled_offset
            # t
            sampled_time = np.ones([n_sample_v, 1], dtype=np.float32) * t
            # mask_status
            mask_prob = self.config["mask_prob"]
            sampled_mask = (np.random.uniform(size=[n_sample_v, 1]) > mask_prob).astype(np.float32)
            # Concaten
            output[i, :, :] = np.concatenate([sampled_pos, sampled_velocity, sampled_time, sampled_mask], axis=-1)
        return output
        

class MultiDeformGraphSampler:
    """ Multiple animations are presented in the same scene
    """
    def __init__(self, anime_file_list, dum_path, config_file, force_overlap=False):
        self.singe_deform_graph_sampler_list = list()
        for anime_file in anime_file_list:
            single_deform_graph_sampler = SingleDeformGraphSampler(anime_file, dum_path, config_file)
            self.singe_deform_graph_sampler_list.append(single_deform_graph_sampler)
        self.force_overlap = force_overlap

    def sample(self):
        # Take two animation by random
        [idx1, idx2] = random.sample(range(len(self.singe_deform_graph_sampler_list)), 2)
        single_sample_1 = self.singe_deform_graph_sampler_list[idx1].sample()
        single_sample_2 = self.singe_deform_graph_sampler_list[idx2].sample()
        
        if not self.force_overlap:
            return np.concatenate([single_sample_1, single_sample_2], axis=1)
        else:
            spatial_feature_1 = SpatialFeature(single_sample_1)
            spatial_feature_2 = SpatialFeature(single_sample_2)
            
            random_rot_1 = T.Rotation.random()
            random_rot_2 = T.Rotation.random()
            
            return np.concatenate([
                spatial_feature_1.rotate(random_rot_1).toOrigin().data, 
                spatial_feature_2.rotate(random_rot_2).toOrigin().data], axis=1)


if __name__ == "__main__":
    # Test
    anime_file_list = ["./code/example.anime", "./code/example.anime"]
    dump_path = "./code/example"
    config_file = "./code/example/config.json"
    deform_graph_sampler =  MultiDeformGraphSampler(anime_file_list, dump_path, config_file, force_overlap=True)

    output = deform_graph_sampler.sample()

    # Visualize: Generate node-graph visualization / the python interface for Easy3DViewer
    # Test visualization
    from visualization.graph_visualizer import *
    from visualization.context import *

    context = Context()
    context.setDir("code/data", dir_prefix="frame_")
    for i in range(output.shape[0]):
        context.open(i)
        context.addGraph("test_graph", size=2, normal_len=1)  # Normal is of scale 1
        SaveGraph(
            vertices=output[i, :, :3],
            vertex_weight=output[i, :, 7:8],
            normals=output[i, :, 3:6],
            file_name=context.at("test_graph")
        )
        context.close()