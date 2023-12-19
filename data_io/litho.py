import glob
from data_io.base_class import BASE_DATASET

__all__ = ['Litho']

class Litho(BASE_DATASET):
    def __init__(self, root, modalities=["A", "B"], learn_mode="train", extract_slice=[0, 1], noise_type='normal',
                 transform_data=None, client_weights=[1.0], data_mode='paired', data_num=6000, data_paired_weight=0.2, seed=3, 
                 data_moda_ratio=0.5, data_moda_case='case1', dataset_splited=True, assigned_data=False, assigned_images=None):

        super(Litho, self).__init__(root, modalities=modalities, learn_mode=learn_mode, extract_slice=extract_slice, noise_type=noise_type,
                                  transform_data=transform_data, client_weights=client_weights, data_mode=data_mode, data_num=data_num,
                                  data_paired_weight=data_paired_weight, data_moda_ratio=data_moda_ratio, data_moda_case=data_moda_case,
                                  dataset_splited=dataset_splited, seed=seed)

        # infer assigned images
        self.fedmed_dataset = assigned_images
        self._get_transform_modalities()

        if not assigned_data:
            self._check_noise_type()   
            self._check_sanity()
            self._generate_dataset()
            self._generate_client_indice()

    def _check_noise_type(self):
        return super()._check_noise_type()

    def _get_transform_modalities(self):
        return super()._get_transform_modalities()

    def _check_sanity(self):
        files_A = sorted(glob.glob("%s/%s/*" % (self.dataset_path, 'A')))
        files_B = sorted(glob.glob("%s/%s/*" % (self.dataset_path, 'B')))

        A = [f.split('/')[-1][:-4] for f in files_A]
        B = [f.split('/')[-1][:-4] for f in files_B]

        for x in A:
            if x in B:
                self.files.append(x)

    def _generate_dataset(self):
        return super()._generate_dataset()

    def _generate_client_indice(self):
        return super()._generate_client_indice()