import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from skimage import data
import pandas as pd
from itertools import product
from skimage.feature import greycomatrix, greycoprops

def normalize(x, scale=256):
    scale-=1
    x = (((x-x.min())/(x.max()-x.min()))*scale)
    return x

class MSMFE:
    def __init__(self, ref, imgs=None, vmin=0, vmax=255, nbit=8, ks=5,
                features = ['Autocorrelation', 'ClusterProminence', 'ClusterShade', 'ClusterTendency', 'Contrast',
                            'Correlation', 'Dissimilarity', 'DifferenceEntropy', 'DifferenceVariance', 'Energy',
                            'Entropy', 'Id', 'Idm', 'Idmn', 'Idn', 'Imc1', 'Imc2', 'InverseVariance', 'JointAverage',
                            'MCC', 'MaximumProbability', 'SumAverage', 'SumEntropy', 'SumSquares']):

        ref = self.normalize(ref)
        if imgs is not None:
            self.keys = imgs.keys()
            for key in imgs.keys():
                imgs[key] = self.normalize(imgs[key])

        self.vmin = vmin
        self.vmax = vmax
        self.nbit = nbit
        self.ks   = ks
        self.glcm_ref = self.fast_glcm(ref)
        self.glcm_imgs = {}
        self.features = features
        self.error = {}
        self.img_feature_maps = {}
        self.feature_maps_ref = self.feature_maps(self.glcm_ref, features)
        self.imgs = imgs

    def get_names(self):
        names = list(self.keys) + ['_Reference']
        return names

    def normalize(self, img, scale=255):
        img = (img - img.min())/(img.max()-img.min())
        img *= scale
        #img = img.astype(np.uint8)
        return img

    def get_feature_maps(self):

        if self.imgs is not None:
            for key in self.keys:
                glcm = self.fast_glcm(self.imgs[key])
                self.img_feature_maps[key] = self.feature_maps(glcm, self.features)

            self.img_feature_maps['Reference'] = self.feature_maps_ref

            return self.img_feature_maps
        else:
            return self.feature_maps_ref

    def get_error(self):
        for key in self.keys:
            glcm = self.fast_glcm(self.imgs[key])
            self.img_feature_maps[key] = self.feature_maps(glcm, self.features)

        error_df = pd.DataFrame(index=self.keys, columns=self.features)
        for feature in self.features:
            for key in self.keys:
                img = self.img_feature_maps[key][feature]
                ref = self.feature_maps_ref[feature]
                error = ((ref - img) ** 2).mean()
                error_df.at[key, feature] = error
        return error_df

    def get_saliency(self, feature):
        saliencies = []
        for key in self.keys:
            img = self.feature_maps[feature][key]
            ref = self.feature_maps_ref[feature]
            saliencies.append((ref - img) ** 2)
        saliencies = np.asarray(saliencies)
        return saliencies

    def calculate_matrix(self, img, voxelCoordinates=None):
        quant = normalize(img, scale=self.nbit).astype(np.int8)
        degrees  = [0, np.pi/4, np.pi/2, (3*np.pi)]
        distance = [1]
        P_glcm = greycomatrix(quant, distance, degrees, levels=self.nbit)
        P_glcm = np.moveaxis(P_glcm, -2, 0)
        P_glcm = P_glcm.astype(np.float32)

        sumP_glcm = np.sum(P_glcm, (1, 2)).astype(np.float32)

        sumP_glcm[sumP_glcm == 0] = np.nan
        P_glcm /= sumP_glcm[:, None, None, :]
        P_glcm = np.moveaxis(P_glcm, -1, 0).squeeze()
        return P_glcm

    def fast_glcm(self, img, conv=True, scale=False):

        min, max = self.vmin, self.vmax
        h,w = img.shape

        # digitize
        bins = np.linspace(min, max, self.nbit+1)[1:]

        gl   = np.digitize(img, bins) - 1
        gl.shape

        shifts = np.zeros((4, h, w))

        shifts[0] = np.append(       gl[:, 1:],        gl[:, -1:], axis=1) # one
        shifts[1] = np.append(       gl[1:, :],        gl[-1:, :], axis=0) # two
        shifts[2] = np.append(shifts[0][1:, :], shifts[0][-1:, :], axis=0) # three
        shifts[3] = np.append(shifts[0][:1, :], shifts[0][:-1, :], axis=0) # four

        glcm = np.zeros((4, self.nbit, self.nbit, h, w), dtype=np.uint8)
        for n, shift in enumerate(shifts):
            for i in range(self.nbit):
                for j in range(self.nbit):
                    mask = ((gl==i) & (shift==j))
                    glcm[n, i, j, mask] = 1

            if conv:
                kernel = np.ones((self.ks, self.ks), dtype=np.uint8)
                for i in range(self.nbit):
                    for j in range(self.nbit):
                        glcm[n, i, j] = cv2.filter2D(glcm[n, i, j], -1, kernel)

            glcm = glcm.astype(np.float32)

        if scale:
            matrix = self.calculate_matrix(img)
            glcm = matrix[:, :, :, None, None] * glcm

        glcm = np.moveaxis(glcm, 0, -1)
        return glcm

    def get_means(self, img, glcm):
        h,w = img.shape

        mean_i = np.zeros((h,w), dtype=np.float32)
        for i in range(self.nbit):
            for j in range(self.nbit):
                mean_i += glcm[i,j] * i / (self.nbit)**2

        mean_j = np.zeros((h,w), dtype=np.float32)
        for j in range(self.nbit):
            for i in range(self.nbit):
                mean_j += glcm[i,j] * j / (self.nbit)**2

        return mean_i, mean_j

    def get_stds(self, img, glcm):
        h,w = img.shape

        mean_i, mean_j = self.get_means(img, glcm)

        std_i = np.zeros((h,w), dtype=np.float32)
        for i in range(self.nbit):
            for j in range(self.nbit):
                std_i += (glcm[i,j] * i - mean_i)**2
        std_i = np.sqrt(std_i)

        std_j = np.zeros((h,w), dtype=np.float32)
        for j in range(self.nbit):
            for i in range(self.nbit):
                std_j += (glcm[i,j] * j - mean_j)**2
        std_j = np.sqrt(std_j)

        return mean_i, mean_j, std_i, std_j

    def get_max(self, glcm):
            max_  = np.max(glcm, axis=(0,1))
            return(max_)

    def feature_maps(self, glcm, features):
        glcm = normalize(glcm, scale=2)

        eps = np.spacing(1)

        bitVector = np.arange(0,self.nbit,1)
        i, j = np.meshgrid(bitVector, bitVector, indexing='ij', sparse=True)
        iAddj = i + j
        iSubj = np.abs(i-j)

        ux = i[:, :, None, None, None] * glcm
        uy = j[:, :, None, None, None] * glcm

        px = np.sum(glcm, 1)
        px = px[:, None, :, :, :]
        py = np.sum(glcm, 0)
        py = py[None, :, :, :, :]

        ux = np.sum((i[:, :, None, None, None] * glcm), (0, 1))
        ux = normalize(ux, scale=self.nbit)
        uy = np.sum((j[:, :, None, None, None] * glcm), (0, 1))
        uy = normalize(uy, scale=self.nbit)

        kValuesSum  = np.arange(0, (self.nbit * 2)-1, dtype='float')
        #kValuesSum = np.arange(2, (self.nbit * 2) + 1, dtype='float')

        kDiagIntensity = np.array([iAddj == k for k in  kValuesSum])
        GLCMDiagIntensity = np.array([kDiagIntensity[int(k)][:, :, None, None, None] * glcm for k in kValuesSum])
        pxAddy = np.sum(GLCMDiagIntensity, (1, 2))

        kValuesDiff = np.arange(0, self.nbit, dtype='float')
        #kValuesDiff = np.arange(0, self.nbit, dtype='float')
        kDiagContrast = np.array([iSubj == k for k in  kValuesDiff])
        GLCMDiagIntensity = np.array([kDiagContrast[int(k)][:, :, None, None, None] * glcm for k in kValuesDiff])
        pxSuby = np.sum(GLCMDiagIntensity, (1, 2))

        HXY = (-1) * np.sum((glcm * np.log2(glcm + eps)), (0, 1))

        features_dict = {}

        if 'Autocorrelation' in features:
            ac = np.sum(glcm * (i * j)[:, :, None, None, None], (0, 1))
            features_dict['Autocorrelation'] = np.nanmean(ac, -1)

        if 'ClusterProminence' in features:
            cp = np.sum((glcm * (((i + j)[:, :, None, None, None] - ux - uy) ** 4)), (0, 1))
            features_dict['ClusterProminence'] = np.nanmean(cp, -1)

        if 'ClusterShade' in features:
            cs = np.sum((glcm * (((i + j)[:, :, None, None, None] - ux - uy) ** 3)), (0, 1))
            features_dict['ClusterShade'] = np.nanmean(cs, -1)

        if 'ClusterTendency' in features:
            ct = np.sum((glcm * (((i + j)[:, :, None, None, None] - ux - uy) ** 2)), (0, 1))
            features_dict['ClusterTendency'] = np.nanmean(ct, -1)

        if 'Contrast' in features:
            cont = np.sum((glcm * ((np.abs(i - j))[:, :, None, None, None] ** 2)), (0, 1))
            features_dict['Contrast'] = np.nanmean(cont, -1)

        if 'Correlation' in features:
            # shape = (Nv, 1, 1, angles)
            sigx = np.sum(glcm * ((i[:, :, None, None, None] - ux) ** 2), (0, 1), keepdims=True) ** 0.5
            # shape = (Nv, 1, 1, angles)
            sigy = np.sum(glcm * ((j[:, :, None, None, None] - uy) ** 2), (0, 1), keepdims=True) ** 0.5

            corm = np.sum(glcm * (i[:, :, None, None, None] - ux) * (j[:, :, None, None, None] - uy), (0, 1), keepdims=True)
            corr = corm / (sigx * sigy + eps)
            corr[sigx * sigy == 0] = 1  # Set elements that would be divided by 0 to 1.
            features_dict['Correlation'] = np.nanmean(corr, (0, 1, -1))

        if 'DifferenceAverage' in features:
            features_dict['DifferenceAverage'] = np.sum((kValuesDiff[:, None, None, None] * pxSuby), (0, -1))

        if 'DifferenceEntropy' in features:
            features_dict['DifferenceEntropy'] = (-1) * np.sum((pxSuby * np.log2(pxSuby + eps)), (0, -1))

        if 'DifferenceVariance' in features:
            diffavg = np.sum((kValuesDiff[:, None, None, None] * pxSuby), 0, keepdims=True)
            diffvar = np.sum((pxSuby * ((kValuesDiff[:, None, None, None] - diffavg) ** 2)), (0, -1))
            features_dict['DifferenceVariance'] = diffvar

        if 'Energy' in features:
            sum_squares = np.sum((glcm ** 2), (0, 1))
            features_dict['Energy'] = np.nanmean(sum_squares, -1)

        if 'Entropy' in features:
            features_dict['Entropy'] = np.sum(HXY, -1)

        if 'Id' in features:
            features_dict['Id'] = np.sum(pxSuby / (1 + kValuesDiff[:, None, None, None]), (0, -1))

        if 'Idm' in features:
            features_dict['Idm'] = np.sum(pxSuby / (1 + (kValuesDiff[:, None, None, None] ** 2)), (0, -1))

        if 'Idmn' in features:
            features_dict['Idmn'] = np.sum(pxSuby / (1 + ((kValuesDiff[:, None, None, None] ** 2) / (self.nbit ** 2))), (0,-1))

        if 'Idn' in features:
            features_dict['Idn'] = np.sum(pxSuby / (1 + (kValuesDiff[:, None, None, None] / self.nbit)), (0, -1))

        if 'Imc1' in features:
            # entropy of px # shape = (Nv, angles)
            HX = (-1) * np.sum((px * np.log2(px + eps)), (0, 1))
            # entropy of py # shape = (Nv, angles)
            HY = (-1) * np.sum((py * np.log2(py + eps)), (0, 1))
            # shape = (Nv, angles)
            HXY1 = (-1) * np.sum((glcm * np.log2(px * py + eps)), (0, 1))

            div = np.fmax(HX, HY)

            imc1 = HXY - HXY1
            imc1[div != 0] /= div[div != 0]
            imc1[div == 0] = 0  # Set elements that would be divided by 0 to 0

            features_dict['Imc1'] = np.nanmean(imc1, -1)

            #print('IMC1:', features_dict['Imc1'].shape)

        if 'Imc2' in features:
            # shape = (Nv, angles)
            HXY2 = (-1) * np.sum(((px * py) * np.log2(px * py + eps)), (0, 1))

            imc2 = (1 - np.e ** (-2 * (HXY2 - HXY)))

            min = imc2.min()
            imc2 += np.abs(min)
            #print(imc2.min(), imc2.max())

            imc2 = imc2 ** 0.5

            imc2[HXY2 == HXY] = 0

            features_dict['Imc2'] = np.nanmean(imc2, -1)

        if 'InverseVariance' in features:
            features_dict['InverseVariance'] = np.sum(pxSuby[1:, :, :, :] / kValuesDiff[1:, None, None, None] ** 2, (0, -1))  # Skip k = 0 (division by 0)

        if 'JointAverage' in features:
            features_dict['JointAverage'] = ux.mean(-1)

        if 'MCC' in features:
            # Calculate Q (shape (i, i, d)). To prevent division by 0, add epsilon (such a division can occur when in a ROI
            # along a certain angle, voxels with gray level i do not have neighbors
            nom = glcm[:, :, :, :, :] * glcm[:, :, :, :, :]
            den =   px[:, 0, :, :, :] *   py[:, 0, :, :, :]
            den = np.expand_dims(den, 1)

            Q = (nom /  (den + eps))  # sum over k (4th axis --> index 3)

            for gl in range(1, glcm.shape[1]):
                Q += ((glcm[:, None, gl, :, :] * glcm[None, :, gl, :, :]) /  # slice: v, i, j, k, d
                        (px[:, None,  0, :, :] *   py[None, :, gl, :, :] + eps))  # sum over k (4th axis --> index 3)

            # calculation of eigenvalues if performed on last 2 dimensions, therefore, move the angles dimension (d) forward
            Q_eigenValue = np.linalg.eigvals(Q.transpose((2, 3, 4, 0, 1)))
            Q_eigenValue.sort()  # sorts along last axis --> eigenvalues, low to high

            if Q_eigenValue.shape[3] < 2:
                return 1  # flat region

            MCC = np.sqrt(Q_eigenValue[:, :, :,-2])  # 2nd highest eigenvalue

            features_dict['MCC'] = np.nanmean(MCC, 2).real

        if 'MaximumProbability' in features:
            maxprob = np.amax(glcm, (0, 1))
            features_dict['MaximumProbability'] = np.nanmean(maxprob, -1)

        if 'SumAverage' in features:
            sumavg = np.sum((kValuesSum[:, None, None, None] * pxAddy), 0)
            features_dict['SumAverage'] = np.nanmean(sumavg, -1)

        if 'SumEntropy' in features:
            sumentr = (-1) * np.sum((pxAddy * np.log2(pxAddy + eps)), 0)
            features_dict['SumEntropy'] = np.nanmean(sumentr, -1)

        if 'SumSquares' in features:
            ix = (i[:, :, None, None, None] - ux) ** 2
            ss = np.sum(glcm * ix, (0, 1))
            features_dict['SumSquares'] = np.nanmean(ss, -1)

        return features_dict

if __name__ == '__main__':

    ref = cv2.imread(r'C:\Users\william\Feature Maps and Saliency\Figura-4-Motor-R-de-Data-Mining-usado-para-analisis.png', 0)
    ref = cv2.resize(ref, (160, 114))

    features = ['Autocorrelation', 'ClusterProminence', 'ClusterShade', 'ClusterTendency',
                'Contrast', 'Correlation', 'DifferenceAverage', 'DifferenceEntropy', 'DifferenceVariance',
                'Energy', 'Entropy', 'Imc1', 'Imc2', 'Id', 'Idm', 'Idmn', 'Idn', 'InverseVariance',
                'MaximumProbability', 'MCC', 'SumAverage', 'SumEntropy', 'SumSquares']

    msmfe = MSMFE(ref, features=features)

    feature_maps = msmfe.get_feature_maps()

    plt.figure(figsize=(36,18))
    for i, key in enumerate(feature_maps.keys()):
        img = feature_maps[key]
        plt.title(key + ' ' + str(img.min())+ " " +str(img.max()))

        plt.imshow(img[:, :])
        plt.show()

