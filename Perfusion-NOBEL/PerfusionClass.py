import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import re
import tkinter as tk
from PIL import Image
import cv2 as cv2


def load_dicom_on_array(dicom_folder):
    import pydicom as dc
    import os
    dicom_files = [f for f in os.listdir(dicom_folder)]
    # The next line is specific for file names following the structure: MRI[num]
    # Comment if the naming convention is different to avoid any errors
    dicom_files.sort(key=lambda f: int(re.sub('\D', '', f)))  
    images = np.array([cv2.GaussianBlur(dc.read_file(dicom_folder + img).pixel_array, (3, 3), 0.5)
                       for img in dicom_files])
    return images


def signal_recovery_calculations(tensor, S_pre, S_post, S_min, initial_frame=0):
    shifted_S_pre = [i - initial_frame for i in S_pre]
    shifted_S_post = [i - initial_frame for i in S_post]
    shifted_S_min = S_min - initial_frame

    SR_num = np.mean(tensor[shifted_S_post], axis=0) - np.mean(tensor[shifted_S_pre], axis=0)
    SR_denom = np.mean(tensor[shifted_S_pre], axis=0)
    SR = 100 * SR_num / SR_denom

    PSR_num = np.mean(tensor[shifted_S_post], axis=0) - tensor[shifted_S_min]
    PSR_denom = np.mean(tensor[shifted_S_pre], axis=0) - tensor[shifted_S_min]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        PSR = 100 * PSR_num / PSR_denom
    return SR, PSR


def CreateConcentrationCurves(tensor, Dictionary):
    slices = Dictionary["FramesForBaseline"]
    DeleteNegative = Dictionary["DeleteNegative"]
    Kmr = Dictionary['Kmr']
    TE = Dictionary['TE']

    S_0 = np.mean(tensor[:slices], axis=0)
    # S_j = tensor[slices:]
    S_j = tensor

    k = np.mean(S_j, axis=1)
    w = np.mean(k, axis=1)
    c_j = -(Kmr / TE) * np.log(S_j / S_0)

    # DeleteNegative = Dictionary['DeleteNegative']
    if DeleteNegative:
        c_j = np.where(c_j <= 0, 0, c_j)
    p = np.mean(c_j, axis=1)
    q = np.mean(p, axis=1)

    y = q
    x = list(range(0, len(y)))

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("Time")
    ax.set_ylabel("C(t)")
    ax.set_title("CA Concentration Curve")
    plt.show()
    # plt.savefig("c_j.jpg")
    plt.close()

    y1 = w
    x1 = list(range(0, len(y1)))

    fig, ax = plt.subplots()
    ax.plot(x1, y1)
    ax.set_xlabel("Time")
    ax.set_ylabel("S(t)")
    ax.set_title("Signal intensity")
    plt.show()
    # plt.savefig("S_j.jpg")
    plt.close()
    return c_j


def GammaVariate(t, t0, r, b, C0):
    tr = t - t0
    y = C0 * (tr ** r) * np.exp((-tr) / b)
    return np.where(t > t0, y, 0)


def GammaVariateLoss(Parameters, Inputs):
    t0 = Parameters[0]
    r = Parameters[1]
    b = Parameters[2]
    C0 = Parameters[3]
    Time = Inputs[0]
    Intensity = Inputs[1]
    Predictions = GammaVariate(Time, t0, r, b, C0)
    return ((Predictions - Intensity) ** 2).mean()


def FitGammaVariate(c_Art, root, UpperLimit=None, InitialParameters=[1, 1, 1, 1]):
    time_var = np.array([i for i in range(c_Art.shape[0])])
    if UpperLimit is not None:
        time_var = time_var[:UpperLimit]
        aif4fitt = c_Art[:UpperLimit]
    else:
        aif4fitt = c_Art
    InputList = [time_var, aif4fitt]

    from scipy.optimize import minimize
    Optmizer = minimize(
        GammaVariateLoss, InitialParameters, InputList,
        # method='nelder-mead',
        # method='SLSQP',
        method='L-BFGS-B',
        options={'xtol': 1e-9, 'disp': True, 'maxiter': 100000}
        )
    t0 = Optmizer.x[0]
    r = Optmizer.x[1]
    b = Optmizer.x[2]
    C0 = Optmizer.x[3]

    fitted_aif = GammaVariate(time_var, t0, r, b, C0)

    plt.figure()
    plt.plot(time_var, aif4fitt, 'o')
    plt.plot(time_var, fitted_aif, '-')
    plt.legend(['AIF', 'GammaVariateAIF'])
    plt.xlabel("Time")
    plt.ylabel(r"C$_a$ (t)")
    plt.title("Gamma Variate function fitting output in the time interval [0, {}]".format(UpperLimit))
    plt.show()
  
    time_var = np.array([i for i in range(c_Art.shape[0])])
    fitted_aif = GammaVariate(time_var, t0, r, b, C0)

    plt.figure()
    plt.plot(time_var, fitted_aif, '-')
    plt.plot(time_var, c_Art, 'o', markersize=2)
    plt.legend(['AIF', 'GammaVariateAIF'])
    plt.xlabel("Time")
    plt.ylabel(r"C$_a$ (t)")
    plt.title("Gamma Variate function fitting output in the entire curve")
    plt.show()
    root.destroy()
    return fitted_aif


def SVD4AIF(aif):
    # Array_aif=[]
    array_AIF = np.array([np.roll(aif, i) for i in range(aif.shape[0])]).T

    Lambda = np.tril(array_AIF)
    u, s, vh = np.linalg.svd(Lambda)
    # from scipy import linalg
    # u, s, vh = linalg.svd(Lambda, lapack_driver='gesvd')
    return u, s, vh, Lambda


def SolveDeConvolution(c, U, S, Vh, f_l):
    invS = S ** -1
    invSigma = np.where(invS == np.inf, 0, invS) 
    inv_AIF = np.dot(Vh.T, np.dot(np.diag(f_l * invSigma), U.T))
    k = np.dot(inv_AIF, c)
    return k


def SolveDeconvOnImage(c, u, s, v, f, Mask_Brain):
    # ks_t = []
    new_c = np.zeros(c.shape)
    for i in range(c.shape[1]):
        for j in range(c.shape[2]):
            if Mask_Brain[i, j] != np.inf:
                # step_C = np.where(c[:, i,j] > 0.01*np.max(c[:, i,j]), c[:, i,j], 0)
                # new_c[:, i,j] = SolveDeConvolution(step_C, u, s, v, f)
                new_c[:, i, j] = SolveDeConvolution(c[:, i, j], u, s, v, f)
    return new_c


def CreateImage(array, Mask_Brain, name, unidades, SAVE):
    B = np.copy(array)
    Mask_Brain_c = np.copy(Mask_Brain)
    Mask_Brain_c[Mask_Brain == np.inf] = np.nan
    # fig = plt.figure()
    plot = B * Mask_Brain_c
    plt.imshow(plot, cmap='jet')
    plt.colorbar()
    plt.title(unidades)
    if SAVE:
        im = Image.fromarray(plot)
        im.save(name, format="TIFF", save_all=True)
    plt.show()


def Save2D(ArrayNP, File, Mask_Brain=None, unidades=None, Plot=False):
    # Save file in NPY format
    np.save(File + '.npy', ArrayNP)
    # Save file in CSV format
    np.savetxt(File + '.csv', ArrayNP, delimiter=',')
    if Plot:
        CreateImage(ArrayNP, Mask_Brain, File, unidades, True)


def Save1D(ArrayNP, File):
    # Save file in NPY format
    np.save(File + '.npy', ArrayNP)
    # Save file in CSV format
    np.savetxt(File + '.csv', ArrayNP, delimiter=',')


def continue1(root, bool):
    if bool:
        root.destroy()
    else:
        root.destroy()
        sys.exit()


class Perfusion:

    def __init__(self):
        # self.FramesForBaseline = InputDictionary['FramesForBaseline']
        # self.DeleteNegative = InputDictionary['DeleteNegative']
        # self.Kmr = InputDictionary['Kmr']
        # self.TE = InputDictionary['TE']
        # self.ConcentrationDict={
        #    'FramesForBaseline': self.FramesForBaseline,
        #    'DeleteNegative': self.DeleteNegative,
        #    'Kmr': self.Kmr, 'TE': self.TE
        # }
        # self.rho_Voi = InputDictionary['rho_Voi']
        # self.DeltaT = InputDictionary['DeltaT']
        # self.kH = InputDictionary['kH']

        self.BaseFolder = None
        self.kH = None
        self.DeltaT = None
        self.rho_Voi = None
        self.map_CBV = None
        self.smoothAIF = None
        self.aif = None
        self.c_Art = None
        self.AIF_Mask = None
        self.AIF_File = None
        self.Mask_Brain = None
        self.MaskBrain_File = None
        self.Kmr = None
        self.TE = None
        self.c_MRI = None
        self.S_min = None
        self.S_post = None
        self.S_pre = None
        self.initialFrame = None
        self.DicomFolder = None
        self.arrayMRI = np.array([])
        self.arrayMRI_cut = np.array([])
        self.U = np.array([])
        self.S = np.array([])
        self.Vh = np.array([])
        self.AIF_matrix = np.array([])
        self.Regularization = {}
        self.SR = np.array([])
        self.PSR = np.array([])
        self.LoadedPerfusionConstants = False
        self.SaveBaseClass = False

    def LoadDicom(self, InputDictionary):
        self.DicomFolder = InputDictionary['Dicom_Folder']
        self.arrayMRI = load_dicom_on_array(self.DicomFolder)
        # The initial frames are deleted because they can be unstable
        self.initialFrame = InputDictionary['DeleteInitialFrames']
        self.arrayMRI_cut = self.arrayMRI[self.initialFrame:]
        print("Number of images registered: {}".format(len(self.arrayMRI)))

    def SignalRecovery(self, S_pre, S_post, S_min):
        if self.arrayMRI_cut.size == 0:
            raise ValueError('No image tensor found. The LoadDicom method must be called before.')
        # Signal Recovery
        self.S_pre = [i - self.initialFrame for i in S_pre]
        self.S_post = [i - self.initialFrame for i in S_post]
        self.S_min = S_min
        self.SR, self.PSR = signal_recovery_calculations(self.arrayMRI_cut, self.S_pre, self.S_post, self.S_min)

    def CreateConcentrationCurves(self, ConcentrationDict):
        if self.arrayMRI_cut.size == 0:
            raise ValueError('No image tensor found. The LoadDicom method must be called before.')
        # Concentration curves
        self.c_MRI = CreateConcentrationCurves(self.arrayMRI_cut, ConcentrationDict)
        self.TE = ConcentrationDict['TE']
        self.Kmr = ConcentrationDict['Kmr']

    def LoadBrainMask(self, MaskBrain_File):
        import os.path
        self.MaskBrain_File = MaskBrain_File
        extensionBM = os.path.splitext(self.MaskBrain_File)[1]
        if extensionBM == '.png':
            self.Mask_Brain = np.array(Image.open(self.MaskBrain_File))
            self.Mask_Brain = self.Mask_Brain.astype(int)
        elif extensionBM == '.npy':
            self.Mask_Brain = np.load(self.MaskBrain_File)
        else:
            raise ValueError('The file format must be .png or .npy')
        self.Mask_Brain = np.where(self.Mask_Brain == 0, np.inf, self.Mask_Brain)

    def LoadAIFMask(self, AIF_File):
        import os.path
        self.AIF_File = AIF_File
        extensionBM = os.path.splitext(self.AIF_File)[1]
        if extensionBM == '.png':
            self.AIF_Mask = np.array(Image.open(self.AIF_File))
            self.AIF_Mask = self.AIF_Mask.astype(int)
        elif extensionBM == '.npy':
            self.AIF_Mask = np.load(self.AIF_File)
        else:
            raise ValueError('The file format must be .png or .npy')
        self.AIF_Mask = np.where(self.AIF_Mask == 0, np.inf, self.AIF_Mask)
        self.c_Art = np.mean(self.c_MRI[:, self.AIF_Mask != np.inf], axis=1)
        self.aif = self.c_Art

    def LoadAIF(self, AIF_File):
        self.aif = np.load(AIF_File)
        self.c_Art = self.aif

    def SmoothAIF(self):
        self.smoothAIF = np.zeros(self.c_Art.shape)
        self.smoothAIF[0] = self.c_Art[0]
        for i in range(1, self.smoothAIF.shape[0] - 1):
            self.smoothAIF[i] = (self.c_Art[i - 1] + 4 * self.c_Art[i] + self.c_Art[i + 1]) / 6.0
        self.aif = self.smoothAIF

    def LoadConcentrationTensor(self, ConcentrationFolder):
        self.c_MRI = np.load(ConcentrationFolder)

    def PlotBrainMask(self):
        # fig = plt.figure()
        plt.matshow(self.Mask_Brain)

    def PlotAIFMask(self):
        # fig = plt.figure()
        plt.matshow(self.AIF_Mask)

    def PlotAIF(self):
        # fig = plt.figure()
        plt.plot(self.c_Art)

    def FitGammaVariate(self, root, UpperLimit=None, InitialParameters=[1, 1, 1, 1]):
        self.aif = FitGammaVariate(
            self.c_Art, root,
            UpperLimit=UpperLimit,
            InitialParameters=InitialParameters
        )

    def widget_gammafit(self):
        fig, axes = plt.subplots()
        axes.plot(self.c_Art)
        axes.set_xlabel("Time")
        axes.set_ylabel(r"C$_a$(t)")
        axes.set_title("CA concentration in the AIF/VOF region")
        plt.show()

        root2 = tk.Tk()
        root2.title("AIF/VOF fitting to a gamma variate function")
        root2.geometry('500x200')

        upperlimvar = tk.IntVar()
        t0var = tk.IntVar()

        upperlimlabel = tk.Label(root2, text="Upper limit:", width=24)
        upperlimlabel.grid(padx=3, pady=5, row=0, column=0)

        upperlim = tk.Entry(root2, textvariable=upperlimvar, width=24)
        upperlim.grid(padx=3, pady=5, row=0, column=1)

        t0label = tk.Label(root2, text="t0:", width=24)
        t0label.grid(padx=3, pady=5, row=3, column=0)

        t0 = tk.Entry(root2, textvariable=t0var, width=24)
        t0.grid(padx=3, pady=5, row=3, column=1)

        startButton = tk.Button(root2, text="Start", width=8,
                                command=lambda: self.FitGammaVariate(root2, upperlimvar.get(),
                                                                     [t0var.get(), 1, 1, 1]))
        startButton.grid(padx=5, pady=5, row=11, column=1)
        root2.mainloop()
        return

    def continue2(self, root, bool):
        root.destroy()
        while bool:
            self.widget_gammafit()

            root1 = tk.Tk()
            root1.geometry('400x150')
            root1.title("AIF/VOF fitting to a gamma variate function")

            label1 = tk.Label(root1, text="¿Would you like to reintroduce the initial parameters?", width=50)
            label1.grid(padx=5, pady=5, row=0, columnspan=2)

            buttonYes = tk.Button(root1, text="Yes", width=8, command=lambda: self.continue2(root1, True))
            buttonYes.grid(padx=5, pady=5, row=2, column=0)

            buttonNo = tk.Button(root1, text="No", width=8, command=lambda: self.continue2(root1, False))
            buttonNo.grid(padx=5, pady=5, row=2, column=1)

            root1.mainloop()
            return

    def ShiftAIF(self, factor=0.1):
        Shift = np.argmax(self.aif > factor * np.max(self.aif))
        print('¡¡ATTENTION!! The {} initial frames were deleted (Cart ==0)'.format(Shift))
        self.aif = self.aif[Shift:]
        self.c_MRI = self.c_MRI[Shift:]
        self.c_Art = self.c_Art[Shift:]

    def SVD(self):
        try:
            self.ShiftAIF(factor=0.01)
            self.U, self.S, self.Vh, self.AIF_matrix = SVD4AIF(self.aif)
        except AttributeError:
            print('The arterial input function (AIF) isnt defined. The LoadAIFMask method must be called.')

    def LoadPerfusionConstants(self, PerfusionConstants):
        # Perfusion constants
        self.rho_Voi = PerfusionConstants['rho_Voi']
        self.DeltaT = PerfusionConstants['DeltaT']
        self.kH = PerfusionConstants['kH']
        self.LoadedPerfusionConstants = True

    def ResidueCalculation(self, reg_method, rel_Lambda, PerfusionConstants=None):
        lambda_th = rel_Lambda * self.S[0]
        # The regularization method is selected and the coefficients are obtained
        if reg_method == 'Tikhonov':
            f_j = (self.S ** 2) / (self.S ** 2 + lambda_th ** 2)
        elif reg_method == 'TSVD':
            f_j = self.S > lambda_th
        else:
            raise ValueError('The regularization methods available are: Tikhonov o TSVD')

        residue_dict = {
            'reg_method': reg_method,
            'rel_Lambda': rel_Lambda,
            'lambda_th': lambda_th,
            'f_j': f_j
        }

        if PerfusionConstants is None:
            if not self.LoadedPerfusionConstants:
                raise ValueError('No perfusion constants found. The corresponding dictionary must be provided')
            else:
                residue_dict.update({
                    'rho_Voi': self.rho_Voi, 'DeltaT': self.DeltaT, 'kH': self.kH
                })

        else:
            self.LoadPerfusionConstants(PerfusionConstants)
            residue_dict.update({
                'rho_Voi': self.rho_Voi, 'DeltaT': self.DeltaT, 'kH': self.kH
            })

        ks = SolveDeconvOnImage(
            self.c_MRI,
            self.U, self.S, self.Vh,
            f_j,
            self.Mask_Brain
        )

        residue_dict.update({'k_s': ks})

        self.Regularization.update(
            {reg_method + '_' + str(rel_Lambda): residue_dict}
        )

    def CBF(self, key):
        ks = self.Regularization[key]['k_s']
        CBF = np.max(ks, axis=0)
        self.Regularization[key].update({'CBF': (self.kH / self.rho_Voi) * CBF})

    def CBV_Deconvolution(self, key):
        # Deconvolution-based method for the CBV calculation 
        ks = self.Regularization[key]['k_s']
        CBV_deconv = np.trapz(ks, dx=self.DeltaT, axis=0)
        self.Regularization[key].update({'CBV_deconv': (self.kH / self.rho_Voi) * CBV_deconv})

    def MTT_Deconvolution(self, key):
        # MTT is calculated using the result obtained from the CBV_Deconvolution calculations
        try:
            CBV_deconv = self.Regularization[key]['CBV_deconv']
        except:
            raise KeyError('No CBV_deconv was found. The method CBV_Deconvolution must be called')
        try:
            CBF = self.Regularization[key]['CBF']
        except:
            raise KeyError('No CBF was found. The method CBF must be called')
        MTT_deconv = CBV_deconv / CBF
        self.Regularization[key].update({'MTT_deconv': MTT_deconv})

    def CBV(self):
        # Non deconvolution-based method for the CBV calculation
        self.map_CBV = np.trapz(self.c_MRI, dx=self.DeltaT, axis=0) / np.trapz(self.aif, dx=self.DeltaT)
        self.map_CBV = (self.kH / self.rho_Voi) * self.map_CBV

    def MTT(self, key):
        try:
            CBF = self.Regularization[key]['CBF']
        except:
            raise KeyError('No CBF was found. The CBF method must be called')
        MTT = self.map_CBV / CBF
        self.Regularization[key].update({'MTT': MTT})

    def PlotVoxel(self, lambdaKey, pixX=60, pixY=60):
        plt.figure()
        plt.plot(self.aif, '-')
        plt.title("AIF used")
        plt.legend(['AIF'])
        plt.title('AIF')

        ks_1 = self.Regularization[lambdaKey]['k_s'][:, pixX, pixY]
        plt.figure()
        plt.plot(ks_1, '-o', markersize=3)
        plt.legend(['Residue'])
        plt.title('Pixel Residue Function: ({}, {}) [{}]'.format(pixX, pixY, lambdaKey))
        plt.show()

        plt.figure()
        c_tissue_1 = self.c_MRI[:, pixX, pixY]
        plt.plot(c_tissue_1, 'o-', markersize=3)
        plt.plot(np.convolve(self.aif, ks_1, mode='full'), '--')
        plt.legend(['C_tissue', 'Convolution'])
        plt.title('AIF Convolution x Pixel Residue Function: ({}, {}) [{}]'.format(pixX, pixY, lambdaKey))
        plt.show()

    def SaveRegularization(self, key):
        if not self.SaveBaseClass:
            raise ValueError('The analysis Base Data wasnt saved. Execute Ejecuta SaveClass() method')

        reg_method = self.Regularization[key]
        SubFolder = self.BaseFolder + key + '/'
        import os
        if not os.path.exists(SubFolder):
            os.mkdir(SubFolder)
        info = dict({})
        info.update({
            'reg_method': reg_method['reg_method'],
            'rel_Lambda': reg_method['rel_Lambda'],
            'lambda_th': reg_method['lambda_th'],
            'rho_Voi': reg_method['rho_Voi'],
            'DeltaT': reg_method['DeltaT'],
            'kH': reg_method['kH'],

        })
        import json
        with open(SubFolder + 'RegularizationConfig.json', 'w') as fp:
            json.dump(info, fp)
        # The regularization coefficients are saved
        Save1D(reg_method['f_j'], SubFolder + 'f_j_Coefficients')
        Save2D(reg_method['CBF'], SubFolder + 'CBF', self.Mask_Brain, 'CBF [mL/(100*g*s)]: ' + key, True)
        Save2D(reg_method['CBV_deconv'], SubFolder + 'CBV_deconv', self.Mask_Brain,
               'CBV [mL/(100g)] Regularization: ' + key, True)
        Save2D(reg_method['MTT_deconv'], SubFolder + 'MTT_deconv', self.Mask_Brain, 'MTT [s] Regularization: ' + key,
               True)
        Save2D(self.map_CBV, SubFolder + 'CBV_NoDeconv', self.Mask_Brain, 'CBV [mL/(100g)]', True)
        Save2D(reg_method['MTT'], SubFolder + 'MTT_NoDeconv', self.Mask_Brain, 'MTT [s] ' + key, True)
        np.save(SubFolder + 'k_s.npy', reg_method['k_s'])

    def SaveClass(self, BaseFolder):
        import os
        if not os.path.exists(BaseFolder):
            os.mkdir(BaseFolder)
        np.save(BaseFolder + 'MRI_concentration.npy', self.c_MRI)

        Save1D(self.c_Art, BaseFolder + 'Initial_AIFcurve')
        Save1D(self.aif, BaseFolder + 'Used_AIFcurve')
        
        if len(self.SR) != 0:
            Save2D(self.SR, BaseFolder + 'SR', self.Mask_Brain, 'SR [a.u.]')
        if len(self.PSR) != 0:
            Save2D(self.PSR, BaseFolder + 'PSR', self.Mask_Brain, 'PSR [a.u.]')
        Save2D(self.U, BaseFolder + 'U_SVD_Matrix')
        Save2D(self.Vh, BaseFolder + 'Vh_SVD_Matrix')
        Save2D(self.AIF_matrix, BaseFolder + 'AIF_matrix')
        Save1D(self.S, BaseFolder + 'Diagonal_Sigma_SVD')
        
        self.BaseFolder = BaseFolder
        self.SaveBaseClass = True

    def SaveAll(self, BaseFolder):
        self.SaveClass(BaseFolder)
        for key in self.Regularization.keys():
            self.SaveRegularization(key)
