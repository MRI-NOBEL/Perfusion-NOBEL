from InitialParametersClass import GUI1
import matplotlib.pyplot as plt
import tkinter as tk
from PerfusionClass import Perfusion
from PerfusionClass import CreateImage, continue1
from IPython.display import display, HTML

display(HTML("<style>.container {width:100% !important; }</style>"))

if __name__ == '__main__':
    object = GUI1()
    object.widget()
    object.root.mainloop()

InputDictionary = object.InputDictionary

perf = Perfusion()
perf.LoadBrainMask(InputDictionary['MaskBrain_File'])

DicomDictionary = {
    'Dicom_Folder': InputDictionary['Dicom_Folder'],
    'DeleteInitialFrames': InputDictionary['DeleteInitialFrames']
}
perf.LoadDicom(DicomDictionary)

# SIGNAL RECOVERY MAPS
perf.SignalRecovery(InputDictionary['S_pre'], InputDictionary['S_post'], InputDictionary['S_min'])

CreateImage(perf.SR, perf.Mask_Brain, 'SR.tiff', 'SR [.a.u]', True)
plt.close()
CreateImage(perf.PSR, perf.Mask_Brain, 'PSR.tiff', 'PSR [.a.u]', True)
plt.close()


ConcentrationDictionary = {
    'DeleteNegative': InputDictionary['DeleteNegative'],
    'FramesForBaseline': InputDictionary['FramesForBaseline'],
    'TE': InputDictionary['TE'],
    'Kmr': InputDictionary['Kmr']
}

perf.CreateConcentrationCurves(ConcentrationDictionary)

perf.LoadAIFMask(InputDictionary['MaskAIF_File'])
# perf.PlotAIFMask()
# plt.show()

plt.plot(perf.c_Art)
plt.xlabel("Time")
plt.ylabel("C(t)")
plt.title("CA concentration in the AIF/VOF region")
plt.show()

root1 = tk.Tk()
root1.geometry('400x150')
root1.title("CA concentration in the AIF/VOF region")

label1 = tk.Label(root1, text="¿Would you like to continue with the execution?", width=50)
label1.grid(padx=5, pady=5, row=0, columnspan=2)

buttonYes = tk.Button(root1, text="Yes", width=8, command=lambda: continue1(root1, True))
buttonYes.grid(padx=5, pady=5, row=2, column=0)

buttonNo = tk.Button(root1, text="No", width=8, command=lambda: continue1(root1, False))
buttonNo.grid(padx=5, pady=5, row=2, column=1)

root1.mainloop()


# AIF FITTING TO A GAMMA VARIATE FUNCTION
# The concentration curve in the AIF/VOF region is displayed two times
fig, axes = plt.subplots()
axes.plot(perf.c_Art)
axes.set_xlabel("Time")
axes.set_ylabel("C(t)")
axes.set_title("CA concentration in the AIF/VOF region")
plt.show()

root2 = tk.Tk()
root2.title("AIF/VOF fitting to a gamma variate function")
root2.geometry('500x150')

upperlimvar = tk.IntVar()
t0var = tk.IntVar()

upperlimlabel = tk.Label(root2, text="Upper Limit:", width=24)
upperlimlabel.grid(padx=3, pady=5, row=0, column=0)

upperlim = tk.Entry(root2, textvariable=upperlimvar, width=24)
upperlim.grid(padx=3, pady=5, row=0, column=1)

t0label = tk.Label(root2, text="t0:", width=24)
t0label.grid(padx=3, pady=5, row=3, column=0)

t0 = tk.Entry(root2, textvariable=t0var, width=24)
t0.grid(padx=3, pady=5, row=3, column=1)

startButton = tk.Button(root2, text="Start", width=8, command=lambda: perf.FitGammaVariate(root2, upperlimvar.get(),
                                                                                           [t0var.get(), 1, 1, 1]))
startButton.grid(padx=5, pady=5, row=11, column=1)

root2.mainloop()

root3 = tk.Tk()
root3.geometry('400x150')
root3.title("AIF/VOF fitting to a gamma variate function")

label3 = tk.Label(root3, text="¿Would you like to reintroduce the initial parameters?", width=50)
label3.grid(padx=5, pady=5, row=0, columnspan=2)

buttonYes = tk.Button(root3, text="Yes", width=8, command=lambda: perf.continue2(root3, True))
buttonYes.grid(padx=5, pady=5, row=2, column=0)

buttonNo = tk.Button(root3, text="No", width=8, command=lambda: perf.continue2(root3, False))
buttonNo.grid(padx=5, pady=5, row=2, column=1)

root3.mainloop()


# DECONVOLUTION
perf.SVD()

rel_Lambda = 0.2
reg_method = 'Tikhonov'
# reg_method = 'TSVD'

PerfusionConstantsDictionary = {
    'rho_Voi': InputDictionary['rho_Voi'],
    'DeltaT': InputDictionary['DeltaT'],
    'kH': InputDictionary['kH']
}

perf.ResidueCalculation(reg_method, rel_Lambda, PerfusionConstantsDictionary)

key = 'Tikhonov_0.2'
# key = 'TSVD_0.2'
residue_dict = perf.Regularization[key]

residue_dict.keys()
print("Regularization Method: {}".format(residue_dict['reg_method']))
print("Lambda_rel= {}".format(residue_dict['rel_Lambda']))
print("Lambda used in the regularization: {}".format(residue_dict['lambda_th']))


# PERFUSION PARAMETRIC MAPS
# CBF
perf.Regularization.keys()

key = 'Tikhonov_0.2'
# key = 'TSVD_0.2'

# CBF
perf.CBF(key)
perf.Regularization[key]['CBF']

CreateImage(perf.Regularization[key]['CBF'], perf.Mask_Brain, 'CBF_ml-100gs.tiff', 'CBF [mL/(100*g*s)]: '+key, True)
plt.close()

CreateImage(perf.Regularization[key]['CBF'] * 60, perf.Mask_Brain, 'CBF_ml-100gmin.tiff', 'CBF [mL/(100*g*min)]: '+key,
            True)
plt.close()

# CBV
# Non deconvolution-based method
perf.CBV()

CreateImage(perf.map_CBV, perf.Mask_Brain, 'CBV_ml-100g1.tiff', 'CBV [mL/(100g)]', True)
plt.close()

# Deconvolution-based method
perf.CBV_Deconvolution(key)
perf.Regularization[key]['CBV_deconv']
CreateImage(perf.Regularization[key]['CBV_deconv'], perf.Mask_Brain, 'CBV_ml-100gres.tiff',
            'CBV [mL/(100g)] Deconvolution: '+key, True)
plt.close()


# MTT
# Non deconvolution-based method
perf.MTT(key)
CreateImage(perf.Regularization[key]['MTT'], perf.Mask_Brain, 'MTT_1.tiff', 'MTT [s]: '+key, True)
plt.close()

# Deconvolution-based method
perf.MTT_Deconvolution(key)
CreateImage(perf.Regularization[key]['MTT_deconv'], perf.Mask_Brain, 'MTT_res.tiff', 'MTT [s] Deconvolution: '+key, True)
plt.close()

