import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog


class GUI1:
    def __init__(self):
        super().__init__()
        self.root = tk.Tk()
        self.wait = self.root.wait_visibility()
        self.title = self.root.title("Inputs")
        self.geometry = self.root.geometry('500x200')

        self.InputDictionary = None

        self.dicom_folder = tk.StringVar()
        self.brain_mask = tk.StringVar()
        self.aif_mask = tk.StringVar()

        self.s_min = tk.IntVar()

    def widget(self):

        stringVar1 = tk.StringVar()
        stringVar2 = tk.StringVar()
        stringVar3 = tk.StringVar()

        s_mintext = tk.IntVar()

        folderlabel = tk.Label(self.root, text="DICOM folder:", width=24)
        folderlabel.grid(padx=3, pady=5, row=0, column=0)

        self.dicom_folder = tk.Entry(self.root, textvariable=stringVar1, width=24)
        self.dicom_folder.grid(padx=3, pady=5, row=0, column=1)
        buttonGetFolder = tk.Button(self.root, text="...", width=8,
                                    command=lambda: self.folderNameToEntry(stringVar1, "DICOM Folder"))
        buttonGetFolder.grid(padx=5, pady=5, row=0, column=2)

        filelabel1 = tk.Label(self.root, text="Brain Mask File:", width=24)
        filelabel1.grid(padx=3, pady=5, row=3, column=0)

        self.brain_mask = tk.Entry(self.root, textvariable=stringVar2, width=24)
        self.brain_mask.grid(padx=3, pady=5, row=3, column=1)
        buttonGetFile1 = tk.Button(self.root, text="...", width=8,
                                   command=lambda: self.fileNameToEntry(stringVar2, "Brain Mask File"))
        buttonGetFile1.grid(padx=5, pady=5, row=3, column=2)

        filelabel2 = tk.Label(self.root, text="Reference Vase Mask File:", width=24)
        filelabel2.grid(padx=3, pady=5, row=5, column=0)

        self.aif_mask = tk.Entry(self.root, textvariable=stringVar3, width=24)
        self.aif_mask.grid(padx=3, pady=5, row=5, column=1)
        buttonGetFile2 = tk.Button(self.root, text="...", width=8,
                                   command=lambda: self.fileNameToEntry(stringVar3, "Reference Vase Mask File"))
        buttonGetFile2.grid(padx=5, pady=5, row=5, column=2)

        sminlabel = tk.Label(self.root, text="S_min:", width=24)
        sminlabel.grid(padx=3, pady=5, row=7, column=0)
        self.s_min = tk.Entry(self.root, textvariable=s_mintext, width=24)
        self.s_min.grid(padx=3, pady=5, row=7, column=1)

        self.startButton = tk.Button(self.root, text="Start", width=8, command=lambda: self.inputDict())
        self.startButton.grid(padx=5, pady=5, row=11, column=1)

    def fileNameToEntry(self, stringVar, title):
        files = [('All Files', '*.*'), ('png Files', '*.png'), ('Numpy Files', '*.npy')]
        filename = filedialog.askopenfilename(initialdir="/", title=title, filetypes=files,
                                              defaultextension=files)
        filename = filename.strip()

        if len(filename) == 0:
            messagebox.showinfo("show info", "you must select a folder")
            return
        else:
            stringVar.set(filename)

    def folderNameToEntry(self, stringVar, title):
        foldername = filedialog.askdirectory(initialdir="/", title=title)
        foldername = foldername.strip()

        if len(foldername) == 0:
            messagebox.showinfo("show info", "you must select a file")
            return
        else:
            stringVar.set(foldername)

    def inputDict(self):
        self.InputDictionary = {
            'Dicom_Folder': self.dicom_folder.get() + "\\",
            'ConcentrationFile': None,
            'MaskBrain_File': self.brain_mask.get(),
            'MaskAIF_File': self.aif_mask.get(),

            'DeltaT': 1,  # Time interval between frames [s],
            'DeleteInitialFrames': 4,

            # Concentration Curve parameters
            'DeleteNegative': True,
            'FramesForBaseline': 15,
            'TE': 0.00617,  # Echo time [s]
            'Kmr': 1.0,     # Proportionality constant relating signal intensity and tracer concentration
            'kH': 0.1369 * (0.55 / 0.75),    # Constant grouping tisular parameters for absolute quantification
            'rho_Voi': 0.0104,  # Apparent brain density [100g/ml]

            # Signal Recovery Parameters
            'S_pre': [5, 20],       # Default values: [5, 20]
            'S_post': [60, 80],     # Default values: [60, 80]
            'S_min': int(self.s_min.get()),
        }
        self.root.destroy()
        return self.InputDictionary
