import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# -----------------------
# HARROUCHE BASMA
# -----------------------
def dft2(img):
    return np.fft.fft2(img)

def idft2(F):
    return np.real(np.fft.ifft2(F))

def amplitude_spectrum(F):
    return np.log(1 + np.abs(np.fft.fftshift(F)))

def show_image(img, title):
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()
 #masque ferequential
"""masque circulaire permet d'slectionner les basses et les hautes frequence 
en fonction de leur distance par rapport au centre """
def circular_mask(shape, r, mode):
    M, N = shape
    mask = np.zeros((M, N))
    cx, cy = M // 2, N // 2
    for i in range(M):
        for j in range(N):
            d = np.sqrt((i - cx)**2 + (j - cy)**2)
            if mode == "low":
                mask[i, j] = 1 if d <= r else 0
            else:
                mask[i, j] = 0 if d <= r else 1
    return mask

def mean_filter_mask(size, shape):
    mask = np.zeros(shape)
    mask[:size, :size] = 1 / (size * size)
    return np.fft.fft2(mask)
"""convlution dans domaine spatiale = multiplication dans domaine frequentiele"""

def gaussian_filter_mask(size, sigma, shape):
    mask = np.zeros(shape)
    k = size // 2
    for i in range(size):
        for j in range(size):
            x, y = i - k, j - k
            mask[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    mask /= np.sum(mask)
    return np.fft.fft2(mask)
# Image filtrée = TF⁻¹( TF(Image) × TF(Gaussian) )



class DFTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DFT Image Processing")

        self.image = None
        self.F = None

        tk.Button(root, text="Charger Image", command=self.load).pack(fill="x")
        tk.Button(root, text="DFT", command=self.apply_dft).pack(fill="x")
        tk.Button(root, text="DFT Inverse", command=self.apply_idft).pack(fill="x")
        tk.Button(root, text="Filtrage", command=self.filter_menu).pack(fill="x")


    def load(self):
        path = filedialog.askopenfilename()
        if path:
            img = Image.open(path).convert("L")
            self.image = np.array(img, dtype=float)
            show_image(self.image, "Image originale")

    # DFT-----------------------------------
    def apply_dft(self):
        if self.image is not None:
            self.F = dft2(self.image)
            amp = amplitude_spectrum(self.F)
            show_image(amp, "Amplitude Spectrum")


    # IDFT---------------------------------------
    def apply_idft(self):
        if self.F is not None:
            img = idft2(self.F)
            show_image(img, "Image reconstruite")


    def filter_menu(self):
        win = tk.Toplevel(self.root)
        win.title("Choix du filtrage")

        tk.Label(win, text="Choisir le type de filtrage").pack(pady=10)

        tk.Button(win, text="Filtrage circulaire (rayon)",
                  command=lambda: self.circular_menu(win)).pack(fill="x", padx=20, pady=5)

        tk.Button(win, text="Filtrage par masque",
                  command=lambda: self.mask_menu(win)).pack(fill="x", padx=20, pady=5)



    # filtrage circulaire--------------------------------------------------------------------------

    def circular_menu(self, parent):
        parent.destroy()
        win = tk.Toplevel(self.root)
        win.title("Filtrage circulaire")

        tk.Label(win, text="Rayon r").pack()
        r_entry = tk.Entry(win)
        r_entry.pack()

        mode = tk.StringVar(value="low")
        tk.Radiobutton(win, text="Passe-bas", variable=mode, value="low").pack()
        tk.Radiobutton(win, text="Passe-haut", variable=mode, value="high").pack()

        def apply():
            r = int(r_entry.get())
            F = np.fft.fftshift(dft2(self.image))
            H = circular_mask(self.image.shape, r, mode.get())
            G = F * H # Théorème de convolution
            img = idft2(np.fft.ifftshift(G))
            show_image(img, "Image filtrée (circulaire)")
            win.destroy()
        tk.Button(win, text="Appliquer", command=apply).pack(pady=10)

    """filtrage frequentiele permet de separ directe des basses et hautes frequence
     et est plus efficace pour des filtres de grande taill"""
    
    def mask_menu(self, parent):
        parent.destroy()
        win = tk.Toplevel(self.root)
        win.title("Filtrage par masque")

        tk.Button(win, text="Filtre moyenneur",
                  command=lambda: self.mean_menu(win)).pack(fill="x", padx=20, pady=5)

        tk.Button(win, text="Filtre gaussien",
                  command=lambda: self.gaussian_menu(win)).pack(fill="x", padx=20, pady=5)


    # moyenneu

    def mean_menu(self, parent):
        parent.destroy()
        win = tk.Toplevel(self.root)
        win.title("Filtre moyenneur")

        tk.Label(win, text="Taille du voisinage").pack()
        size_entry = tk.Entry(win)
        size_entry.pack()

        def apply():
            size = int(size_entry.get())
            F = dft2(self.image)
            H = mean_filter_mask(size, self.image.shape)
            img = idft2(F * H)
            show_image(img, "Image filtrée (moyenneur)")
            win.destroy()

        # Image filtree = TF^-1( TF(Image) * TF(Moyenneur) )

        tk.Button(win, text="Appliquer", command=apply).pack(pady=10)


    def gaussian_menu(self, parent):
        parent.destroy()
        win = tk.Toplevel(self.root)
        win.title("Filtre gaussien")

        tk.Label(win, text="Taille du voisinage").pack()
        size_entry = tk.Entry(win)
        size_entry.pack()

        tk.Label(win, text="Variance (σ²)").pack()
        sigma_entry = tk.Entry(win)
        sigma_entry.pack()

        def apply():
            size = int(size_entry.get())
            sigma = float(sigma_entry.get())
            F = dft2(self.image)
            H = gaussian_filter_mask(size, sigma, self.image.shape)
            img = idft2(F * H)
            show_image(img, "Image filtrée (gaussien)")
            win.destroy()

        # Image filtree = TF^-1( TF(Image) * TF(Gaussian) )

        tk.Button(win, text="Appliquer", command=apply).pack(pady=10)



root = tk.Tk()
app = DFTApp(root)
root.mainloop()

