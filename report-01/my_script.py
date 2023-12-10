import os
import numpy as np
import cv2
import matplotlib.pyplot as plt 

class MyImage:
    
    def __init__(self, filepath):
        self.filepath = filepath
        
        # ファイルの読み込み
        self.original_img = cv2.imread(filepath)
        self.reload()
        
    def reload(self):
        self.img = self.original_img.copy()
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) # BGR → RGB への変換
        return self.img
        
    def resize(self, size):
        self.img = cv2.resize(self.img, dsize=tuple(size[::-1]), interpolation=cv2.INTER_AREA)
        return self.img
    
    def get_grayscale(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def img_info(self):
        print("*==============================================*")
        print(" 原画像情報")
        self._img_info(self.original_img)
        print()
        print("*==============================================*")
        print(" 現在の状態")
        self._img_info(self.img)

    def _img_info(self, img):
        # パラメータの取得
        ndim  = img.ndim  # プレーン数
        shape = img.shape # (height, width, channels)
        size  = img.size  # 全要素数（H x W x P）
        dtype = img.dtype # データ型

        h = shape[0] # 行（縦）
        w = shape[1] # 列（横）
        p = shape[2] if len(shape) > 2 else 1 # プレーン

        print("*----------------------------------------------*")
        print(" ファイル名：",self.filepath)
        print(" プレーン数：",ndim)
        print(" サイズ　　：",shape)
        print("   縦（行）：",h)
        print(" 　横（列）：",w)
        print("   色　　　：",p)
        print(" 全要素数　：",size, end="")
        print(' （{} x {} x {} = {}）'.format(h, w, p, h*w*p)) # 書式付き文字列は format 関数で出力できる
        print(" データ型　：",dtype)
        print("*----------------------------------------------*")

    @classmethod
    def normalize_spectrum(cls, spectrum, shifted=False):
        """FFT で得られた周波数スペクトルをグラフ用に正規化して可視化した画像を得る関数"""
        
        if shifted:
            fshift = spectrum
        else:
            fshift = np.fft.fftshift(spectrum)
        fabs = np.abs(fshift) # 複素数の絶対値を計算
        A = np.log10(fabs + 1) # logをとるパワーをデシベルであらわすには20をかけるが正規化するのでかけない
        B = np.amax(A)
        normalized_shifted_spectrum = np.divide(A, B, out=np.zeros_like(A), where=B!=0) * 255 # [0,1]に正規化してから8bitに直す

        return normalized_shifted_spectrum

    @classmethod
    def spectrum(cls, img):
        spectrum = np.fft.fft2(img)
        shifted_spectrum = np.fft.fftshift(spectrum) # 第1象限 ⇔ 第3象限，第2象限 ⇔ 第4象限を交換
        return shifted_spectrum

    @classmethod
    def spectrum_img(cls, img):
        """画像をスペクトル変換して fftshift させた画像を取得

        Parameter: 
        ----------
        img: np.array
            original input image
        """
        shifted_spectrum = cls.spectrum(img)
        shifted_spectrum = cls.normalize_spectrum(shifted_spectrum, shifted = True) # スペクトル表示用に正規化

        return shifted_spectrum
    
    @classmethod
    def add_sp_noise(cls, img, sp_ratio = 0.5, amount = 0.01):
        """ゴマ塩ノイズの付加"""

        output = img.copy()
        row, col = output.shape[:2]
        img_size = row * col

        # salt（white）
        num_salt = np.ceil(amount * img_size * sp_ratio)
        coords = tuple(np.random.randint(0, i-1, int(num_salt)) for i in output.shape[:2])
        output[coords] = 255

        # pepper（black）
        num_pepper = np.ceil(amount * img_size * (1. - sp_ratio))
        coords = tuple(np.random.randint(0, i-1, int(num_pepper)) for i in output.shape[:2])
        output[coords] = 0

        return output

    @classmethod
    def add_gaussian_noise(cls, img, mean=0, sigma=25):
        """ガウシアンノイズの付加"""

        row, col = img.shape[:2]
        ch = img.shape[2] if len(img.shape) > 2 else 1
        img_size = row * col * ch
        
        gauss = np.random.normal(mean, sigma, img_size)
        output = img.reshape(img_size)
        output = np.minimum((output + gauss).astype('uint16'), 255).astype('uint8')
        output = output.reshape(img.shape)
        
        return output

def normalize_spectrum(spectrum, shifted = False):
    """fft で得られた周波数スペクトルをグラフ用に正規化して可視化した画像を得る関数"""
    
    if shifted:
        fshift = spectrum
    else:
        fshift = np.fft.fftshift(spectrum)
    fabs = np.abs(fshift) # 複素数の絶対値を計算
    normalized_spectrum = np.log10(fabs + 1) # logをとるパワーをデシベルであらわすには20をかけるが正規化するのでかけない
    normalized_spectrum = np.uint8(normalized_spectrum / np.amax(normalized_spectrum) * 255) # [0,1]に正規化してから8bitに直す
    
    return normalized_spectrum


def plot_spec_img(shifted_spectrum, img, src_img=None, title=None):
    """スペクトル画像と原画像を並べて表示する関数
      左：右の画像のスペクトル
      右：原画像
      [optional] 左上：参照用のオリジナル画像（src_imgで指定）
    """
    plt.figure(figsize=(16,9))
    rc = 120
    offset = 0
    if src_img is not None:
        rc = 220
        plt.subplot(rc+1)
        plt.imshow(src_img, cmap='gray', vmin = 0, vmax = 255)
        plt.title("original image")
        offset = 2

    comment = ""
    if title is not None:
        comment = " of {}".format(title)

    plt.subplot(rc+offset+1)
    plt.imshow(shifted_spectrum, cmap='gray', vmin = 0, vmax = 255)
    plt.title("spectrum{}".format(comment))

    plt.subplot(rc+offset+2)
    plt.imshow(img, cmap='gray')
    plt.title("result{}".format(comment))


def frequency_filter(kernel, unshifted_spectrum, show_img=True, title=None):
    """線形カーネルの周波数解析を行い，その周波数スペクトルを表示する．
    さらに，そのカーネルを用いた周波数領域での演算を行い，フィルタ適用後の周波数スペクトルと結果画像も並べて表示する

    Parameters
    ----------
    kernel：解析・適用する空間フィルタリングカーネル
    unshifted_spectrum：適用対象の画像（未 fftshift）
    """

    r, c = kernel.shape[:2]
    h, w = unshifted_spectrum.shape[:2]

    #=======================================
    # ここの処理がポイント
    kernel_img = np.zeros((h, w), np.float32)
    kernel_img[0:r,0:c] = kernel
    #=======================================

    # spectrum of filter kernel
    kernel_spectrum = np.fft.fft2(kernel_img)
    shifted_kernel_spectrum = np.fft.fftshift(kernel_spectrum)
    shifted_kernel_spectrum_img = normalize_spectrum(shifted_kernel_spectrum, shifted=True)

    # spectrum filtered img
    filtered_spectrum = kernel_spectrum * unshifted_spectrum
    filtered_spectrum_img = normalize_spectrum(filtered_spectrum, shifted=False)

    filtered_img = np.fft.ifft2(filtered_spectrum)
    filtered_img = np.abs(filtered_img).astype('uint8')

    if show_img:
      fig = plt.figure(figsize=(12,10))

      plt.subplot(121)
      plt.imshow(shifted_kernel_spectrum_img, cmap="gray", interpolation="gaussian")
      title_suffix = ""
      if title is not None:
          title_suffix = " of {}".format(title)
      plt.title("spectrum{}".format(title_suffix))

      plt.subplot(122)
      plt.imshow(filtered_spectrum_img, cmap="gray")
      plt.title("spectrum of filtered image")

      fig = plt.figure(figsize=(12,10))

      plt.subplot(111)
      plt.imshow(filtered_img, cmap="gray")
      plt.title("filtered image")

    return filtered_img, filtered_spectrum, kernel_spectrum

