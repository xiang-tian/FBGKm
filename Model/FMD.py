import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import firwin, hilbert, correlate
from scipy.linalg import inv
from sklearn.metrics import mean_squared_error
from scipy.linalg import inv, pinv
from scipy.signal import hilbert, correlate
import sys
from scipy.linalg import LinAlgError 

def load_excel_data(file_path, sheet_name=0, column_name=None):

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        if column_name is None:
            data = df.iloc[:, 0].values
        else:
            data = df[column_name].values
        return data.astype(float)
    except Exception as e:
        print(f"读取Excel文件出错: {e}")
        return None

def FMD(fs, x, FilterSize=100, CutNum=10, ModeNum=3, MaxIterNum=100):

    freq_bound = np.arange(0, 1, 1/CutNum)
    temp_filters = np.zeros((FilterSize, CutNum))
    for n in range(len(freq_bound)):

        temp_filters[:, n] = firwin(
            FilterSize, 
            [freq_bound[n] + np.finfo(float).eps, 
             freq_bound[n] + 1/CutNum - np.finfo(float).eps],
            window='hann', 
            pass_zero=False
        )

    result = [[None]*5 for _ in range(CutNum+1)]
    result[0] = ['迭代计数', '迭代结果', '相关矩阵', '比较模式数', '停止数']
    
    temp_sig = np.tile(x, (CutNum, 1)).T

    itercount = 1
    while True:
        iternum = 2
        if itercount == 1:
            iternum = MaxIterNum - (CutNum - ModeNum) * iternum
        
        result[itercount][0] = iternum

        for n in range(temp_filters.shape[1]):
            f_init = temp_filters[:, n]
            y_Iter, f_Iter, k_Iter, T_Iter = xxc_mckd(
                fs, temp_sig[:, n], f_init, iternum, None, 1, 0
            )
            

            result[itercount][1] = [
                y_Iter[:, -1],  
                f_Iter[:, -1],  
                k_Iter[:, -1], 
                np.abs(np.fft.fft(f_Iter))[:FilterSize//2], 
                (np.argmax(np.abs(np.fft.fft(f_Iter))[:FilterSize//2]) - 1) * (fs/FilterSize), 
                T_Iter  
            ]

        X = result[itercount][1]
        for n in range(len(X)):
            temp_sig[:, n] = X[n][0]
            temp_filters[:, n] = X[n][1]

        CorrMatrix = np.abs(np.corrcoef(temp_sig, rowvar=False))
        CorrMatrix = np.triu(CorrMatrix, 1)

        I, J, _ = max_IJ(CorrMatrix)
        Location = [I, J]

        XI = X[I][0] - np.mean(X[I][0])
        XJ = X[J][0] - np.mean(X[J][0])
        T_1 = result[itercount][1][I][5]
        KI = CK(XI, T_1, 1)
        T_2 = result[itercount][1][J][5]
        KJ = CK(XJ, T_2, 1)

        output = J if KI > KJ else I

        X.pop(output)
        temp_sig = np.delete(temp_sig, output, axis=1)
        temp_filters = np.delete(temp_filters, output, axis=1)

        result[itercount][2] = CorrMatrix
        result[itercount][3] = Location
        result[itercount][4] = output

        if temp_filters.shape[1] == ModeNum - 1:
            break
            
        itercount += 1

    Final_Mode = np.zeros((len(result[itercount][1][0][0]), len(result[itercount][1])))
    for nn in range(len(result[itercount][1])):
        Final_Mode[:, nn] = result[itercount][1][nn][0]
    
    return Final_Mode


def xxc_mckd(fs, x, f_init, termIter, T=None, M=1, plotMode=0):

    x = np.nan_to_num(x.astype(np.float64), nan=0.0, 
                     posinf=np.finfo(np.float64).max, 
                     neginf=np.finfo(np.float64).min)
    x = x - np.mean(x)  

    if T is None:
        try:
            xxenvelope = np.abs(hilbert(x)) - np.mean(np.abs(hilbert(x)))
            xxenvelope = np.nan_to_num(xxenvelope)
            T = int(np.clip(TT(xxenvelope, fs), 1, len(x)//2))
        except:
            T = max(10, len(x)//20)  

    L = len(x)
    XmT = np.zeros((L, M+1, M))
    
    for m in range(M):
        XmT[:, 0, m] = x.ravel()
        for l in range(1, L):
            if l - T >= 0:
                XmT[l, 1:, m] = np.nan_to_num(XmT[l-T, :-1, m])

    X_matrix = XmT[:, :, 0] @ XmT[:, :, 0].T
    X_matrix = np.nan_to_num(X_matrix)

    reg_param = 1e-8 * np.trace(X_matrix)/X_matrix.shape[0] * np.eye(X_matrix.shape[0])
    X_matrix += reg_param

    cond_threshold = 1/sys.float_info.epsilon
    if np.linalg.cond(X_matrix) > cond_threshold:
        reg_param = 1e-6 * np.trace(X_matrix)/X_matrix.shape[0] * np.eye(X_matrix.shape[0])
        X_matrix += reg_param
    
    try:
        Xinv = inv(X_matrix)
    except LinAlgError:
        try:
            Xinv = pinv(X_matrix)
        except:
            Xinv = np.eye(X_matrix.shape[0]) * 1e-10 

    if termIter is None:
        termIter = 30
    if plotMode is None:
        plotMode = 0
    if M is None:
        M = 3
    if T is None:
        xxenvelope = np.abs(hilbert(x)) - np.mean(np.abs(hilbert(x)))
        T = TT(xxenvelope, fs)
    T = round(T)
    

    x = x.flatten()
    L = len(f_init)
    N = len(x)

    XmT = np.zeros((L, N, M + 1))
    for m in range(M + 1):
        for l in range(L):
            if l == 0:
                XmT[l, (m * T):, m] = x[:N - m * T]
            else:
                XmT[l, 1:, m] = XmT[l - 1, :-1, m]

    Xinv = inv(XmT[:, :, 0] @ XmT[:, :, 0].T)

    f = f_init
    ck_best = 0
    y = np.zeros(N)
    ckIter = []

    for n in range(termIter):

        y = (f.T @ XmT[:, :, 0]).T

        yt = np.zeros((N, M))
        for m in range(M):
            if m == 0:
                yt[:, m] = y
            else:
                yt[T:, m] = yt[:-T, m - 1]

        alpha = np.zeros((N, M + 1))
        for m in range(M + 1):
            mask = np.ones(M + 1, dtype=bool)
            mask[m] = False
            alpha[:, m] = (np.prod(yt[:, mask], axis=1) ** 2) * yt[:, m]
        
        beta = np.prod(yt, axis=1)
        
        # 计算Xalpha
        Xalpha = np.zeros(L)
        for m in range(M + 1):
            Xalpha += XmT[:, :, m] @ alpha[:, m]

        f = np.sum(y ** 2) / (2 * np.sum(beta ** 2)) * Xinv @ Xalpha
        f = f / np.sqrt(np.sum(f ** 2))

        ck = np.sum(np.prod(yt, axis=1) ** 2) / (np.sum(y ** 2) ** (M + 1))
        ckIter.append(ck)

        if ck > ck_best:
            ck_best = ck

        xyenvelope = np.abs(hilbert(y)) - np.mean(np.abs(hilbert(y)))
        T = TT(xyenvelope, fs)
        T = round(T)

        XmT = np.zeros((L, N, M + 1))
        for m in range(M + 1):
            for l in range(L):
                if l == 0:
                    XmT[l, (m * T):, m] = x[:N - m * T]
                else:
                    XmT[l, 1:, m] = XmT[l - 1, :-1, m]

        Xinv = inv(XmT[:, :, 0] @ XmT[:, :, 0].T)

    y_final = np.zeros((N, termIter))
    f_final = np.zeros((L, termIter))
    for i in range(termIter):
        y_final[:, i] = np.convolve(x, f_final[:, i], mode='same')
    
    return y_final, f_final, np.array(ckIter), T

def TT(y, fs):
    M = fs
    NA = correlate(y, y, mode='full')[len(y)-1:]

    zeroposi = 0 
    sample1 = NA[0]
    for lag in range(1, len(NA)):
        sample2 = NA[lag]
        if (sample1 > 0 and sample2 < 0) or (sample1 == 0 or sample2 == 0):
            zeroposi = lag
            break
        sample1 = sample2

    if zeroposi >= len(NA):
        zeroposi = len(NA) - 1
    NA = NA[zeroposi:]
    if len(NA) == 0:
        return max(1, round(fs/10))

    max_position = np.argmax(NA)

    T = zeroposi + max_position
    return max(1, T) 

def CK(x, T, M=2):

    x = x.flatten()
    N = len(x)
    x_shift = np.zeros((M + 1, N))
    x_shift[0, :] = x
    for m in range(1, M + 1):
        x_shift[m, T:] = x_shift[m - 1, :-T]
    ck = np.sum(np.prod(x_shift, axis=0) ** 2) / np.sum(x ** 2) ** (M + 1)
    return ck

def max_IJ(X):

    temp = np.max(X, axis=0)
    J = np.argmax(temp)
    I = np.argmax(X[:, J])
    M = X[I, J]
    return I, J, M

def plot_results(original_signal, modes, fs, save_path=None):
    time = np.arange(len(original_signal)) / fs
    plt.figure(figsize=(12, 8))

    plt.subplot(len(modes) + 1, 1, 1)
    plt.plot(time, original_signal, 'b', linewidth=1.5)
    plt.title('原始信号')
    plt.xlabel('时间 (s)')
    plt.ylabel('幅值')
    plt.grid(True)

    for i in range(modes.shape[1]):
        plt.subplot(len(modes) + 1, 1, i + 2)
        plt.plot(time, modes[:, i], 'g', linewidth=1.5)
        plt.title(f'模式 {i+1}')
        plt.xlabel('时间 (s)')
        plt.ylabel('幅值')
        plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def evaluate_modes(original_signal, modes):

    reconstructed = np.sum(modes, axis=1)
    mse = mean_squared_error(original_signal, reconstructed)
    energy_ratios = []
    total_energy = np.sum(original_signal**2)
    for i in range(modes.shape[1]):
        mode_energy = np.sum(modes[:, i]**2)
        energy_ratios.append(mode_energy / total_energy)
    

    corr_matrix = np.corrcoef(modes, rowvar=False)
    np.fill_diagonal(corr_matrix, 0)  
    max_corr = np.max(np.abs(corr_matrix))
    
    return {
        'MSE': mse,
        'Energy_Ratios': energy_ratios,
        'Max_Mode_Correlation': max_corr
    }

if __name__ == "__main__":

    file_path = './data.xlsx'
    column_name = "gonglv"  
    fs = 1000  
    
    data = load_excel_data(file_path, column_name=column_name)
    if data is None:
        print("数据加载失败，请检查文件路径和列名")
        exit()

    print("开始特征模式分解...")
    modes = FMD(fs, data, FilterSize=100, CutNum=10, ModeNum=3, MaxIterNum=100)
    print("绘制分解结果...")
    plot_results(data, modes, fs, save_path="FMD_results.png")
    print("计算评价指标...")
    metrics = evaluate_modes(data, modes)
    print("\n分解结果评价指标:")
    print(f"重构均方误差(MSE): {metrics['MSE']:.4f}")
    print("各模式能量占比:")
    for i, ratio in enumerate(metrics['Energy_Ratios']):
        print(f"  模式{i+1}: {ratio*100:.2f}%")
    print(f"模式间最大相关系数: {metrics['Max_Mode_Correlation']:.4f}")
