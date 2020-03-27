# 误差（Errors）

误差的基本概念在高中物理里面应该有所涉及，这里就不仔细展开了。
对于科学计算而言，我们所关注的主要是相对误差(relative error)和绝对误差(absolute error)。

如果我们把一个数据的实际值记做$x$, 把它的近似值极为$\hat{x}$。
其中，这个近似$\hat{x}$，或者说是误差产生的原因主要有：
* 测量中的误差；
* 运算和计算机表达中的不精确所引入的机器误差或者舍入误差（round-off error）；
* 数值方法和离散化带来的误差（discretization error）。

尤其是后面两种，是科学计算中要面临的两个主要的误差来源， 下面我们会对它们进行逐一分析。
因此，在评估和使用数值方法时，我们需要系统的衡量这两项误差，并且要__掌握它们的来源__，对它们的__大小有足够的控制__。



## 范数（Norm）

在了解误差以前，我们先来回顾一下线性代数中的一个常规概念，**范数**。

科学计算中经常会涉及到向量（vector），矩阵（matrix）和张量（tensor）。
计算与它们有关的误差，需要使用更为一般化的“绝对值”函数，也就是__范数__ $\|\cdot\|$。

下面是几个常见的范数，其中向量由小写字母表示，矩阵由大写字母表示。

* 1-范数：$\|v\|_1=(|v_1|+|v_2|+\cdots+|v_N|)$
* 2-范数：$\|v\|_2=(v_1ˆ2+v_2ˆ2+\cdots+v_Nˆ2)ˆ{\frac{1}{2}}$
* p-范数：$\|v\|_p=(v_1ˆp+v_2ˆp+\cdots+v_Nˆp)ˆ{\frac{1}{p}},\quad p>0$
* $\infty$-范数，$\|v\|_{\infty}=\max_{1\le i \le N}|v_i|$
* 矩阵的p-范数：$\|A\|_p=\max_{\|u\|_p=1}\|Au\|_p$
* Frobenius-范数：$\|A\|_F=(\sum_{i=1}ˆm\sum_{j=1}ˆn|a_{ij}|ˆ2)ˆ{\frac{1}{2}}$

这些范数可以用Matlab的`norm()`函数或者用Python中的[`numpy.linalg.norm()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html)函数进行计算。

那么绝对误差的用范数表达式为$e=\|x-\hat{x}\|$，相对误差为$\epsilon=\frac{\|x-\hat{x}\|}{\|x\|}$。

### 例子1： `absrelerror(corr, approx)`函数使用默认的2-范数，计算`corr`和`approx`之间的绝对和相对误差。


```python
import numpy as np

def absrelerror(corr=None, approx=None):
    """ 
    Illustrates the relative and absolute error.
    The program calculates the absolute and relative error. 
    For vectors and matrices, numpy.linalg.norm() is used.
    
    Parameters
    ----------
    corr : float, list, numpy.ndarray, optional
        The exact value(s)
    approx : float, list, numpy.ndarray, optional
        The approximated value(s)
        
    Returns
    -------
    None
    """
    
    print('*----------------------------------------------------------*')
    print('This program illustrates the absolute and relative error.')
    print('*----------------------------------------------------------*')

    # Check if the values are given, if not ask to input
    if corr is None:
        corr = float(input('Give the correct, exact number: '))
    if approx is None:
        approx = float(input('Give the approximated, calculated number: '))

    # be default 2-norm/Frobenius-norm is used
    abserror = np.linalg.norm(corr - approx)
    relerror = abserror/np.linalg.norm(corr)

    # Output
    print(f'Absolute error: {abserror}')
    print(f'Relative error: {relerror}')
```

## 舍入误差（Round-off error）

舍入误差，有时也称为机器精度（machine epsilon），简单理解就是**由于用计算机有限的内存来表达实数轴上面无限多的数，而引入的近似误差。**

换一个角度来看，这个误差应该不超过计算机中可以表达的相邻两个实数之间的间隔。
通过一些巧妙的设计（例如 [IEEE 浮点数标准 754-2019](https://ieeexplore.ieee.org/document/8766229)），这个间隔的绝对值可以随计算机所要表达的值的大小变化，并始终保持它的相对值在一个较小的范围。
关于浮点数标准，我们会在后面的__计算机算术__里面具体解释和分析。

### 例子2：机器精度
使用之前定义的`absrelerror()`函数，在Python3环境下，运行下面的程序。


```python
a = 4/3
b = a-1
c = b + b +b
print(f'c = {c}')
absrelerror(c, 1)
```

    c = 0.9999999999999998
    *----------------------------------------------------------*
    This program illustrates the absolute and relative error.
    *----------------------------------------------------------*
    Absolute error: 2.220446049250313e-16
    Relative error: 2.2204460492503136e-16


可以观察到，这时的机器精度约为$10ˆ{-16}$。

值得注意的是，如果我们直接令`b=1/3`，再求和，则不会得到任何误差。
事实上，这个舍入误差出现在`b=a-1`时，并且延续到了后面的求和计算。
至于具体的原理，也会在__计算机算术__和浮点数标准中详细介绍。

### 例子3: 矩阵运算中的误差

下面定义一个测试函数`testErrA(n)`，随机生成$n\times n$矩阵$A$，然后计算$Aˆ{-1}A$与单位矩阵$I$之间的误差。

理论上，这两个值应该完全相等，但是由于存在舍入误差，$Aˆ{-1}A$并不完全等于单位矩阵。


```python
# Generate a random nxn matrix and compute A^{-1}*A which should be I analytically
def testErrA(n = 10):
    A = np.random.rand(n,n)
    Icomp = np.matmul(np.linalg.inv(A), A)
    Iexact = np.eye(n)
    absrelerror(Iexact, Icomp)
```

调用函数可得到类似如下的输出：


```python
testErrA()
```

    *----------------------------------------------------------*
    This program illustrates the absolute and relative error.
    *----------------------------------------------------------*
    Absolute error: 1.1600263611094952e-14
    Relative error: 3.668325446942974e-15


由于矩阵$A$的随机性，这里的输出并不会完全一致，但相对误差始终保持在$10ˆ{-14}$至$10ˆ{-16}$这个范围。
我们也可以尝试不同大小的矩阵，误差也会随着矩阵尺寸变大而增大。

**注1**：对于矩阵求逆运算的机器精度估算，涉及到矩阵的条件数（condition number），这个会在后面求解线性方程时具体分析。

**注2**：矩阵求逆的运算复杂度约为$\mathcal{O}(nˆ2)$或以上，因此继续增大$n$有可能会让程序的运行时间大大增加。

## 离散化误差（Discretization error）（D顾名思义，iscre是由数值方法的离散化所引入的误差。通常情况下，离散化误差的大小与离散化尺寸直接相关。
以前向差分（forward difference）为例，函数$f(x)$的一阶导数可以近似为：
$$f'(x)\approx \frac{f(x+h)-f(x)}{h}，$$
其中，$h$为网格尺寸或步长。
通过泰勒展开（Taylor expansion）可知，前向差分的离散化误差为$\mathcal{O}(h)$。

类似的，对于中央差分（central difference）和五点差分（five-points difference）
$$f'(x)\approx \frac{f(x+h)-f(x-h)}{2h}，$$
$$f'(x)\approx \frac{-f(x+2h)+8f(x+h)-8f(x-h)-f(x)}{12h}，$$
tization error）

我们提到的另外一类误差就是离散化误差。从名字可知，这类误差是由数值方法中的各种离散li zi z化所引入的。
看下面的这个


```python
h = 0.1
# The exact solution
N = 400
l = 0
u = 2
x = np.linspace(l, u, N)
f_exa = np.exp(x)

# check if h is too large or too small
if h > 1 or h < 1e-5:
    h = 0.5

# compute the numerical derivatives
xh = np.linspace(l, u, int(abs(u-l)/h))
fprimF = ForwardDiff(np.exp, xh, h)
```


```python
import matplotlib.pyplot as plt
# Plot
fig, ax = plt.subplots(1)
ax.plot(x, f_exa, color='blue')
ax.plot(xh, fprimF, 'ro', clip_on=False)
ax.set_xlim([0,2])
ax.set_ylim([1,max(fprimF)])
ax.set_xlabel(r'$x$')
ax.set_ylabel('Derivatives')
ax.set_title('Discretization Errors')
ax.legend(['Exact Derivatives','Calculated Derivatives'])

# if saveFigure:
#     filename = 'DiscretizationError_h' + str(h) + '.pdf'
#     fig.savefig(filename, format='pdf', dpi=1000, bbox_inches='tight')

```




    <matplotlib.legend.Legend at 0x1231fee90>




![svg](SCB1_errors_files/SCB1_errors_13_1.svg)


### 小结

舍入误差


```python

```


```python

```


```python
def ForwardDiff(fx, x, h=0.001):
    """
    ForwardDiff(@fx, x, h);
    Use forward difference to approximatee the derivative of function fx
    in points x, and with step length h
    The function fx must be defined as a function handle with input
    parameter x and the derivative as output parameter
    
    Parameters
    ----------
    fx : function
        A function defined as fx(x)
    x : float, list, numpy.ndarray
        The point(s) of function fx to compute the derivatives
    h : float, optional
        The step size
    
    Returns
    -------
    float, list, numpy.ndarray: The numerical derivatives of fx at x with 
        the same size as x and the type if from fx()
    """
    return (fx(x+h) - fx(x))/h
```


```python

```


```python

```


```python

```


```python

```


```python

def testErrA(n = 10):
    A = np.random.rand(n,n)
    Icomp = np.matmul(np.linalg.inv(A),A)
    Iexact = np.eye(n)
    absrelerror(Iexact, Icomp)
```

完整代码见github: [measureErrors.py](https://github.com/enigne/ScientificComputingBridging/blob/master/Lab/L2/measureErrors.py)


```python

```
