# Attention Mechanisms in Computer Vision:   A Survey

###### Meng-Hao Guo, Tian-Xing Xu, Jiang-Jiang Liu, Zheng-Ning Liu, Peng-T ao Jiang, T ai-Jiang Mu, Song-Hai Zhang, Ralph R. Martin, Ming-Ming Cheng, Senior Member, IEEE, Shi-Min Hu, Senior Member, IEEE,

<br>

### 一、摘要：

#### 1.1 启示：

人类可以自然有效地在复杂场景中找到突出区域。

#### 1.2 注意力机制：

这种关注机制可以被视为基于输入图像的**特征的动态权重调整过程**。

#### 1.3 主要内容：

全面回顾了计算机视觉中的各种注意机制，并根据方法对它们进行了分类，如**通道注意**、**空间注意**、**时间注意**和**分支注意**

<br>

### 二、介绍：

#### 2.1 注意力机制：

将注意力**转移到**图像中**最重要的区域**并**忽略不相关部分**的方法称为注意力机制。在视觉系统中，注意力机制可以被视为**动态选择过程**，其通过根据输入的**重要性自适应加权特征**来实现。

#### 2.2 注意力机制取得的进展：

注意力机制已经**在许多视觉任务中提供了好处**，例如图像分类、对象检测、语义分割、人脸识别、人物重新识别、动作识别、少数显示学习、医学图像处理、图像生成、姿势估计、超分辨率、3D视觉和多模态任务。

#### **2.3 **对深度学习时代计算机视觉中基于注意力的模型的历史的简要总结。 

![image-20221120104902550](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120104902550.png)

Brief summary of **key developments** in **attention in computer vision**, which have loosely occurred in four phases. 

- Phase 1 **adopted RNNs to construct attention**, a representative method being **RAM** 
- Phase 2 **explicitly predicted important regions**, a representative method being **STN** 
- Phase 3 **implicitly completed the attention process**, a representative method being **SENet** . 
- Phase 4 used **self-attention methods** 

##### 2.3.1 RAM：

将**深度神经网络与注意力机制**相结合的开创性工作。它**反复预测重要区域**，并通过策略梯度以**端到端**的方式更新整个网络。这个阶段递归神经网络（RNN）是注意力机制的必要工具

##### 2.3.2 STN：

引入了一**个子网络**来**预测**用于选择输入中重要区域的**仿射变换**。**明确预测歧视性输入**特征是第二阶段的主要特征

##### 2.3.3 SENeT：

提出新的信道注意网络，能隐式地和自适应地预测潜在的关键特征。

##### 2.3.4 self-attention ：

先在NLP方向取得巨大进展，后引入CV。提出了一种新颖的非本地网络，在视频理解和对象检测方面取得了巨大成功，它们提高了速度、结果质量和泛化能力。基于注意力的模型**有可能取代卷积神经网络**，成为**计算机视觉中**更强大和通用的架构。

#### 2.4.小结：

##### 2.4.1 本文的目的是总结和分类当前计算机视觉中的注意力方法。

##### 2.4.2 关键、次要符号解释

![image-20221120143442280](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120143442280.png)

##### 2.4.3 主要注意力方法：

![image-20221120143757657](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120143757657.png)

Attention **mechanisms** can be categorised **according to data domain**. These include four fundamental categories of 

- **channel attention,**

- **spatial attention,** 

- ##### temporal attention and branch attention, 

- two **hybrid** categories, combining **channel & spatial attention** and **spatial & temporal attention**.

-  ∅ means such combinations do not (yet) exist.

##### 2.4.4 对各类方法的进一步解释：

![image-20221120144317042](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120144317042.png)

- **Channel**, **spatial** and **temporal** attention can be regarded as **operating on different domains.** 

- **C** represents the **channel domain,** **H** and **W** represent **spatial domains**, and **T** means the **temporal domain**.
- **Branch attention** is **complementary** to these.

##### 2.4.5 对注意力方法进行分类（六类）：

- 四个基本类别：通道注意力（注意什么）、空间注意力（注意哪里）、时间注意力（何时注意）和分支通道（注意什么），
- 两个混合组合类别：通道和空间注意力和空间和时间注意力。

##### 2.4.6 对各类别注意力方法和相关工作的总结：

![image-20221120150021125](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120150021125.png)

##### 2.4.7 本文的主要贡献：

- 对视觉注意力方法进行了**系统综述**，涵盖了注意力机制的统一描述、视觉注意力机制的发展以及当前的研究，
- 根据注意力方法的数据域对其进行了**分类**，使我们能够独立于其特定应用将视觉注意力方法联系起来
- 对未来视觉注意力**研究的建议**。

##### 2.4.8 本文其他主要结构

第2节考虑了相关调查，第3节是我们调查的主体。第4节给出了未来研究的建议，最后，第5节给出了结论。 

### 三、OTHER SURVEYS

#### 3.1将本文和其他调查进行比较

##### 3.1.1 调查侧重点

这些调查回顾了注意方法和视觉变换器。Chaudhari等人[140]对深度神经网络中的注意力模型进行了一项调查，重点研究了它们在**自然语言处理中的应用**，而我们的工作**侧重于计算机视觉**。

##### 3.1.2调查范围

- 三项更具体的调查[141]、[142]、[143]总结了**视觉变压器的发展**，而我们的论文**更全面地回顾了视觉中的注意机制**，而不仅仅是自我注意机制。

- Wang等人[144]对计算机视觉中的**注意力模型**进行了调查，但它**只考虑了基于RNN的注意力模型**，这只是我们调查的一部分。

##### 3.1.3本文调查的创新点：

与之前的调查不同，我们**提供了一种分类**，根据数据域而非应用领域对各种注意力方法进行分类。这样做可以让我们集中精力 在方法本身，而不是将它们作为其他任务的补充。

### 四、计算机视觉中的注意力方法

#### 4.1总述：

- 总结了基于人类视觉系统识别过程的**注意力机制的一般形式**

- 回顾了**各种类型的注意力模型**
- 更深入地介绍了**该类别的注意力策略**，考虑到其在**动机**、**形式**和**功能**方面的发展。 

#### 4.2注意力机制的一般形式

##### 4.2.1将注意力机制转化为公式：

**一般形式**：g（x）可以表示产生与关注辨别区域的过程相对应的注意。f（g（x），x）表示基于**与处理关键区域**和**获取信息一致的关注度g（x）**来处理输入x。

![image-20221120160610252](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120160610252.png)

##### 4.2.2对于模型：self-attention、squeeze-and-excitation(SE)

- self-attention

![image-20221120161219693](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120161219693.png)

g（x）：全连接层后接softmax变换

- squeeze-and-excitation(SE)

![image-20221120161236246](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120161236246.png)

g（x）：全局平均池化->多层感知机->sigmoid

#### 4.3对于各种注意力机制的介绍

##### 4.3.1Channel Attention

##### （1）⭐总述：

在深度神经网络中，**不同特征图**中的**不同通道**通常代表**不同的对象**[50]。通道注意力**自适应**地**重新校准每个通道的权重**，并且可以被视为**一个对象选择过程，从而确定要关注什么（需要注意的通道权重更高）**。

![image-20221120213322806](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120213322806.png)

##### （2）SENet：

- SENet的**核心**是一个**挤压和激励（SE）块**，用于**收集全局信息、捕获信道关系并提高表示能力**。
- **SE块分为两个部分**，**挤压模块**和**激励模块**。通过**全局平均池（即一阶统计）在挤压模块中收集全局空间信息**。**激励模块通过使用完全连接的层和非线性层（ReLU和sigmoid）来捕获通道关系并输出注意力向量。**然后，⭐**通过乘以关注向量中的对应元素来缩放输入特征的每个通道。**
- 公式化：

![image-20221120163652644](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120163652644.png)

- 优点：SE块在抑制噪声的同时起到强调重要信道的作用，低计算资源要求（可在每个剩余单元后面添加）。
- 缺点：**挤压模块**中，**全局平均池太简单**，**无法捕获复杂的全局信息**。在**激励模块**中，**完全连接的层增加了模型的复杂性**。

##### （3）GSoP-Net

- 使用**全局二阶池（GSoP）**块来建模高阶统计，同时收集全局信息，从而改进挤压模块。与SE模块一样，GSoP模块也具有挤压模块和激励模块。在**挤压模块**中，GSoP块首先使用**1x1卷积将信道数量从c减少到c0**（c0＜c），然后**计算不同信道的c0×c0协方差矩阵以获得它们的相关性。**接下来，**对协方差矩阵执行逐行归一化。**归一化协方差矩阵中的**每个（i，j）显式地将信道i与信道j相关联。**在**激励模块**中，GSoP块**执行逐行卷积以保持结构信息并输出向量。**然后应用一个**全连通层和一个S形函数**得到一个**c维注意力向量**。最后将**输入特征与关注向量相乘**

![image-20221120185030795](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120185030795.png)

Conv（·）减少了信道数量，Cov（·）计算协方差矩阵，RC（·）表示逐行卷积。

- 优点：提高了收集全局信息的能力
- 增大了计算量

##### （4）SRM

- 它的主要贡献是**样式池**，它利用输入特征的**均值和标准差**来**提高其捕获全局信息的能力**。它还采用了一个轻量级的全连接（CFC）层来代替原来的全连接层，以**减少计算需求**（也可在每个剩余单元后面添加）。
- 用**结合了全局平均池和全局标准差池的样式池**（SP（·））来**收集全局信息**。然后，使用**信道方向全连接**（CFC（·））层（即**每个信道全连接**）、**批量归一化BN**和**S形函数σ**来提供关注向量。最后，如在SE块中，输入特征乘以关注向量。

![image-20221120190914671](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120190914671.png)

##### （5）GCT

- 由于激励模块中全连接层的计算需求和参数数量，在每个卷积层之后使用SE块是不切实际的。此外，使用**完全连接的层**来建模信道关系是一个**隐式**过程。
- 门控信道变换（GCT），可以有效地收集信息，同时**显式地建模信道关系**。与以前的方法不同，GCT首先通过**计算每个信道的l2范数**来**收集全局信息**。接下来，应用可**学习向量α来缩放特征**。然后通过**渠道规范化**采用**竞争机制**来**实现渠道之间的互动**。与其他常见的归一化方法一样，应用**可学习的尺度参数γ和偏差β来重新缩放归一化**。
- 采用**tanh**激活来**控制注意力向量**
- 将输入和注意力向量相乘，同时**添加标识连接** 

![image-20221120191710307](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120191710307.png)

其中α、β和γ是可训练参数。范数（·）表示每个信道的L2范数。CN是信道归一化。GCT块比SE块具有更少的参数，并且由于它是

- 优点：轻量级，可以在**CNN的每个卷积层之后**添加。

##### （6）ECANet

- 为了**避免高模型复杂性**，SENet**减少了通道的数量**。然而，该策略**无法直接建模权重向量和输入之间的对应关系**，从而降低了结果的质量。
- 信道关注（ECA）块，该块**使用1D卷积**来**确定信道之间的交互**，**而不是降维**。ECA块使用**聚集全局空间信息**的挤压模块和用于**建模跨信道交互**的有效激励模块。与间接对应不同，ECA块**只考虑每个信道与其k近邻之间的直接交互**，以**控制模型复杂性**。

![image-20221120193121134](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120193121134.png)

Conv1D（·）表示在信道域上具有形状为k的核的1D卷积，以模拟局部跨信道交互。参数k决定交互的覆盖范围，在ECA中，内核大小k由信道维度C自适应地确定

![image-20221120193307158](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120193307158.png)

γ和b是超参数|x|odd表示x的最接近的奇函数

- 优点：比SENet更加高效，可以容易地合并到各种CNN中。

##### （7） FcaNet

- 仅在挤压模块中使用全局平均池限制了表示能力
- 证明了全局平均池是**离散余弦变换**（DCT）的一种特殊情况，并利用这一观察结果提出了一种**新的多谱信道关注**。给定输入特征图X∈  RC×H×W，多光谱通道注意力首先**将X分成多个部分xi**∈ RC0×H×W。然后**将2D DCT应用于每个零件xi**。注意，2D  DCT可以使用**预处理结果来减少计算**。在处理每个部分之后，所有结果都被连接到一个向量中。最后全连接层，使用Relu激活和sigmoid来获取注意力向量（像SE块中那样）。

![image-20221120201141284](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120201141284.png)

Group（·）表示将输入分成多个组，DCT（·）是2D离散余弦变换。

- 该模型在分类任务中取得了优异的性能。 

##### （8）EncNet

- 提出了**包含语义编码丢失**（SE丢失）的**上下文编码模块（CEM）**，以建模**场景上下文**和**对象类别概率之间的关系**，从而**利用全局场景上下文信息进行语义分割。**
- 给定输入特征图X∈  RC×H×W，CEM首先在训练阶段**学习K个聚类中心**D={d1，…，dK}和一组**平滑因子**S={s1，…，sK}。接下来，它使用**软分配权重**对输入中的**局部描述符**和**对应的聚类中心**之间的差异求和，**以获得置换不变描述符**。然后，为了计算效率，它将聚合应用于K个簇中心的描述符，而不是级联。

![image-20221120203048204](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120203048204.png)

dk∈ RC和sk∈  R是可学习的参数。φ表示ReLU激活的批量归一化。除了按通道缩放向量外，还将紧凑的上下文描述符e应用于计算SE损失以正则化训练，这改进了小对象的分割。

- 优点：CEM不仅**增强了与类别相关的特征图**，还通过**合并SE损失**，**迫使网络同等地考虑大小对象**。由于其轻量级架构，CEM可以**应用于各种主干网，计算开销很低**。

##### （9）Bilinear Attention

- 提出了一种新的**双线性关注块（双关注）**，以捕获**每个通道内的局部成对特征交互**，同时**保留空间信息**。双注意使用注意中的注意（AiA）机制来**捕获二阶统计信息**（以获取高阶信息）：外部点方向通道注意向量是根据内部通道注意的输出计算的。对于给定输入特征图X，双注意首先使用**双线性池来捕获二阶信息** 。

- 双注意块使用双线性池来建模**沿着每个通道的局部成对特征交互**，同时**保留空间信息**。使用所提出的AiA，与其他基于注意力的模型相比，该模型**更关注高阶统计信息**。双注意可以被结合到任何CNN主干中，以提高其代表能力，同时抑制噪声。 

##### 4.3.2 Spatial Attention

（1）⭐总述：

空间注意力可以被视为一种**自适应的空间区域选择机制**：关注哪里。如图4所示，RAM[31]、STN[32]、GENet[61]和Non-Local[15]是**不同类型空间注意力方法的代表**。RAM表示**基于RNN**的方法。STN表示那些**使用子网络来明确预测相关区域**的人。GENet表示那些**隐式使用子网络来预测软掩模以选择重要区域**的网络。**Non-Local**表示与self-attention相关的方法。我们首先总结了具有代表性的空间注意机制，并将过程g（x）和f（g（x，x）描述为表4中的等式1，然后根据图4讨论它们。

##### （2）RAM

- 采用**RNN**[147]和**强化学习**（RL）[148]，以**使网络了解关注的位置**。RAM开创了使用RNN进行视觉注意的先河，RAM有三个关键元素：（A）**一个窥视传感器**，（B）**一个窥探网络**和（C）**一个RNN模型**。
- 窥视传感器获取**坐标**lt−1和**图像**Xt。它输出多个分辨率补丁ρ（Xt，lt−1）  以lt为中心−1.扫视网络fg（θ（g））包括扫视传感器，并输出输入坐标lt的特征表示gt−1和图像Xt。RNN模型考虑gt和内部状态ht−并输出下一个中心坐标lt和例如softmax处的动作，从而产生图像分类任务。由于整个过程是不可区分的，它**在更新过程中应用强化学习策略**。这提供了一种简单但有效的方法来将网络聚焦于关键区域，从而**减少了网络执行的计算次数**，特别是对于大输入，同时改进了图像分类结果。 

![image-20221120214518065](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120214518065.png)

RAM中的注意过程[31]。（A） ：一瞥传感器将图像和中心坐标作为输入，并输出多个分辨率补丁。（B）  ：一瞥网络包括一瞥传感器，以图像和中心坐标作为输入并输出特征向量。（C） 整个网络周期性地使用一瞥网络，输出预测结果以及下一个中心坐标。图取自[31]。 

























（3）第一阶段RNN、第二阶段STN、第三阶段SeNet、第四阶段self-attention 

（4）将现有的注意力方法分为六类，包括四个**基本类别**：**channels**注意力（关注什么[50]）、**spatial**注意力（关注哪里）、**temporal**注意力（何时关注）和**branch** **channels**（关注哪些），以及两个混合的组合类别：**channels**和**spatial**注意力以及**spatial**和**temporal**注意力。

<br>

##### 计算机视觉中的注意力机制：

##### 1.总述：

将“专注于辨别区域，并快速处理这些区域”这一过程抽象为：

![image-20221115111808471](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221115111808471.png)

- g（x）表示**产生**与关注辨别区域的过程相对应的**注意力。**

- f（g（x），x）表示基于与处理关键区域和获取信息一致的关注度g（x）来**处理输入**x

<br>

##### （1）对于self-attention:

g(x) and f(g(x), x) can be written as
Q, K, V = Linear(x) 
g(x) = Softmax(QK) 
f(g(x), x) = g(x)V

##### （2）For SE:

g(x) and f(g(x), x) can be written as
g(x) = Sigmoid(MLP(GAP(x))) 
f(g(x), x) = g(x)x

<br>

##### 2.Channel Attention

（1）在**深度神经网络**中，不同特征图中的不同通道通常表示不同的对象。**想到卷积神经网络的多通道输出，每个输出通道都有一个卷积核，每个通道提取一类图像的特征，最后将所有提取到的特征加权组合到一起。**

（2）通道注意力**自适应地重新校准每个通道的权重**，并可以被视为一个对象选择过程，**从而确定要注意什么**（需要注意的权重更高？）

<br>

##### 代表模型：

##### SeNet:

- Se块：用于收集全局信息、捕获通道关系和提高表示能力
- squeeze module

global average pooling:收集全局信息

- excitation module

fully-connected layers and non-linear layers (ReLU and sigmoid).：捕获通道关系，输出注意力向量

缺点：

squeeze module中： **global average pooling is too simple to capture complex global information.** 

excitation module中：fully-connected layers increase the **complexity** of the
model.

<br>

##### GSoP-Net

相比SeNet的优点：using a **global second-order pooling** (GSoP) block to model high-order statistics while gathering global information.

<br>

- Representative channel attention **mechanisms** ordered by **category** and **publication date**. Their **key aims are to emphasize important channels and**
  **capture global information**. 

- **Application** areas include: Cls = classification, Det = detection, SSeg = semantic segmentation, ISeg = instance
  segmentation, ST = style transfer, Action = action recognition.

- g(x) and f(g(x), x) are the **attention process described by Eq**. 1（公式1）. Ranges means the
  **ranges of attention map**. S or H means **soft or hard attention.**
- (A) channel-wise product. (I) emphasize important channels, (II) capture global
  information.

![image-20221120103004395](C:\Users\china\AppData\Roaming\Typora\typora-user-images\image-20221120103004395.png)

