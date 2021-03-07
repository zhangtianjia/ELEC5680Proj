# CVPR20201 Oral

[1] UP-DETR: Unsupervised Pre-training for Object Detection with Transformers<br>
[paper](https://arxiv.org/pdf/2011.09094.pdf)|[code](https://github.com/dddzg/up-detr)<br>
解读：[无监督预训练检测器](https://www.zhihu.com/question/432321109/answer/1606004872)<br><br>
Object detection with transformers (DETR) reaches competitive performance with Faster R-CNN via a transformer encoder-decoder architecture. Inspired by the great success of pre-training transformers in natural language processing, we propose a pretext task named random query patch detection to unsupervisedly pre-train DETR (UP-DETR) for object detection. Specifically, we randomly crop patches from the given image and then feed them as queries to the decoder. The model is pre-trained to detect these query patches from the original image. During the pre-training, we address two critical issues: multi-task learning and multi-query localization. (1) To trade-off multi-task learning of classification and localization in the pretext task, we freeze the CNN backbone and propose a patch feature reconstruction branch which is jointly optimized with patch detection. (2) To perform multi-query localization, we introduce UP-DETR from single-query patch and extend it to multi-query patches with object query shuffle and attention mask. In our experiments, UP-DETR significantly boosts the performance of DETR with faster convergence and higher precision on PASCAL VOC and COCO datasets. The code will be available soon.
[2] Image-to-image Translation via Hierarchical Style Disentanglement(通过分层样式分解实现图像到图像的翻译)<br>
[paper](https://arxiv.org/abs/2103.01456)|[code](https://github.com/imlixinyang/HiSD)<br><br>
Recently, image-to-image translation has made significant progress in achieving both multi-label (\ie, translation conditioned on different labels) and multi-style (\ie, generation with diverse styles) tasks. However, due to the unexplored independence and exclusiveness in the labels, existing endeavors are defeated by involving uncontrolled manipulations to the translation results. In this paper, we propose Hierarchical Style Disentanglement (HiSD) to address this issue. Specifically, we organize the labels into a hierarchical tree structure, in which independent tags, exclusive attributes, and disentangled styles are allocated from top to bottom. Correspondingly, a new translation process is designed to adapt the above structure, in which the styles are identified for controllable translations. Both qualitative and quantitative results on the CelebA-HQ dataset verify the ability of the proposed HiSD. We hope our method will serve as a solid baseline and provide fresh insights with the hierarchically organized annotations for future research in image-to-image translation. The code has been released at this https URL.
[3] Less is More: CLIPBERT for Video-and-Language Learning via Sparse Sampling(通过稀疏采样进行视频和语言学习)<br>
[paper](https://arxiv.org/pdf/2102.06183.pdf)|[code](https://github.com/jayleicn/ClipBERT)<br><br>
The canonical approach to video-and-language learning (e.g., video question answering) dictates a neural model to learn from offline-extracted dense video features from vision models and text features from language models. These feature extractors are trained independently and usually on tasks different from the target domains, rendering these fixed features sub-optimal for downstream tasks. Moreover, due to the high computational overload of dense video features, it is often difficult (or infeasible) to plug feature extractors directly into existing approaches for easy finetuning. To provide a remedy to this dilemma, we propose a generic framework CLIPBERT that enables affordable endto-end learning for video-and-language tasks, by employing sparse sampling, where only a single or a few sparsely sampled short clips from a video are used at each training step. Experiments on text-to-video retrieval and video question answering on six datasets demonstrate that CLIPBERT outperforms (or is on par with) existing methods that exploit full-length videos, suggesting that end-to-end learning with just a few sparsely sampled clips is often more
accurate than using densely extracted offline features from full-length videos, proving the proverbial less-is-more principle. Videos in the datasets are from considerably different domains and lengths, ranging from 3-second genericdomain GIF videos to 180-second YouTube human activity videos, showing the generalization ability of our approach. Comprehensive ablation studies and thorough analyses are provided to dissect what factors lead to this success.

[4]PREDATOR: Registration of 3D Point Clouds with Low Overlap(预测器：低重叠的3D点云的注册)<br>
[paper](https://arxiv.org/pdf/2011.13005.pdf)|[code](https://github.com/ShengyuH/OverlapPredator)|[project](https://overlappredator.github.io/)<br><br>
We introduce PREDATOR, a model for pairwise pointcloud registration with deep attention to the overlap region. Different from previous work, our model is specifically designed to handle (also) point-cloud pairs with low overlap. Its key novelty is an overlap-attention block for early information exchange between the latent encodings of the two point clouds. In this way the subsequent decoding of the latent representations into per-point features is conditioned on the respective other point cloud, and thus can predict which points are not only salient, but also lie in the overlap region between the two point clouds. The ability to focus
on points that are relevant for matching greatly improves performance: PREDATOR raises the rate of successful registrations by more than 20% in the low-overlap scenario,
and also sets a new state of the art for the 3DMatch benchmark with 89% registration recall. 

[5]PatchmatchNet: Learned Multi-View Patchmatch Stereo(学习多视图立体声)<br>
[paper](https://arxiv.org/abs/2012.01411)|[code](https://github.com/FangjinhuaWang/PatchmatchNet)
We present PatchmatchNet, a novel and learnable cascade formulation of Patchmatch for high-resolution multi-view stereo. With high computation speed and low memory requirement, PatchmatchNet can process higher resolution imagery and is more suited to run on resource limited devices than competitors that employ 3D cost volume regularization. For the first time we introduce an iterative multi-scale Patchmatch in an end-to-end trainable architecture and improve the Patchmatch core algorithm with a novel and learned adaptive propagation and evaluation scheme for each iteration. Extensive experiments show a very competitive performance and generalization for our method on DTU, Tanks & Temples and ETH3D, but at a significantly higher efficiency than all existing top-performing models: at least two and a half times faster than state-of-the-art methods with twice less memory usage.

[6] Categorical Depth Distribution Network for Monocular 3D Object Detection(用于单目三维目标检测的分类深度分布网络)<br>
[paper](https://arxiv.org/abs/2103.01100)<br><br>

[7] MultiBodySync: Multi-Body Segmentation and Motion Estimation via 3D Scan Synchronization(通过3D扫描同步进行多主体分割和运动估计)<br>
[paper](https://arxiv.org/pdf/2101.06605.pdf)|[code](https://github.com/huangjh-pub/multibody-sync)<br><br>
We present MultiBodySync, a novel, end-to-end trainable multi-body motion segmentation and rigid registration framework for multiple input 3D point clouds. The
two non-trivial challenges posed by this multi-scan multibody setting that we investigate are: (i) guaranteeing correspondence and segmentation consistency across multiple input point clouds capturing different spatial arrangements of bodies or body parts; and (ii) obtaining robust motionbased rigid body segmentation applicable to novel object categories.We propose an approach to address these issues that incorporates spectral synchronization into an iterative deep declarative network, so as to simultaneously recover consistent correspondences as well as motion segmentation. At the same time, by explicitly disentangling the correspondence and motion segmentation estimation modules, we achieve strong generalizability across different object categories. Our extensive evaluations demonstrate that our method is effective on various datasets ranging from rigid parts in articulated objects to individually moving objects in a 3D scene, be it single-view or full point clouds

[8] Exploring Data-Efficient 3D Scene Understanding with Contrastive Scene Contexts(探索具有对比场景上下文的数据高效3D场景理解)<br>
[paper](http://arxiv.org/abs/2012.09165)|[project](http://sekunde.github.io/project_efficient)|[video](http://youtu.be/E70xToZLgs4)<br><br>
The rapid progress in 3D scene understanding has come with growing demand for data; however, collecting and annotating 3D scenes (e.g. point clouds) are notoriously hard. For example, the number of scenes (e.g. indoor rooms) that can be accessed and scanned might be limited; even given sufficient data, acquiring 3D labels (e.g. instance masks) requires intensive human labor. In this paper, we explore data-efficient learning for 3D point cloud. As a first step towards this direction, we propose Contrastive Scene Contexts, a 3D pre-training method that makes use of both point-level correspondences and spatial contexts in a scene. Our method achieves state-of-the-art results on a suite of benchmarks where training data or labels are scarce. Our study reveals that exhaustive labelling of 3D point clouds might be unnecessary; and remarkably, on ScanNet, even using 0.1% of point labels, we still achieve 89% (instance segmentation) and 96% (semantic segmentation) of the baseline performance that uses full annotations.

[9] Real-Time High Resolution Background Matting(实时高分辨率背景抠像)<br>
[paper](https://arxiv.org/abs/2012.07810)|[code](https://github.com/PeterL1n/BackgroundMattingV2)|[project](https://grail.cs.washington.edu/projects/background-matting-v2/)|[video](https://youtu.be/oMfPTeYDF9g)<br><br>
We introduce a real-time, high-resolution background replacement technique which operates at 30fps in 4K resolution, and 60fps for HD on a modern GPU. Our technique is based on background matting, where an additional frame of the background is captured and used in recovering the alpha matte and the foreground layer. The main challenge is to compute a high-quality alpha matte, preserving strand-level hair details, while processing high-resolution images in real-time. To achieve this goal, we employ two neural networks; a base network computes a low-resolution result which is refined by a second network operating at high-resolution on selective patches. We introduce two largescale video and image matting datasets: VideoMatte240K and PhotoMatte13K/85. Our approach yields higher quality results compared to the previous state-of-the-art in background matting, while simultaneously yielding a dramatic boost in both speed and resolution.
















