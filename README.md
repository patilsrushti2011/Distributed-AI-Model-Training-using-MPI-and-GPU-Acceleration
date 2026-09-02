# Distributed-AI-Model-Training-using-MPI-and-GPU-Acceleration
CIFAR-10 image classification using CPU, GPU and distributed MPI training with Slurm scheduling.
# Distributed AI Model Training using MPI and GPU Acceleration

## Overview

This project demonstrates scalable deep learning training on High Performance Computing (HPC) systems using CPU, GPU, and distributed multi-GPU execution.

The implementation focuses on CIFAR-10 image classification using the MobileNetV2 architecture with input resolution resized to 224×224. The project evaluates three execution modes:

- Serial CPU Training  
- Single GPU Training  
- Distributed Multi-GPU Training using MPI and PyTorch Distributed Data Parallel (DDP)

The objective is to compare performance, scalability, and accuracy across these modes while maintaining consistent model quality.

---

## Objectives

- Build an efficient CIFAR-10 classifier using MobileNetV2  
- Implement serial CPU and single GPU training for baseline performance  
- Develop distributed training using MPI and PyTorch DDP  
- Analyze speedup and scalability on HPC infrastructure  
- Deploy trained model using Gradio for inference  
- Demonstrate best practices in distributed machine learning  

---
## Technologies Used

- Python  
- PyTorch & torchvision  
- MobileNetV2  
- CIFAR-10 Dataset  
- MPI (mpi4py)  
- PyTorch Distributed Data Parallel (DDP)  
- CUDA & cuDNN  
- Slurm Workload Manager  
- Gradio (Inference UI)  

---

## Methodology

- CIFAR-10 images resized from 32×32 to 224×224  
- Strong data augmentation (Random Crop, Flip, Rotation, Color Jitter)  
- Transfer learning using ImageNet pretrained MobileNetV2  
- Adam optimizer with learning rate scheduling  
- DistributedSampler for parallel data loading  
- Gradient synchronization using DDP  
- Model checkpointing based on best validation accuracy  

---

## Training Modes

### 1. CPU Serial Training
Used as baseline for performance comparison.

### 2. Single GPU Training
Accelerated training using CUDA-enabled GPU.

### 3. Distributed GPU Training (MPI + DDP)
Multi-process training across multiple GPUs with synchronized gradients.

---

## Performance Results (CIFAR-10)

| Execution Mode | Hardware | Total Time (Minutes) | Best Accuracy |
|---------------|----------|----------------------|---------------|
| CPU Serial | Single CPU | 515.92 | 96.17% |
| GPU | Single GPU | 178.30 | 96.03% |
| Distributed GPU | 2 GPUs (MPI) | 87.01 | 95.58% |

Distributed training achieved nearly **6× speedup** over CPU while maintaining comparable accuracy.

---

## Inference Deployment

The trained model is deployed using Gradio, allowing users to upload images and receive top-3 predictions interactively.

---

## Key Contributions

- CIFAR-10 classification using MobileNetV2 with 224×224 resolution  
- CPU, GPU, and distributed MPI implementations  
- HPC-based performance benchmarking  
- Web-based inference interface  
- Comparative scalability analysis  

---

## Conclusion

This project demonstrates how HPC systems significantly accelerate deep learning workloads. GPU acceleration reduced training time by 2.9×, while distributed MPI-based training achieved nearly 6× speedup compared to serial CPU execution, proving the effectiveness of distributed AI frameworks for scalable model training.

---

## Future Enhancements

- Species-level classification  
- Larger biodiversity datasets  
- Advanced architectures (ResNet, EfficientNet, Vision Transformers)  
