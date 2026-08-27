import torch
import torch.nn.functional as F
import numpy as np

class ImageAnalyzer:
    """Extract image characteristics to detect transformations"""

    def __init__(self):
        pass

    def extract_features(self, images):
        """Extract 5D feature vector describing image characteristics

        Args:
            images: Tensor of shape (B, 3, H, W) in range [-1, 1] (normalized)

        Returns:
            features: Tensor of shape (B, 5) with values in [0, 1]
                [compression_score, blur_score, noise_score, color_variance, resolution_score]
        """
        B = images.shape[0]
        features = []

        for i in range(B):
            img = images[i]

            # 1. Compression artifacts score (using high-frequency noise)
            compression_score = self._estimate_compression(img)

            # 2. Blur estimation (Laplacian variance)
            blur_score = self._estimate_blur(img)

            # 3. Noise level (std deviation in smooth regions)
            noise_score = self._estimate_noise(img)

            # 4. Color distribution stats (saturation variance)
            color_variance = self._estimate_color_jitter(img)

            # 5. Resolution consistency (high-frequency content)
            resolution_score = self._estimate_resolution(img)

            features.append([compression_score, blur_score, noise_score,
                           color_variance, resolution_score])

        return torch.tensor(features, dtype=torch.float32, device=images.device)

    def _estimate_compression(self, img):
        """Estimate JPEG compression level (0=clean, 1=heavy compression)"""
        # Use high-frequency noise as proxy
        # Apply Laplacian filter and measure response
        laplacian = torch.tensor([[[0, 1, 0], [1, -4, 1], [0, 1, 0]]], dtype=torch.float32).to(img.device)
        laplacian = laplacian.repeat(3, 1, 1, 1)

        filtered = F.conv2d(img.unsqueeze(0), laplacian, padding=1, groups=3)
        noise_level = filtered.abs().mean().item()

        # Normalize to [0, 1] (higher noise = less compression)
        compression_score = 1.0 - min(noise_level * 10, 1.0)
        return compression_score

    def _estimate_blur(self, img):
        """Estimate blur level (0=sharp, 1=very blurry)"""
        # Laplacian variance method
        laplacian = torch.tensor([[[0, 1, 0], [1, -4, 1], [0, 1, 0]]], dtype=torch.float32).to(img.device)
        laplacian = laplacian.repeat(3, 1, 1, 1)

        filtered = F.conv2d(img.unsqueeze(0), laplacian, padding=1, groups=3)
        variance = filtered.var().item()

        # Normalize: high variance = sharp, low variance = blurry
        blur_score = 1.0 - min(variance * 100, 1.0)
        return blur_score

    def _estimate_noise(self, img):
        """Estimate additive noise level (0=clean, 1=very noisy)"""
        # Use local variance in smooth regions
        # Simple approximation: global std after smoothing
        smoothed = F.avg_pool2d(img.unsqueeze(0), kernel_size=5, stride=1, padding=2)
        noise = (img.unsqueeze(0) - smoothed).abs().mean().item()

        noise_score = min(noise * 20, 1.0)
        return noise_score

    def _estimate_color_jitter(self, img):
        """Estimate color jitter/adjustment (0=normal, 1=heavily adjusted)"""
        # Measure saturation variance
        # Convert to HSV-like representation (simplified)
        r, g, b = img[0], img[1], img[2]
        max_rgb = torch.max(torch.stack([r, g, b]), dim=0)[0]
        min_rgb = torch.min(torch.stack([r, g, b]), dim=0)[0]

        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
        sat_variance = saturation.var().item()

        color_variance = min(sat_variance * 5, 1.0)
        return color_variance

    def _estimate_resolution(self, img):
        """Estimate resolution quality (0=upscaled/low-res, 1=high-res)"""
        # Measure high-frequency content
        # Use Sobel filter
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).to(img.device)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).to(img.device)
        sobel_x = sobel_x.repeat(3, 1, 1, 1)
        sobel_y = sobel_y.repeat(3, 1, 1, 1)

        grad_x = F.conv2d(img.unsqueeze(0), sobel_x, padding=1, groups=3)
        grad_y = F.conv2d(img.unsqueeze(0), sobel_y, padding=1, groups=3)

        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2).mean().item()

        resolution_score = min(gradient_magnitude * 2, 1.0)
        return resolution_score
