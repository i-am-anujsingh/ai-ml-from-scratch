import numpy as np
from numpy.lib.stride_tricks import as_strided

class Convolution_Layer:
    def __init__(self, k, num_of_filters):
        self.num_of_filters = num_of_filters
        self.kernel = (
            np.random.randn(num_of_filters, k, k)
            * np.sqrt(2 / (k * k))
        ).astype(np.float32)
        self.biases = np.zeros(num_of_filters, dtype=np.float32)

    def convolve(self, patch, f):
        return np.sum(patch * self.kernel[f], axis=(1, 2)) + self.biases[f]
    
    def forward(self, batch):
        self.input = batch
        s, h, w = batch.shape
        k = self.kernel.shape[-1]

        self.output = np.zeros((s, (h-k)+1, (w-k)+1, self.num_of_filters), dtype=np.float32)

        for f in range(self.num_of_filters):
            for i in range(h-k+1):
                for j in range(w-k+1):
                    patch = batch[:, i:i+k, j:j+k]
                    self.output[:,i, j,f] = self.convolve(patch, f)
        return self.output
        
    def backward(self, dvalues):
        self.dbias = np.sum(dvalues, axis=(0,1,2))
        self.dkernel = np.zeros_like(self.kernel)
        self.dinputs = np.zeros_like(self.input)
        _, out_h, out_w, _ = dvalues.shape
        k = self.kernel.shape[-1]
        for f in range(self.num_of_filters):
            for i in range(out_h):
                for j in range(out_w):
                    patch = self.input[:,i:i+k, j:j+k]
                    gradient = dvalues[:, i, j,f][:, None, None]
                    self.dkernel[f] += np.sum(patch * gradient, axis=0)
                    self.dinputs[:, i:i+k, j:j+k] += (
                        gradient * self.kernel[f]
                    )
        return self.dinputs

class FastConvolution_Layer:
    def __init__(self, kernel_size, num_of_filters):
        self.num_of_filters = num_of_filters
        self.k = kernel_size
        
        # He initialization
        self.kernel = np.random.randn(num_of_filters, kernel_size, kernel_size) * \
                      np.sqrt(2.0 / (kernel_size * kernel_size))
        self.kernel = self.kernel.astype(np.float32)
        
        self.biases = np.zeros(num_of_filters, dtype=np.float32)
        
    def forward(self, batch):
        """
        batch shape: (batch_size, height, width)
        """
        self.input = batch  # save for backward
        b, h, w = batch.shape
        k = self.k
        
        # Output shape
        out_h = h - k + 1
        out_w = w - k + 1
        
        # === im2col: Extract all patches ===
        # Create view with sliding windows
        shape = (b, out_h, out_w, k, k)
        strides = batch.strides + (batch.strides[1], batch.strides[2])
        
        patches = as_strided(batch, shape=shape, strides=strides)
        # patches shape: (b, out_h, out_w, k, k)
        
        # Reshape for matrix multiplication
        patches_flat = patches.reshape(b * out_h * out_w, k * k)
        kernels_flat = self.kernel.reshape(self.num_of_filters, k * k).T   # (k*k, num_filters)
        
        # Matrix multiply
        output_flat = patches_flat @ kernels_flat + self.biases
        output = output_flat.reshape(b, out_h, out_w, self.num_of_filters)
        
        self.output = output
        return output
    
    def backward(self, dvalues):
        """
        dvalues shape: (batch_size, out_h, out_w, num_filters)
        """
        b, out_h, out_w, _ = dvalues.shape
        k = self.k
        _, h, w = self.input.shape
        
        # 1. Gradient w.r.t biases
        self.dbias = dvalues.sum(axis=(0, 1, 2))
        
        # 2. Gradient w.r.t kernel
        self.dkernel = np.zeros_like(self.kernel)
        
        # Extract patches again
        shape = (b, out_h, out_w, k, k)
        strides = self.input.strides + (self.input.strides[1], self.input.strides[2])
        patches = as_strided(self.input, shape=shape, strides=strides)
        patches_flat = patches.reshape(b * out_h * out_w, k * k)
        
        dvalues_flat = dvalues.reshape(b * out_h * out_w, self.num_of_filters)
        
        # Efficient kernel gradient
        self.dkernel = (patches_flat.T @ dvalues_flat).T.reshape(self.num_of_filters, k, k)
        
        # 3. Gradient w.r.t input (full convolution with rotated kernel)
        self.dinputs = np.zeros_like(self.input)
        
        dvalues_padded = np.pad(dvalues, ((0,0), (k-1, k-1), (k-1, k-1), (0,0)), mode='constant')
        
        # Rotate kernel 180 degrees for full convolution
        kernel_rot = np.flip(self.kernel, axis=(1, 2))
        
        for f in range(self.num_of_filters):
            for i in range(h):
                for j in range(w):
                    patch = dvalues_padded[:, i:i+k, j:j+k, f]
                    self.dinputs[:, i, j] += np.sum(patch * kernel_rot[f], axis=(1, 2))
        
        return self.dinputs