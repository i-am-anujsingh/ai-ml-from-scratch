import numpy as np
np.random.seed(0)

class Convolution_Layer:
    def __init__(self, k):
        self.kernel = np.random.randn(k,k).astype(np.float32)
        self.biases = np.float32(np.random.randn())

    def convolve(self, patch):
        return np.sum(patch * self.kernel) + self.biases
    
    def forward(self, image):
        self.input = image
        h, w = image.shape
        k = self.kernel.shape[0]

        self.output = np.zeros(((h-k)+1, (w-k)+1), dtype=np.float32)

        for i in range(h-k+1):
            for j in range(w-k+1):
                patch = image[i:i+k, j:j+k]
                self.output[i, j] = self.convolve(patch)

        return self.output
        
    def backward(self, dvalues):
        dbias = np.sum(dvalues)
        dkernel = np.zeros_like(self.kernel)
        out_h, out_w = dvalues.shape
        k = self.kernel.shape[0]
        for i in range(out_h):
            for j in range(out_w):
                patch = self.input[i:i+k, j:j+k]
                gradient = dvalues[i, j]
                dkernel += patch * gradient

        self.dbias = dbias
        self.dkernel = dkernel

class MaxPooling_Layer:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, inputs):
        self.inputs = inputs
        h, w = inputs.shape
        p = self.pool_size
        out_h = h // p
        out_w = w // p
        self.output = np.zeros((out_h, out_w), dtype=np.float32)

        for i in range(out_h):
            for j in range(out_w):

                patch = inputs[
                    i*p:(i+1)*p,
                    j*p:(j+1)*p
                ]

                self.output[i, j] = np.max(patch)

        return self.output
    
    def backward(self, dinputs):
        p = self.pool_size
        h,w = self.inputs.shape
        self.dinputs = np.zeros_like(self.inputs)
        out_h, out_w = dinputs.shape
        for i in range(out_h):
            for j in range(out_w):
                patch = self.inputs[
                    i*p:(i+1)*p,
                    j*p:(j+1)*p
                ]
                max_indx = np.unravel_index(np.argmax(patch), patch.shape)

                self.dinputs[
                    i*p + max_indx[0],
                    j*p +  max_indx[1]
                ] = dinputs[i,j]
        return self.dinputs

class Flatten_Layer:
    def forward(self, inputs):
        self.input_shape = inputs.shape
        self.output = inputs.flatten()
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs = dvalues.reshape(self.input_shape)
        return self.dinputs

