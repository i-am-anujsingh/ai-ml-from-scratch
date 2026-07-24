import numpy as np
np.random.seed(0)

class MaxPooling_Layer:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, h, w, nof = inputs.shape
        p = self.pool_size
        out_h = h // p
        out_w = w // p

        self.output = np.zeros((batch_size, out_h, out_w, nof), dtype=np.float32)

        for i in range(out_h):
            for j in range(out_w):
                patch = inputs[
                    :,
                    i*p:(i+1)*p,
                    j*p:(j+1)*p,
                    :
                ]
                self.output[:,i, j,:] = np.max(patch, axis=(1, 2))
        return self.output
    
    def backward(self, dvalues):
        p = self.pool_size
        self.dinputs = np.zeros_like(self.inputs)

        s, out_h, out_w, channels = dvalues.shape

        for i in range(out_h):
            for j in range(out_w):
                patch = self.inputs[
                    :,
                    i*p:(i+1)*p,
                    j*p:(j+1)*p,
                    :
                ]

                for n in range(s):
                    for c in range(channels):
                        max_pos = np.unravel_index(
                            np.argmax(patch[n,:,:,c]),
                            (p, p)
                        )

                        self.dinputs[
                            n,
                            i*p + max_pos[0],
                            j*p + max_pos[1],
                            c
                        ] = dvalues[n, i, j,c]
        
        return self.dinputs
