TODOS:

- [] Change the matrix representation to `type Matrix[M, N: static int] = array[0..N, array[0..M, int]]]`
- [] Alter `categoricalCrossEntropyLoss`, `softmax`, `ReLU`, `dReLU`, `matmul`, `*`, `+`, `-`, and `transpose` to become macros that unroll the loops
- [] Compare performance to existing operation
