# z_1 = ReLU(W_1 * x + b_1)
# z_2 = W_2 * z_1 + b_2
# o = softmax(z_2)
# L(o, y) = categoricalCrossEntropyLoss(o, y)
#
# dL/dz_2 = o - y
#
# dL/dW_2 = dL/dz_2 * dz_2/dW_2 = (o - y) * ReLU(W_1 * x + b_1) -> (1, 10) * (1, 24)^T
# dL/dW_1 = dL/dz_2 * dz_2/dz_1 * dz_1/dW_1 = (o - y) * W_2 * ReLU'(W_1 * x + b_1) * x -> (784, 1) * (1, 10) * (24, 10)^T @ (1, 24)
# dL/db_2 = dL/dz_2 * dz_2/db_2 = (o - y) * 1
# dL/db_1 = dL/dz_2 * dz_2/dz_1 * dz_1/db_1 = (o - y) * W_2 * ReLu'(W_1 * x + b_1) * 1


import std/[strutils, sequtils, math, random, assertions]

proc clamp(x, lo, hi: float64): float64 =
  max(lo, min(x, hi))

proc categoricalCrossEntropyLoss(probs: seq[seq[float64]], targetIndex: uint8): float64 =
  let eps = 1e-15
  let p = clamp(probs[targetIndex][0], eps, 1.0 - eps)
  result = -ln(p)

proc softmax(xs: seq[seq[float64]]): seq[seq[float64]] =
  var m = xs[0][0]
  for x in xs:
    if x[0] > m: m = x[0]

  var s = 0.0
  result = newSeq[seq[float64]](xs.len)

  for i, x in xs:
    let e = exp(x[0] - m)
    result[i] = @[e]
    s += e

  for i in 0 ..< result.len:
    result[i][0] /= s

proc ReLU (xs: seq[seq[float64]]): seq[seq[float64]] = xs.mapIt(it.mapIt(max(0, it)))

proc dReLU (xs: seq[seq[float64]]): seq[seq[float64]] =
  result = newSeqWith(xs.len, newSeq[float64](xs[0].len))

  for i, x in xs:
    for j, a in x:
      if a <= 0:
        result[i][j] = 0
      else:
        result[i][j] = 1

proc matmul (a, b: seq[seq[float64]]): seq[seq[float64]] =
  if len(b) != len(a[0]):
    echo $len(b[0]) & " x " & $len(b)
    echo $len(a[0]) & " x " & $len(a)
    raise newException(ValueError, "Matrix dimensionality is wrong")

  result = newSeqWith(len(a), newSeq[float64](b[0].len))

  for j in 0..<len(a):
    for i in 0..<len(b[0]):
      var sum: float64 = 0.0
      for k in 0..<len(a[0]):
        sum += b[k][i] * a[j][k]
      result[j][i] = sum

proc `*` (a: float64, b: seq[seq[float64]]): seq[seq[float64]] = b.mapIt(it.mapIt(it * a))

proc `*` (a, b: seq[seq[float64]]): seq[seq[float64]] =
  result = newSeqWith(a.len, newSeq[float64](a[0].len))

  for i in 0..<len(a):
    for j in 0..<a[0].len:
      result[i][j] = a[i][j] * b[i][j]

proc `+`(a, b: seq[seq[float64]]): seq[seq[float64]] =
  result = newSeqWith(len(a), newSeq[float64](a[0].len))

  for i in 0..<len(a):
    for j in 0..<len(a[0]):
      result[i][j] = a[i][j] + b[i][j]

proc `-`(a, b: seq[seq[float64]]): seq[seq[float64]] =
  result = newSeqWith(len(a), newSeq[float64](a[0].len))

  for i in 0..<len(a):
    for j in 0..<len(a[0]):
      result[i][j] = a[i][j] - b[i][j]

proc transpose (a: seq[seq[float64]]): seq[seq[float64]] =
  result = @[]

  for i in 0..<len(a[0]):
    var r = newSeq[float64]()
    for j in 0..<len(a):
      r.add(a[j][i])
    result.add(r)

type MNISTData = object
  label: uint8
  image: seq[seq[float64]]

type NN = object
  W1: seq[seq[float64]]
  W2: seq[seq[float64]]
  B1: seq[seq[float64]]
  B2: seq[seq[float64]]

proc forward (nn: NN, x: seq[seq[float64]]): seq[seq[float64]] =
  let z1 = ReLU(matmul(nn.W1, x) + nn.B1)
  let z2 = matmul(nn.W2, z1) + nn.B2
  return softmax(z2)

proc backward (nn: var NN, o: seq[seq[float64]], y: seq[seq[float64]], x: seq[seq[float64]], alpha: float64): void =
  let dW2 = matmul((o - y), ReLU(matmul(nn.W1, x) + nn.B1).transpose())
  let dW1 = matmul(matmul(nn.W2.transpose(), (o - y)) * dReLU(matmul(nn.W1, x) + nn.B1), x.transpose())
  let dB2 = o - y
  let dB1 = matmul(nn.W2.transpose(), (o - y)) * dReLU(matmul(nn.W1, x) + nn.B1)

  nn.W2 = nn.W2 - alpha * dW2
  nn.W1 = nn.W1 - alpha * dW1
  nn.B2 = nn.B2 - alpha * dB2
  nn.B1 = nn.B1 - alpha * dB1

proc init (nn: var NN): void =
  randomize()

  proc initializeRandomWeights (row, column: int): seq[seq[float64]] =
    result = newSeqWith(row, newSeq[float64](column))

    for i in 0..<row:
      for j in 0..<column:
        result[i][j] = rand(2.0) - 1

  nn.W1 = initializeRandomWeights(128, 784)
  nn.W2 = initializeRandomWeights(10, 128)
  nn.B1 = initializeRandomWeights(128, 1)
  nn.B2 = initializeRandomWeights(10, 1)

proc loadMNISTData (path: string): seq[MNISTData] =
  let f = open(path)
  defer: f.close()

  result = newSeq[MNISTData]()

  var data: string
  while f.readLine(data):
    let label: uint8 = uint8(parseInt(data[0..0]))
    let data: seq[seq[float64]] = @[map[string, float64](data[2..len(data)-1].rsplit(","), proc (x: string): float64 = float64(parseInt(x)) / 255)].transpose()

    assert label >= 0 and label <= 9

    result.add(MNISTData(label: label, image: data))

var trainingData = loadMNISTData("mnist/mnist_train.csv")
var testData = loadMNISTData("mnist/mnist_test.csv")

var nn = NN()
nn.init()

for epoch in 1..20:
  trainingData.shuffle()
  var alpha = 0.01'f64
  for data in trainingData:
    let output = nn.forward(data.image)
    var expected: seq[seq[float64]] = repeat(@[0.0'f64], 10)
    expected[data.label][0] = 1.0'f64
    nn.backward(output, expected, data.image, alpha)
    let loss = categoricalCrossEntropyLoss(output, data.label)

    if loss.isNaN():
      echo output
      echo expected
      break
  alpha *= 0.99

var count = 0
for data in testData:
  let output = nn.forward(data.image)
  let result = maxIndex(output.transpose()[0])

  if data.label == uint8(result):
    count += 1

echo $(float64(count) / float64(testData.len) * 100) & "% accuracy"
