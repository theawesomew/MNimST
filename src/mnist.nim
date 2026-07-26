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

proc matmul (a, b: seq[seq[float64]]): seq[seq[float64]] =
  result = @[]

  if len(b) != len(a[0]):
    echo $len(b[0]) & " x " & $len(b)
    echo $len(a[0]) & " x " & $len(a)
    raise newException(ValueError, "Matrix dimensionality is wrong")

  for j in 0..<len(a):
    var r = newSeq[float64]()
    for i in 0..<len(b[0]):
      var sum: float64 = 0.0
      for k in 0..<len(a[0]):
        sum += b[k][i] * a[j][k]
      r.add(sum)
    result.add(r)

proc `*` (a: float64, b: seq[seq[float64]]): seq[seq[float64]] = b.map(proc (x: seq[float64]): seq[float64] = x.map(proc (y: float64): float64 = y * a))

proc `+`(a, b: seq[seq[float64]]): seq[seq[float64]] =
  result = @[]

  for i in 0..<len(a):
    var r = newSeq[float64]()
    for j in 0..<len(a[0]):
      r.add(a[i][j] + b[i][j])
    result.add(r)

proc `-`(a, b: seq[seq[float64]]): seq[seq[float64]] =
  result = @[]

  for i in 0..<len(a):
    var r = newSeq[float64]()
    for j in 0..<len(a[0]):
      r.add(a[i][j] - b[i][j])
    result.add(r)

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
  let z1 = matmul(nn.W1, x) + nn.B1
  let z2 = matmul(nn.W2, z1) + nn.B2
  return softmax(z2)

proc backward (nn: var NN, o: seq[seq[float64]], y: seq[seq[float64]], x: seq[seq[float64]], alpha: float64): void =
  let dW2 = matmul((o - y), (matmul(nn.W1, x) + nn.B1).transpose())
  let dW1 = matmul(matmul(nn.W2.transpose(), (o - y)), x.transpose())
  let dB2 = o - y
  let dB1 = matmul(nn.W2.transpose(), (o - y))

  nn.W2 = nn.W2 - alpha * dW2
  nn.W1 = nn.W1 - alpha * dW1
  nn.B2 = nn.B2 - alpha * dB2
  nn.B1 = nn.B1 - alpha * dB1

proc init (nn: var NN): void =
  randomize()
  nn.W1 = repeat(repeat(float64(0.0), 784), 24).map(proc (x: seq[float64]): seq[float64] {.closure.} = map(x, proc (y: float64): float64 = rand(2.0) - 1.0))
  nn.W2 = repeat(repeat(float64(0.0), 24), 10).map(proc (x: seq[float64]): seq[float64] {.closure.} = map(x, proc (y: float64): float64 = rand(2.0) - 1.0))
  nn.B1 = repeat(@[float64(0.0)], 24).map(proc (x: seq[float64]): seq[float64] = @[float64(rand(2.0) - 1.0)])
  nn.B2 = repeat(@[float64(0.0)], 10).map(proc (x: seq[float64]): seq[float64] = @[float64(rand(2.0) - 1.0)])

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

let trainingData = loadMNISTData("mnist/mnist_train.csv")
let testData = loadMNISTData("mnist/mnist_test.csv")

var nn = NN()
nn.init()

for data in trainingData:
  let output = nn.forward(data.image)
  var expected: seq[seq[float64]] = repeat(@[0.0'f64], 10)
  expected[data.label][0] = 1.0'f64
  nn.backward(output, expected, data.image, 0.001'f64)
  let loss = categoricalCrossEntropyLoss(output, data.label)

  if loss.isNaN():
    echo output
    echo expected
    break

var count = 0
for data in testData:
  let output = nn.forward(data.image)
  let result = maxIndex(output.transpose()[0])

  if data.label == uint8(result):
    count += 1

echo $(float64(count) / float64(testData.len) * 100) & "% accuracy"
