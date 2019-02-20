import numpy as np

np.set_printoptions(precision=3)


def normalize(np_array):
    norm = lambda x: x / np.sum(np_array)
    return norm(np_array)


def forward(currentState, evidence, transitionModel):
    return normalize(np.matmul(evidence, transitionModel.dot(currentState)))


def backward(currentState, evidence, transitionModel):
    return normalize(np.matmul(currentState, np.matmul(evidence, transitionModel)))


def smoothed(forward, backward):
    out = []
    for f, b in zip(forward, backward):
        out.append(normalize(f * b))
    return out


def nicePrint(msg, arr):
    print(msg)
    for elem in arr:
        print(elem)


def calculateProbability(ev, transitionModel):
    fv = [np.array([0.5, 0.5])]
    bv = [np.array([1.0, 1.0])]
    for i in range(1, len(ev)):
        fv.append(forward(fv[i - 1], ev[i], transitionModel))
        bv.append(backward(bv[i - 1], ev[i], transitionModel))
    smooth = smoothed(fv, np.flipud(bv)) # Reversing the backward calculation with np.flipud()
    nicePrint("\nNormalized forward messages: ", fv)
    nicePrint("\nNormalized backward messages: ", smooth)
    return smooth[-1]


def main():
    umbrellaTransitionModel = np.array([[0.7, 0.3], [0.3, 0.7]])
    umbrellaEvidenceTrue = np.array([[0.9, 0], [0, 0.2]])
    umbrellaEvidenceFalse = np.array([[0.1, 0], [0, 0.8]])

    ev = [None, umbrellaEvidenceTrue, umbrellaEvidenceTrue]
    calculateProbability(ev, umbrellaTransitionModel)

    ev = [
        None,
        umbrellaEvidenceTrue,
        umbrellaEvidenceTrue,
        umbrellaEvidenceFalse,
        umbrellaEvidenceTrue,
        umbrellaEvidenceTrue,
    ]
    calculateProbability(ev, umbrellaTransitionModel)


if __name__ == "__main__":
    main()
