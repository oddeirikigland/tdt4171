import numpy as np


def forward(currentState, evidence, transitionModel):
    prediction = np.matmul(evidence, transitionModel.dot(currentState))
    normalize = lambda x: x / np.sum(prediction)
    return normalize(prediction)


def calculateProbability(ev, transitionModel):
    fv = [np.array([0.5, 0.5])]
    t = len(ev)
    for i in range(1, t):
        # print(
        #     "0:{}, evi: {}, trans: {}, curr: {}\n".format(
        #         i, ev[i], transitionModel, fv[i - 1]
        #     )
        # )
        fv.append(forward(fv[i - 1], ev[i], transitionModel))
    print("Probability of rain at day {}: {}".format(t - 1, round(fv[t - 1][0], 3)))



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
