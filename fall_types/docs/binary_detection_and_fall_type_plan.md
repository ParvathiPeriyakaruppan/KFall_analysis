# Existing Binary Fall Detection and Fall-Type Prediction plan

## Purpose

The current KFall system answers one question:

> Is this sensor window a fall or not a fall?

The planned extension is to keep that existing answer and, only when a fall is
detected, provide additional information about the fall direction and context.
This document describes both stages and the proposed design.

## 1. Current binary prediction system

The existing models are trained as binary classifiers:

| Class | Meaning |
|---:|---|
| 0 | ADL / not a fall |
| 1 | Fall |

The ConvLSTM baseline uses the nine raw sensor channels:

- `AccX`, `AccY`, `AccZ`
- `GyrX`, `GyrY`, `GyrZ`
- `EulerX`, `EulerY`, `EulerZ`

The input is divided into short time windows. For the ConvLSTM baseline, a
window contains 50 frames. The model processes the window as follows:

```text
9-channel sensor window
          |
          v
1D convolution blocks
          |
          v
LSTM sequence model
          |
          v
last hidden feature vector
          |
          v
binary fully connected layer
          |
          v
ADL / Fall logits
```

The final layer produces two logits. A softmax converts them to class
probabilities, and the class with the larger probability is selected. During
continuous evaluation, predictions from multiple windows belonging to one
trial are aggregated using the existing persistence logic before a final
trial-level fall decision is made.

Conceptually, the current output is:

```text
fall_probability = P(fall | sensor window)
not_fall_probability = P(ADL | sensor window)
```

The existing model is therefore a detector, not a fall-type classifier.

## 2. Why the current output cannot identify fall type

The final binary layer has only two output values:

```python
self.fc = nn.Linear(hidden_size, 2)
```

Those two outputs represent ADL and fall. They do not directly represent:

- forward fall,
- backward fall,
- lateral fall,
- sitting fall,
- standing fall,
- walking fall,
- running fall, or
- fainting-related fall.

Once the internal feature vector has been converted into only two logits, the
binary output alone does not contain an explicit fall-type decision. Reading a
fall type from the two probabilities would not be reliable.

## 3. Planned fall-type output from the existing system

The proposed solution is to reuse the existing convolutional and temporal
feature extractor. The detector remains responsible for the binary decision,
while two lightweight prediction heads use the same learned feature vector:

```text
                    +--> binary head --> ADL / Fall
                    |
sensor window --> shared existing backbone --> direction head --> direction
                    |
                    +--> context head --> context
```

For the ConvLSTM, the shared feature is the final LSTM representation currently
used immediately before the binary fully connected layer:

```python
features = out[:, -1, :]
```

The planned heads are:

```python
fall_logits = fall_head(features)
direction_logits = direction_head(features)
context_logits = context_head(features)
```

The direction head can predict:

```text
forward / backward / lateral / unspecified
```

The context head can predict:

```text
sitting / standing / walking / running / fainting / unspecified
```

The system should expose direction and context only for a positive fall
decision:

```text
if fall_probability < fall_threshold:
    output = ADL / no fall
else:
    output = ADL / Fall + direction + context
```

This preserves the validated binary detector and avoids training a completely
new fall-detection model from the beginning.

## 4. How the new heads would be trained

The KFall label workbooks provide task descriptions. These descriptions are
used to create task-level direction and context labels, which are stored in
`fall_type_metadata.csv`.

Examples:

| Description | Direction | Context |
|---|---|---|
| Forward fall when trying to sit down | forward | sitting |
| Backward fall while sitting, caused by fainting | backward | fainting |
| Forward fall while walking caused by a trip | forward | walking |
| Forward fall while jogging caused by a trip | forward | running |
| Forward lateral fall while walking caused by a slip | lateral | walking |

The binary fall head can be retained from the existing checkpoint. The shared
backbone can initially be frozen, and only the direction/context heads trained.
This is much cheaper than training a second complete detector. A later fine-
tuning stage can unfreeze selected backbone layers if the type heads need more
signal.

ADL windows do not receive a direction or context target because those labels
are meaningful only after a fall has been identified.

## 5. Expected inference output

For an unknown sensor window, the desired output is:

```text
fall_probability: 0.97
fall_detected: true

direction:
  forward: 0.81
  backward: 0.06
  lateral: 0.11
  unspecified: 0.02

context:
  fainting: 0.62
  sitting: 0.21
  walking: 0.12
  standing: 0.05
```

If the binary detector says ADL, the type result should be suppressed:

```text
fall_probability: 0.08
fall_detected: false
direction: not applicable
context: not applicable
```

If the fall probability is positive but the type confidence is low, the system
should return `uncertain` instead of forcing an incorrect type.

## 6. What the current analysis suggests

The event-aligned analysis used 2,315 valid fall trials and tested the extracted
features with subject-grouped validation:

| Output | Balanced accuracy | Meaning |
|---|---:|---|
| Direction | 94.9% +/- 2.4% | Direction appears strongly represented in the sensor features. |
| Context | 70.2% +/- 4.9% | Context is only moderately separable and has substantial overlap. |

The strongest directional evidence came from orientation-change and
gyroscope-magnitude features. This is consistent with the physical idea that
different fall directions produce different rotational and orientation
responses, while signed accelerometer axes can vary with sensor placement.

Context is harder because fainting, sitting, standing, walking, and running
can share portions of the same motion pattern. The context head should therefore
report probabilities and support `uncertain` results.

## 7. Important implementation caution

The planned extension is not simply a change from `n_classes=2` to a larger
single output layer. Direction and context are different labels and should be
modeled as separate heads. A multi-task loss is a suitable design:

```text
total_loss = binary_loss + direction_loss + context_loss
```

Direction/context losses should be calculated only for fall samples with valid
labels. The binary loss should continue to use both ADL and fall samples.

Evaluation must remain subject-grouped. Windows from the same subject must not
be split between training and testing, otherwise the reported fall-type
performance can be overly optimistic.

## Conclusion

The existing models can be extended to provide fall type without discarding the
current binary detector. The recommended design is a shared existing backbone
with three outputs: binary fall detection, direction classification, and
context classification. Direction is currently promising; context should be
treated as a lower-confidence supplementary output rather than a guaranteed
label.