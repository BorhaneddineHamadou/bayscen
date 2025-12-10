# LLM Prompts for Bayesian Network Structure Elicitation

This document contains the GPT-4 prompts used to derive causal relationships for BayScen's Bayesian Network structure learning.

**LLM Used**: GPT-4 (gpt-4-0613)

**Methodology Reference**: Ban et al. (2025). "Integrating large language model for improved causal discovery." *IEEE Transactions on Artificial Intelligence*.

---

## Prompt 1: Variable Definition and Understanding

### Objective
Understand the meaning of each variable in the context of autonomous vehicle testing scenarios.

### Prompt

```
You are an expert on autonomous vehicle testing scenarios, particularly focusing on weather and road conditions that affect vehicle behavior and safety.

You are investigating the cause-and-effect relationships between the following variables in your field.

Variable names and their possible values are presented as follows. Please understand the real meaning of each variable according to their possible values and the context of autonomous vehicle testing scenarios, and explain them in order.

network weather_road_conditions {
}

variable Road_Friction {
    type discrete;
    values { 0.1, 0.2, 0.4, 0.8, 1.0 };
    description "Coefficient ranging from 0 (no friction/ice) to 1.0 (maximum friction/dry asphalt)";
}

variable Cloudiness {
    type discrete;
    values { 0, 20, 40, 60, 80, 100 };
    description "Percentage of sky covered by clouds";
}

variable Precipitation {
    type discrete;
    values { 0, 20, 40, 60, 80, 100 };
    description "Intensity or amount of precipitation";
}

variable Wetness {
    type discrete;
    values { 0, 20, 40, 60, 80, 100 };
    description "Level of moisture on road surfaces";
}

variable Precipitation_Deposits {
    type discrete;
    values { 0, 20, 40, 60, 80, 100 };
    description "Accumulation of precipitation on surfaces";
}

variable Wind_Intensity {
    type discrete;
    values { 0, 20, 40, 60, 80, 100 };
    description "Strength of wind conditions";
}

variable Fog_Density {
    type discrete;
    values { 0, 20, 40, 60, 80, 100 };
    description "Thickness or concentration of fog";
}

variable Fog_Distance {
    type discrete;
    values { 0, 20, 40, 60, 80, 100 };
    description "Visibility distance in foggy conditions";
}

[For Scenario 2 only:]
variable Time_of_Day {
    type discrete;
    values { -90, -60, -30, 0, 30, 60, 90 };
    description "Time relative to solar noon in degrees (likely representing sun angle)";
}

Please provide precise definitions for each variable that clarify their meaning in the context of autonomous vehicle testing scenarios.
```

**Purpose**: Establish clear understanding of what each variable represents before eliciting causal relationships.

---

## Prompt 2: Causal Extraction

### Objective
Identify all direct causal relationships between pairs of variables.

### Prompt

```
Based on your explanation, analyze the causality between each pair of variables in the form of <edge>variable1->variable2</edge>, and give the reason.

Ensure direct causality of outputs, not indirect causality due to common causes.

For each potential edge, consider:
1. Is there a direct physical/meteorological mechanism?
2. Does variable1 directly cause changes in variable2?
3. Or is the relationship mediated through other variables?

Provide ONLY direct causal edges.
```

**Purpose**: Generate a comprehensive list of candidate causal relationships with justifications.

**Example LLM Output**:
```
<edge>Precipitation->Wetness</edge>
Reason: Precipitation directly causes water to accumulate on road surfaces, increasing wetness levels. This is a direct physical effect.

<edge>Precipitation->Precipitation_Deposits</edge>
Reason: Precipitation directly causes accumulation of water, snow, or ice on surfaces. The intensity and duration of precipitation determines deposit levels.

<edge>Wetness->Road_Friction</edge>
Reason: Water on road surfaces directly reduces tire-road contact friction. Increased wetness decreases available friction coefficient.
```

---

## Prompt 3: Causal Validation

### Objective
Verify specific causal edges proposed from domain knowledge or initial extraction.

### Prompt

```
Based on your explanation, check whether the following causal statements are correct, and give the reasons:

<edge>Precipitation->Wetness</edge>
<edge>Precipitation->Precipitation_Deposits</edge>
<edge>Precipitation->Road_Friction</edge>
<edge>Wetness->Road_Friction</edge>
<edge>Precipitation_Deposits->Road_Friction</edge>
<edge>Precipitation_Deposits->Wetness</edge>
<edge>Fog_Density->Fog_Distance</edge>
<edge>Precipitation->Fog_Distance</edge>
<edge>Wind_Intensity->Fog_Density</edge>

[For Scenario 2, additional edges:]
<edge>Time_of_Day->Cloudiness</edge>
<edge>Time_of_Day->Wind_Intensity</edge>

For each edge:
1. State whether it represents a DIRECT causal relationship
2. If correct, explain the physical mechanism
3. If incorrect, explain why (e.g., indirect causation, reverse causation, no causation)
```

**Purpose**: Validate proposed edges and identify which are correct and direct causal relationships.

**Example LLM Output**:
```
<edge>Precipitation->Precipitation_Deposits</edge>: Correct. An increase in precipitation directly leads to an increase in the accumulation of precipitation on surfaces. The more it rains or snows, the more accumulation there will be on the road surface.

<edge>Wetness->Road_Friction</edge>: Correct. An increase in road surface wetness directly leads to a decrease in road friction. The wetter the road surface, the less friction there is between the road and the vehicle's tires, making the road more slippery.

<edge>Precipitation_Deposits->Road_Friction</edge>: Correct. An increase in precipitation deposits directly leads to a decrease in road friction. The more accumulation of snow or ice on the road, the less friction there is between the road and the vehicle's tires.

<edge>Cloudiness->Time_of_Day</edge>: Incorrect. Cloudiness and Time_of_Day are not causally related. Cloudiness refers to the percentage of the sky covered by clouds, while Time_of_Day refers to the time relative to solar noon. Although cloudiness can affect the perceived brightness during the day, it does not cause a change in the time of day.
```

---

## Converting LLM Output to Bi-CaMML Priors

The LLM output is converted to Bi-CaMML soft constraints:

### Arc Constraints (Strong Causal Relationships)

From LLM validation, edges marked as "DIRECT" with high confidence:

```
Precipitation => Wetness 0.99999
Precipitation => Precipitation_Deposits 0.99999
Wetness => Road_Friction 0.99999
Precipitation_Deposits => Road_Friction 0.99999
Fog_Density => Fog_Distance 0.99999
Wind_Intensity => Precipitation_Deposits 0.99999
Wind_Intensity => Fog_Density 0.99999
Time_of_Day => Fog_Density 0.99999
```

**Interpretation**:
- `=>` denotes ancestral constraint (can be direct or indirect path)
- `0.99999` means very high confidence (soft constraint, not hard)
- Data can still override if evidence is strong

### Why Soft Constraints?

Following Ban et al. (2025):
- LLMs can make mistakes
- Real-world data may reveal unexpected dependencies
- Soft constraints guide but don't force the structure
- Allows data-driven learning to correct LLM errors

---

**Mitigation**: Use as soft constraints, not hard rules. Let data override when evidence is strong.

---

## Citation

If using these prompts, cite:

```bibtex
@article{ban2025integrating,
  title={Integrating large language model for improved causal discovery},
  author={Ban, Tong and Chen, Liang and others},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2025}
}
```