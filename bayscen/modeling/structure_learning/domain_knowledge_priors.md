# Domain Knowledge Priors for BayScen Structure Learning

This document describes the domain knowledge priors used with Bi-CaMML to learn
the Bayesian Network structure for BayScen, following the error-tolerant
knowledge-guided framework of Ban et al. (2025).

Domain knowledge is encoded as **soft ancestral constraints** — directed paths
that are strongly encouraged but not enforced. If observational data strongly
contradicts a prior relationship, the data-driven term overrides it.

**Methodology reference**: Ban et al. (2025). "Integrating large language model
for improved causal discovery." *IEEE Transactions on Artificial Intelligence*.

---

## Causal Relationships Derived from Domain Knowledge

The following physical causal relationships are well-established in meteorology
and autonomous driving literature and serve as soft constraints for structure
learning:

### Weather-Surface Relationships

| Cause | Effect | Physical mechanism |
|-------|--------|--------------------|
| Precipitation | Wetness | Rain/snow directly accumulates on road surfaces |
| Precipitation | Precipitation_Deposits | Precipitation creates surface water and contamination |
| Wetness | Road_Friction | Wet roads reduce tire-road contact friction |
| Precipitation_Deposits | Road_Friction | Surface deposits (ice, water) reduce friction |
| Precipitation_Deposits | Wetness | Deposits contribute to overall surface wetness |

### Atmospheric Relationships

| Cause | Effect | Physical mechanism |
|-------|--------|--------------------|
| Fog_Density | Fog_Distance | Denser fog reduces visibility distance |
| Wind_Intensity | Fog_Density | Strong wind disperses fog concentration |
| Precipitation | Fog_Density | Precipitation reduces fog (washout effect) |
| Cloudiness | Precipitation | Cloud cover precedes and enables precipitation |

### Temporal/Solar Relationships (Scenarios 2 & 3 only)

| Cause | Effect | Physical mechanism |
|-------|--------|--------------------|
| Sun_Altitude_Angle | Cloudiness | Daytime heating drives cloud formation patterns |
| Sun_Altitude_Angle | Wind_Intensity | Solar-driven thermal gradients affect wind |

---

## Bi-CaMML Soft Ancestral Constraints

The constraints below are fed into Bi-CaMML's prior specification. A confidence
of 0.99999 means a near-certain soft constraint; data can still override if the
evidence is strong.

```
# Scenario 1 (Vehicle–Vehicle) — 8 environmental variables

Precipitation              => Wetness                     0.99999
Precipitation              => Precipitation_Deposits       0.99999
Wetness                    => Road_Friction                0.99999
Precipitation_Deposits     => Road_Friction                0.99999
Fog_Density                => Fog_Distance                 0.99999
Wind_Intensity             => Fog_Density                  0.99999
Cloudiness                 => Precipitation                0.99999
```

```
# Scenarios 2 & 3 — additional constraints involving Sun_Altitude_Angle

Sun_Altitude_Angle         => Cloudiness                  0.99999
Sun_Altitude_Angle         => Wind_Intensity              0.99999
```

---

## Tier Constraints (Temporal Ordering)

Bi-CaMML tier constraints encode a global causal ordering:

```
Sun_Altitude_Angle
  < Cloudiness, Wind_Intensity
    < Precipitation
      < Fog_Density, Fog_Distance
        < Wetness, Precipitation_Deposits
          < Road_Friction
```

This ordering reflects the physical causal chain: solar conditions influence
cloud formation and wind; clouds and wind influence precipitation; precipitation
drives surface conditions; surface conditions determine friction.

---

## Running Bi-CaMML

1. **Install**: `git clone https://github.com/CausalAILab/Bi-CaMML`
2. **Prepare data**: use `data/processed/bayscen_final_data.csv`
3. **Load priors**: paste contents from `bicamml_priors.txt` into the Bi-CaMML
   "Expert Priors" interface
4. **Run learning**: Bi-CaMML outputs the learned DAG structure
5. **Save edges**: copy the edge list to `learned_structures/scenario{N}_structure.txt`

---

## References

- Ban et al. (2025). "Integrating large language model for improved causal
  discovery." *IEEE Transactions on Artificial Intelligence*.
- Wallace, Korb & Dai (1996). "Causal discovery via MML." *ICML*.
- Heckerman, Geiger & Chickering (1995). "Learning Bayesian networks." *Machine
  Learning*, 20(3):197–243.
