"""
Default parameter values for the synthetic data generator.

These values define the 'baseline' realistic batch. They are illustrative
rather than calibrated against real Fab data; in deployment, parameters
would be fitted to measured batches.
"""

from dataclasses import dataclass, field


@dataclass
class LengthDistributionParams:
    """
    Parameters for the lognormal length distribution.

    The lognormal distribution is parameterised by mu and sigma, which are
    the mean and standard deviation of the underlying normal distribution,
    not of length itself.

    With mu = 1.1 and sigma = 0.4:
      - Median length: exp(mu) ~ 3.0 mm
      - Mean length:   exp(mu + sigma^2 / 2) ~ 3.3 mm
      - ~95% of fibres fall within roughly 1.5-6.5 mm
      - A long right tail extends past 10 mm but is rare

    These values place the distribution within the system's specified
    measurement range (1-10 mm) while reflecting the
    short-fibre-heavy character of mechanically recycled textile streams.
    """
    distribution: str = "lognormal"   # "lognormal" or "weibull"
    mu: float = 1.1                   # used when distribution == "lognormal"
    sigma: float = 0.4                # used when distribution == "lognormal"
    weibull_shape: float = 1.8        # used when distribution == "weibull"
    weibull_scale: float = 3.5        # used when distribution == "weibull"
    min_length_mm: float = 0.5        # truncation lower bound
    max_length_mm: float = 12.0       # truncation upper bound (slightly above spec)
    
    
    
@dataclass
class WidthDistributionParams:
    """
    Parameters for the lognormal width distribution.

    Width is sampled independently of length. The decision to treat
    width and length as uncorrelated reflects the absence of published
    correlation data for recycled textile fibres at this size scale,
    and the heterogeneous origins of fibres in a recycled feedstock
    stream.

    With mu = -2.30 and sigma = 0.18:
      - Median width:  exp(mu) ~ 0.100 mm
      - Mean width:    exp(mu + sigma^2 / 2) ~ 0.102 mm
      - ~95% of fibres fall within roughly 0.07-0.14 mm
      - CV (std/mean) ~ 18%, consistent with reported fibre-diameter
        variability for textile fibre populations.

    These values place the distribution within the system's specified
    measurement range (0.05-0.2 mm) while reflecting
    the relatively low width variability typical of fibre populations
    of common origin.

    Only lognormal is offered for width: the literature
    consistently supports lognormal for fibre diameter regardless of
    fibre type or manufacturing technique, and no
    comparable evidence base exists for alternative distributions in
    this context.
    """
    mu: float = -2.30                  # underlying-normal mean
    sigma: float = 0.18                # underlying-normal standard deviation
    min_width_mm: float = 0.04         # truncation lower bound
    max_width_mm: float = 0.25         # truncation upper bound
    
    
    
@dataclass
class MeasurementNoiseParams:
    """
    Parameters for the measurement-noise model applied to true fibre
    dimensions to simulate SS4's segmentation output.

    The model is additive Gaussian noise applied independently to
    length and width:

        L_measured = L_true + epsilon_L,   epsilon_L ~ N(0, sigma_L)
        w_measured = w_true + epsilon_w,   epsilon_w ~ N(0, sigma_w)

    Default values reflect the typical sub-pixel edge-localisation
    accuracy of industrial machine-vision system,
    scaled to representative pixel sizes for the
    target measurement range.

    With sigma_L = 0.05 mm and sigma_w = 0.005 mm:
      - For a 3 mm fibre, length noise is ~1.7% (within SYS-PERF-02
        tolerance of +/- 5%).
      - For a 0.10 mm fibre, width noise is ~5% (within SYS-PERF-03
        tolerance of +/- 15%).

    The asymmetry reflects the physical reality that the same absolute
    pixel-level noise has a much larger fractional impact on small
    measurements (width) than on large ones (length). This is also
    why the SYS-PERF specifications themselves allow a wider
    tolerance for width.

    These values are illustrative engineering estimates, not
    empirically calibrated against SS4's specific algorithm.
    Empirical characterisation in deployment would refine them.

    No post-noise clipping is applied: measured values may
    occasionally fall slightly outside the specified measurement
    range, mirroring real measurement behaviour.
    """
    sigma_length_mm: float = 0.05
    sigma_width_mm: float = 0.005
    
    
@dataclass
class QualityMetadataParams:
    """
    Parameters for the quality-metadata model.

    Two fields are modelled stochastically per fibre:

    1. reconstructed_fraction: a two-stage model.
       - Stage 1: Bernoulli draw with probability p_reconstructed
         determines whether the fibre's mask was reconstructed at all.
       - Stage 2: if reconstructed, the fraction of the mask that
         was reconstructed is drawn from Beta(alpha, beta).
       Fibres not flagged as reconstructed have
       reconstructed_fraction = 0.

    Default values are engineering assumptions, not literature-derived
    figures. They reflect the expected behaviour of a moderately
    well-dispersed sample feeding a reconstruction-capable
    segmentation pipeline:

    - p_reconstructed = 0.2: roughly one in five fibres is expected to
      have some degree of reconstruction. Higher values would indicate
      poor sample dispersion (a Sub-system 2 problem); lower values
      would indicate sparse imaging (low throughput).
    - Beta(2, 8): mean 0.2, right-skewed, mass concentrated below 0.3.
      Reflects the expectation that mild reconstruction is the typical
      case and severe reconstruction (>50% of mask) is rare.

    2. pixel_size_mm: held constant per batch at the calibration value.
       Variability in pixel_size_mm across batches (calibration drift)
       is a Sub-system 3 / Sub-system 5 calibration concern handled
       separately in Module 4, not in the synthetic generator.

    The default value of 0.000977 mm/px matches the example
    calibration in the SS4 demonstration output, which
    represents typical industrial machine-vision pixel scales for
    the target fibre size range.
    """
    p_reconstructed: float = 0.2
    beta_alpha: float = 2.0
    beta_beta: float = 8.0
    pixel_size_mm: float = 0.000977
    
    
@dataclass
class DatasetGenerationParams:
    """
    Parameters for multi-image dataset generation.

    The 'variation_scale' parameter is the central design choice: it
    is a single dial controlling per-image variation across the
    dataset.

    - variation_scale = 0: every image is generated from the same
      underlying distribution parameters with the same n_fibres.
      This produces a 'clean experiment' dataset useful for
      sample-size analysis and other studies where a single
      variable must be isolated.

    - variation_scale > 0 (typical range 0-1): every image is
      generated from slightly different distribution parameters,
      and n_fibres varies via a Poisson draw around the target.
      This simulates the realistic case where each image
      represents a different (slightly different) batch of fibres
      from the same broad feedstock.

    Variation only affects the length and width distribution
    parameters and n_fibres. Reconstruction, measurement noise, and
    pixel_size_mm are not varied across images: these reflect
    properties of the upstream pipeline (SS1, SS2, SS4) rather than
    of the feedstock, and are stable system characteristics.

    Concrete effect of variation_scale at typical values (each image
    samples Gaussian-perturbed copies of the default parameters):

    - 0.0 : no variation (clean experiment)
    - 0.1 : mild variation; length mu by +/-0.04, width mu by +/-0.018
    - 0.5 : moderate variation; length mu by +/-0.20, width mu by +/-0.09
    - 1.0 : extreme variation; length mu by +/-0.40, width mu by +/-0.18

    These ranges produce realistic-looking batch-to-batch variation
    consistent with what would be expected from natural feedstock
    inhomogeneity.
    """
    n_images: int = 50
    target_n_fibres: int = 500
    variation_scale: float = 0.0
    
    
@dataclass(frozen=True)
class LiteratureGroundedTargetSpec:
    """
    Specification for the literature-grounded synthetic target.
 
    The target represents an MOR-like quantity (MPa-equivalent) for
    an MDF board pressed from the analysed fibre batch. Its
    structural form is fixed and grounded in the Module 5 evidence base:
 
        y = beta_0
            + beta_length * S(D50_L; L_sat)
            + beta_ar     * S(D50_AR; AR_sat)
            - beta_fines  * f_fine_count
            - beta_cv     * CV_L
            + epsilon,    epsilon ~ N(0, sigma_eps)
 
    where S(x; x_sat) = min(x, x_sat) is a saturating ramp.
 
    Coefficient magnitudes are illustrative and chosen to produce
    realistic-range MOR values (~4-26 MPa) across the descriptor
    space the synthetic generator can produce. They are not measured
    coefficients; deployment would require fitting them against
    paired Fab fibre-characterisation and board-test data.
    """
    name: str = "synthetic_MOR"
 
    # Intercept (MPa)
    beta_0: float = 12.0
 
    # Length contribution: beta_length * min(D50_L, length_sat)
    # D50_L in mm; output in MPa.
    beta_length: float = 2.0
    length_sat_mm: float = 6.0
 
    # Aspect ratio contribution: beta_ar * min(D50_AR, ar_sat)
    # D50_AR dimensionless; output in MPa.
    beta_ar: float = 0.10
    ar_sat: float = 50.0
 
    # Fines penalty: -beta_fines * f_fine_count
    # f_fine_count is in [0, 1]; output in MPa.
    beta_fines: float = 30.0
 
    # Heterogeneity penalty: -beta_cv * CV_L
    # CV_L dimensionless; output in MPa.
    beta_cv: float = 8.0
 
    # Gaussian noise standard deviation (MPa).
    noise_std: float = 1.5
 
    # Threshold defining a fine, in mm. Matches Module 3's default
    # and SYS-PERF-01 lower bound.
    fines_threshold_mm: float = 1.0
 
 
@dataclass(frozen=True)
class TargetGenerationParams:
    """
    Top-level configuration for synthetic-target generation.
 
    Backward-compatible class name: batch_assembly.py and
    dataset_generator.py import this name.
 
    Disabled by default; enable explicitly when training data is
    needed for Module 6.
    """
    enabled: bool = False
    spec: LiteratureGroundedTargetSpec = field(
        default_factory=LiteratureGroundedTargetSpec
    )
    
    
@dataclass(frozen=True)
class DiverseDatasetParams:
    """
    Configuration for generate_diverse_dataset.
 
    Each per-image regime is drawn from a tuple of independent
    uniform distributions with the bounds given here. Defaults are
    engineering choices that span the descriptor space the synthetic
    generator can plausibly produce: from short, fines-heavy batches
    at one corner to long, well-bonded batches at the other.
 
    Coverage rationale (per descriptor that drives the synthetic
    target):
      D50_length    : controlled by length_mu_range; default spans
                      roughly 1.5-4.5 mm of median length.
      CV_length     : controlled by length_sigma_range; default
                      spans roughly 0.25-0.7 of length CV.
      D50_aspect    : controlled by width_mu_range; default spans
                      roughly 15-50 of median aspect ratio.
      fines_fraction: emerges from the joint draw - low length_mu
                      values in the chosen range produce batches
                      with measurable fines.
    """
    n_images: int = 200
    target_n_fibres: int = 500
 
    # Per-image parameter ranges. Each image draws one value from
    # each range (Latin Hypercube if scipy.stats.qmc is available;
    # uniform random otherwise).
    length_mu_range: tuple = (0.4, 1.5)
    length_sigma_range: tuple = (0.25, 0.65)
    width_mu_range: tuple = (-2.6, -2.0)
    p_reconstructed_range: tuple = (0.05, 0.40)
 
    # Whether to use Latin Hypercube sampling for better coverage
    # at fixed n_images. Falls back to uniform random if scipy
    # is unavailable.
    use_latin_hypercube: bool = True
