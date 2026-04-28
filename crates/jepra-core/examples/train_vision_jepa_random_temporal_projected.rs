#[path = "support/projected_temporal.rs"]
mod projected_temporal;
#[path = "support/temporal_vision.rs"]
mod temporal_vision;

use jepra_core::{
    BottleneckPredictor, Linear, Predictor, PredictorModule, ProjectedVisionJepa,
    ResidualBottleneckPredictor, SignedAngularRadialObjectiveReport,
    SignedBankSoftmaxObjectiveReport, SignedCenteredRadiusScalarObjectiveReport,
    SignedMarginObjectiveReport, SignedRadialCalibrationReport, StateRadiusPredictor, Tensor,
    projection_stats, representation_stats,
};
use projected_temporal::{
    PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO, PROJECTED_VALIDATION_BASE_SEED,
    PROJECTED_VALIDATION_BATCHES, PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO,
    ProjectedSignedCandidateCentroidIntegration, ProjectedSignedCandidateRadiusHeadIntegration,
    ProjectedSignedCandidateSelectorHeadIntegration,
    ProjectedSignedCandidateSelectorHeadObjectiveReport,
    ProjectedSignedCandidateSelectorOutputCouplingReport, ProjectedSignedCandidateSelectorProbe,
    ProjectedSignedCandidateSelectorReadoutDiagnostics,
    ProjectedSignedCandidateUnitMixHeadIntegration, ProjectedSignedCandidateUnitMixObjectiveReport,
    ProjectedSignedObjectiveErrorBreakdown, ProjectedSignedPredictionBankMargin,
    ProjectedSignedPredictionBankUnitGeometry, ProjectedSignedPredictionCounterfactualMetrics,
    ProjectedSignedPredictionGeometryCounterfactual, ProjectedSignedStateSeparability,
    ProjectedSignedTargetBankSeparability, ProjectedSignedVelocityBankBreakdown,
    ProjectedVelocityBankRanking, projected_signed_angular_radial_objective_loss_and_grad,
    projected_signed_angular_radial_objective_report_from_base_seed,
    projected_signed_bank_softmax_objective_loss_and_grad,
    projected_signed_bank_softmax_objective_report_from_base_seed,
    projected_signed_candidate_centroid_integration_from_base_seed,
    projected_signed_candidate_radius_delta_loss_and_grad,
    projected_signed_candidate_radius_head_features,
    projected_signed_candidate_radius_head_integration_from_base_seed,
    projected_signed_candidate_radius_logit_head_features,
    projected_signed_candidate_radius_logit_head_integration_from_base_seed,
    projected_signed_candidate_radius_logit_mixing_loss_and_grad,
    projected_signed_candidate_selector_active_normalized_stable_hard_full_output_coupling_loss_and_grad,
    projected_signed_candidate_selector_hard_full_output_coupling_loss_and_grad,
    projected_signed_candidate_selector_hard_full_output_coupling_report_from_base_seed,
    projected_signed_candidate_selector_head_features,
    projected_signed_candidate_selector_head_integration_from_base_seed,
    projected_signed_candidate_selector_head_loss_and_grad,
    projected_signed_candidate_selector_probe_from_base_seed,
    projected_signed_candidate_selector_stable_hard_full_output_coupling_loss_and_grad,
    projected_signed_candidate_unit_mix_head_features,
    projected_signed_candidate_unit_mix_head_integration_from_base_seed,
    projected_signed_candidate_unit_mix_loss_and_grad,
    projected_signed_margin_objective_loss_and_grad,
    projected_signed_margin_objective_report_from_base_seed,
    projected_signed_objective_error_breakdown_from_base_seed,
    projected_signed_prediction_bank_margin_from_base_seed,
    projected_signed_prediction_bank_unit_geometry_from_base_seed,
    projected_signed_prediction_geometry_counterfactual_from_base_seed,
    projected_signed_radial_calibration_loss_and_grad,
    projected_signed_radial_calibration_report_from_base_seed,
    projected_signed_state_separability_from_base_seed,
    projected_signed_target_bank_separability_from_base_seed,
    projected_signed_velocity_bank_breakdown_from_base_seed,
    projected_validation_batch_losses_from_base_seed_for_task,
    projected_velocity_bank_ranking_from_base_seed,
};
use temporal_vision::{
    CompactEncoderMode, PredictorMode, SIGNED_VELOCITY_BANK_CANDIDATE_DX, TemporalTaskMode,
    assert_required_motion_modes_for_task, assert_temporal_contract,
    assert_temporal_experiment_improved, make_compact_frozen_encoder,
    make_compact_frozen_encoder_signed_direction,
    make_compact_frozen_encoder_signed_direction_magnitude, make_compact_frozen_encoder_stronger,
    make_frozen_encoder, make_train_batch_for_config, make_validation_batch_for_config,
    make_validation_batch_with_required_motion_modes_for_config, print_batch_summary_for_task,
    print_motion_mode_summary_for_task, print_representation_stats,
};

const PROJECTION_DIM: usize = 4;
const TRAIN_BASE_SEED: u64 = 11_000;
const NUM_STEPS: usize = 300;
const LOG_EVERY: usize = 25;
const PROJECTOR_LR: f32 = 0.005;
const PREDICTOR_LR: f32 = 0.02;
const REGULARIZER_WEIGHT: f32 = 1e-4;
const DEFAULT_SIGNED_CANDIDATE_CENTROID_TEMPERATURE: f32 = 0.25;
const DEFAULT_SIGNED_CANDIDATE_RADIUS_HEAD_TEMPERATURE: f32 = 0.05;
const DEFAULT_SIGNED_CANDIDATE_RADIUS_HEAD_LR: f32 = 0.02;
const DEFAULT_SIGNED_CANDIDATE_RADIUS_LOGIT_HEAD_LR: f32 = 0.01;
const DEFAULT_SIGNED_CANDIDATE_RADIUS_HEAD_WEIGHT: f32 = 1.0;
const DEFAULT_SIGNED_CANDIDATE_UNIT_MIX_HEAD_TEMPERATURE: f32 = 0.05;
const DEFAULT_SIGNED_CANDIDATE_UNIT_MIX_HEAD_LR: f32 = 0.01;
const DEFAULT_SIGNED_CANDIDATE_UNIT_MIX_HEAD_WEIGHT: f32 = 1.0;
const DEFAULT_SIGNED_CANDIDATE_SELECTOR_HEAD_TEMPERATURE: f32 = 0.05;
const DEFAULT_SIGNED_CANDIDATE_SELECTOR_HEAD_LR: f32 = 0.05;
const DEFAULT_SIGNED_CANDIDATE_SELECTOR_HEAD_WEIGHT: f32 = 1.0;
const DEFAULT_SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_FLOOR: f32 = 1.0;
const DEFAULT_SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_WEIGHT: f32 = 0.1;
const DEFAULT_SIGNED_CANDIDATE_SELECTOR_HEAD_KL_WEIGHT: f32 = 0.0;
const DEFAULT_SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING_WEIGHT: f32 = 1.0;
const DEFAULT_SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING_WARMUP_STEPS: usize = 0;
const DEFAULT_SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING_MIN_CONFIDENCE: f32 = 0.0;
const DEFAULT_SIGNED_CANDIDATE_SELECTOR_PROBE_TEMPERATURE: f32 = 0.05;
const DEFAULT_SIGNED_CANDIDATE_SELECTOR_PROBE_STEPS: usize = 200;
const DEFAULT_SIGNED_CANDIDATE_SELECTOR_PROBE_LR: f32 = 0.05;
const SIGNED_CANDIDATE_RADIUS_HEAD_FEATURE_DIM: usize = PROJECTION_DIM * 2 + 4;
const SIGNED_CANDIDATE_RADIUS_LOGIT_HEAD_FEATURE_DIM: usize =
    PROJECTION_DIM + SIGNED_VELOCITY_BANK_CANDIDATE_DX.len() * 2;
const SIGNED_CANDIDATE_RADIUS_LOGIT_HEAD_OUTPUT_DIM: usize =
    SIGNED_VELOCITY_BANK_CANDIDATE_DX.len();
const SIGNED_CANDIDATE_UNIT_MIX_HEAD_FEATURE_DIM: usize =
    SIGNED_CANDIDATE_RADIUS_LOGIT_HEAD_FEATURE_DIM;
const SIGNED_CANDIDATE_UNIT_MIX_HEAD_OUTPUT_DIM: usize = SIGNED_VELOCITY_BANK_CANDIDATE_DX.len();
const SIGNED_CANDIDATE_SELECTOR_HEAD_FEATURE_DIM: usize =
    SIGNED_CANDIDATE_UNIT_MIX_HEAD_FEATURE_DIM;
const SIGNED_CANDIDATE_SELECTOR_HEAD_OUTPUT_DIM: usize = SIGNED_VELOCITY_BANK_CANDIDATE_DX.len();

#[derive(Debug, Clone, Copy)]
struct SignedCandidateCentroidIntegrationRunConfig {
    enabled: bool,
    softmax_temperature: f32,
}

#[derive(Debug, Clone, Copy)]
struct SignedCandidateRadiusHeadRunConfig {
    enabled: bool,
    mode: SignedCandidateRadiusHeadMode,
    softmax_temperature: f32,
    learning_rate: f32,
    loss_weight: f32,
}

#[derive(Debug, Clone, Copy)]
struct SignedCandidateUnitMixHeadRunConfig {
    enabled: bool,
    softmax_temperature: f32,
    learning_rate: f32,
    loss_weight: f32,
}

#[derive(Debug, Clone, Copy)]
struct SignedCandidateSelectorHeadRunConfig {
    enabled: bool,
    softmax_temperature: f32,
    learning_rate: f32,
    loss_weight: f32,
    entropy_floor: f32,
    entropy_weight: f32,
    kl_weight: f32,
}

#[derive(Debug, Clone, Copy)]
struct SignedCandidateSelectorOutputCouplingRunConfig {
    mode: SignedCandidateSelectorOutputMode,
    loss_weight: f32,
    warmup_steps: usize,
    freeze_selector_after_warmup: bool,
    min_confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignedCandidateSelectorOutputMode {
    Off,
    HardFull,
    StableHardFull,
    ActiveNormalizedStableHardFull,
}

impl SignedCandidateSelectorOutputMode {
    fn from_str(value: &str) -> Self {
        match value {
            "off" => Self::Off,
            "hard-full" => Self::HardFull,
            "stable-hard-full" => Self::StableHardFull,
            "active-normalized-stable-hard-full" => Self::ActiveNormalizedStableHardFull,
            raw => panic!(
                "signed candidate selector output must be off|hard-full|stable-hard-full|active-normalized-stable-hard-full, got {}",
                raw
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::HardFull => "hard-full",
            Self::StableHardFull => "stable-hard-full",
            Self::ActiveNormalizedStableHardFull => "active-normalized-stable-hard-full",
        }
    }

    fn enabled(self) -> bool {
        self != Self::Off
    }

    fn requires_correct_selector(self) -> bool {
        matches!(
            self,
            Self::StableHardFull | Self::ActiveNormalizedStableHardFull
        )
    }

    fn active_normalized(self) -> bool {
        self == Self::ActiveNormalizedStableHardFull
    }
}

#[derive(Debug, Clone, Copy)]
struct SignedCandidateSelectorProbeRunConfig {
    enabled: bool,
    softmax_temperature: f32,
    probe_steps: usize,
    learning_rate: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignedCandidateRadiusHeadMode {
    ScalarResidual,
    LogitResidual,
}

impl SignedCandidateRadiusHeadMode {
    fn as_str(&self) -> &'static str {
        match self {
            Self::ScalarResidual => "scalar-residual",
            Self::LogitResidual => "logit-residual",
        }
    }
}

fn make_projector() -> Linear {
    Linear::new(
        Tensor::new(
            vec![
                1.0, 0.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, 1.0, //
                0.0, 0.0, 1.0, 1.0,
            ],
            vec![3, PROJECTION_DIM],
        ),
        Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![PROJECTION_DIM]),
    )
}

fn make_predictor() -> Predictor {
    Predictor::new(
        Linear::randn(PROJECTION_DIM, 8, 0.1, 21_000),
        Linear::randn(8, PROJECTION_DIM, 0.1, 21_001),
    )
}

fn make_bottleneck_predictor() -> BottleneckPredictor {
    BottleneckPredictor::new(
        Linear::randn(PROJECTION_DIM, 8, 0.1, 21_100),
        Linear::randn(8, 2, 0.1, 21_101),
        Linear::randn(2, PROJECTION_DIM, 0.1, 21_102),
    )
}

fn make_residual_bottleneck_predictor(residual_delta_scale: f32) -> ResidualBottleneckPredictor {
    ResidualBottleneckPredictor::new_scaled(
        BottleneckPredictor::new(
            Linear::randn(PROJECTION_DIM, 8, 0.1, 21_100),
            Linear::randn(8, 2, 0.1, 21_101),
            Linear::new(
                Tensor::zeros(vec![2, PROJECTION_DIM]),
                Tensor::zeros(vec![PROJECTION_DIM]),
            ),
        ),
        residual_delta_scale,
    )
}

fn make_state_radius_predictor() -> StateRadiusPredictor {
    StateRadiusPredictor::new(
        make_predictor(),
        Linear::randn(PROJECTION_DIM, 4, 0.1, 21_200),
        Linear::randn(4, 1, 0.01, 21_201),
    )
}

fn make_signed_candidate_radius_head(mode: SignedCandidateRadiusHeadMode) -> Linear {
    match mode {
        SignedCandidateRadiusHeadMode::ScalarResidual => Linear::new(
            Tensor::zeros(vec![SIGNED_CANDIDATE_RADIUS_HEAD_FEATURE_DIM, 1]),
            Tensor::zeros(vec![1]),
        ),
        SignedCandidateRadiusHeadMode::LogitResidual => Linear::new(
            Tensor::zeros(vec![
                SIGNED_CANDIDATE_RADIUS_LOGIT_HEAD_FEATURE_DIM,
                SIGNED_CANDIDATE_RADIUS_LOGIT_HEAD_OUTPUT_DIM,
            ]),
            Tensor::zeros(vec![SIGNED_CANDIDATE_RADIUS_LOGIT_HEAD_OUTPUT_DIM]),
        ),
    }
}

fn make_signed_candidate_unit_mix_head() -> Linear {
    Linear::new(
        Tensor::zeros(vec![
            SIGNED_CANDIDATE_UNIT_MIX_HEAD_FEATURE_DIM,
            SIGNED_CANDIDATE_UNIT_MIX_HEAD_OUTPUT_DIM,
        ]),
        Tensor::zeros(vec![SIGNED_CANDIDATE_UNIT_MIX_HEAD_OUTPUT_DIM]),
    )
}

fn make_signed_candidate_selector_head() -> Linear {
    Linear::new(
        Tensor::zeros(vec![
            SIGNED_CANDIDATE_SELECTOR_HEAD_FEATURE_DIM,
            SIGNED_CANDIDATE_SELECTOR_HEAD_OUTPUT_DIM,
        ]),
        Tensor::zeros(vec![SIGNED_CANDIDATE_SELECTOR_HEAD_OUTPUT_DIM]),
    )
}

impl SignedCandidateCentroidIntegrationRunConfig {
    fn from_args_and_env() -> Self {
        let args = std::env::args().collect::<Vec<_>>();
        let enabled = args
            .iter()
            .any(|arg| arg == "--signed-candidate-centroid-integration")
            || parse_bool_env("JEPRA_SIGNED_CANDIDATE_CENTROID_INTEGRATION").unwrap_or(false);
        let softmax_temperature = parse_f32_arg(&args, "--signed-candidate-centroid-temperature")
            .or_else(|| parse_positive_f32_env("JEPRA_SIGNED_CANDIDATE_CENTROID_TEMPERATURE"))
            .unwrap_or(DEFAULT_SIGNED_CANDIDATE_CENTROID_TEMPERATURE);

        assert!(
            softmax_temperature.is_finite() && softmax_temperature > 0.0,
            "signed candidate-centroid temperature must be finite and positive"
        );

        Self {
            enabled,
            softmax_temperature,
        }
    }
}

impl SignedCandidateRadiusHeadRunConfig {
    fn from_args_and_env() -> Self {
        let args = std::env::args().collect::<Vec<_>>();
        let scalar_flag = args
            .iter()
            .any(|arg| arg == "--signed-candidate-radius-head");
        let logit_flag = args
            .iter()
            .any(|arg| arg == "--signed-candidate-radius-logit-head")
            || parse_bool_env("JEPRA_SIGNED_CANDIDATE_RADIUS_LOGIT_HEAD").unwrap_or(false);
        let enabled = scalar_flag
            || logit_flag
            || parse_bool_env("JEPRA_SIGNED_CANDIDATE_RADIUS_HEAD").unwrap_or(false);
        let default_mode = if logit_flag {
            SignedCandidateRadiusHeadMode::LogitResidual
        } else {
            SignedCandidateRadiusHeadMode::ScalarResidual
        };
        let parsed_mode = parse_candidate_radius_head_mode(&args).unwrap_or(default_mode);
        assert!(
            !(logit_flag && parsed_mode == SignedCandidateRadiusHeadMode::ScalarResidual),
            "--signed-candidate-radius-logit-head cannot be combined with scalar-residual mode"
        );
        let softmax_temperature =
            parse_f32_arg(&args, "--signed-candidate-radius-head-temperature")
                .or_else(|| {
                    parse_positive_f32_env("JEPRA_SIGNED_CANDIDATE_RADIUS_HEAD_TEMPERATURE")
                })
                .unwrap_or(DEFAULT_SIGNED_CANDIDATE_RADIUS_HEAD_TEMPERATURE);
        let learning_rate = parse_f32_arg(&args, "--signed-candidate-radius-head-lr")
            .or_else(|| parse_positive_f32_env("JEPRA_SIGNED_CANDIDATE_RADIUS_HEAD_LR"))
            .unwrap_or(match parsed_mode {
                SignedCandidateRadiusHeadMode::ScalarResidual => {
                    DEFAULT_SIGNED_CANDIDATE_RADIUS_HEAD_LR
                }
                SignedCandidateRadiusHeadMode::LogitResidual => {
                    DEFAULT_SIGNED_CANDIDATE_RADIUS_LOGIT_HEAD_LR
                }
            });
        let loss_weight = parse_f32_arg(&args, "--signed-candidate-radius-head-weight")
            .or_else(|| parse_positive_f32_env("JEPRA_SIGNED_CANDIDATE_RADIUS_HEAD_WEIGHT"))
            .unwrap_or(DEFAULT_SIGNED_CANDIDATE_RADIUS_HEAD_WEIGHT);

        assert!(
            softmax_temperature.is_finite() && softmax_temperature > 0.0,
            "signed candidate-radius head temperature must be finite and positive"
        );
        assert!(
            learning_rate.is_finite() && learning_rate > 0.0,
            "signed candidate-radius head lr must be finite and positive"
        );
        assert!(
            loss_weight.is_finite() && loss_weight > 0.0,
            "signed candidate-radius head weight must be finite and positive"
        );

        Self {
            enabled,
            mode: parsed_mode,
            softmax_temperature,
            learning_rate,
            loss_weight,
        }
    }
}

impl SignedCandidateUnitMixHeadRunConfig {
    fn from_args_and_env() -> Self {
        let args = std::env::args().collect::<Vec<_>>();
        let enabled = args
            .iter()
            .any(|arg| arg == "--signed-candidate-unit-mix-head")
            || parse_bool_env("JEPRA_SIGNED_CANDIDATE_UNIT_MIX_HEAD").unwrap_or(false);
        let softmax_temperature = parse_f32_arg(&args, "--signed-candidate-unit-mix-temperature")
            .or_else(|| parse_positive_f32_env("JEPRA_SIGNED_CANDIDATE_UNIT_MIX_TEMPERATURE"))
            .unwrap_or(DEFAULT_SIGNED_CANDIDATE_UNIT_MIX_HEAD_TEMPERATURE);
        let learning_rate = parse_f32_arg(&args, "--signed-candidate-unit-mix-lr")
            .or_else(|| parse_positive_f32_env("JEPRA_SIGNED_CANDIDATE_UNIT_MIX_LR"))
            .unwrap_or(DEFAULT_SIGNED_CANDIDATE_UNIT_MIX_HEAD_LR);
        let loss_weight = parse_f32_arg(&args, "--signed-candidate-unit-mix-weight")
            .or_else(|| parse_positive_f32_env("JEPRA_SIGNED_CANDIDATE_UNIT_MIX_WEIGHT"))
            .unwrap_or(DEFAULT_SIGNED_CANDIDATE_UNIT_MIX_HEAD_WEIGHT);

        assert!(
            softmax_temperature.is_finite() && softmax_temperature > 0.0,
            "signed candidate unit-mix temperature must be finite and positive"
        );
        assert!(
            learning_rate.is_finite() && learning_rate > 0.0,
            "signed candidate unit-mix lr must be finite and positive"
        );
        assert!(
            loss_weight.is_finite() && loss_weight > 0.0,
            "signed candidate unit-mix weight must be finite and positive"
        );

        Self {
            enabled,
            softmax_temperature,
            learning_rate,
            loss_weight,
        }
    }
}

impl SignedCandidateSelectorHeadRunConfig {
    fn from_args_and_env() -> Self {
        let args = std::env::args().collect::<Vec<_>>();
        let enabled = args
            .iter()
            .any(|arg| arg == "--signed-candidate-selector-head")
            || parse_bool_env("JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD").unwrap_or(false);
        let softmax_temperature =
            parse_f32_arg(&args, "--signed-candidate-selector-head-temperature")
                .or_else(|| {
                    parse_positive_f32_env("JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_TEMPERATURE")
                })
                .unwrap_or(DEFAULT_SIGNED_CANDIDATE_SELECTOR_HEAD_TEMPERATURE);
        let learning_rate = parse_f32_arg(&args, "--signed-candidate-selector-head-lr")
            .or_else(|| parse_positive_f32_env("JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_LR"))
            .unwrap_or(DEFAULT_SIGNED_CANDIDATE_SELECTOR_HEAD_LR);
        let loss_weight = parse_f32_arg(&args, "--signed-candidate-selector-head-weight")
            .or_else(|| parse_positive_f32_env("JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_WEIGHT"))
            .unwrap_or(DEFAULT_SIGNED_CANDIDATE_SELECTOR_HEAD_WEIGHT);
        let entropy_floor =
            parse_nonnegative_f32_arg(&args, "--signed-candidate-selector-head-entropy-floor")
                .or_else(|| {
                    parse_nonnegative_f32_env("JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_FLOOR")
                })
                .unwrap_or(DEFAULT_SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_FLOOR);
        let entropy_weight =
            parse_nonnegative_f32_arg(&args, "--signed-candidate-selector-head-entropy-weight")
                .or_else(|| {
                    parse_nonnegative_f32_env("JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_WEIGHT")
                })
                .unwrap_or(DEFAULT_SIGNED_CANDIDATE_SELECTOR_HEAD_ENTROPY_WEIGHT);
        let kl_weight =
            parse_nonnegative_f32_arg(&args, "--signed-candidate-selector-head-kl-weight")
                .or_else(|| {
                    parse_nonnegative_f32_env("JEPRA_SIGNED_CANDIDATE_SELECTOR_HEAD_KL_WEIGHT")
                })
                .unwrap_or(DEFAULT_SIGNED_CANDIDATE_SELECTOR_HEAD_KL_WEIGHT);

        assert!(
            softmax_temperature.is_finite() && softmax_temperature > 0.0,
            "signed candidate selector head temperature must be finite and positive"
        );
        assert!(
            learning_rate.is_finite() && learning_rate > 0.0,
            "signed candidate selector head lr must be finite and positive"
        );
        assert!(
            loss_weight.is_finite() && loss_weight > 0.0,
            "signed candidate selector head weight must be finite and positive"
        );
        assert!(
            entropy_floor.is_finite() && entropy_floor >= 0.0,
            "signed candidate selector head entropy floor must be finite and non-negative"
        );
        assert!(
            entropy_weight.is_finite() && entropy_weight >= 0.0,
            "signed candidate selector head entropy weight must be finite and non-negative"
        );
        assert!(
            kl_weight.is_finite() && kl_weight >= 0.0,
            "signed candidate selector head kl weight must be finite and non-negative"
        );

        Self {
            enabled,
            softmax_temperature,
            learning_rate,
            loss_weight,
            entropy_floor,
            entropy_weight,
            kl_weight,
        }
    }
}

impl SignedCandidateSelectorOutputCouplingRunConfig {
    fn from_args_and_env() -> Self {
        let args = std::env::args().collect::<Vec<_>>();
        let output_mode = parse_arg_value(&args, "--signed-candidate-selector-output")
            .map(str::to_owned)
            .or_else(|| std::env::var("JEPRA_SIGNED_CANDIDATE_SELECTOR_OUTPUT").ok())
            .unwrap_or_else(|| "off".to_owned());
        let output_mode = output_mode.trim().to_ascii_lowercase();
        let legacy_enabled = args
            .iter()
            .any(|arg| arg == "--signed-candidate-selector-output-coupling")
            || parse_bool_env("JEPRA_SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING").unwrap_or(false);
        let mode = if legacy_enabled && output_mode == "off" {
            SignedCandidateSelectorOutputMode::HardFull
        } else {
            SignedCandidateSelectorOutputMode::from_str(&output_mode)
        };
        let loss_weight =
            parse_f32_arg(&args, "--signed-candidate-selector-output-coupling-weight")
                .or_else(|| {
                    parse_positive_f32_env("JEPRA_SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING_WEIGHT")
                })
                .unwrap_or(DEFAULT_SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING_WEIGHT);
        let warmup_steps =
            parse_nonnegative_usize_arg(&args, "--signed-candidate-selector-output-warmup-steps")
                .or_else(|| {
                    parse_nonnegative_usize_env(
                        "JEPRA_SIGNED_CANDIDATE_SELECTOR_OUTPUT_WARMUP_STEPS",
                    )
                })
                .unwrap_or(DEFAULT_SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING_WARMUP_STEPS);
        let freeze_selector_after_warmup = args
            .iter()
            .any(|arg| arg == "--signed-candidate-selector-output-freeze-selector-after-warmup")
            || parse_bool_env(
                "JEPRA_SIGNED_CANDIDATE_SELECTOR_OUTPUT_FREEZE_SELECTOR_AFTER_WARMUP",
            )
            .unwrap_or(false);
        let min_confidence =
            parse_nonnegative_f32_arg(&args, "--signed-candidate-selector-output-min-confidence")
                .or_else(|| {
                    parse_nonnegative_f32_env(
                        "JEPRA_SIGNED_CANDIDATE_SELECTOR_OUTPUT_MIN_CONFIDENCE",
                    )
                })
                .unwrap_or(DEFAULT_SIGNED_CANDIDATE_SELECTOR_OUTPUT_COUPLING_MIN_CONFIDENCE);

        assert!(
            loss_weight.is_finite() && loss_weight > 0.0,
            "signed candidate selector output coupling weight must be finite and positive"
        );
        assert!(
            min_confidence.is_finite() && (0.0..=1.0).contains(&min_confidence),
            "signed candidate selector output min confidence must be in [0, 1]"
        );
        assert!(
            !mode.enabled() || !freeze_selector_after_warmup || warmup_steps > 0,
            "freezing selector after output warmup requires positive warmup steps"
        );

        Self {
            mode,
            loss_weight,
            warmup_steps,
            freeze_selector_after_warmup,
            min_confidence,
        }
    }

    fn coupling_active_at_step(self, step: usize) -> bool {
        self.mode.enabled() && step > self.warmup_steps
    }

    fn selector_training_enabled_at_step(self, step: usize) -> bool {
        !self.mode.enabled() || !self.freeze_selector_after_warmup || step <= self.warmup_steps
    }
}

impl SignedCandidateSelectorProbeRunConfig {
    fn from_args_and_env() -> Self {
        let args = std::env::args().collect::<Vec<_>>();
        let enabled = args
            .iter()
            .any(|arg| arg == "--signed-candidate-selector-probe")
            || parse_bool_env("JEPRA_SIGNED_CANDIDATE_SELECTOR_PROBE").unwrap_or(false);
        let softmax_temperature =
            parse_f32_arg(&args, "--signed-candidate-selector-probe-temperature")
                .or_else(|| {
                    parse_positive_f32_env("JEPRA_SIGNED_CANDIDATE_SELECTOR_PROBE_TEMPERATURE")
                })
                .unwrap_or(DEFAULT_SIGNED_CANDIDATE_SELECTOR_PROBE_TEMPERATURE);
        let probe_steps = parse_usize_arg(&args, "--signed-candidate-selector-probe-steps")
            .or_else(|| parse_positive_usize_env("JEPRA_SIGNED_CANDIDATE_SELECTOR_PROBE_STEPS"))
            .unwrap_or(DEFAULT_SIGNED_CANDIDATE_SELECTOR_PROBE_STEPS);
        let learning_rate = parse_f32_arg(&args, "--signed-candidate-selector-probe-lr")
            .or_else(|| parse_positive_f32_env("JEPRA_SIGNED_CANDIDATE_SELECTOR_PROBE_LR"))
            .unwrap_or(DEFAULT_SIGNED_CANDIDATE_SELECTOR_PROBE_LR);

        assert!(
            softmax_temperature.is_finite() && softmax_temperature > 0.0,
            "signed candidate selector probe temperature must be finite and positive"
        );
        assert!(
            probe_steps > 0,
            "signed candidate selector probe steps must be positive"
        );
        assert!(
            learning_rate.is_finite() && learning_rate > 0.0,
            "signed candidate selector probe lr must be finite and positive"
        );

        Self {
            enabled,
            softmax_temperature,
            probe_steps,
            learning_rate,
        }
    }
}

fn parse_candidate_radius_head_mode(args: &[String]) -> Option<SignedCandidateRadiusHeadMode> {
    parse_arg_value(args, "--signed-candidate-radius-head-mode")
        .map(parse_candidate_radius_head_mode_value)
        .or_else(|| {
            std::env::var("JEPRA_SIGNED_CANDIDATE_RADIUS_HEAD_MODE")
                .ok()
                .map(|raw_mode| parse_candidate_radius_head_mode_value(&raw_mode))
        })
}

fn parse_candidate_radius_head_mode_value(raw_mode: &str) -> SignedCandidateRadiusHeadMode {
    match raw_mode {
        "scalar" | "scalar-residual" => SignedCandidateRadiusHeadMode::ScalarResidual,
        "logit" | "logits" | "logit-residual" => SignedCandidateRadiusHeadMode::LogitResidual,
        _ => panic!(
            "unsupported signed candidate-radius head mode: {} (expected scalar-residual|logit-residual)",
            raw_mode
        ),
    }
}

fn reduction_thresholds_for_run_config(
    run_config: temporal_vision::TemporalRunConfig,
) -> (f32, f32) {
    match (run_config.compact_encoder_mode, run_config.predictor_mode) {
        (CompactEncoderMode::Disabled, PredictorMode::Baseline) => (
            PROJECTED_TRAIN_LOSS_MAX_REDUCTION_RATIO,
            PROJECTED_VALIDATION_LOSS_MAX_REDUCTION_RATIO,
        ),
        _ => (1.0, 1.0),
    }
}

fn main() {
    let run_config =
        temporal_vision::TemporalRunConfig::from_args(TRAIN_BASE_SEED, NUM_STEPS, LOG_EVERY, 0.0);
    let signed_candidate_centroid_integration_config =
        SignedCandidateCentroidIntegrationRunConfig::from_args_and_env();
    let signed_candidate_radius_head_config =
        SignedCandidateRadiusHeadRunConfig::from_args_and_env();
    let signed_candidate_unit_mix_head_config =
        SignedCandidateUnitMixHeadRunConfig::from_args_and_env();
    let signed_candidate_selector_head_config =
        SignedCandidateSelectorHeadRunConfig::from_args_and_env();
    let signed_candidate_selector_output_coupling_config =
        SignedCandidateSelectorOutputCouplingRunConfig::from_args_and_env();
    let signed_candidate_selector_probe_config =
        SignedCandidateSelectorProbeRunConfig::from_args_and_env();

    assert!(
        !signed_candidate_selector_output_coupling_config
            .mode
            .enabled()
            || signed_candidate_selector_head_config.enabled,
        "signed candidate selector output coupling requires --signed-candidate-selector-head"
    );
    assert!(
        !signed_candidate_selector_output_coupling_config
            .mode
            .enabled()
            || run_config.temporal_task_mode == TemporalTaskMode::SignedVelocityTrail,
        "signed candidate selector output coupling is only supported for signed-velocity-trail projected runs"
    );

    println!(
        "temporal run config | train_base_seed {} | steps {} | log_every {}",
        run_config.train_base_seed, run_config.total_steps, run_config.log_every
    );
    println!(
        "temporal run config | target projection momentum {} -> {} | warmup {} steps",
        run_config.target_projection_momentum_start,
        run_config.target_projection_momentum_end,
        run_config.target_projection_momentum_warmup_steps
    );
    println!(
        "temporal run config | task {}",
        run_config.temporal_task_mode.as_str()
    );
    println!(
        "temporal run config | encoder variant {}",
        run_config.compact_encoder_mode.as_str()
    );
    println!(
        "temporal run config | predictor mode {}",
        run_config.predictor_mode.as_str()
    );
    println!(
        "temporal run config | residual delta scale {}",
        run_config.residual_delta_scale
    );
    println!(
        "temporal run config | projector drift weight {}",
        run_config.projector_drift_weight
    );
    println!(
        "temporal run config | signed margin weight {} | bank_gap {} | sign_gap {} | speed_gap {} | bank_weight {} | sign_weight {} | speed_weight {}",
        run_config.signed_margin_weight,
        run_config.signed_margin_config.bank_gap,
        run_config.signed_margin_config.sign_gap,
        run_config.signed_margin_config.speed_gap,
        run_config.signed_margin_config.bank_weight,
        run_config.signed_margin_config.sign_weight,
        run_config.signed_margin_config.speed_weight
    );
    println!(
        "temporal run config | signed bank softmax weight {} | temperature {}",
        run_config.signed_bank_softmax_weight, run_config.signed_bank_softmax_config.temperature,
    );
    println!(
        "temporal run config | signed radial weight {}",
        run_config.signed_radial_weight
    );
    println!(
        "temporal run config | signed angular-radial weight {} | angular_weight {} | radius_weight {}",
        run_config.signed_angular_radial_weight,
        run_config.signed_angular_radial_config.angular_weight,
        run_config.signed_angular_radial_config.radial_weight
    );
    println!(
        "temporal run config | signed candidate-centroid integration enabled {} | softmax_temperature {}",
        signed_candidate_centroid_integration_config.enabled,
        signed_candidate_centroid_integration_config.softmax_temperature
    );
    println!(
        "temporal run config | signed candidate-radius head enabled {} | mode {} | softmax_temperature {} | lr {} | weight {}",
        signed_candidate_radius_head_config.enabled,
        signed_candidate_radius_head_config.mode.as_str(),
        signed_candidate_radius_head_config.softmax_temperature,
        signed_candidate_radius_head_config.learning_rate,
        signed_candidate_radius_head_config.loss_weight
    );
    println!(
        "temporal run config | signed candidate unit-mix head enabled {} | softmax_temperature {} | lr {} | weight {}",
        signed_candidate_unit_mix_head_config.enabled,
        signed_candidate_unit_mix_head_config.softmax_temperature,
        signed_candidate_unit_mix_head_config.learning_rate,
        signed_candidate_unit_mix_head_config.loss_weight
    );
    println!(
        "temporal run config | signed candidate selector head enabled {} | softmax_temperature {} | lr {} | weight {} | entropy_floor {} | entropy_weight {} | kl_weight {}",
        signed_candidate_selector_head_config.enabled,
        signed_candidate_selector_head_config.softmax_temperature,
        signed_candidate_selector_head_config.learning_rate,
        signed_candidate_selector_head_config.loss_weight,
        signed_candidate_selector_head_config.entropy_floor,
        signed_candidate_selector_head_config.entropy_weight,
        signed_candidate_selector_head_config.kl_weight
    );
    println!(
        "temporal run config | signed candidate selector output mode {} | detached_target hard-full | weight {} | warmup_steps {} | freeze_selector_after_warmup {} | min_confidence {}",
        signed_candidate_selector_output_coupling_config
            .mode
            .as_str(),
        signed_candidate_selector_output_coupling_config.loss_weight,
        signed_candidate_selector_output_coupling_config.warmup_steps,
        signed_candidate_selector_output_coupling_config.freeze_selector_after_warmup,
        signed_candidate_selector_output_coupling_config.min_confidence
    );
    println!(
        "temporal run config | signed candidate selector probe enabled {} | softmax_temperature {} | steps {} | lr {}",
        signed_candidate_selector_probe_config.enabled,
        signed_candidate_selector_probe_config.softmax_temperature,
        signed_candidate_selector_probe_config.probe_steps,
        signed_candidate_selector_probe_config.learning_rate
    );

    match run_config.predictor_mode {
        PredictorMode::Baseline => run_with_predictor(
            run_config,
            signed_candidate_centroid_integration_config,
            signed_candidate_radius_head_config,
            signed_candidate_unit_mix_head_config,
            signed_candidate_selector_head_config,
            signed_candidate_selector_output_coupling_config,
            signed_candidate_selector_probe_config,
            make_predictor(),
        ),
        PredictorMode::Bottleneck => run_with_predictor(
            run_config,
            signed_candidate_centroid_integration_config,
            signed_candidate_radius_head_config,
            signed_candidate_unit_mix_head_config,
            signed_candidate_selector_head_config,
            signed_candidate_selector_output_coupling_config,
            signed_candidate_selector_probe_config,
            make_bottleneck_predictor(),
        ),
        PredictorMode::ResidualBottleneck => run_with_predictor(
            run_config,
            signed_candidate_centroid_integration_config,
            signed_candidate_radius_head_config,
            signed_candidate_unit_mix_head_config,
            signed_candidate_selector_head_config,
            signed_candidate_selector_output_coupling_config,
            signed_candidate_selector_probe_config,
            make_residual_bottleneck_predictor(run_config.residual_delta_scale),
        ),
        PredictorMode::StateRadius => run_with_predictor(
            run_config,
            signed_candidate_centroid_integration_config,
            signed_candidate_radius_head_config,
            signed_candidate_unit_mix_head_config,
            signed_candidate_selector_head_config,
            signed_candidate_selector_output_coupling_config,
            signed_candidate_selector_probe_config,
            make_state_radius_predictor(),
        ),
    }
}

fn run_with_predictor<P>(
    run_config: temporal_vision::TemporalRunConfig,
    signed_candidate_centroid_integration_config: SignedCandidateCentroidIntegrationRunConfig,
    signed_candidate_radius_head_config: SignedCandidateRadiusHeadRunConfig,
    signed_candidate_unit_mix_head_config: SignedCandidateUnitMixHeadRunConfig,
    signed_candidate_selector_head_config: SignedCandidateSelectorHeadRunConfig,
    signed_candidate_selector_output_coupling_config: SignedCandidateSelectorOutputCouplingRunConfig,
    signed_candidate_selector_probe_config: SignedCandidateSelectorProbeRunConfig,
    predictor: P,
) where
    P: PredictorModule,
{
    let (train_probe_t, train_probe_t1) = make_train_batch_for_config(run_config, 0);
    let (train_probe_next_t, train_probe_next_t1) = make_train_batch_for_config(run_config, 1);
    let (val_probe_t, val_probe_t1) =
        make_validation_batch_for_config(run_config, PROJECTED_VALIDATION_BASE_SEED, 0);
    let (mixed_val_probe_t, mixed_val_probe_t1, mixed_val_probe_seed) =
        make_validation_batch_with_required_motion_modes_for_config(
            run_config,
            PROJECTED_VALIDATION_BASE_SEED,
            1,
        );

    assert_temporal_contract(&train_probe_t, &train_probe_t1);
    assert_temporal_contract(&train_probe_next_t, &train_probe_next_t1);
    assert_temporal_contract(&val_probe_t, &val_probe_t1);
    assert_temporal_contract(&mixed_val_probe_t, &mixed_val_probe_t1);

    assert_ne!(train_probe_t.data, train_probe_next_t.data);
    assert_ne!(train_probe_t.data, val_probe_t.data);
    assert_ne!(val_probe_t.data, mixed_val_probe_t.data);
    assert_required_motion_modes_for_task(
        run_config.temporal_task_mode,
        &mixed_val_probe_t,
        &mixed_val_probe_t1,
    );

    print_batch_summary_for_task(
        "train probe",
        run_config.temporal_task_mode,
        &train_probe_t,
        &train_probe_t1,
    );
    print_batch_summary_for_task(
        "validation probe",
        run_config.temporal_task_mode,
        &val_probe_t,
        &val_probe_t1,
    );
    print_batch_summary_for_task(
        "validation mixed probe",
        run_config.temporal_task_mode,
        &mixed_val_probe_t,
        &mixed_val_probe_t1,
    );
    print_motion_mode_summary_for_task(
        "validation mixed probe",
        mixed_val_probe_seed,
        run_config.temporal_task_mode,
        &mixed_val_probe_t,
        &mixed_val_probe_t1,
    );

    let encoder = match run_config.compact_encoder_mode {
        CompactEncoderMode::Disabled => make_frozen_encoder(),
        CompactEncoderMode::Base => make_compact_frozen_encoder(),
        CompactEncoderMode::Stronger => make_compact_frozen_encoder_stronger(),
        CompactEncoderMode::SignedDirection => make_compact_frozen_encoder_signed_direction(),
        CompactEncoderMode::SignedDirectionMagnitude => {
            make_compact_frozen_encoder_signed_direction_magnitude()
        }
    };
    let online_projector = make_projector();
    let target_projector = online_projector.clone();
    let mut model =
        ProjectedVisionJepa::new(encoder, online_projector, target_projector, predictor)
            .with_target_projection_momentum(run_config.target_projection_momentum_at_step(1));
    let mut signed_candidate_radius_head = signed_candidate_radius_head_config
        .enabled
        .then(|| make_signed_candidate_radius_head(signed_candidate_radius_head_config.mode));
    let mut signed_candidate_unit_mix_head = signed_candidate_unit_mix_head_config
        .enabled
        .then(make_signed_candidate_unit_mix_head);
    let mut signed_candidate_selector_head = signed_candidate_selector_head_config
        .enabled
        .then(make_signed_candidate_selector_head);

    let _initial_mixed_val_z_t = model.encode(&mixed_val_probe_t);
    let initial_projection_t = model.project_latent(&train_probe_t);
    let initial_prediction = model.predict_next_projection(&train_probe_t);
    let initial_target = model.target_projection(&train_probe_t1);
    let (initial_train_prediction_loss, initial_train_regularizer_loss, initial_train_total_loss) =
        model.losses(&train_probe_t, &train_probe_t1, REGULARIZER_WEIGHT);
    let (initial_val_prediction_loss, initial_val_regularizer_loss, initial_val_total_loss) =
        projected_validation_batch_losses_from_base_seed_for_task(
            &model,
            REGULARIZER_WEIGHT,
            PROJECTED_VALIDATION_BASE_SEED,
            PROJECTED_VALIDATION_BATCHES,
            run_config.temporal_task_mode,
        );
    let initial_projection_drift = model.target_projection_drift();
    let (initial_projection_mean_abs, initial_projection_var_mean) =
        projection_stats(&initial_projection_t);
    let initial_velocity_bank_ranking = maybe_projected_velocity_bank_ranking(&model, run_config);
    let initial_signed_velocity_bank_breakdown =
        maybe_projected_signed_velocity_bank_breakdown(&model, run_config);
    let initial_signed_target_bank_separability =
        maybe_projected_signed_target_bank_separability(&model, run_config);
    let initial_signed_prediction_bank_margin =
        maybe_projected_signed_prediction_bank_margin(&model, run_config);
    let initial_signed_prediction_bank_unit_geometry =
        maybe_projected_signed_prediction_bank_unit_geometry(&model, run_config);
    let initial_signed_prediction_geometry_counterfactual =
        maybe_projected_signed_prediction_geometry_counterfactual(&model, run_config);
    let initial_signed_candidate_centroid_integration =
        maybe_projected_signed_candidate_centroid_integration(
            &model,
            run_config,
            signed_candidate_centroid_integration_config,
        );
    let initial_signed_candidate_radius_head_integration =
        maybe_projected_signed_candidate_radius_head_integration(
            &model,
            signed_candidate_radius_head.as_ref(),
            run_config,
            signed_candidate_radius_head_config,
        );
    let initial_signed_candidate_unit_mix_head_integration =
        maybe_projected_signed_candidate_unit_mix_head_integration(
            &model,
            signed_candidate_unit_mix_head.as_ref(),
            run_config,
            signed_candidate_unit_mix_head_config,
        );
    let initial_signed_candidate_selector_head_integration =
        maybe_projected_signed_candidate_selector_head_integration(
            &model,
            signed_candidate_selector_head.as_ref(),
            run_config,
            signed_candidate_selector_head_config,
        );
    let initial_signed_candidate_selector_output_coupling =
        maybe_projected_signed_candidate_selector_output_coupling_report(
            &model,
            signed_candidate_selector_head.as_ref(),
            run_config,
            signed_candidate_selector_head_config,
            signed_candidate_selector_output_coupling_config,
        );
    let initial_signed_candidate_selector_probe = maybe_projected_signed_candidate_selector_probe(
        &model,
        run_config,
        signed_candidate_selector_probe_config,
    );
    let initial_signed_margin_objective_report =
        maybe_projected_signed_margin_objective_report(&model, run_config);
    let initial_signed_bank_softmax_objective_report =
        maybe_projected_signed_bank_softmax_objective_report(&model, run_config);
    let initial_signed_radial_calibration_report =
        maybe_projected_signed_radial_calibration_report(&model, run_config);
    let initial_signed_angular_radial_objective_report =
        maybe_projected_signed_angular_radial_objective_report(&model, run_config);
    let initial_signed_state_separability =
        maybe_projected_signed_state_separability(&model, run_config);

    println!(
        "initial | projection sample0 {:?} | target {:?}",
        &initial_projection_t.data[0..PROJECTION_DIM],
        &initial_target.data[0..PROJECTION_DIM]
    );
    println!(
        "initial | proj mean_abs {:.6} | var_mean {:.6}",
        initial_projection_mean_abs, initial_projection_var_mean
    );
    print_representation_stats(
        "initial prediction",
        &representation_stats(&initial_prediction),
    );
    print_representation_stats("initial target", &representation_stats(&initial_target));
    println!(
        "initial | train pred {:.6} | reg {:.6} | total {:.6}",
        initial_train_prediction_loss, initial_train_regularizer_loss, initial_train_total_loss
    );
    println!(
        "initial | val pred {:.6} | reg {:.6} | total {:.6}",
        initial_val_prediction_loss, initial_val_regularizer_loss, initial_val_total_loss
    );
    println!("initial | target drift {:.6}", initial_projection_drift);
    print_velocity_bank_ranking("initial", initial_velocity_bank_ranking);
    print_signed_velocity_bank_breakdown("initial", initial_signed_velocity_bank_breakdown);
    print_signed_target_bank_separability("initial", initial_signed_target_bank_separability);
    print_signed_prediction_bank_margin("initial", initial_signed_prediction_bank_margin);
    print_signed_prediction_bank_unit_geometry(
        "initial",
        initial_signed_prediction_bank_unit_geometry,
    );
    print_signed_prediction_geometry_counterfactual(
        "initial",
        initial_signed_prediction_geometry_counterfactual,
    );
    print_signed_candidate_centroid_integration(
        "initial",
        initial_signed_candidate_centroid_integration,
    );
    print_signed_candidate_radius_head_integration(
        "initial",
        initial_signed_candidate_radius_head_integration,
    );
    print_signed_candidate_unit_mix_head_integration(
        "initial",
        initial_signed_candidate_unit_mix_head_integration,
    );
    print_signed_candidate_selector_head_integration(
        "initial",
        initial_signed_candidate_selector_head_integration,
    );
    print_signed_candidate_selector_output_coupling_report(
        "initial",
        initial_signed_candidate_selector_output_coupling,
    );
    print_signed_candidate_selector_probe("initial", initial_signed_candidate_selector_probe);
    print_signed_margin_objective_report("initial", initial_signed_margin_objective_report);
    print_signed_bank_softmax_objective_report(
        "initial",
        initial_signed_bank_softmax_objective_report,
    );
    print_signed_radial_calibration_report("initial", initial_signed_radial_calibration_report);
    print_signed_angular_radial_objective_report(
        "initial",
        initial_signed_angular_radial_objective_report,
    );
    print_signed_state_separability("initial", initial_signed_state_separability);

    let experiment_summary = temporal_vision::run_temporal_experiment_with_summary(
        run_config,
        &mut model,
        initial_train_total_loss,
        initial_val_total_loss,
        |model, step, should_log| {
            let (x_t, x_t1) = make_train_batch_for_config(run_config, step as u64);
            let momentum = run_config.target_projection_momentum_at_step(step);
            model.set_target_projection_momentum(momentum);
            let signed_margin_step = maybe_projected_signed_margin_objective_loss_and_grad(
                model, run_config, &x_t, &x_t1,
            );
            let signed_bank_softmax_step =
                maybe_projected_signed_bank_softmax_objective_loss_and_grad(
                    model, run_config, &x_t, &x_t1,
                );
            let signed_radial_step = maybe_projected_signed_radial_calibration_loss_and_grad(
                model, run_config, &x_t, &x_t1,
            );
            let signed_angular_radial_step =
                maybe_projected_signed_angular_radial_objective_loss_and_grad(
                    model, run_config, &x_t, &x_t1,
                );
            let signed_candidate_radius_head_step =
                maybe_train_projected_signed_candidate_radius_head(
                    model,
                    signed_candidate_radius_head.as_mut(),
                    run_config,
                    signed_candidate_radius_head_config,
                    &x_t,
                    &x_t1,
                );
            let signed_candidate_unit_mix_head_step =
                maybe_train_projected_signed_candidate_unit_mix_head(
                    model,
                    signed_candidate_unit_mix_head.as_mut(),
                    run_config,
                    signed_candidate_unit_mix_head_config,
                    &x_t,
                    &x_t1,
                );
            let signed_candidate_selector_head_step =
                if signed_candidate_selector_output_coupling_config
                    .selector_training_enabled_at_step(step)
                {
                    maybe_train_projected_signed_candidate_selector_head(
                        model,
                        signed_candidate_selector_head.as_mut(),
                        run_config,
                        signed_candidate_selector_head_config,
                        &x_t,
                        &x_t1,
                    )
                } else {
                    None
                };
            let signed_candidate_selector_output_coupling_step =
                maybe_projected_signed_candidate_selector_output_coupling_loss_and_grad(
                    model,
                    signed_candidate_selector_head.as_ref(),
                    run_config,
                    signed_candidate_selector_head_config,
                    signed_candidate_selector_output_coupling_config,
                    step,
                    &x_t,
                    &x_t1,
                );
            let extra_prediction_loss = signed_margin_step
                .as_ref()
                .map(|(report, _)| run_config.signed_margin_weight * report.weighted_loss)
                .unwrap_or(0.0)
                + signed_bank_softmax_step
                    .as_ref()
                    .map(|(report, _)| run_config.signed_bank_softmax_weight * report.loss)
                    .unwrap_or(0.0)
                + signed_radial_step
                    .as_ref()
                    .map(|(report, _)| run_config.signed_radial_weight * report.loss)
                    .unwrap_or(0.0)
                + signed_angular_radial_step
                    .as_ref()
                    .map(|(report, _)| run_config.signed_angular_radial_weight * report.loss)
                    .unwrap_or(0.0)
                + signed_candidate_selector_output_coupling_step
                    .as_ref()
                    .map(|(report, _)| {
                        signed_candidate_selector_output_coupling_config.loss_weight * report.loss
                    })
                    .unwrap_or(0.0);
            let mut extra_prediction_grad = None;
            if let Some((_, grad)) = signed_margin_step.as_ref() {
                add_scaled_extra_prediction_grad(
                    &mut extra_prediction_grad,
                    grad,
                    run_config.signed_margin_weight,
                );
            }
            if let Some((_, grad)) = signed_bank_softmax_step.as_ref() {
                add_scaled_extra_prediction_grad(
                    &mut extra_prediction_grad,
                    grad,
                    run_config.signed_bank_softmax_weight,
                );
            }
            if let Some((_, grad)) = signed_radial_step.as_ref() {
                add_scaled_extra_prediction_grad(
                    &mut extra_prediction_grad,
                    grad,
                    run_config.signed_radial_weight,
                );
            }
            if let Some((_, grad)) = signed_angular_radial_step.as_ref() {
                add_scaled_extra_prediction_grad(
                    &mut extra_prediction_grad,
                    grad,
                    run_config.signed_angular_radial_weight,
                );
            }
            if let Some((_, grad)) = signed_candidate_selector_output_coupling_step.as_ref() {
                add_scaled_extra_prediction_grad(
                    &mut extra_prediction_grad,
                    grad,
                    signed_candidate_selector_output_coupling_config.loss_weight,
                );
            }
            let (prediction_loss, regularizer_loss, projector_drift_loss, total_loss) =
                if run_config.encoder_learning_rate > 0.0 {
                    model.step_with_extra_prediction_grad(
                        &x_t,
                        &x_t1,
                        REGULARIZER_WEIGHT,
                        run_config.projector_drift_weight,
                        extra_prediction_loss,
                        extra_prediction_grad.as_ref(),
                        PREDICTOR_LR,
                        PROJECTOR_LR,
                        run_config.encoder_learning_rate,
                    )
                } else {
                    model.step_with_extra_prediction_grad(
                        &x_t,
                        &x_t1,
                        REGULARIZER_WEIGHT,
                        run_config.projector_drift_weight,
                        extra_prediction_loss,
                        extra_prediction_grad.as_ref(),
                        PREDICTOR_LR,
                        PROJECTOR_LR,
                        0.0,
                    )
                };

            if should_log {
                let current_momentum = model.target_projection_momentum();
                let (val_prediction_loss, val_regularizer_loss, val_total_loss) =
                    projected_validation_batch_losses_from_base_seed_for_task(
                        model,
                        REGULARIZER_WEIGHT,
                        PROJECTED_VALIDATION_BASE_SEED,
                        PROJECTED_VALIDATION_BATCHES,
                        run_config.temporal_task_mode,
                    );

                temporal_vision::print_projected_temporal_train_val_metrics(
                    step,
                    prediction_loss,
                    regularizer_loss,
                    total_loss,
                    current_momentum,
                    model.target_projection_drift(),
                    val_prediction_loss,
                    val_regularizer_loss,
                    val_total_loss,
                );
                if run_config.projector_drift_weight > 0.0 {
                    println!(
                        "step {:03} | projector drift regularizer {:.6}",
                        step, projector_drift_loss
                    );
                }
                if let Some((report, _)) = signed_margin_step {
                    print_signed_margin_objective_report(
                        &format!("step {:03} train", step),
                        Some(report),
                    );
                }
                if let Some((report, _)) = signed_bank_softmax_step {
                    print_signed_bank_softmax_objective_report(
                        &format!("step {:03} train", step),
                        Some(report),
                    );
                }
                if let Some((report, _)) = signed_radial_step {
                    print_signed_radial_calibration_report(
                        &format!("step {:03} train", step),
                        Some(report),
                    );
                }
                if let Some((report, _)) = signed_angular_radial_step {
                    print_signed_angular_radial_objective_report(
                        &format!("step {:03} train", step),
                        Some(report),
                    );
                }
                print_signed_centered_radius_scalar_report(
                    &format!("step {:03} train", step),
                    signed_candidate_radius_head_step,
                );
                print_signed_candidate_unit_mix_objective_report(
                    &format!("step {:03} train", step),
                    signed_candidate_unit_mix_head_step,
                );
                print_signed_candidate_selector_head_objective_report(
                    &format!("step {:03} train", step),
                    signed_candidate_selector_head_step,
                );
                print_signed_candidate_selector_output_coupling_report(
                    &format!("step {:03} train", step),
                    signed_candidate_selector_output_coupling_step
                        .as_ref()
                        .map(|(report, _)| *report),
                );
            }

            total_loss
        },
        |model| {
            projected_validation_batch_losses_from_base_seed_for_task(
                model,
                REGULARIZER_WEIGHT,
                PROJECTED_VALIDATION_BASE_SEED,
                PROJECTED_VALIDATION_BATCHES,
                run_config.temporal_task_mode,
            )
            .2
        },
    );

    let final_projection_t = model.project_latent(&train_probe_t);
    let final_pred = model.predict_next_projection(&train_probe_t);
    let final_target = model.target_projection(&train_probe_t1);
    let final_projection_drift = model.target_projection_drift();
    let (final_train_prediction_loss, final_train_regularizer_loss, final_train_total_loss) =
        model.losses(&train_probe_t, &train_probe_t1, REGULARIZER_WEIGHT);
    let (final_val_prediction_loss, final_val_regularizer_loss, final_val_total_loss) =
        projected_validation_batch_losses_from_base_seed_for_task(
            &model,
            REGULARIZER_WEIGHT,
            PROJECTED_VALIDATION_BASE_SEED,
            PROJECTED_VALIDATION_BATCHES,
            run_config.temporal_task_mode,
        );
    let (final_projection_mean_abs, final_projection_var_mean) =
        projection_stats(&final_projection_t);
    let final_velocity_bank_ranking = maybe_projected_velocity_bank_ranking(&model, run_config);
    let final_signed_velocity_bank_breakdown =
        maybe_projected_signed_velocity_bank_breakdown(&model, run_config);
    let final_signed_target_bank_separability =
        maybe_projected_signed_target_bank_separability(&model, run_config);
    let final_signed_prediction_bank_margin =
        maybe_projected_signed_prediction_bank_margin(&model, run_config);
    let final_signed_prediction_bank_unit_geometry =
        maybe_projected_signed_prediction_bank_unit_geometry(&model, run_config);
    let final_signed_prediction_geometry_counterfactual =
        maybe_projected_signed_prediction_geometry_counterfactual(&model, run_config);
    let final_signed_candidate_centroid_integration =
        maybe_projected_signed_candidate_centroid_integration(
            &model,
            run_config,
            signed_candidate_centroid_integration_config,
        );
    let final_signed_candidate_radius_head_integration =
        maybe_projected_signed_candidate_radius_head_integration(
            &model,
            signed_candidate_radius_head.as_ref(),
            run_config,
            signed_candidate_radius_head_config,
        );
    let final_signed_candidate_unit_mix_head_integration =
        maybe_projected_signed_candidate_unit_mix_head_integration(
            &model,
            signed_candidate_unit_mix_head.as_ref(),
            run_config,
            signed_candidate_unit_mix_head_config,
        );
    let final_signed_candidate_selector_head_integration =
        maybe_projected_signed_candidate_selector_head_integration(
            &model,
            signed_candidate_selector_head.as_ref(),
            run_config,
            signed_candidate_selector_head_config,
        );
    let final_signed_candidate_selector_output_coupling =
        maybe_projected_signed_candidate_selector_output_coupling_report(
            &model,
            signed_candidate_selector_head.as_ref(),
            run_config,
            signed_candidate_selector_head_config,
            signed_candidate_selector_output_coupling_config,
        );
    let final_signed_candidate_selector_probe = maybe_projected_signed_candidate_selector_probe(
        &model,
        run_config,
        signed_candidate_selector_probe_config,
    );
    let final_signed_objective_error_breakdown =
        maybe_projected_signed_objective_error_breakdown(&model, run_config);
    let final_signed_margin_objective_report =
        maybe_projected_signed_margin_objective_report(&model, run_config);
    let final_signed_bank_softmax_objective_report =
        maybe_projected_signed_bank_softmax_objective_report(&model, run_config);
    let final_signed_radial_calibration_report =
        maybe_projected_signed_radial_calibration_report(&model, run_config);
    let final_signed_angular_radial_objective_report =
        maybe_projected_signed_angular_radial_objective_report(&model, run_config);
    let final_signed_state_separability =
        maybe_projected_signed_state_separability(&model, run_config);
    let (train_reduction_threshold, validation_reduction_threshold) =
        reduction_thresholds_for_run_config(run_config);

    assert_temporal_experiment_improved(
        "projected",
        initial_train_total_loss,
        final_train_total_loss,
        initial_val_total_loss,
        final_val_total_loss,
        train_reduction_threshold,
        validation_reduction_threshold,
    );

    println!(
        "projected run summary | steps {} | train {:.6} -> {:.6} (Δ {:+.6}, improved={}) | val {:.6} -> {:.6} (Δ {:+.6}, improved={}) | target drift {:.6} -> {:.6} (Δ {:+.6})",
        experiment_summary.config.total_steps,
        experiment_summary.initial_train_loss,
        experiment_summary.final_train_loss,
        experiment_summary.train_delta(),
        experiment_summary.train_improved(),
        experiment_summary.initial_validation_loss,
        experiment_summary.final_validation_loss,
        experiment_summary.validation_delta(),
        experiment_summary.validation_improved(),
        initial_projection_drift,
        final_projection_drift,
        final_projection_drift - initial_projection_drift,
    );

    println!(
        "\nfinal | projection sample0 {:?} | pred {:?} | target {:?}",
        &final_projection_t.data[0..PROJECTION_DIM],
        &final_pred.data[0..PROJECTION_DIM],
        &final_target.data[0..PROJECTION_DIM]
    );
    println!(
        "final | proj mean_abs {:.6} | var_mean {:.6}",
        final_projection_mean_abs, final_projection_var_mean
    );
    print_representation_stats("final prediction", &representation_stats(&final_pred));
    print_representation_stats("final target", &representation_stats(&final_target));
    println!(
        "final | train pred {:.6} | reg {:.6} | total {:.6}",
        final_train_prediction_loss, final_train_regularizer_loss, final_train_total_loss
    );
    println!(
        "final | val pred {:.6} | reg {:.6} | total {:.6}",
        final_val_prediction_loss, final_val_regularizer_loss, final_val_total_loss
    );
    println!("final | target drift {:.6}", final_projection_drift);
    print_velocity_bank_ranking("final", final_velocity_bank_ranking);
    print_signed_velocity_bank_breakdown("final", final_signed_velocity_bank_breakdown);
    print_signed_target_bank_separability("final", final_signed_target_bank_separability);
    print_signed_prediction_bank_margin("final", final_signed_prediction_bank_margin);
    print_signed_prediction_bank_unit_geometry("final", final_signed_prediction_bank_unit_geometry);
    print_signed_prediction_geometry_counterfactual(
        "final",
        final_signed_prediction_geometry_counterfactual,
    );
    print_signed_candidate_centroid_integration(
        "final",
        final_signed_candidate_centroid_integration,
    );
    print_signed_candidate_radius_head_integration(
        "final",
        final_signed_candidate_radius_head_integration,
    );
    print_signed_candidate_unit_mix_head_integration(
        "final",
        final_signed_candidate_unit_mix_head_integration,
    );
    print_signed_candidate_selector_head_integration(
        "final",
        final_signed_candidate_selector_head_integration,
    );
    print_signed_candidate_selector_output_coupling_report(
        "final",
        final_signed_candidate_selector_output_coupling,
    );
    print_signed_candidate_selector_probe("final", final_signed_candidate_selector_probe);
    print_signed_objective_error_breakdown("final", final_signed_objective_error_breakdown);
    print_signed_margin_objective_report("final", final_signed_margin_objective_report);
    print_signed_bank_softmax_objective_report("final", final_signed_bank_softmax_objective_report);
    print_signed_radial_calibration_report("final", final_signed_radial_calibration_report);
    print_signed_angular_radial_objective_report(
        "final",
        final_signed_angular_radial_objective_report,
    );
    print_signed_state_separability("final", final_signed_state_separability);
}

fn scale_tensor(tensor: &Tensor, scale: f32) -> Tensor {
    Tensor {
        data: tensor.data.iter().map(|value| value * scale).collect(),
        shape: tensor.shape.clone(),
    }
}

fn add_scaled_extra_prediction_grad(accumulator: &mut Option<Tensor>, grad: &Tensor, scale: f32) {
    if scale == 0.0 {
        return;
    }

    let scaled_grad = scale_tensor(grad, scale);
    if let Some(existing) = accumulator {
        existing.add_inplace(&scaled_grad);
    } else {
        *accumulator = Some(scaled_grad);
    }
}

fn maybe_train_projected_signed_candidate_radius_head<P>(
    model: &ProjectedVisionJepa<P>,
    radius_head: Option<&mut Linear>,
    run_config: temporal_vision::TemporalRunConfig,
    head_config: SignedCandidateRadiusHeadRunConfig,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> Option<SignedCenteredRadiusScalarObjectiveReport>
where
    P: PredictorModule,
{
    let radius_head = radius_head?;
    assert!(head_config.enabled, "candidate radius head missing config");
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed candidate-radius head is only supported for signed-velocity-trail projected runs"
    );

    let (features, report, grad_out) = match head_config.mode {
        SignedCandidateRadiusHeadMode::ScalarResidual => {
            let features = projected_signed_candidate_radius_head_features(
                model,
                x_t,
                head_config.softmax_temperature,
            );
            let radius_delta = radius_head.forward(&features);
            let (report, grad_delta) = projected_signed_candidate_radius_delta_loss_and_grad(
                model,
                x_t,
                x_t1,
                &radius_delta,
                head_config.softmax_temperature,
            );

            (features, report, grad_delta)
        }
        SignedCandidateRadiusHeadMode::LogitResidual => {
            let features = projected_signed_candidate_radius_logit_head_features(
                model,
                x_t,
                head_config.softmax_temperature,
            );
            let residual_logits = radius_head.forward(&features);
            let (report, grad_logits) =
                projected_signed_candidate_radius_logit_mixing_loss_and_grad(
                    model,
                    x_t,
                    x_t1,
                    &residual_logits,
                    head_config.softmax_temperature,
                );

            (features, report, grad_logits)
        }
    };
    let scaled_grad = scale_tensor(&grad_out, head_config.loss_weight);
    let grads = radius_head.backward(&features, &scaled_grad);
    radius_head.sgd_step(&grads, head_config.learning_rate);

    Some(report)
}

fn maybe_train_projected_signed_candidate_unit_mix_head<P>(
    model: &ProjectedVisionJepa<P>,
    unit_mix_head: Option<&mut Linear>,
    run_config: temporal_vision::TemporalRunConfig,
    head_config: SignedCandidateUnitMixHeadRunConfig,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> Option<ProjectedSignedCandidateUnitMixObjectiveReport>
where
    P: PredictorModule,
{
    let unit_mix_head = unit_mix_head?;
    assert!(
        head_config.enabled,
        "candidate unit-mix head missing config"
    );
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed candidate unit-mix head is only supported for signed-velocity-trail projected runs"
    );

    let features = projected_signed_candidate_unit_mix_head_features(
        model,
        x_t,
        head_config.softmax_temperature,
    );
    let residual_logits = unit_mix_head.forward(&features);
    let (report, grad_logits) = projected_signed_candidate_unit_mix_loss_and_grad(
        model,
        x_t,
        x_t1,
        &residual_logits,
        head_config.softmax_temperature,
    );
    let scaled_grad = scale_tensor(&grad_logits, head_config.loss_weight);
    let grads = unit_mix_head.backward(&features, &scaled_grad);
    unit_mix_head.sgd_step(&grads, head_config.learning_rate);

    Some(report)
}

fn maybe_train_projected_signed_candidate_selector_head<P>(
    model: &ProjectedVisionJepa<P>,
    selector_head: Option<&mut Linear>,
    run_config: temporal_vision::TemporalRunConfig,
    head_config: SignedCandidateSelectorHeadRunConfig,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> Option<ProjectedSignedCandidateSelectorHeadObjectiveReport>
where
    P: PredictorModule,
{
    let selector_head = selector_head?;
    assert!(
        head_config.enabled,
        "candidate selector head missing config"
    );
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed candidate selector head is only supported for signed-velocity-trail projected runs"
    );

    let features = projected_signed_candidate_selector_head_features(
        model,
        x_t,
        head_config.softmax_temperature,
    );
    let selector_logits = selector_head.forward(&features);
    let (report, grad_logits) = projected_signed_candidate_selector_head_loss_and_grad(
        model,
        x_t,
        x_t1,
        &selector_logits,
        head_config.softmax_temperature,
        head_config.entropy_floor,
        head_config.entropy_weight,
        head_config.kl_weight,
    );
    let scaled_grad = scale_tensor(&grad_logits, head_config.loss_weight);
    let grads = selector_head.backward(&features, &scaled_grad);
    selector_head.sgd_step(&grads, head_config.learning_rate);

    Some(report)
}

fn maybe_projected_signed_candidate_selector_output_coupling_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    selector_head: Option<&Linear>,
    run_config: temporal_vision::TemporalRunConfig,
    head_config: SignedCandidateSelectorHeadRunConfig,
    coupling_config: SignedCandidateSelectorOutputCouplingRunConfig,
    step: usize,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> Option<(ProjectedSignedCandidateSelectorOutputCouplingReport, Tensor)>
where
    P: PredictorModule,
{
    if !coupling_config.coupling_active_at_step(step) {
        return None;
    }
    assert!(
        head_config.enabled,
        "signed candidate selector output coupling requires selector head config"
    );
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed candidate selector output coupling is only supported for signed-velocity-trail projected runs"
    );
    let selector_head =
        selector_head.expect("enabled selector output coupling missing selector head");

    let features = projected_signed_candidate_selector_head_features(
        model,
        x_t,
        head_config.softmax_temperature,
    );
    let selector_logits = selector_head.forward(&features);

    Some(if coupling_config.mode.active_normalized() {
        projected_signed_candidate_selector_active_normalized_stable_hard_full_output_coupling_loss_and_grad(
            model,
            x_t,
            x_t1,
            &selector_logits,
            coupling_config.min_confidence,
        )
    } else if coupling_config.mode.requires_correct_selector() {
        projected_signed_candidate_selector_stable_hard_full_output_coupling_loss_and_grad(
            model,
            x_t,
            x_t1,
            &selector_logits,
            coupling_config.min_confidence,
        )
    } else {
        projected_signed_candidate_selector_hard_full_output_coupling_loss_and_grad(
            model,
            x_t,
            x_t1,
            &selector_logits,
            coupling_config.min_confidence,
        )
    })
}

fn parse_arg_value<'a>(args: &'a [String], flag: &'a str) -> Option<&'a str> {
    args.windows(2).find_map(|window| {
        if window[0] == flag {
            Some(window[1].as_str())
        } else {
            None
        }
    })
}

fn parse_f32_arg(args: &[String], flag: &str) -> Option<f32> {
    parse_arg_value(args, flag)
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn parse_nonnegative_f32_arg(args: &[String], flag: &str) -> Option<f32> {
    parse_arg_value(args, flag)
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
}

fn parse_usize_arg(args: &[String], flag: &str) -> Option<usize> {
    parse_arg_value(args, flag)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn parse_nonnegative_usize_arg(args: &[String], flag: &str) -> Option<usize> {
    parse_arg_value(args, flag).and_then(|value| value.parse::<usize>().ok())
}

fn parse_positive_f32_env(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn parse_nonnegative_f32_env(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
}

fn parse_positive_usize_env(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn parse_nonnegative_usize_env(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

fn parse_bool_env(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            raw => panic!("{} must be boolean-like, got {}", name, raw),
        })
}

fn maybe_projected_velocity_bank_ranking<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<ProjectedVelocityBankRanking>
where
    P: PredictorModule,
{
    if !matches!(
        run_config.temporal_task_mode,
        TemporalTaskMode::VelocityTrail | TemporalTaskMode::SignedVelocityTrail
    ) {
        return None;
    }

    Some(projected_velocity_bank_ranking_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
        run_config.temporal_task_mode,
    ))
}

fn print_velocity_bank_ranking(tag: &str, ranking: Option<ProjectedVelocityBankRanking>) {
    if let Some(ranking) = ranking {
        println!(
            "{} | velocity bank mrr {:.6} | top1 {:.6} | mean_rank {:.6} | samples {} | candidates {}",
            tag, ranking.mrr, ranking.top1, ranking.mean_rank, ranking.samples, ranking.candidates
        );
    }
}

fn maybe_projected_signed_velocity_bank_breakdown<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<ProjectedSignedVelocityBankBreakdown>
where
    P: PredictorModule,
{
    if run_config.temporal_task_mode != TemporalTaskMode::SignedVelocityTrail {
        return None;
    }

    Some(projected_signed_velocity_bank_breakdown_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
    ))
}

fn print_signed_velocity_bank_breakdown(
    tag: &str,
    breakdown: Option<ProjectedSignedVelocityBankBreakdown>,
) {
    if let Some(breakdown) = breakdown {
        println!(
            "{} | signed velocity bank neg_mrr {:.6} | pos_mrr {:.6} | slow_mrr {:.6} | fast_mrr {:.6} | sign_top1 {:.6} | speed_top1 {:.6} | samples {} | true_neg_best_neg {} | true_neg_best_pos {} | true_pos_best_neg {} | true_pos_best_pos {} | true_slow_best_slow {} | true_slow_best_fast {} | true_fast_best_slow {} | true_fast_best_fast {}",
            tag,
            breakdown.negative_mrr,
            breakdown.positive_mrr,
            breakdown.slow_mrr,
            breakdown.fast_mrr,
            breakdown.sign_top1,
            breakdown.speed_top1,
            breakdown.samples,
            breakdown.true_neg_best_neg,
            breakdown.true_neg_best_pos,
            breakdown.true_pos_best_neg,
            breakdown.true_pos_best_pos,
            breakdown.true_slow_best_slow,
            breakdown.true_slow_best_fast,
            breakdown.true_fast_best_slow,
            breakdown.true_fast_best_fast
        );
    }
}

fn maybe_projected_signed_target_bank_separability<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<ProjectedSignedTargetBankSeparability>
where
    P: PredictorModule,
{
    if run_config.temporal_task_mode != TemporalTaskMode::SignedVelocityTrail {
        return None;
    }

    Some(projected_signed_target_bank_separability_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
    ))
}

fn print_signed_target_bank_separability(
    tag: &str,
    separability: Option<ProjectedSignedTargetBankSeparability>,
) {
    if let Some(separability) = separability {
        println!(
            "{} | target bank separability oracle_mrr {:.6} | top1 {:.6} | true_dist {:.6} | true_dist_max {:.6} | nearest_wrong {:.6} | nearest_wrong_min {:.6} | margin {:.6} | margin_min {:.6} | neg_nearest_wrong {:.6} | pos_nearest_wrong {:.6} | slow_nearest_wrong {:.6} | fast_nearest_wrong {:.6} | sign_margin {:.6} | speed_margin {:.6} | samples {}",
            tag,
            separability.oracle_mrr,
            separability.oracle_top1,
            separability.true_distance,
            separability.max_true_distance,
            separability.nearest_wrong_distance,
            separability.min_nearest_wrong_distance,
            separability.margin,
            separability.min_margin,
            separability.negative_nearest_wrong_distance,
            separability.positive_nearest_wrong_distance,
            separability.slow_nearest_wrong_distance,
            separability.fast_nearest_wrong_distance,
            separability.sign_margin,
            separability.speed_margin,
            separability.samples,
        );
    }
}

fn maybe_projected_signed_prediction_bank_margin<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<ProjectedSignedPredictionBankMargin>
where
    P: PredictorModule,
{
    if run_config.temporal_task_mode != TemporalTaskMode::SignedVelocityTrail {
        return None;
    }

    Some(projected_signed_prediction_bank_margin_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
    ))
}

fn print_signed_prediction_bank_margin(
    tag: &str,
    margin: Option<ProjectedSignedPredictionBankMargin>,
) {
    if let Some(margin) = margin {
        println!(
            "{} | signed prediction bank margin true_distance {:.6} | nearest_wrong_distance {:.6} | margin {:.6} | min_margin {:.6} | positive_margin_rate {:.6} | sign_margin {:.6} | speed_margin {:.6} | samples {}",
            tag,
            margin.true_distance,
            margin.nearest_wrong_distance,
            margin.margin,
            margin.min_margin,
            margin.positive_margin_rate,
            margin.sign_margin,
            margin.speed_margin,
            margin.samples,
        );
    }
}

fn maybe_projected_signed_prediction_bank_unit_geometry<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<ProjectedSignedPredictionBankUnitGeometry>
where
    P: PredictorModule,
{
    if run_config.temporal_task_mode != TemporalTaskMode::SignedVelocityTrail {
        return None;
    }

    Some(
        projected_signed_prediction_bank_unit_geometry_from_base_seed(
            model,
            PROJECTED_VALIDATION_BASE_SEED,
            PROJECTED_VALIDATION_BATCHES,
        ),
    )
}

fn print_signed_prediction_bank_unit_geometry(
    tag: &str,
    geometry: Option<ProjectedSignedPredictionBankUnitGeometry>,
) {
    if let Some(geometry) = geometry {
        println!(
            "{} | signed prediction bank unit geometry mrr {:.6} | top1 {:.6} | true_distance {:.6} | nearest_wrong_distance {:.6} | margin {:.6} | positive_margin_rate {:.6} | sign_margin {:.6} | speed_margin {:.6} | prediction_center_norm {:.6} | true_target_center_norm {:.6} | samples {} | candidates {}",
            tag,
            geometry.mrr,
            geometry.top1,
            geometry.true_distance,
            geometry.nearest_wrong_distance,
            geometry.margin,
            geometry.positive_margin_rate,
            geometry.sign_margin,
            geometry.speed_margin,
            geometry.prediction_center_norm,
            geometry.true_target_center_norm,
            geometry.samples,
            geometry.candidates,
        );
    }
}

fn maybe_projected_signed_prediction_geometry_counterfactual<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<ProjectedSignedPredictionGeometryCounterfactual>
where
    P: PredictorModule,
{
    if run_config.temporal_task_mode != TemporalTaskMode::SignedVelocityTrail {
        return None;
    }

    Some(
        projected_signed_prediction_geometry_counterfactual_from_base_seed(
            model,
            PROJECTED_VALIDATION_BASE_SEED,
            PROJECTED_VALIDATION_BATCHES,
        ),
    )
}

fn print_signed_prediction_geometry_counterfactual(
    tag: &str,
    counterfactual: Option<ProjectedSignedPredictionGeometryCounterfactual>,
) {
    if let Some(counterfactual) = counterfactual {
        println!(
            "{} | signed prediction geometry counterfactual oracle_radius_mrr {:.6} | oracle_radius_top1 {:.6} | oracle_radius_margin {:.6} | oracle_radius_positive_margin_rate {:.6} | oracle_radius_sign_margin {:.6} | oracle_radius_speed_margin {:.6} | oracle_radius_norm_ratio {:.6} | oracle_angle_mrr {:.6} | oracle_angle_top1 {:.6} | oracle_angle_margin {:.6} | oracle_angle_positive_margin_rate {:.6} | oracle_angle_sign_margin {:.6} | oracle_angle_speed_margin {:.6} | oracle_angle_norm_ratio {:.6} | support_global_rescale_mrr {:.6} | support_global_rescale_top1 {:.6} | support_global_rescale_margin {:.6} | support_global_rescale_positive_margin_rate {:.6} | support_global_rescale_sign_margin {:.6} | support_global_rescale_speed_margin {:.6} | support_global_rescale_norm_ratio {:.6} | support_norm_ratio {:.6} | support_samples {} | query_samples {} | candidates {}",
            tag,
            counterfactual.oracle_radius.mrr,
            counterfactual.oracle_radius.top1,
            counterfactual.oracle_radius.margin,
            counterfactual.oracle_radius.positive_margin_rate,
            counterfactual.oracle_radius.sign_margin,
            counterfactual.oracle_radius.speed_margin,
            counterfactual.oracle_radius.norm_ratio,
            counterfactual.oracle_angle.mrr,
            counterfactual.oracle_angle.top1,
            counterfactual.oracle_angle.margin,
            counterfactual.oracle_angle.positive_margin_rate,
            counterfactual.oracle_angle.sign_margin,
            counterfactual.oracle_angle.speed_margin,
            counterfactual.oracle_angle.norm_ratio,
            counterfactual.support_global_rescale.mrr,
            counterfactual.support_global_rescale.top1,
            counterfactual.support_global_rescale.margin,
            counterfactual.support_global_rescale.positive_margin_rate,
            counterfactual.support_global_rescale.sign_margin,
            counterfactual.support_global_rescale.speed_margin,
            counterfactual.support_global_rescale.norm_ratio,
            counterfactual.support_norm_ratio,
            counterfactual.support_samples,
            counterfactual.query_samples,
            counterfactual.candidates,
        );
    }
}

fn maybe_projected_signed_candidate_centroid_integration<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
    integration_config: SignedCandidateCentroidIntegrationRunConfig,
) -> Option<ProjectedSignedCandidateCentroidIntegration>
where
    P: PredictorModule,
{
    if !integration_config.enabled {
        return None;
    }
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed candidate-centroid integration is only supported for signed-velocity-trail projected runs"
    );

    Some(
        projected_signed_candidate_centroid_integration_from_base_seed(
            model,
            PROJECTED_VALIDATION_BASE_SEED,
            PROJECTED_VALIDATION_BATCHES,
            integration_config.softmax_temperature,
        ),
    )
}

fn print_signed_candidate_centroid_integration(
    tag: &str,
    integration: Option<ProjectedSignedCandidateCentroidIntegration>,
) {
    if let Some(integration) = integration {
        println!(
            "{} | signed candidate-centroid integration mean_radius_mrr {:.6} | mean_radius_top1 {:.6} | mean_radius_margin {:.6} | mean_radius_positive_margin_rate {:.6} | mean_radius_sign_margin {:.6} | mean_radius_speed_margin {:.6} | mean_radius_norm_ratio {:.6} | nearest_unit_radius_mrr {:.6} | nearest_unit_radius_top1 {:.6} | nearest_unit_radius_margin {:.6} | nearest_unit_radius_positive_margin_rate {:.6} | nearest_unit_radius_sign_margin {:.6} | nearest_unit_radius_speed_margin {:.6} | nearest_unit_radius_norm_ratio {:.6} | softmax_radius_mrr {:.6} | softmax_radius_top1 {:.6} | softmax_radius_margin {:.6} | softmax_radius_positive_margin_rate {:.6} | softmax_radius_sign_margin {:.6} | softmax_radius_speed_margin {:.6} | softmax_radius_norm_ratio {:.6} | softmax_temperature {:.6} | samples {} | candidates {}",
            tag,
            integration.mean_radius.mrr,
            integration.mean_radius.top1,
            integration.mean_radius.margin,
            integration.mean_radius.positive_margin_rate,
            integration.mean_radius.sign_margin,
            integration.mean_radius.speed_margin,
            integration.mean_radius.norm_ratio,
            integration.nearest_unit_radius.mrr,
            integration.nearest_unit_radius.top1,
            integration.nearest_unit_radius.margin,
            integration.nearest_unit_radius.positive_margin_rate,
            integration.nearest_unit_radius.sign_margin,
            integration.nearest_unit_radius.speed_margin,
            integration.nearest_unit_radius.norm_ratio,
            integration.softmax_radius.mrr,
            integration.softmax_radius.top1,
            integration.softmax_radius.margin,
            integration.softmax_radius.positive_margin_rate,
            integration.softmax_radius.sign_margin,
            integration.softmax_radius.speed_margin,
            integration.softmax_radius.norm_ratio,
            integration.softmax_temperature,
            integration.samples,
            integration.candidates,
        );
    }
}

fn maybe_projected_signed_candidate_radius_head_integration<P>(
    model: &ProjectedVisionJepa<P>,
    radius_head: Option<&Linear>,
    run_config: temporal_vision::TemporalRunConfig,
    head_config: SignedCandidateRadiusHeadRunConfig,
) -> Option<ProjectedSignedCandidateRadiusHeadIntegration>
where
    P: PredictorModule,
{
    let radius_head = radius_head?;
    assert!(head_config.enabled, "candidate radius head missing config");
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed candidate-radius head is only supported for signed-velocity-trail projected runs"
    );

    Some(match head_config.mode {
        SignedCandidateRadiusHeadMode::ScalarResidual => {
            projected_signed_candidate_radius_head_integration_from_base_seed(
                model,
                radius_head,
                PROJECTED_VALIDATION_BASE_SEED,
                PROJECTED_VALIDATION_BATCHES,
                head_config.softmax_temperature,
            )
        }
        SignedCandidateRadiusHeadMode::LogitResidual => {
            projected_signed_candidate_radius_logit_head_integration_from_base_seed(
                model,
                radius_head,
                PROJECTED_VALIDATION_BASE_SEED,
                PROJECTED_VALIDATION_BATCHES,
                head_config.softmax_temperature,
            )
        }
    })
}

fn print_signed_candidate_radius_head_integration(
    tag: &str,
    integration: Option<ProjectedSignedCandidateRadiusHeadIntegration>,
) {
    if let Some(integration) = integration {
        println!(
            "{} | signed candidate-radius head integration learned_radius_mrr {:.6} | learned_radius_top1 {:.6} | learned_radius_margin {:.6} | learned_radius_positive_margin_rate {:.6} | learned_radius_sign_margin {:.6} | learned_radius_speed_margin {:.6} | learned_radius_norm_ratio {:.6} | scalar_loss {:.6} | scalar_prediction_radius {:.6} | scalar_target_radius {:.6} | scalar_radius_ratio {:.6} | softmax_temperature {:.6} | samples {} | candidates {}",
            tag,
            integration.learned_radius.mrr,
            integration.learned_radius.top1,
            integration.learned_radius.margin,
            integration.learned_radius.positive_margin_rate,
            integration.learned_radius.sign_margin,
            integration.learned_radius.speed_margin,
            integration.learned_radius.norm_ratio,
            integration.scalar_report.loss,
            integration.scalar_report.prediction_radius,
            integration.scalar_report.target_radius,
            integration.scalar_report.radius_ratio,
            integration.softmax_temperature,
            integration.samples,
            integration.candidates,
        );
    }
}

fn maybe_projected_signed_candidate_unit_mix_head_integration<P>(
    model: &ProjectedVisionJepa<P>,
    unit_mix_head: Option<&Linear>,
    run_config: temporal_vision::TemporalRunConfig,
    head_config: SignedCandidateUnitMixHeadRunConfig,
) -> Option<ProjectedSignedCandidateUnitMixHeadIntegration>
where
    P: PredictorModule,
{
    let unit_mix_head = unit_mix_head?;
    assert!(
        head_config.enabled,
        "candidate unit-mix head missing config"
    );
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed candidate unit-mix head is only supported for signed-velocity-trail projected runs"
    );

    Some(
        projected_signed_candidate_unit_mix_head_integration_from_base_seed(
            model,
            unit_mix_head,
            PROJECTED_VALIDATION_BASE_SEED,
            PROJECTED_VALIDATION_BATCHES,
            head_config.softmax_temperature,
        ),
    )
}

fn maybe_projected_signed_candidate_selector_head_integration<P>(
    model: &ProjectedVisionJepa<P>,
    selector_head: Option<&Linear>,
    run_config: temporal_vision::TemporalRunConfig,
    head_config: SignedCandidateSelectorHeadRunConfig,
) -> Option<ProjectedSignedCandidateSelectorHeadIntegration>
where
    P: PredictorModule,
{
    if !head_config.enabled {
        return None;
    }
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed candidate selector head is only supported for signed-velocity-trail projected runs"
    );
    let selector_head = selector_head.expect("enabled candidate selector head missing head");

    Some(
        projected_signed_candidate_selector_head_integration_from_base_seed(
            model,
            selector_head,
            PROJECTED_VALIDATION_BASE_SEED,
            PROJECTED_VALIDATION_BATCHES,
            head_config.softmax_temperature,
            run_config.total_steps,
            head_config.learning_rate,
            head_config.entropy_floor,
            head_config.entropy_weight,
            head_config.kl_weight,
        ),
    )
}

fn maybe_projected_signed_candidate_selector_output_coupling_report<P>(
    model: &ProjectedVisionJepa<P>,
    selector_head: Option<&Linear>,
    run_config: temporal_vision::TemporalRunConfig,
    head_config: SignedCandidateSelectorHeadRunConfig,
    coupling_config: SignedCandidateSelectorOutputCouplingRunConfig,
) -> Option<ProjectedSignedCandidateSelectorOutputCouplingReport>
where
    P: PredictorModule,
{
    if !coupling_config.mode.enabled() {
        return None;
    }
    assert!(
        head_config.enabled,
        "signed candidate selector output coupling requires selector head config"
    );
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed candidate selector output coupling is only supported for signed-velocity-trail projected runs"
    );
    let selector_head =
        selector_head.expect("enabled selector output coupling missing selector head");

    Some(
        projected_signed_candidate_selector_hard_full_output_coupling_report_from_base_seed(
            model,
            selector_head,
            PROJECTED_VALIDATION_BASE_SEED,
            PROJECTED_VALIDATION_BATCHES,
            head_config.softmax_temperature,
            coupling_config.min_confidence,
            coupling_config.mode.requires_correct_selector(),
            coupling_config.mode.active_normalized(),
        ),
    )
}

fn print_signed_candidate_unit_mix_head_integration(
    tag: &str,
    integration: Option<ProjectedSignedCandidateUnitMixHeadIntegration>,
) {
    if let Some(integration) = integration {
        println!(
            "{} | signed candidate unit-mix head integration learned_mix_mrr {:.6} | learned_mix_top1 {:.6} | learned_mix_margin {:.6} | learned_mix_positive_margin_rate {:.6} | learned_mix_sign_margin {:.6} | learned_mix_speed_margin {:.6} | learned_mix_norm_ratio {:.6} | objective_loss {:.6} | objective_prediction_radius {:.6} | objective_target_radius {:.6} | objective_radius_ratio {:.6} | objective_entropy {:.6} | objective_max_weight {:.6} | objective_true_weight {:.6} | softmax_temperature {:.6} | samples {} | candidates {}",
            tag,
            integration.learned_mix.mrr,
            integration.learned_mix.top1,
            integration.learned_mix.margin,
            integration.learned_mix.positive_margin_rate,
            integration.learned_mix.sign_margin,
            integration.learned_mix.speed_margin,
            integration.learned_mix.norm_ratio,
            integration.objective_report.loss,
            integration.objective_report.prediction_radius,
            integration.objective_report.target_radius,
            integration.objective_report.radius_ratio,
            integration.objective_report.entropy,
            integration.objective_report.max_weight,
            integration.objective_report.true_weight,
            integration.softmax_temperature,
            integration.samples,
            integration.candidates,
        );
    }
}

fn print_signed_candidate_selector_head_integration(
    tag: &str,
    integration: Option<ProjectedSignedCandidateSelectorHeadIntegration>,
) {
    if let Some(integration) = integration {
        let legacy_readout = integration.readout_diagnostics.selector_soft_unit_mix;
        println!(
            "{} | signed candidate selector head integration learned_selector_mrr {:.6} | learned_selector_top1 {:.6} | learned_selector_margin {:.6} | learned_selector_positive_margin_rate {:.6} | learned_selector_sign_margin {:.6} | learned_selector_speed_margin {:.6} | learned_selector_norm_ratio {:.6} | objective_loss {:.6} | objective_ce {:.6} | objective_entropy_reg {:.6} | objective_kl_to_prior {:.6} | objective_entropy {:.6} | objective_true_probability {:.6} | objective_max_probability {:.6} | softmax_temperature {:.6} | selector_steps {} | lr {:.6} | entropy_floor {:.6} | entropy_weight {:.6} | kl_weight {:.6} | support_samples {} | query_samples {} | candidates {}",
            tag,
            legacy_readout.mrr,
            legacy_readout.top1,
            legacy_readout.margin,
            legacy_readout.positive_margin_rate,
            legacy_readout.sign_margin,
            legacy_readout.speed_margin,
            legacy_readout.norm_ratio,
            integration.objective_report.loss,
            integration.objective_report.cross_entropy_loss,
            integration.objective_report.entropy_regularization_loss,
            integration.objective_report.kl_to_prior_loss,
            integration.objective_report.entropy,
            integration.objective_report.true_probability,
            integration.objective_report.max_probability,
            integration.softmax_temperature,
            integration.selector_steps,
            integration.learning_rate,
            integration.entropy_floor,
            integration.entropy_regularization_weight,
            integration.kl_to_prior_weight,
            integration.support_samples,
            integration.query_samples,
            integration.candidates,
        );
        print_signed_candidate_selector_readout_diagnostics(tag, integration.readout_diagnostics);
        println!(
            "{} | signed candidate selector head integration objective_loss {:.6} | objective_ce {:.6} | objective_entropy_reg {:.6} | objective_kl_to_prior {:.6} | objective_entropy {:.6} | objective_true_probability {:.6} | objective_max_probability {:.6} | softmax_temperature {:.6} | selector_steps {} | lr {:.6} | entropy_floor {:.6} | entropy_weight {:.6} | kl_weight {:.6} | support_samples {} | query_samples {} | samples {} | candidates {}",
            tag,
            integration.objective_report.loss,
            integration.objective_report.cross_entropy_loss,
            integration.objective_report.entropy_regularization_loss,
            integration.objective_report.kl_to_prior_loss,
            integration.objective_report.entropy,
            integration.objective_report.true_probability,
            integration.objective_report.max_probability,
            integration.softmax_temperature,
            integration.selector_steps,
            integration.learning_rate,
            integration.entropy_floor,
            integration.entropy_regularization_weight,
            integration.kl_to_prior_weight,
            integration.support_samples,
            integration.query_samples,
            integration.samples,
            integration.candidates,
        );
    }
}

fn print_signed_candidate_selector_readout_diagnostics(
    tag: &str,
    readout: ProjectedSignedCandidateSelectorReadoutDiagnostics,
) {
    print_signed_candidate_selector_readout_metrics(
        tag,
        "base_prediction",
        readout.base_prediction,
    );
    print_signed_candidate_selector_readout_metrics(
        tag,
        "selector_soft_unit_mix",
        readout.selector_soft_unit_mix,
    );
    print_signed_candidate_selector_readout_metrics(
        tag,
        "selector_soft_full",
        readout.selector_soft_full,
    );
    print_signed_candidate_selector_readout_metrics(
        tag,
        "selector_hard_full",
        readout.selector_hard_full,
    );
    print_signed_candidate_selector_readout_metrics(
        tag,
        "selector_soft_radius",
        readout.selector_soft_radius,
    );
    print_signed_candidate_selector_readout_metrics(
        tag,
        "selector_hard_radius",
        readout.selector_hard_radius,
    );
}

fn print_signed_candidate_selector_readout_metrics(
    tag: &str,
    field: &str,
    metrics: ProjectedSignedPredictionCounterfactualMetrics,
) {
    println!(
        "{} | signed candidate selector readout {}_mrr {:.6} | {}_top1 {:.6} | {}_margin {:.6} | {}_positive_margin_rate {:.6} | {}_sign_margin {:.6} | {}_speed_margin {:.6} | {}_norm_ratio {:.6}",
        tag,
        field,
        metrics.mrr,
        field,
        metrics.top1,
        field,
        metrics.margin,
        field,
        metrics.positive_margin_rate,
        field,
        metrics.sign_margin,
        field,
        metrics.speed_margin,
        field,
        metrics.norm_ratio,
    );
}

fn print_signed_candidate_selector_output_coupling_report(
    tag: &str,
    report: Option<ProjectedSignedCandidateSelectorOutputCouplingReport>,
) {
    if let Some(report) = report {
        println!(
            "{} | signed candidate selector output coupling loss {:.6} | detached_target hard-full | selector_max_probability {:.6} | active_rate {:.6} | base_prediction_top1 {:.6} | base_prediction_margin {:.6} | base_prediction_norm_ratio {:.6} | selector_hard_full_top1 {:.6} | selector_hard_full_margin {:.6} | selector_hard_full_norm_ratio {:.6} | active_samples {} | samples {} | candidates {}",
            tag,
            report.loss,
            report.selector_max_probability,
            report.active_rate,
            report.base_prediction_top1,
            report.base_prediction_margin,
            report.base_prediction_norm_ratio,
            report.selector_hard_full_top1,
            report.selector_hard_full_margin,
            report.selector_hard_full_norm_ratio,
            report.active_samples,
            report.samples,
            report.candidates,
        );
    }
}

fn maybe_projected_signed_candidate_selector_probe<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
    probe_config: SignedCandidateSelectorProbeRunConfig,
) -> Option<ProjectedSignedCandidateSelectorProbe>
where
    P: PredictorModule,
{
    if !probe_config.enabled {
        return None;
    }
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed candidate selector probe is only supported for signed-velocity-trail projected runs"
    );

    Some(projected_signed_candidate_selector_probe_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
        probe_config.softmax_temperature,
        probe_config.probe_steps,
        probe_config.learning_rate,
    ))
}

fn print_signed_candidate_selector_probe(
    tag: &str,
    probe: Option<ProjectedSignedCandidateSelectorProbe>,
) {
    if let Some(probe) = probe {
        println!(
            "{} | signed candidate selector probe prior_loss {:.6} | prior_mrr {:.6} | prior_top1 {:.6} | prior_true_probability {:.6} | prior_entropy {:.6} | support_trained_loss {:.6} | support_trained_mrr {:.6} | support_trained_top1 {:.6} | support_trained_true_probability {:.6} | support_trained_entropy {:.6} | query_trained_loss {:.6} | query_trained_mrr {:.6} | query_trained_top1 {:.6} | query_trained_true_probability {:.6} | query_trained_entropy {:.6} | softmax_temperature {:.6} | steps {} | lr {:.6} | support_samples {} | query_samples {} | candidates {}",
            tag,
            probe.prior_anchor.loss,
            probe.prior_anchor.mrr,
            probe.prior_anchor.top1,
            probe.prior_anchor.true_probability,
            probe.prior_anchor.entropy,
            probe.support_trained_probe.loss,
            probe.support_trained_probe.mrr,
            probe.support_trained_probe.top1,
            probe.support_trained_probe.true_probability,
            probe.support_trained_probe.entropy,
            probe.trained_probe.loss,
            probe.trained_probe.mrr,
            probe.trained_probe.top1,
            probe.trained_probe.true_probability,
            probe.trained_probe.entropy,
            probe.softmax_temperature,
            probe.probe_steps,
            probe.learning_rate,
            probe.support_samples,
            probe.query_samples,
            probe.candidates,
        );
    }
}

fn print_signed_candidate_unit_mix_objective_report(
    tag: &str,
    report: Option<ProjectedSignedCandidateUnitMixObjectiveReport>,
) {
    if let Some(report) = report {
        println!(
            "{} | signed candidate unit-mix objective loss {:.6} | prediction_radius {:.6} | target_radius {:.6} | radius_ratio {:.6} | entropy {:.6} | max_weight {:.6} | true_weight {:.6} | samples {}",
            tag,
            report.loss,
            report.prediction_radius,
            report.target_radius,
            report.radius_ratio,
            report.entropy,
            report.max_weight,
            report.true_weight,
            report.samples,
        );
    }
}

fn print_signed_candidate_selector_head_objective_report(
    tag: &str,
    report: Option<ProjectedSignedCandidateSelectorHeadObjectiveReport>,
) {
    if let Some(report) = report {
        println!(
            "{} | signed candidate selector head objective loss {:.6} | ce {:.6} | entropy_reg {:.6} | kl_to_prior {:.6} | entropy {:.6} | true_probability {:.6} | max_probability {:.6} | samples {} | candidates {}",
            tag,
            report.loss,
            report.cross_entropy_loss,
            report.entropy_regularization_loss,
            report.kl_to_prior_loss,
            report.entropy,
            report.true_probability,
            report.max_probability,
            report.samples,
            report.candidates,
        );
    }
}

fn print_signed_centered_radius_scalar_report(
    tag: &str,
    report: Option<SignedCenteredRadiusScalarObjectiveReport>,
) {
    if let Some(report) = report {
        println!(
            "{} | signed centered radius scalar objective loss {:.6} | prediction_radius {:.6} | target_radius {:.6} | radius_ratio {:.6} | samples {}",
            tag,
            report.loss,
            report.prediction_radius,
            report.target_radius,
            report.radius_ratio,
            report.samples,
        );
    }
}

fn maybe_projected_signed_objective_error_breakdown<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<ProjectedSignedObjectiveErrorBreakdown>
where
    P: PredictorModule,
{
    if run_config.temporal_task_mode != TemporalTaskMode::SignedVelocityTrail {
        return None;
    }

    Some(projected_signed_objective_error_breakdown_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
    ))
}

fn maybe_projected_signed_margin_objective_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> Option<(SignedMarginObjectiveReport, Tensor)>
where
    P: PredictorModule,
{
    if run_config.signed_margin_weight == 0.0 {
        return None;
    }
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed margin objective is only supported for signed-velocity-trail projected runs"
    );

    Some(projected_signed_margin_objective_loss_and_grad(
        model,
        x_t,
        x_t1,
        run_config.signed_margin_config,
    ))
}

fn maybe_projected_signed_margin_objective_report<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<SignedMarginObjectiveReport>
where
    P: PredictorModule,
{
    if run_config.signed_margin_weight == 0.0 {
        return None;
    }
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed margin objective is only supported for signed-velocity-trail projected runs"
    );

    Some(projected_signed_margin_objective_report_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
        run_config.signed_margin_config,
    ))
}

fn maybe_projected_signed_bank_softmax_objective_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> Option<(SignedBankSoftmaxObjectiveReport, Tensor)>
where
    P: PredictorModule,
{
    if run_config.signed_bank_softmax_weight == 0.0 {
        return None;
    }
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed bank softmax objective is only supported for signed-velocity-trail projected runs"
    );

    Some(projected_signed_bank_softmax_objective_loss_and_grad(
        model,
        x_t,
        x_t1,
        run_config.signed_bank_softmax_config,
    ))
}

fn maybe_projected_signed_bank_softmax_objective_report<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<SignedBankSoftmaxObjectiveReport>
where
    P: PredictorModule,
{
    if run_config.signed_bank_softmax_weight == 0.0 {
        return None;
    }
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed bank softmax objective is only supported for signed-velocity-trail projected runs"
    );

    Some(
        projected_signed_bank_softmax_objective_report_from_base_seed(
            model,
            PROJECTED_VALIDATION_BASE_SEED,
            PROJECTED_VALIDATION_BATCHES,
            run_config.signed_bank_softmax_config,
        ),
    )
}

fn maybe_projected_signed_radial_calibration_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> Option<(SignedRadialCalibrationReport, Tensor)>
where
    P: PredictorModule,
{
    if run_config.signed_radial_weight == 0.0 {
        return None;
    }
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed radial calibration objective is only supported for signed-velocity-trail projected runs"
    );

    Some(projected_signed_radial_calibration_loss_and_grad(
        model, x_t, x_t1,
    ))
}

fn maybe_projected_signed_radial_calibration_report<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<SignedRadialCalibrationReport>
where
    P: PredictorModule,
{
    if run_config.signed_radial_weight == 0.0 {
        return None;
    }
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed radial calibration objective is only supported for signed-velocity-trail projected runs"
    );

    Some(projected_signed_radial_calibration_report_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
    ))
}

fn maybe_projected_signed_angular_radial_objective_loss_and_grad<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
    x_t: &Tensor,
    x_t1: &Tensor,
) -> Option<(SignedAngularRadialObjectiveReport, Tensor)>
where
    P: PredictorModule,
{
    if run_config.signed_angular_radial_weight == 0.0 {
        return None;
    }
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed angular-radial objective is only supported for signed-velocity-trail projected runs"
    );

    Some(projected_signed_angular_radial_objective_loss_and_grad(
        model,
        x_t,
        x_t1,
        run_config.signed_angular_radial_config,
    ))
}

fn maybe_projected_signed_angular_radial_objective_report<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<SignedAngularRadialObjectiveReport>
where
    P: PredictorModule,
{
    if run_config.signed_angular_radial_weight == 0.0 {
        return None;
    }
    assert_eq!(
        run_config.temporal_task_mode,
        TemporalTaskMode::SignedVelocityTrail,
        "signed angular-radial objective is only supported for signed-velocity-trail projected runs"
    );

    Some(
        projected_signed_angular_radial_objective_report_from_base_seed(
            model,
            PROJECTED_VALIDATION_BASE_SEED,
            PROJECTED_VALIDATION_BATCHES,
            run_config.signed_angular_radial_config,
        ),
    )
}

fn maybe_projected_signed_state_separability<P>(
    model: &ProjectedVisionJepa<P>,
    run_config: temporal_vision::TemporalRunConfig,
) -> Option<ProjectedSignedStateSeparability>
where
    P: PredictorModule,
{
    if run_config.temporal_task_mode != TemporalTaskMode::SignedVelocityTrail {
        return None;
    }

    Some(projected_signed_state_separability_from_base_seed(
        model,
        PROJECTED_VALIDATION_BASE_SEED,
        PROJECTED_VALIDATION_BATCHES,
    ))
}

fn print_signed_objective_error_breakdown(
    tag: &str,
    breakdown: Option<ProjectedSignedObjectiveErrorBreakdown>,
) {
    if let Some(breakdown) = breakdown {
        println!(
            "{} | signed objective error breakdown all_loss {:.6} | dx_neg2_loss {:.6} | dx_neg1_loss {:.6} | dx_pos1_loss {:.6} | dx_pos2_loss {:.6} | neg_loss {:.6} | pos_loss {:.6} | slow_loss {:.6} | fast_loss {:.6} | sign_gap {:.6} | speed_gap {:.6} | samples {} | dx_neg2_samples {} | dx_neg1_samples {} | dx_pos1_samples {} | dx_pos2_samples {}",
            tag,
            breakdown.all_loss,
            breakdown.dx_neg2_loss,
            breakdown.dx_neg1_loss,
            breakdown.dx_pos1_loss,
            breakdown.dx_pos2_loss,
            breakdown.neg_loss,
            breakdown.pos_loss,
            breakdown.slow_loss,
            breakdown.fast_loss,
            breakdown.sign_gap,
            breakdown.speed_gap,
            breakdown.samples,
            breakdown.dx_neg2_samples,
            breakdown.dx_neg1_samples,
            breakdown.dx_pos1_samples,
            breakdown.dx_pos2_samples,
        );
    }
}

fn print_signed_state_separability(
    tag: &str,
    separability: Option<ProjectedSignedStateSeparability>,
) {
    if let Some(separability) = separability {
        println!(
            "{} | signed state separability latent_mrr {:.6} | latent_top1 {:.6} | latent_sign_top1 {:.6} | latent_mean_rank {:.6} | projection_mrr {:.6} | projection_top1 {:.6} | projection_sign_top1 {:.6} | projection_mean_rank {:.6} | support_samples {} | query_samples {} | candidates {}",
            tag,
            separability.latent_mrr,
            separability.latent_top1,
            separability.latent_sign_top1,
            separability.latent_mean_rank,
            separability.projection_mrr,
            separability.projection_top1,
            separability.projection_sign_top1,
            separability.projection_mean_rank,
            separability.support_samples,
            separability.query_samples,
            separability.candidates,
        );
    }
}

fn print_signed_margin_objective_report(tag: &str, report: Option<SignedMarginObjectiveReport>) {
    if let Some(report) = report {
        println!(
            "{} | signed margin objective bank_loss {:.6} | sign_loss {:.6} | speed_loss {:.6} | weighted_loss {:.6} | active_bank_rate {:.6} | active_sign_rate {:.6} | active_speed_rate {:.6} | samples {}",
            tag,
            report.bank_loss,
            report.sign_loss,
            report.speed_loss,
            report.weighted_loss,
            report.active_bank_pairs as f32 / report.bank_pairs as f32,
            report.active_sign_pairs as f32 / report.sign_pairs as f32,
            report.active_speed_pairs as f32 / report.speed_pairs as f32,
            report.samples,
        );
    }
}

fn print_signed_bank_softmax_objective_report(
    tag: &str,
    report: Option<SignedBankSoftmaxObjectiveReport>,
) {
    if let Some(report) = report {
        println!(
            "{} | signed bank softmax objective loss {:.6} | top1 {:.6} | true_probability {:.6} | samples {}",
            tag, report.loss, report.top1, report.mean_true_probability, report.samples,
        );
    }
}

fn print_signed_radial_calibration_report(
    tag: &str,
    report: Option<SignedRadialCalibrationReport>,
) {
    if let Some(report) = report {
        println!(
            "{} | signed radial calibration loss {:.6} | prediction_norm {:.6} | target_norm {:.6} | norm_ratio {:.6} | samples {}",
            tag,
            report.loss,
            report.prediction_norm,
            report.target_norm,
            report.norm_ratio,
            report.samples,
        );
    }
}

fn print_signed_angular_radial_objective_report(
    tag: &str,
    report: Option<SignedAngularRadialObjectiveReport>,
) {
    if let Some(report) = report {
        println!(
            "{} | signed angular-radial objective loss {:.6} | angular_loss {:.6} | radial_loss {:.6} | cosine {:.6} | prediction_norm {:.6} | target_norm {:.6} | norm_ratio {:.6} | samples {}",
            tag,
            report.loss,
            report.angular_loss,
            report.radial_loss,
            report.cosine,
            report.prediction_norm,
            report.target_norm,
            report.norm_ratio,
            report.samples,
        );
    }
}
